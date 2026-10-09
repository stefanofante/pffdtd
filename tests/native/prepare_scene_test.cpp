// Native integration contracts for JSON/DEF preparation, including the solver
// HDF5 loader. No reference calculations are delegated to Python.
#include "prepare_scene.h"
#include <nlohmann/json.hpp>
#include <hdf5.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

using namespace pffdtd_prepare;
using nlohmann::json;
namespace {
std::size_t checks=0;
void check(bool value,const std::string &message)
{ ++checks; if (!value) throw std::runtime_error(message); }

template<class F> void rejects(F action,const char *message)
{
   bool rejected=false;
   try { action(); } catch (const std::exception &) { rejected=true; }
   check(rejected,message);
}

void save_json(const std::string &file,const json &document)
{
   std::ofstream out(file);
   check(out.good(),"cannot create test JSON");
   out<<document.dump(2)<<'\n';
   check(out.good(),"cannot save test JSON");
}

json cube_json(bool rigid=false)
{
   const json points={{0,0,0},{4,0,0},{4,4,0},{0,4,0},
                      {0,0,4},{4,0,4},{4,4,4},{0,4,4},
                      {1,1,1},{1.00001,1,1},{1,1.00001,1}};
   const json triangles={{0,2,1},{0,3,2},{4,5,6},{4,6,7},
                         {0,1,5},{0,5,4},{1,2,6},{1,6,5},
                         {2,3,7},{2,7,6},{3,0,4},{3,4,7},{8,9,10}};
   json sides=json::array();
   for (unsigned i=0;i<13;++i) sides.push_back(rigid ? 0 : 3);
   json document;
   document["mats_hash"][rigid ? "_RIGID" : "Wall"]={{"pts",points},{"tris",triangles},{"sides",sides}};
   document["sources"]={{{"xyz",{1.35,1.45,1.55}},{"name","source1"}},
                        {{"xyz",{2.15,2.25,2.35}},{"name","source2"}}};
   document["receivers"]={{{"xyz",{2.5,2.6,2.7}},{"name","receiver1"}},
                          {{"xyz",{1.5,2.4,2.1}},{"name","receiver2"}},
                          {{"xyz",{2.5,2.6,2.7}},{"name","receiver duplicate"}}};
   return document;
}

void save_material(const std::string &file,const std::vector<double> &def,
                   hsize_t columns=3,hid_t type=H5T_IEEE_F64LE)
{
   hid_t f=H5Fcreate(file.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT);
   check(f>=0,"cannot create test material");
   const hsize_t dims[]={columns ? def.size()/columns : 0,columns};
   hid_t s=H5Screate_simple(2,dims,nullptr);
   hid_t d=H5Dcreate2(f,"DEF",type,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
   const double dummy=0;
   check(s>=0 && d>=0 && H5Dwrite(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,
                                def.empty() ? &dummy : def.data())>=0,"cannot write test material");
   H5Dclose(d); H5Sclose(s); H5Fclose(f);
}

void check_prepared(const PreparedScene &scene,const PrepareOptions &options)
{
   check(scene.sources.size()==8 && scene.receivers.size()==24 && scene.steps==7,"communication counts");
   check(scene.mesh.triangles_input==13 && scene.mesh.triangles_removed==1 &&
         scene.mesh.geometry.triangles.size()==12,"triangle area pruning");
   check(!scene.boundary.empty(),"missing cube boundary");
   const bool fcc=options.fcc;
   const unsigned nn=fcc ? 12 : 6;
   const auto dims=scene.physical_grid.dims();
   const auto directions=grid_directions(nn);
   // Undo the storage layout independently and verify every physical link.
   std::vector<BoundaryRecord> physical=scene.boundary;
   for (auto &record:physical) {
      const int64_t index=scene.layout.unmap_index(record.index);
      uint16_t mask=0;
      for (unsigned k=0;k<nn;++k)
         if (record.adjacency&transform_adjacency(scene.layout,index,uint16_t(1)<<k,nn))
            mask|=uint16_t(1)<<k;
      record.index=index; record.adjacency=mask;
   }
   std::sort(physical.begin(),physical.end(),[](const BoundaryRecord &a,const BoundaryRecord &b){return a.index<b.index;});
   for (const auto &record:physical) {
      const auto p=grid_coord(dims,record.index);
      check(!fcc || even_fcc_coord(p),"boundary FCC parity");
      check(record.adjacency || record.material==-1,"isolated nodes must be rigid");
      for (unsigned k=0;k<nn;++k) {
         const GridCoord q{{p[0]+directions[k][0],p[1]+directions[k][1],p[2]+directions[k][2]}};
         const int64_t index=grid_index(dims,q);
         const auto found=std::lower_bound(physical.begin(),physical.end(),index,
                          [](const BoundaryRecord &r,int64_t i){return r.index<i;});
         const bool forward=(record.adjacency&(uint16_t(1)<<k))!=0;
         const bool reverse=found==physical.end() || found->index!=index ||
                           (found->adjacency&(uint16_t(1)<<(k^1)))!=0;
         check(forward==reverse,"physical link reciprocity across layout");
      }
   }
   const auto source=interpolate8(scene.physical_grid,scene.mesh.sources[options.source_num-1].xyz);
   for (std::size_t s=0;s<scene.sources.size();++s) {
      const int64_t physical_index=scene.layout.unmap_index(scene.sources[s]);
      const auto corner=std::find(source.indices.begin(),source.indices.end(),physical_index);
      check(corner!=source.indices.end(),"source storage reorder mapping");
      const std::size_t k=corner-source.indices.begin();
      const double amplitude=source.weights[k]*(fcc ? 0.5 : 1)*scene.constants.l2/scene.constants.h;
      const double y0=amplitude*(options.differentiated ? 2/scene.constants.ts : 1);
      for (int64_t n=0;n<scene.steps;++n) {
         const double expected=options.differentiated ? (n==0 ? y0 : (n&1 ? -2*y0 : 2*y0)) : (n==0 ? y0 : 0);
         const double actual=scene.input[s*scene.steps+n];
         check(std::abs(actual-expected)<=1e-12*std::max(1.0,std::abs(expected)),"true bilinear impulse recurrence");
      }
   }
   for (std::size_t r=0;r<scene.mesh.receivers.size();++r) {
      GridPoint moment{{0,0,0}};
      double sum=0;
      for (unsigned k=0;k<8;++k) {
         const std::size_t sorted=scene.reorder[r*8+k];
         const int64_t physical_index=scene.layout.unmap_index(scene.receivers[sorted]);
         const auto p=grid_coord(dims,physical_index);
         const double weight=scene.weights[r*8+k];
         sum+=weight;
         for (unsigned a=0;a<3;++a) moment[a]+=weight*scene.physical_grid.axes[a][p[a]];
      }
      check(std::abs(sum-1)<1e-14,"receiver weights partition unity");
      for (unsigned a=0;a<3;++a)
         check(std::abs(moment[a]-scene.mesh.receivers[r].xyz[a])<1e-13,"receiver coordinate reproduction and inverse reorder");
   }
}

void near_reconciliation()
{
   PreparedScene scene;
   scene.physical_grid=build_grid(GridPoint{{0,0,0}},GridPoint{{4,4,4}},0.2,false);
   Triangle triangle;
   check(precompute_triangle(triangle,{0,0,0},{0,4,0},{0,0,4},0,3,0),"near triangle precompute");
   scene.mesh.geometry=build_bvh({triangle});
   BoundaryRecord isolated;
   isolated.index=grid_index(scene.physical_grid.dims(),GridCoord{{8,8,8}});
   isolated.adjacency=0; isolated.triangle=0; isolated.near_boundary=true; isolated.material=-1;
   scene.boundary={isolated};
   reconcile_boundary(scene);
   check(scene.boundary.size()==7 && scene.added_boundary_nodes==6 && scene.reconciled_links==6,
         "near isolation repairs exactly six endpoints without propagation");
   for (const auto &record:scene.boundary)
      check(record.material==-1,"artificial incident endpoint must stay rigid");
}
} // namespace

int main(int argc,char **argv)
{
   try {
      check(argc==2,"usage: prepare_scene_test NEW_DIR");
      const std::string root=argv[1];
      check(mkdir(root.c_str(),0755)==0,"test directory must be new");
      const json model=cube_json();
      const std::string jsonfile=root+"/cube.json",materialfile=root+"/wall.h5";
      save_json(jsonfile,model);
      save_material(materialfile,{1e-4,2,1000,2e-4,3,2000});
      const SceneMesh mesh=import_scene(jsonfile);
      const auto materials=import_materials(mesh,{{"Wall",materialfile}});
      check(materials.size()==1 && materials[0].def.size()==6,"passive material import");
      rejects([&]{import_materials(mesh,{});},"missing material accepted");
      rejects([&]{import_materials(mesh,{{"Wrong",materialfile}});},"wrong material label accepted");
      unsigned cases=0;
      for (bool fcc:{false,true})
         for (bool differentiated:{false,true}) {
            PrepareOptions options;
            options.spacing=0.2; options.steps=7; options.fcc=fcc;
            options.differentiated=differentiated; options.target_precision=differentiated ? 32 : 64;
            options.fold=differentiated; options.rotate=differentiated; options.source_num=differentiated ? 2 : 1;
            options.threads=2;
            auto scene=prepare_scene(mesh,materials,options);
            check_prepared(scene,options);
            const std::string output=root+"/case"+std::to_string(cases++);
            write_prepared(scene,output);
            rejects([&]{write_prepared(scene,output);},"existing output directory overwritten");
         }
      near_reconciliation();
      PrepareOptions options; options.spacing=0.2; options.steps=7; options.threads=2;
      options.differentiated=false;
      rejects([&]{prepare_scene(mesh,materials,options);},"FP32 undifferentiated source accepted");
      options.differentiated=true; options.source_num=3;
      rejects([&]{prepare_scene(mesh,materials,options);},"source number beyond table accepted");
      options.source_num=1;
      for (double temperature:{-20.0,0.0,20.0,50.0}) {
         options.temperature=temperature;
         const auto c=make_constants(options);
         check(std::abs(c.c-343.2*std::sqrt((temperature+273.15)/293.15))<1e-12,"Kelvin sound speed");
         check(c.ts>0 && c.l2<=1.0/3,"finite Cartesian constants");
      }
      options.temperature=51;
      rejects([&]{make_constants(options);},"invalid temperature accepted");
      for (unsigned bad=0;bad<6;++bad) {
         auto changed=model;
         if (bad==0) changed["mats_hash"]["Wall"]["tris"][0][0]=-1;
         if (bad==1) changed["mats_hash"]["Wall"]["tris"][0][0]=999999;
         if (bad==2) changed["mats_hash"]["Wall"]["tris"][0][0]=0.5;
         if (bad==3) changed["mats_hash"]["Wall"]["sides"][0]=4;
         if (bad==4) changed["sources"][0]["xyz"][0]=0;
         if (bad==5) changed["receivers"]=json::array();
         const std::string file=root+"/bad"+std::to_string(bad)+".json";
         save_json(file,changed);
         rejects([&]{import_scene(file);},"malformed JSON accepted");
      }
      const std::vector<std::vector<double>> invalid={{-1,1,1},{0,0,0},std::vector<double>(39,1)};
      for (std::size_t i=0;i<invalid.size();++i) {
         const std::string file=root+"/badmaterial"+std::to_string(i)+".h5";
         save_material(file,invalid[i]);
         rejects([&]{import_materials(mesh,{{"Wall",file}});},"invalid DEF accepted");
      }
      const std::string shape=root+"/badshape.h5";
      save_material(shape,{1,2,3,4},2);
      rejects([&]{import_materials(mesh,{{"Wall",shape}});},"invalid DEF columns accepted");
      const std::string type=root+"/badtype.h5";
      save_material(type,{1,2,3},3,H5T_IEEE_F32LE);
      rejects([&]{import_materials(mesh,{{"Wall",type}});},"float32 DEF accepted");
      const std::string empty=root+"/empty.h5";
      save_material(empty,{});
      const auto empty_material=import_materials(mesh,{{"Wall",empty}});
      check(empty_material[0].def.empty(),"zero-branch material rejected");
      options.temperature=20; options.source_num=1;
      const auto empty_scene=prepare_scene(mesh,empty_material,options);
      write_prepared(empty_scene,root+"/zero-branch-output"); ++cases;
      const std::string extreme=root+"/large-inertance.h5";
      save_material(extreme,{1e50,2,0});
      const auto extreme_material=import_materials(mesh,{{"Wall",extreme}});
      rejects([&]{prepare_scene(mesh,extreme_material,options);},"FP32 underflowing material gain accepted");
      options.target_precision=64;
      const auto extreme_scene=prepare_scene(mesh,extreme_material,options);
      write_prepared(extreme_scene,root+"/fp64-gain-output"); ++cases;
      const std::string rigidfile=root+"/rigid.json";
      save_json(rigidfile,cube_json(true));
      const auto rigid=import_scene(rigidfile);
      check(rigid.material_labels.empty() && import_materials(rigid,{}).empty(),"rigid-only material contract");
      options.target_precision=32;
      const auto rigid_scene=prepare_scene(rigid,{},options);
      write_prepared(rigid_scene,root+"/rigid-output"); ++cases;
      std::printf("prepare_scene: PASS %zu checks, %u original native loader HDF round-trips; Cart/FCC folded/unfolded, trilinear moments, true bilinear impulse, reciprocal graph and negative JSON/DEF/output guards.\n",checks,cases);
      return 0;
   }
   catch (const std::exception &e) {
      std::fprintf(stderr,"prepare_scene: FAIL: %s\n",e.what()); return 1;
   }
}
