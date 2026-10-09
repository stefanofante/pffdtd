// Native preparation of exported triangle meshes and passive DEF materials.
// Surface adjacency is a graph of air nodes; meshes need not be watertight.
#ifndef PRECISION
#define PRECISION 2
#endif
#ifndef USING_CUDA
#define USING_CUDA false
#endif
#include <fdtd_data.h>
#include "prepare_scene.h"
#include <nlohmann/json.hpp>
#ifdef __CUDACC__
#include "prepare_cuda.h"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PFFDTD_BUILD_REVISION
#define PFFDTD_BUILD_REVISION "unknown"
#endif

namespace {
using namespace pffdtd_prepare;
using nlohmann::json;

void require(bool condition,const std::string &message)
{ if (!condition) throw std::runtime_error(message); }

std::runtime_error system_error(const std::string &message)
{ return std::runtime_error(message+": "+std::strerror(errno)); }

int64_t checked_product(int64_t a,int64_t b,const char *name)
{
   require(a>=0 && b>=0 && (!b || a<=INT64_MAX/b),std::string(name)+" overflows int64");
   return a*b;
}

std::size_t host_count(int64_t count,std::size_t size,const char *name)
{
   require(count>=0 && static_cast<uint64_t>(count)<=SIZE_MAX/size,
           std::string(name)+" exceeds host address space");
   return static_cast<std::size_t>(count);
}

int64_t total_points(const std::array<int64_t,3> &dims)
{ return checked_product(checked_product(dims[0],dims[1],"grid"),dims[2],"grid"); }

std::array<int64_t,3> coordinates(int64_t index,const std::array<int64_t,3> &dims)
{
   const int64_t yz=checked_product(dims[1],dims[2],"grid stride");
   return {{index/yz,(index%yz)/dims[2],index%dims[2]}};
}

int64_t flatten(const std::array<int64_t,3> &p,const std::array<int64_t,3> &dims)
{ return (p[0]*dims[1]+p[1])*dims[2]+p[2]; }

bool away_from_abc(int64_t index,const std::array<int64_t,3> &dims)
{
   if (index<0 || index>=total_points(dims)) return false;
   const auto p=coordinates(index,dims);
   for (unsigned a=0;a<3;++a) if (p[a]<=1 || p[a]>=dims[a]-2) return false;
   return true;
}

uint16_t all_links(bool fcc) { return static_cast<uint16_t>(fcc ? 4095 : 63); }

std::array<double,3> json_point(const json &value,const std::string &where)
{
   require(value.is_array() && value.size()==3,where+" must contain three coordinates");
   std::array<double,3> result;
   for (unsigned a=0;a<3;++a) {
      require(value[a].is_number(),where+" coordinate must be numeric");
      result[a]=value[a].get<double>();
      require(std::isfinite(result[a]),where+" coordinate must be finite");
   }
   return result;
}

uint64_t json_integer(const json &value,const std::string &where)
{
   require(value.is_number_integer(),where+" must be an integer");
   if (value.is_number_unsigned()) return value.get<uint64_t>();
   const int64_t result=value.get<int64_t>();
   require(result>=0,where+" must be nonnegative");
   return static_cast<uint64_t>(result);
}

class H5Handle {
   hid_t id;
   herr_t (*closer)(hid_t);
public:
   H5Handle(hid_t value,herr_t (*close)(hid_t)):id(value),closer(close)
   { require(id>=0,"HDF5 operation failed"); }
   ~H5Handle() { if (id>=0) closer(id); }
   operator hid_t() const { return id; }
   H5Handle(const H5Handle &)=delete;
   H5Handle &operator=(const H5Handle &)=delete;
   void close()
   {
      const hid_t old=id; id=-1;
      require(old>=0 && closer(old)>=0,"Cannot close HDF5 object");
   }
};

void dataset(hid_t file,const char *name,hid_t stored,hid_t memory,
             const std::vector<hsize_t> &dims,const void *data)
{
   H5Handle space(dims.empty() ? H5Screate(H5S_SCALAR) :
                  H5Screate_simple(static_cast<int>(dims.size()),dims.data(),nullptr),H5Sclose);
   H5Handle value(H5Dcreate2(file,name,stored,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),H5Dclose);
   const double dummy=0;
   require(H5Dwrite(value,memory,H5S_ALL,H5S_ALL,H5P_DEFAULT,data ? data : &dummy)>=0,
           std::string("Cannot write dataset ")+name);
}
void i64(hid_t f,const char *n,int64_t v) { dataset(f,n,H5T_STD_I64LE,H5T_NATIVE_INT64,{},&v); }
void i8(hid_t f,const char *n,int8_t v) { dataset(f,n,H5T_STD_I8LE,H5T_NATIVE_INT8,{},&v); }
void f64(hid_t f,const char *n,double v) { dataset(f,n,H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{},&v); }
void string_dataset(hid_t f,const char *n,const std::string &v)
{
   H5Handle type(H5Tcopy(H5T_C_S1),H5Tclose);
   require(H5Tset_size(type,v.size()+1)>=0 && H5Tset_strpad(type,H5T_STR_NULLTERM)>=0,
           "Cannot create HDF5 string type");
   dataset(f,n,type,type,{},v.c_str());
}

class OutputDirectory {
   std::string path;
   std::vector<std::string> owned;
   bool committed=false;
public:
   explicit OutputDirectory(const std::string &name):path(name)
   { if (mkdir(path.c_str(),0755)!=0) throw system_error("Cannot create new directory "+path); }
   void track(const std::string &file) { owned.push_back(file); }
   void commit() { committed=true; }
   ~OutputDirectory()
   {
      if (!committed) {
         for (const auto &file:owned) unlink(file.c_str());
         rmdir(path.c_str());
      }
   }
};

// The legacy loader uses assertions. Validate in a child so a failed assertion
// or allocation cannot leave a half-written published directory in the parent.
void validate_loader(const PreparedScene &scene,const std::string &directory,bool verbose)
{
   require(std::fflush(nullptr)==0,"Cannot flush streams before native loader validation");
   const pid_t pid=fork();
   if (pid<0) throw system_error("Cannot start native loader validation");
   if (pid==0) {
      try {
         const struct rlimit no_core={0,0};
         require(setrlimit(RLIMIT_CORE,&no_core)==0,"Cannot disable loader core dumps");
         require(chdir(directory.c_str())==0,"Cannot enter prepared directory");
         if (!verbose) {
            const int fd=open("/dev/null",O_WRONLY);
            require(fd>=0 && dup2(fd,STDOUT_FILENO)>=0,"Cannot quiet loader output");
            close(fd);
         }
         SimData data{};
         load_sim_data(&data);
         const auto dims=scene.layout.output_dims();
         require(data.Nx==dims[0] && data.Ny==dims[1] && data.Nz==dims[2] &&
                 data.Nt==scene.steps && data.Ns==static_cast<int64_t>(scene.sources.size()) &&
                 data.Nr==static_cast<int64_t>(scene.receivers.size()) &&
                 data.Nb==static_cast<int64_t>(scene.boundary.size()) &&
                 data.Nm==static_cast<int>(scene.materials.size()),"Loaded counts differ");
         const bool fcc=scene.physical_grid.fcc_flag!=0;
         const int expected_flag=fcc ? (scene.layout.fold_fcc ? 2 : 1) : 0;
         require(data.fcc_flag==expected_flag && data.NN==(fcc ? 12 : 6) &&
                 data.l==scene.constants.l && data.l2==scene.constants.l2,
                 "Loaded lattice coefficients differ");
         int64_t nbl=0;
         for (std::size_t b=0;b<scene.boundary.size();++b) {
            const auto &record=scene.boundary[b];
            require(data.bn_ixyz[b]==record.index && data.adj_bn[b]==record.adjacency,
                    "Loaded boundary indices/adjacency differ");
            if (record.material>=0) {
               const double factor=fcc ? 0.5/std::sqrt(2.0) : 1.0;
               require(nbl<data.Nbl && data.bnl_ixyz[nbl]==record.index &&
                       data.mat_bnl[nbl]==record.material &&
                       data.ssaf_bnl[nbl]==static_cast<Real>(static_cast<Real>(factor)*record.saf),
                       "Loaded boundary material/SAF differs");
               ++nbl;
            }
         }
         require(data.Nbl==nbl,"Loaded lossy boundary count differs");
         for (std::size_t s=0;s<scene.sources.size();++s)
            require(data.in_ixyz[s]==scene.sources[s] &&
                    !std::binary_search(data.bna_ixyz,data.bna_ixyz+data.Nba,scene.sources[s]),
                    "Loaded source differs or intersects ABC");
         for (std::size_t r=0;r<scene.receivers.size();++r)
            require(data.out_ixyz[r]==scene.receivers[r] && data.out_reorder[r]==scene.reorder[r] &&
                    !std::binary_search(data.bna_ixyz,data.bna_ixyz+data.Nba,scene.receivers[r]),
                    "Loaded receiver/reorder differs or intersects ABC");
         require(std::memcmp(data.in_sigs,scene.input.data(),scene.input.size()*sizeof(double))==0,
                 "Loaded input signal differs");
         for (int m=0;m<data.Nm;++m) {
            require(data.Mb[m]==static_cast<int>(scene.materials[m].def.size()/3),
                    "Loaded material branch count differs");
            require(std::isfinite(data.mat_beta[m]),"Loaded material beta is not finite");
            for (int b=0;b<data.Mb[m];++b) {
               const auto &q=data.mat_quads[m*MMb+b];
               require(std::isfinite(q.b) && std::isfinite(q.bd) && std::isfinite(q.bDh) &&
                       std::isfinite(q.bFh) && q.b>0 && std::abs(q.bd)<=1 && q.bDh>=0 && q.bFh>=0,
                       "Loaded material coefficient is not finite/passive");
            }
         }
         free_sim_data(&data);
         std::fflush(nullptr);
         _exit(EXIT_SUCCESS);
      }
      catch (const std::exception &e) {
         std::fprintf(stderr,"Native loader validation: %s\n",e.what());
         std::fflush(nullptr); _exit(EXIT_FAILURE);
      }
   }
   int status=0;
   pid_t result;
   do { result=waitpid(pid,&status,0); } while (result<0 && errno==EINTR);
   if (result<0) throw system_error("Cannot wait for native loader validation");
   require(WIFEXITED(status) && WEXITSTATUS(status)==0,
           "Original native solver loader rejected prepared HDF5 inputs (status "+std::to_string(status)+")");
}

} // namespace

namespace pffdtd_prepare {

SceneMesh import_scene(const std::string &filename,double min_triangle_area)
{
   require(std::isfinite(min_triangle_area) && min_triangle_area>=0,"Minimum triangle area must be finite and nonnegative");
   std::ifstream input(filename);
   require(input.good(),"Cannot open model JSON "+filename);
   json document;
   input>>document;
   require(document.is_object() && document.contains("mats_hash") && document["mats_hash"].is_object(),
           "JSONRoomExport must contain mats_hash object");
   require(document.contains("sources") && document["sources"].is_array() && !document["sources"].empty() &&
           document.contains("receivers") && document["receivers"].is_array() && !document["receivers"].empty(),
           "Model must contain nonempty sources and receivers arrays");
   const auto &groups=document["mats_hash"];
   require(!groups.empty(),"Model has no material geometry");
   SceneMesh scene;
   scene.bmin.fill(std::numeric_limits<double>::infinity());
   scene.bmax.fill(-std::numeric_limits<double>::infinity());
   std::vector<std::string> labels;
   for (auto it=groups.begin();it!=groups.end();++it) labels.push_back(it.key());
   std::sort(labels.begin(),labels.end());
   const auto rigid=std::find(labels.begin(),labels.end(),"_RIGID");
   if (rigid!=labels.end()) { labels.erase(rigid); labels.push_back("_RIGID"); }
   for (const auto &label:labels) if (label!="_RIGID") scene.material_labels.push_back(label);
   require(scene.material_labels.size()<=MNm,"More than 64 non-rigid materials are unsupported by the solver");
   std::vector<Triangle> triangles;
   uint64_t ordinal=0;
   for (std::size_t m=0;m<labels.size();++m) {
      const std::string &label=labels[m];
      const auto &group=groups[label];
      require(group.is_object() && group.contains("pts") && group["pts"].is_array() &&
              group.contains("tris") && group["tris"].is_array() &&
              group.contains("sides") && group["sides"].is_array(),
              "Material "+label+" needs pts, tris and sides arrays");
      require(group["tris"].size()==group["sides"].size(),"Triangle/side count differs for "+label);
      std::vector<Vec3> vertices;
      vertices.reserve(group["pts"].size());
      for (const auto &vertex:group["pts"]) {
         const auto xyz=json_point(vertex,"Vertex in "+label);
         vertices.push_back(Vec3{xyz[0],xyz[1],xyz[2]});
         for (unsigned a=0;a<3;++a) {
            scene.bmin[a]=std::min(scene.bmin[a],xyz[a]);
            scene.bmax[a]=std::max(scene.bmax[a],xyz[a]);
         }
      }
      for (std::size_t t=0;t<group["tris"].size();++t,++ordinal) {
         require(ordinal<UINT32_MAX,"Triangle ordinals exceed uint32 range");
         const auto &indices=group["tris"][t];
         require(indices.is_array() && indices.size()==3,"Triangle in "+label+" needs three vertex indices");
         uint64_t ix[3];
         for (unsigned a=0;a<3;++a) {
            ix[a]=json_integer(indices[a],"Triangle index in "+label);
            require(ix[a]<vertices.size(),"Triangle index is outside vertex table for "+label);
         }
         const uint64_t side=json_integer(group["sides"][t],"Triangle side in "+label);
         require(side<=3,"Triangle sidedness must be in 0..3 for "+label);
         require(label!="_RIGID" || side==0,"_RIGID triangles must have side 0");
         Triangle triangle;
         if (precompute_triangle(triangle,vertices[ix[0]],vertices[ix[1]],vertices[ix[2]],
                                  label=="_RIGID" ? -1 : static_cast<int32_t>(m),
                                  static_cast<uint8_t>(side),static_cast<uint32_t>(ordinal),min_triangle_area))
            triangles.push_back(triangle);
         else ++scene.triangles_removed;
      }
   }
   scene.triangles_input=static_cast<std::size_t>(ordinal);
   require(!triangles.empty(),"No nondegenerate triangles survive the area threshold");
   for (unsigned a=0;a<3;++a)
      require(std::isfinite(scene.bmin[a]) && std::isfinite(scene.bmax[a]) && scene.bmin[a]<scene.bmax[a],
              "Model bounds must have a finite positive extent in every axis");
   for (const char *kind:{"sources","receivers"}) {
      auto &points=std::strcmp(kind,"sources")==0 ? scene.sources : scene.receivers;
      for (const auto &entry:document[kind]) {
         require(entry.is_object() && entry.contains("xyz"),std::string(kind)+" entry needs xyz");
         NamedPoint point;
         point.xyz=json_point(entry["xyz"],kind);
         if (entry.contains("name")) {
            require(entry["name"].is_string(),std::string(kind)+" name must be text");
            point.name=entry["name"].get<std::string>();
         }
         for (unsigned a=0;a<3;++a)
            require(point.xyz[a]>scene.bmin[a] && point.xyz[a]<scene.bmax[a],
                    std::string(kind)+" position must be strictly inside the model bounding box");
         points.push_back(point);
      }
   }
   scene.geometry=build_bvh(std::move(triangles));
   return scene;
}

std::vector<MaterialDef> import_materials(const SceneMesh &scene,
                                         const std::map<std::string,std::string> &files)
{
   require(files.size()==scene.material_labels.size(),"Provide exactly one --material label=FILE for every non-rigid model label");
   std::vector<MaterialDef> result;
   for (const auto &label:scene.material_labels) {
      const auto mapping=files.find(label);
      require(mapping!=files.end(),"Missing --material mapping for "+label);
      MaterialDef material; material.label=label; material.filename=mapping->second;
      H5Handle file(H5Fopen(material.filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
      H5Handle data(H5Dopen2(file,"DEF",H5P_DEFAULT),H5Dclose);
      H5Handle space(H5Dget_space(data),H5Sclose);
      H5Handle type(H5Dget_type(data),H5Tclose);
      hsize_t dims[2]={0,0};
      require(H5Tget_class(type)==H5T_FLOAT && H5Tget_size(type)==sizeof(double),
              "DEF must be float64 in "+material.filename);
      require(H5Sget_simple_extent_ndims(space)==2 && H5Sget_simple_extent_dims(space,dims,nullptr)>=0 &&
              dims[0]<=MMb && dims[1]==3,"DEF must have shape [0..12,3] in "+material.filename);
      material.def.resize(static_cast<std::size_t>(dims[0])*3);
      double dummy=0;
      require(H5Dread(data,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,
                      material.def.empty() ? &dummy : material.def.data())>=0,
              "Cannot read DEF in "+material.filename);
      for (std::size_t b=0;b<material.def.size()/3;++b) {
         bool positive=false;
         for (unsigned c=0;c<3;++c) {
            const double value=material.def[b*3+c];
            require(std::isfinite(value) && value>=0,"DEF branch must be finite and nonnegative for "+label);
            positive=positive || value>0;
         }
         require(positive,"DEF branch cannot have D=E=F=0 for "+label);
      }
      result.push_back(std::move(material));
   }
   return result;
}

SimConstants make_constants(const PrepareOptions &options)
{
   require(std::isfinite(options.temperature) && options.temperature>=-20 && options.temperature<=50,
           "Temperature must be in -20..50 Celsius");
   require(std::isfinite(options.humidity) && options.humidity>=10 && options.humidity<=100,
           "Humidity must be in 10..100 percent");
   require(std::isfinite(options.spacing) && options.spacing>=0,"Spacing must be finite and nonnegative");
   SimConstants constants;
   constants.temperature=options.temperature; constants.humidity=options.humidity;
   // Kelvin correction: the former sqrt(Tc/20) becomes invalid below zero C.
   constants.c=343.2*std::sqrt((options.temperature+273.15)/293.15);
   constants.l=(options.fcc ? 1.0 : 1.0/std::sqrt(3.0))*0.999;
   constants.l2=constants.l*constants.l;
   if (options.spacing>0) constants.h=options.spacing;
   else {
      require(std::isfinite(options.fmax) && options.fmax>0 && std::isfinite(options.ppw) && options.ppw>0,
              "fmax and PPW must be finite and positive");
      constants.h=constants.c/(options.fmax*options.ppw);
   }
   constants.ts=constants.h/constants.c*constants.l;
   constants.fs=1.0/constants.ts;
   require(std::isfinite(constants.h) && constants.h>0 && std::isfinite(constants.ts) &&
           constants.ts>0 && std::isfinite(constants.fs) && constants.fs>0,
           "Grid constants overflow or underflow");
   return constants;
}

std::vector<BoundaryRecord> voxelize_cpu(const FlatBvh &geometry,const Grid &grid,unsigned threads)
{
   validate_grid(grid);
   require(grid.h<=0.125*std::numeric_limits<double>::max(),"Mesh link lengths exceed FP64 range");
   const auto dims=grid.dims();
   const bool fcc=grid.fcc_flag!=0;
   require(grid.fcc_flag==0 || grid.fcc_flag==1,"Voxelization requires an unfolded physical grid");
   const int64_t count=total_points(dims);
   host_count(count,sizeof(bool),"native loader grid mask");
   unsigned workers=1;
#ifdef _OPENMP
   workers=threads ? threads : static_cast<unsigned>(omp_get_max_threads());
   require(workers>0 && workers<=static_cast<unsigned>(INT_MAX),"Invalid CPU thread count");
#else
   require(threads<=1,"This preparer was compiled without OpenMP; use --threads 1");
#endif
   workers=std::min(workers,static_cast<unsigned>(std::min<int64_t>(dims[0]-2,INT_MAX)));
   std::vector<std::vector<BoundaryRecord>> partial(workers);
   std::exception_ptr failure;
#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
#endif
   {
      unsigned worker=0;
#ifdef _OPENMP
      worker=static_cast<unsigned>(omp_get_thread_num());
#pragma omp for schedule(static)
#endif
      for (int64_t x=1;x<dims[0]-1;++x) {
         try {
            for (int64_t y=1;y<dims[1]-1;++y)
               for (int64_t z=1;z<dims[2]-1;++z) {
                  if (fcc && ((x+y+z)&1)) continue;
                  const Vec3 point{grid.axes[0][x],grid.axes[1][y],grid.axes[2][z]};
                  const auto classified=classify_node(geometry,point,grid.h,fcc);
                  if (classified.adjacency==all_links(fcc)) continue;
                  BoundaryRecord record;
                  record.index=(x*dims[1]+y)*dims[2]+z;
                  record.adjacency=classified.adjacency;
                  record.triangle=classified.nearest_triangle;
                  record.material=classified.material; record.saf=classified.saf;
                  record.near_boundary=classified.near_boundary!=0;
                  partial[worker].push_back(record);
               }
         }
         catch (...) {
#ifdef _OPENMP
#pragma omp critical(pffdtd_prepare_failure)
#endif
            { if (!failure) failure=std::current_exception(); }
         }
      }
   }
   if (failure) std::rethrow_exception(failure);
   std::vector<BoundaryRecord> result;
   std::size_t size=0;
   for (const auto &part:partial) {
      require(size<=SIZE_MAX-part.size(),"Boundary count exceeds host address space");
      size+=part.size();
   }
   result.reserve(size);
   for (auto &part:partial) result.insert(result.end(),part.begin(),part.end());
   std::sort(result.begin(),result.end(),[](const BoundaryRecord &a,const BoundaryRecord &b){return a.index<b.index;});
   return result;
}

void reconcile_boundary(PreparedScene &scene)
{
   const auto dims=scene.physical_grid.dims();
   const bool fcc=scene.physical_grid.fcc_flag!=0;
   const unsigned nn=fcc ? 12 : 6;
   auto &boundary=scene.boundary;
   std::sort(boundary.begin(),boundary.end(),[](const BoundaryRecord &a,const BoundaryRecord &b){return a.index<b.index;});
   std::unordered_map<int64_t,std::size_t> lookup;
   lookup.reserve(boundary.size());
   for (std::size_t b=0;b<boundary.size();++b)
      require(lookup.emplace(boundary[b].index,b).second,"Voxelizer emitted duplicate boundary indices");
   const std::size_t original=boundary.size();
   for (std::size_t b=0;b<original;++b) {
      // Copy before appending: vector reallocation must not invalidate a record.
      const BoundaryRecord record=boundary[b];
      require(record.triangle<scene.mesh.geometry.triangles.size(),"Boundary has no valid nearest triangle");
      const auto p=coordinates(record.index,dims);
      require(away_from_abc(record.index,dims),"A mesh boundary intersects the absorbing boundary halo");
      for (unsigned k=0;k<nn;++k) {
         if (record.adjacency&(uint16_t(1)<<k)) continue;
         const Vec3 v=mesh_direction(k,fcc);
         const std::array<int64_t,3> q{{p[0]+static_cast<int64_t>(v.x),p[1]+static_cast<int64_t>(v.y),
                                       p[2]+static_cast<int64_t>(v.z)}};
         for (unsigned a=0;a<3;++a)
            require(q[a]>1 && q[a]<dims[a]-2,"A removed mesh link reaches an absorbing boundary cell");
         const int64_t other=flatten(q,dims);
         auto found=lookup.find(other);
         if (found==lookup.end()) {
            BoundaryRecord added;
            added.index=other; added.adjacency=all_links(fcc); added.triangle=record.triangle;
            // Near-node isolation adds graph cuts that need not intersect a
            // triangle. Their previously regular endpoints are kept rigid.
            added.material=-1;
            const std::size_t index=boundary.size();
            boundary.push_back(added);
            found=lookup.emplace(other,index).first;
            ++scene.added_boundary_nodes;
         }
         auto &opposite=boundary[found->second];
         const uint16_t bit=static_cast<uint16_t>(uint16_t(1)<<(k^1));
         if (opposite.adjacency&bit) {
            opposite.adjacency=static_cast<uint16_t>(opposite.adjacency&~bit);
            ++scene.reconciled_links;
         }
      }
   }
   for (auto &record:boundary) {
      const auto &triangle=scene.mesh.geometry.triangles[record.triangle];
      if (!record.adjacency) record.material=-1;
      record.saf=surface_factor(triangle.normal,record.adjacency,fcc);
      require(std::isfinite(record.saf) && record.saf>=0,"Surface factor is invalid after graph reconciliation");
   }
   std::sort(boundary.begin(),boundary.end(),[](const BoundaryRecord &a,const BoundaryRecord &b){return a.index<b.index;});
   // Exact reciprocity, including ordinary nodes absent from the sparse list.
   for (const auto &record:boundary) {
      const auto p=coordinates(record.index,dims);
      for (unsigned k=0;k<nn;++k) {
         const Vec3 v=mesh_direction(k,fcc);
         const std::array<int64_t,3> q{{p[0]+static_cast<int64_t>(v.x),p[1]+static_cast<int64_t>(v.y),
                                       p[2]+static_cast<int64_t>(v.z)}};
         const int64_t other=flatten(q,dims);
         const auto found=std::lower_bound(boundary.begin(),boundary.end(),other,
                     [](const BoundaryRecord &r,int64_t index){return r.index<index;});
         const bool connected=(record.adjacency&(uint16_t(1)<<k))!=0;
         const bool reverse=found==boundary.end() || found->index!=other ||
                            (found->adjacency&(uint16_t(1)<<(k^1)))!=0;
         require(connected==reverse,"Mesh adjacency is not reciprocal after conservative reconciliation");
      }
   }
}

void prepare_comms(PreparedScene &scene,const PrepareOptions &options)
{
   require(options.source_num>=1 && options.source_num<=scene.mesh.sources.size(),"source_num is 1-based and exceeds model sources");
   require(options.steps>=0,"Step count must be nonnegative");
   require(options.target_precision==32 || options.target_precision==64,"Target precision must be 32 or 64");
   require(options.target_precision!=32 || options.differentiated,"FP32 requires a differentiated source");
   scene.differentiated=options.differentiated; scene.target_precision=options.target_precision;
   if (options.steps>0) scene.steps=options.steps;
   else {
      require(std::isfinite(options.duration) && options.duration>0,"Duration must be finite and positive");
      const double count=std::ceil(options.duration/scene.constants.ts);
      require(std::isfinite(count) && count>=1 && count<static_cast<double>(INT64_MAX),"Duration produces an invalid step count");
      scene.steps=static_cast<int64_t>(count);
   }
   const auto require_air_stencil=[&](const Interp8 &stencil,const std::string &point) {
      for (unsigned corner=0;corner<8;++corner) {
         const int64_t index=stencil.indices[corner];
         const auto found=std::lower_bound(scene.boundary.begin(),scene.boundary.end(),index,
                     [](const BoundaryRecord &r,int64_t i){return r.index<i;});
         require(found==scene.boundary.end() || found->index!=index,
                 point+" interpolation corner "+std::to_string(corner)+" intersects a mesh boundary (grid index "+
                 std::to_string(index)+"); refine --spacing or move the point clear of surfaces");
      }
   };
   const auto source=interpolate8(scene.physical_grid,scene.mesh.sources[options.source_num-1].xyz,true);
   require_air_stencil(source,"Source "+std::to_string(options.source_num)+" "+scene.mesh.sources[options.source_num-1].name);
   scene.sources.assign(source.indices.begin(),source.indices.end());
   const int64_t signal_count=checked_product(8,scene.steps,"source signal");
   scene.input.assign(host_count(signal_count,sizeof(double),"source signal"),0);
   const double scale=(options.fcc ? 0.5 : 1.0)*scene.constants.l2/scene.constants.h;
   for (unsigned s=0;s<8;++s) {
      double previous_input=0,previous_output=0;
      for (int64_t n=0;n<scene.steps;++n) {
         const double value=n==0 ? source.weights[s]*scale : 0;
         double output=value;
         if (options.differentiated)
            output=(2.0/scene.constants.ts)*(value-previous_input)-previous_output;
         require(std::isfinite(output),"Differentiated source overflows");
         scene.input[static_cast<std::size_t>(s)*scene.steps+n]=output;
         previous_input=value; previous_output=output;
      }
   }
   require(scene.mesh.receivers.size()<=static_cast<uint64_t>(INT64_MAX)/8,"Receiver count overflows");
   const int64_t nr=static_cast<int64_t>(scene.mesh.receivers.size())*8;
   host_count(checked_product(nr,scene.steps,"native loader output"),sizeof(double),"native loader output");
   scene.receivers.reserve(host_count(nr,sizeof(int64_t),"receiver indices"));
   scene.weights.reserve(host_count(nr,sizeof(double),"receiver weights"));
   for (std::size_t r=0;r<scene.mesh.receivers.size();++r) {
      const auto &receiver=scene.mesh.receivers[r];
      const auto interp=interpolate8(scene.physical_grid,receiver.xyz,true);
      require_air_stencil(interp,"Receiver "+std::to_string(r+1)+" "+receiver.name);
      scene.receivers.insert(scene.receivers.end(),interp.indices.begin(),interp.indices.end());
      scene.weights.insert(scene.weights.end(),interp.weights.begin(),interp.weights.end());
   }
}

void apply_layout(PreparedScene &scene,const PrepareOptions &options)
{
   const auto full_dims=scene.physical_grid.dims();
   const AxisPermutation permutation=options.rotate ? longest_axis_permutation(full_dims) : AxisPermutation{{0,1,2}};
   scene.layout=IndexTransform(full_dims,permutation,options.fcc && options.fold);
   for (auto &record:scene.boundary) {
      const int64_t original=record.index;
      record.adjacency=transform_adjacency(scene.layout,original,record.adjacency,options.fcc ? 12 : 6);
      record.index=scene.layout.map_index(original);
   }
   std::sort(scene.boundary.begin(),scene.boundary.end(),[](const BoundaryRecord &a,const BoundaryRecord &b){return a.index<b.index;});
   for (auto &index:scene.sources) index=scene.layout.map_index(index);
   std::vector<std::size_t> source_order(scene.sources.size());
   for (std::size_t s=0;s<source_order.size();++s) source_order[s]=s;
   std::stable_sort(source_order.begin(),source_order.end(),[&](std::size_t a,std::size_t b){return scene.sources[a]<scene.sources[b];});
   std::vector<int64_t> sources(scene.sources.size());
   std::vector<double> signals(scene.input.size());
   for (std::size_t s=0;s<source_order.size();++s) {
      sources[s]=scene.sources[source_order[s]];
      std::copy(scene.input.begin()+source_order[s]*scene.steps,
                scene.input.begin()+(source_order[s]+1)*scene.steps,signals.begin()+s*scene.steps);
   }
   scene.sources.swap(sources); scene.input.swap(signals);
   for (auto &index:scene.receivers) index=scene.layout.map_index(index);
   std::vector<std::size_t> receiver_order(scene.receivers.size());
   for (std::size_t r=0;r<receiver_order.size();++r) receiver_order[r]=r;
   std::stable_sort(receiver_order.begin(),receiver_order.end(),[&](std::size_t a,std::size_t b){return scene.receivers[a]<scene.receivers[b];});
   std::vector<int64_t> receivers(scene.receivers.size());
   scene.reorder.resize(scene.receivers.size());
   for (std::size_t r=0;r<receiver_order.size();++r) {
      receivers[r]=scene.receivers[receiver_order[r]];
      scene.reorder[receiver_order[r]]=static_cast<int64_t>(r);
   }
   scene.receivers.swap(receivers);
}

void validate_prepared(const PreparedScene &scene)
{
   const auto dims=scene.layout.output_dims();
   const int64_t npts=total_points(dims);
   host_count(npts,sizeof(bool),"native loader grid mask");
   require(scene.steps>=1 && scene.sources.size()==8 && !scene.receivers.empty() &&
           scene.receivers.size()==scene.mesh.receivers.size()*8 &&
           scene.weights.size()==scene.receivers.size() && scene.reorder.size()==scene.receivers.size(),
           "Invalid prepared communication shapes");
   require(scene.target_precision==32 || scene.target_precision==64,"Prepared target precision is invalid");
   require(scene.target_precision!=32 || scene.differentiated,"FP32 input must be differentiated");
   require(scene.input.size()==host_count(checked_product(8,scene.steps,"source signals"),sizeof(double),"source signals"),
           "Invalid prepared input shape");
   require(std::is_sorted(scene.sources.begin(),scene.sources.end()) &&
           std::adjacent_find(scene.sources.begin(),scene.sources.end())==scene.sources.end(),
           "Source grid indices must be sorted and distinct");
   require(std::is_sorted(scene.receivers.begin(),scene.receivers.end()),"Receiver indices must be sorted");
   std::vector<uint8_t> reorder_seen(scene.reorder.size(),0);
   for (int64_t index:scene.reorder) {
      require(index>=0 && static_cast<uint64_t>(index)<reorder_seen.size() && !reorder_seen[index],
              "Receiver reorder is not a permutation");
      reorder_seen[index]=1;
   }
   int64_t previous=-1;
   const uint16_t full=all_links(scene.physical_grid.fcc_flag!=0);
   const bool fcc=scene.physical_grid.fcc_flag!=0;
   const unsigned nn=fcc ? 12 : 6;
   require(scene.layout.full_dims==scene.physical_grid.dims() && (!scene.layout.fold_fcc || fcc),
           "Storage layout does not match physical grid");
   uint16_t direction_bits[2][12]{};
   const auto permuted_dims=scene.layout.permuted_dims();
   for (unsigned reflected=0;reflected<2;++reflected) {
      std::array<int64_t,3> representative{{0,0,0}};
      if (reflected && scene.layout.fold_fcc)
         representative[scene.layout.permutation[1]]=permuted_dims[1]-2;
      const int64_t index=flatten(representative,scene.layout.full_dims);
      for (unsigned k=0;k<nn;++k)
         direction_bits[reflected][k]=transform_adjacency(scene.layout,index,uint16_t(1)<<k,nn);
   }
   for (const auto &record:scene.boundary) {
      require(record.index>previous && record.index<npts && record.adjacency!=full &&
              !(record.adjacency&~full),"Boundary indices/adjacency are invalid");
      require(record.material>=-1 && record.material<static_cast<int32_t>(scene.materials.size()) &&
              std::isfinite(record.saf) && record.saf>=0,"Boundary material/SAF is invalid");
      require(record.adjacency || record.material==-1,"A fully disconnected boundary node must be rigid");
      const int64_t physical=scene.layout.unmap_index(record.index);
      require(away_from_abc(physical,scene.layout.full_dims),"Boundary is outside physical interior or overlaps ABC");
      previous=record.index;
   }
   for (const auto &record:scene.boundary) {
      const int64_t physical=scene.layout.unmap_index(record.index);
      const auto p=coordinates(physical,scene.layout.full_dims);
      const unsigned reflected=scene.layout.fold_fcc && p[scene.layout.permutation[1]]>=permuted_dims[1]/2;
      for (unsigned k=0;k<nn;++k) {
         const Vec3 direction=mesh_direction(k,fcc);
         const std::array<int64_t,3> q{{p[0]+static_cast<int64_t>(direction.x),p[1]+static_cast<int64_t>(direction.y),
                                       p[2]+static_cast<int64_t>(direction.z)}};
         const int64_t other_physical=flatten(q,scene.layout.full_dims);
         const int64_t other=scene.layout.map_index(other_physical);
         const auto found=std::lower_bound(scene.boundary.begin(),scene.boundary.end(),other,
                     [](const BoundaryRecord &r,int64_t i){return r.index<i;});
         const unsigned other_reflected=scene.layout.fold_fcc && q[scene.layout.permutation[1]]>=permuted_dims[1]/2;
         const uint16_t mapped=direction_bits[reflected][k];
         const uint16_t opposite=direction_bits[other_reflected][k^1];
         const bool forward=(record.adjacency&mapped)!=0;
         const bool reverse=found==scene.boundary.end() || found->index!=other || (found->adjacency&opposite)!=0;
         require(forward==reverse,"Stored boundary adjacency fails exact physical reciprocity after layout transform");
      }
   }
   for (const auto *indices:{&scene.sources,&scene.receivers})
      for (int64_t index:*indices) {
         require(index>=0 && index<npts,"Communication index is outside solver grid");
         require(away_from_abc(scene.layout.unmap_index(index),scene.layout.full_dims),
                 "Communication stencil overlaps physical ABC/halo");
         const auto found=std::lower_bound(scene.boundary.begin(),scene.boundary.end(),index,
                     [](const BoundaryRecord &r,int64_t i){return r.index<i;});
         require(found==scene.boundary.end() || found->index!=index,"Communication index intersects boundary");
      }
   for (double v:scene.input) require(std::isfinite(v),"Input signal is not finite");
   for (std::size_t r=0;r<scene.weights.size();r+=8) {
      double sum=0;
      for (unsigned k=0;k<8;++k) {
         require(std::isfinite(scene.weights[r+k]) && scene.weights[r+k]>=0 && scene.weights[r+k]<=1,
                 "Receiver weight is invalid");
         sum+=scene.weights[r+k];
      }
      require(std::abs(sum-1)<=64*std::numeric_limits<double>::epsilon(),"Receiver weights do not sum to one");
   }
   for (const auto &material:scene.materials) {
      require(material.def.size()%3==0 && material.def.size()/3<=MMb,"Material DEF shape is invalid");
      double beta=0;
      float beta32=0;
      for (std::size_t b=0;b<material.def.size()/3;++b) {
         const double d=material.def[b*3]/scene.constants.ts,e=material.def[b*3+1],f=material.def[b*3+2]*scene.constants.ts;
         const double denominator=2*d+e+0.5*f;
         require(std::isfinite(d) && d>=0 && std::isfinite(e) && e>=0 && std::isfinite(f) && f>=0 &&
                 std::isfinite(denominator) && denominator>0,"Discretized material branch is invalid");
         const double gain=1/denominator;
         const double values[]={gain,gain*(2*d-e-0.5*f),gain*d,gain*f};
         for (double value:values)
            require(std::isfinite(value) && (scene.target_precision!=32 || std::isfinite(static_cast<float>(value))),
                    "Material coefficient is not representable in target precision");
         require(gain>0 && (scene.target_precision!=32 || static_cast<float>(gain)>0),"Material gain underflows in target precision");
         beta+=gain;
         if (scene.target_precision==32) beta32+=static_cast<float>(gain);
      }
      require(std::isfinite(beta) && std::isfinite(beta32),"Material beta overflows");
   }
}

PreparedScene prepare_scene(SceneMesh mesh,const std::vector<MaterialDef> &materials,
                            const PrepareOptions &options,Voxelizer voxelizer)
{
   require(voxelizer!=nullptr,"Voxelization backend is missing");
   require(materials.size()==mesh.material_labels.size(),"Material table does not match mesh labels");
   for (std::size_t m=0;m<materials.size();++m)
      require(materials[m].label==mesh.material_labels[m],"Material table label order differs from mesh labels");
   require(options.source_num>=1 && options.source_num<=mesh.sources.size(),"source_num is 1-based and exceeds model sources");
   require(options.steps>=0 && (options.steps>0 || (std::isfinite(options.duration) && options.duration>0)),
           "Specify a positive step count or duration");
   require(options.target_precision==32 || options.target_precision==64,"Target precision must be 32 or 64");
   require(options.target_precision!=32 || options.differentiated,"FP32 requires a differentiated source");
   PreparedScene scene;
   scene.mesh=std::move(mesh); scene.materials=materials;
   scene.constants=make_constants(options);
   scene.physical_grid=build_grid(scene.mesh.bmin,scene.mesh.bmax,scene.constants.h,options.fcc);
   // Fail before classifying the full volume when a communication stencil
   // overlaps geometry. Check reverse cuts too: conservative reconciliation
   // can turn a previously regular point into a rigid boundary endpoint.
   const unsigned nn=options.fcc ? 12 : 6;
   const auto preflight=[&](const NamedPoint &point,const std::string &label) {
      const auto stencil=interpolate8(scene.physical_grid,point.xyz,true);
      const auto dims=scene.physical_grid.dims();
      for (unsigned corner=0;corner<8;++corner) {
         const auto p=coordinates(stencil.indices[corner],dims);
         const Vec3 xyz{scene.physical_grid.axes[0][p[0]],scene.physical_grid.axes[1][p[1]],scene.physical_grid.axes[2][p[2]]};
         bool air=classify_node(scene.mesh.geometry,xyz,scene.constants.h,options.fcc).adjacency==all_links(options.fcc);
         for (unsigned k=0;air && k<nn;++k) {
            const Vec3 direction=mesh_direction(k,options.fcc);
            const std::array<int64_t,3> q{{p[0]+static_cast<int64_t>(direction.x),p[1]+static_cast<int64_t>(direction.y),
                                          p[2]+static_cast<int64_t>(direction.z)}};
            const Vec3 neighbor{scene.physical_grid.axes[0][q[0]],scene.physical_grid.axes[1][q[1]],scene.physical_grid.axes[2][q[2]]};
            air=(classify_node(scene.mesh.geometry,neighbor,scene.constants.h,options.fcc).adjacency&(uint16_t(1)<<(k^1)))!=0;
         }
         require(air,label+" "+point.name+" interpolation corner "+std::to_string(corner)+
                 " intersects a mesh boundary (grid index "+std::to_string(stencil.indices[corner])+
                 "); refine --spacing or move the point clear of surfaces");
      }
   };
   preflight(scene.mesh.sources[options.source_num-1],"Source "+std::to_string(options.source_num));
   for (std::size_t r=0;r<scene.mesh.receivers.size();++r)
      preflight(scene.mesh.receivers[r],"Receiver "+std::to_string(r+1));
   scene.boundary=voxelizer(scene.mesh.geometry,scene.physical_grid,options.threads);
   reconcile_boundary(scene);
   prepare_comms(scene,options);
   apply_layout(scene,options);
   validate_prepared(scene);
   return scene;
}

void write_prepared(const PreparedScene &scene,const std::string &directory,bool verbose)
{
   validate_prepared(scene);
   OutputDirectory output(directory);
   const auto dims=scene.layout.output_dims();
   const bool fcc=scene.physical_grid.fcc_flag!=0;
   const unsigned nn=fcc ? 12 : 6;
   std::vector<int64_t> indices;
   std::vector<int8_t> adjacency,materials,poles;
   std::vector<double> area;
   indices.reserve(scene.boundary.size()); materials.reserve(scene.boundary.size()); area.reserve(scene.boundary.size());
   adjacency.reserve(host_count(checked_product(static_cast<int64_t>(scene.boundary.size()),nn,"boundary adjacency"),1,"boundary adjacency"));
   for (const auto &b:scene.boundary) {
      indices.push_back(b.index); materials.push_back(static_cast<int8_t>(b.material)); area.push_back(b.saf);
      for (unsigned k=0;k<nn;++k) adjacency.push_back(static_cast<int8_t>((b.adjacency>>k)&1));
   }
   for (const auto &m:scene.materials) poles.push_back(static_cast<int8_t>(m.def.size()/3));
   const char *names[]={"sim_consts.h5","vox_out.h5","comms_out.h5","sim_mats.h5","cart_grid.h5"};
   for (unsigned which=0;which<5;++which) {
      const std::string filename=directory+"/"+names[which];
      H5Handle file(H5Fcreate(filename.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
      output.track(filename);
      if (which==0) {
         const auto &c=scene.constants;
         f64(file,"c",c.c); f64(file,"h",c.h); f64(file,"Ts",c.ts); f64(file,"SR",c.fs);
         f64(file,"l",c.l); f64(file,"l2",c.l2); f64(file,"Tc",c.temperature); f64(file,"rh",c.humidity);
         i8(file,"fcc_flag",fcc ? (scene.layout.fold_fcc ? 2 : 1) : 0);
         string_dataset(file,"build_revision",PFFDTD_BUILD_REVISION);
         string_dataset(file,"prepare_backend",scene.backend);
         string_dataset(file,"geometry_policy","double_bvh_fuzzy_edges_reciprocal_union_v1");
         string_dataset(file,"surface_factor_policy","sum_each_blocked_link_v1");
         i64(file,"triangles_input",scene.mesh.triangles_input);
         i64(file,"triangles_retained",scene.mesh.geometry.triangles.size());
         i64(file,"triangles_removed",scene.mesh.triangles_removed);
         i64(file,"reconciled_links",scene.reconciled_links); i64(file,"added_boundary_nodes",scene.added_boundary_nodes);
         i64(file,"target_precision",scene.target_precision);
      }
      else if (which==1) {
         i64(file,"Nx",dims[0]); i64(file,"Ny",dims[1]); i64(file,"Nz",dims[2]); i64(file,"Nb",indices.size());
         const hsize_t nb=indices.size();
         dataset(file,"bn_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{nb},indices.data());
         dataset(file,"adj_bn",H5T_STD_I8LE,H5T_NATIVE_INT8,{nb,nn},adjacency.data());
         dataset(file,"mat_bn",H5T_STD_I8LE,H5T_NATIVE_INT8,{nb},materials.data());
         dataset(file,"saf_bn",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{nb},area.data());
         f64(file,"h",scene.constants.h);
         const char *axes[]={"xv","yv","zv"};
         for (unsigned a=0;a<3;++a) {
            const auto &v=scene.physical_grid.axes[scene.layout.permutation[a]];
            dataset(file,axes[a],H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{v.size()},v.data());
         }
      }
      else if (which==2) {
         const hsize_t ns=scene.sources.size(),nr=scene.receivers.size();
         i64(file,"Ns",ns); i64(file,"Nr",nr); i64(file,"Nt",scene.steps); i8(file,"diff",scene.differentiated ? 1 : 0);
         dataset(file,"in_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{ns},scene.sources.data());
         dataset(file,"out_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{nr},scene.receivers.data());
         dataset(file,"out_reorder",H5T_STD_I64LE,H5T_NATIVE_INT64,{nr},scene.reorder.data());
         dataset(file,"out_alpha",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{scene.mesh.receivers.size(),8},scene.weights.data());
         dataset(file,"in_sigs",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{ns,static_cast<hsize_t>(scene.steps)},scene.input.data());
      }
      else if (which==3) {
         i8(file,"Nmat",static_cast<int8_t>(scene.materials.size()));
         dataset(file,"Mb",H5T_STD_I8LE,H5T_NATIVE_INT8,{poles.size()},poles.data());
         for (std::size_t m=0;m<scene.materials.size();++m) {
            char name[40]; std::snprintf(name,sizeof(name),"mat_%02d_DEF",static_cast<int>(m));
            dataset(file,name,H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{static_cast<hsize_t>(poles[m]),3},scene.materials[m].def.data());
            std::snprintf(name,sizeof(name),"mat_%02d_label",static_cast<int>(m));
            string_dataset(file,name,scene.materials[m].label);
         }
      }
      else {
         f64(file,"h",scene.constants.h);
         const char *axes[]={"xv","yv","zv"};
         for (unsigned a=0;a<3;++a) {
            const auto &v=scene.physical_grid.axes[a];
            dataset(file,axes[a],H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{v.size()},v.data());
         }
      }
      file.close();
   }
   validate_loader(scene,directory,verbose);
   output.commit();
}

} // namespace pffdtd_prepare

#ifndef PFFDTD_PREPARE_NO_MAIN
namespace {
struct CliOptions {
   PrepareOptions numeric;
   std::string model,output;
   std::map<std::string,std::string> materials;
   bool verbose=false;
#ifdef __CUDACC__
   std::string backend="cuda";
#else
   std::string backend="cpu";
#endif
};

void usage(const char *program)
{
   std::printf("Usage: %s --model JSON --output NEW_DIR --material label=FILE ... [options]\n\n"
               "Native JSONRoomExport triangle-mesh preparation; coordinates are metres.\n"
               "  --spacing METRES            override fmax/PPW spacing\n"
               "  --fmax HZ --ppw N           defaults 1000 Hz / 10 points per wavelength\n"
               "  --fcc                       FCC lattice, folded for CUDA by default\n"
               "  --duration SECONDS          default 1; Nt=ceil(duration/Ts)\n"
               "  --steps N                   explicit Nt instead of duration\n"
               "  --source-num N              one physical source, 1-based (default 1)\n"
               "  --temperature C --humidity PERCENT  defaults 20 C / 50 percent\n"
               "  --precision single|double   target solver (default single)\n"
               "  --diff-source --no-diff-source  bilinear differentiation; default on\n"
               "                                 for single, off for double\n"
               "  --backend cpu|cuda          default matches compiled backend\n"
               "  --threads N                 OpenMP CPU threads; 0 uses runtime default\n"
               "  --no-rotate                 preserve original axis order\n"
               "  --no-fold                   unfolded FCC, accepted only by CPU solver\n"
               "  --min-triangle-area M2      default 1e-6; discard smaller/degenerate faces\n"
               "  --verbose --help\n\n"
               "Every non-_RIGID material label needs an explicit reusable HDF5 DEF file.\n"
               "Sides: 0 rigid, 1 back, 2 front, 3 both; winding determines the normal.\n"
               "Nearest triangle selects one material/normal per boundary node. Fuzzy\n"
               "edge tolerance is 1e-3*h, near-node tolerance 1e-6*link length. Near-node\n"
               "hits isolate every incident link; conservative reciprocal union also\n"
               "cuts its neighbor endpoints and keeps newly added endpoints rigid.\n"
               "SAF counts every blocked link. These deliberate edge/near-hit corrections\n"
               "and Kelvin sound-speed correction are not bitwise Python compatibility.\n"
               "Creates five solver-compatible HDF5 files in an exclusively new directory;\n"
               "validates them with the original native loader. No fitting, simulation,\n"
               "Python calculations, air attenuation or plots. CUDA without a device exits 77.\n",program);
}

double number(const char *text,const std::string &option)
{
   require(text && *text,"Missing value for "+option);
   char *end=nullptr; errno=0;
   const double value=std::strtod(text,&end);
   require(!errno && end!=text && *end=='\0' && std::isfinite(value),"Invalid numeric value for "+option);
   return value;
}

uint64_t integer(const char *text,const std::string &option)
{
   require(text && *text && *text!='-' && *text!='+',"Invalid integer value for "+option);
   char *end=nullptr; errno=0;
   const unsigned long long value=std::strtoull(text,&end,10);
   require(!errno && end!=text && *end=='\0',"Invalid integer value for "+option);
   return value;
}

bool parse(int argc,char **argv,CliOptions &options)
{
   bool explicit_diff=false,explicit_spacing=false,explicit_frequency=false,explicit_steps=false,explicit_duration=false;
   for (int i=1;i<argc;++i) {
      const std::string arg(argv[i]);
      if (arg=="--help") { usage(argv[0]); return false; }
      if (arg=="--fcc") options.numeric.fcc=true;
      else if (arg=="--no-rotate") options.numeric.rotate=false;
      else if (arg=="--no-fold") options.numeric.fold=false;
      else if (arg=="--verbose") options.verbose=true;
      else if (arg=="--diff-source" || arg=="--no-diff-source") {
         options.numeric.differentiated=arg=="--diff-source"; explicit_diff=true;
      }
      else {
         require(++i<argc,"Missing value for "+arg);
         const char *value=argv[i];
         if (arg=="--model" || arg=="--json") options.model=value;
         else if (arg=="--output") options.output=value;
         else if (arg=="--backend") options.backend=value;
         else if (arg=="--material") {
            const std::string mapping(value);
            const auto equal=mapping.find('=');
            require(equal!=std::string::npos && equal>0 && equal+1<mapping.size(),"Use --material label=FILE");
            require(options.materials.emplace(mapping.substr(0,equal),mapping.substr(equal+1)).second,
                    "Duplicate --material label "+mapping.substr(0,equal));
         }
         else if (arg=="--precision") {
            const std::string precision(value);
            require(precision=="single" || precision=="double" || precision=="32" || precision=="64",
                    "Precision must be single or double");
            options.numeric.target_precision=(precision=="single" || precision=="32") ? 32 : 64;
         }
         else if (arg=="--steps" || arg=="--source-num" || arg=="--threads") {
            const uint64_t count=integer(value,arg);
            require(count<=static_cast<uint64_t>(INT64_MAX),"Integer value overflows for "+arg);
            if (arg=="--steps") {
               require(count>0,"Steps must be positive"); options.numeric.steps=static_cast<int64_t>(count); explicit_steps=true;
            }
            else {
               require(count<=UINT_MAX,"Integer value exceeds unsigned range for "+arg);
               if (arg=="--source-num") { require(count>0,"Source number must be positive"); options.numeric.source_num=count; }
               else options.numeric.threads=count;
            }
         }
         else if (arg=="--spacing") { options.numeric.spacing=number(value,arg); explicit_spacing=true; require(options.numeric.spacing>0,"Spacing must be positive"); }
         else if (arg=="--fmax") { options.numeric.fmax=number(value,arg); explicit_frequency=true; }
         else if (arg=="--ppw") { options.numeric.ppw=number(value,arg); explicit_frequency=true; }
         else if (arg=="--duration") { options.numeric.duration=number(value,arg); explicit_duration=true; }
         else if (arg=="--temperature") options.numeric.temperature=number(value,arg);
         else if (arg=="--humidity") options.numeric.humidity=number(value,arg);
         else if (arg=="--min-triangle-area") options.numeric.min_triangle_area=number(value,arg);
         else throw std::runtime_error("Unknown argument: "+arg);
      }
   }
   require(!options.model.empty() && !options.output.empty(),"Specify --model JSON and --output NEW_DIR");
   require(!(explicit_spacing && explicit_frequency),"Choose --spacing or --fmax/--ppw");
   require(!(explicit_steps && explicit_duration),"Choose --steps or --duration");
   require(options.backend=="cpu" || options.backend=="cuda","Backend must be cpu or cuda");
   require(!(options.backend=="cuda" && options.numeric.fcc && !options.numeric.fold),"CUDA solver requires folded FCC; --no-fold is CPU-only");
   if (!explicit_diff) options.numeric.differentiated=options.numeric.target_precision==32;
   require(options.numeric.target_precision!=32 || options.numeric.differentiated,"Single precision requires --diff-source");
   return true;
}
} // namespace

int main(int argc,char **argv)
{
   try {
      CliOptions options;
      if (!parse(argc,argv,options)) return EXIT_SUCCESS;
      Voxelizer backend=voxelize_cpu;
      if (options.backend=="cuda") {
#ifdef __CUDACC__
         int count=0;
         const cudaError_t status=cudaGetDeviceCount(&count);
         if (status!=cudaSuccess || count<1) {
            std::fprintf(stderr,"SKIP: CUDA preparation needs an available device (%s).\n",
                         status==cudaSuccess ? "no visible device" : cudaGetErrorString(status));
            return 77;
         }
         backend=voxelize_cuda;
#else
         throw std::runtime_error("This binary has no CUDA backend; use fdtd_prepare_gpu or --backend cpu");
#endif
      }
      struct stat existing;
      if (lstat(options.output.c_str(),&existing)==0)
         throw std::runtime_error("Output directory already exists: "+options.output);
      require(errno==ENOENT,"Cannot inspect output directory "+options.output);
      const auto start=std::chrono::steady_clock::now();
      auto mesh=import_scene(options.model,options.numeric.min_triangle_area);
      auto materials=import_materials(mesh,options.materials);
      auto scene=prepare_scene(std::move(mesh),materials,options.numeric,backend);
      scene.backend=options.backend;
      write_prepared(scene,options.output,options.verbose);
      const auto dims=scene.layout.output_dims();
      const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::printf("Prepared %s: %lld x %lld x %lld, lattice=%s, triangles=%zu/%zu, Nb=%zu, materials=%zu, Ns=%zu, Nr=%zu, Nt=%lld.\n",
                  options.output.c_str(),static_cast<long long>(dims[0]),static_cast<long long>(dims[1]),static_cast<long long>(dims[2]),
                  options.numeric.fcc ? (scene.layout.fold_fcc ? "FCC folded" : "FCC unfolded") : "Cartesian",
                  scene.mesh.geometry.triangles.size(),scene.mesh.triangles_input,scene.boundary.size(),scene.materials.size(),
                  scene.sources.size(),scene.receivers.size(),static_cast<long long>(scene.steps));
      std::printf("h=%.17g m, c=%.17g m/s, Fs=%.17g Hz, source=%u, differentiated=%d, target=FP%d, backend=%s; reciprocal corrections=%zu, added rigid nodes=%zu; native loader PASS; wall=%.6f s.\n",
                  scene.constants.h,scene.constants.c,scene.constants.fs,options.numeric.source_num,scene.differentiated ? 1 : 0,
                  scene.target_precision,scene.backend.c_str(),scene.reconciled_links,scene.added_boundary_nodes,elapsed);
      return EXIT_SUCCESS;
   }
   catch (const std::exception &e) {
      std::fprintf(stderr,"fdtd_prepare: %s\n",e.what());
      return EXIT_FAILURE;
   }
}
#endif
