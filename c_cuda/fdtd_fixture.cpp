// Native analytical Cartesian fixtures for solver/benchmark inputs. This builds
// finite axis-aligned panels directly; it is not a triangle-mesh voxelizer.
#include <fdtd_data.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Options {
   std::string output, self_test;
   int64_t nx=32, ny=32, nz=32, steps=1537, panel_spacing=0, rigid_every=3;
   int poles=11;
   bool mixed_poles=false, verbose=false;
};

void require(bool condition, const std::string &message)
{
   if (!condition) throw std::runtime_error(message);
}

void require(bool condition, const char *message)
{
   if (!condition) throw std::runtime_error(message);
}

std::runtime_error system_error(const std::string &message)
{
   return std::runtime_error(message+": "+std::strerror(errno));
}

void usage(const char *program)
{
   std::printf("Usage: %s --output NEW_DIR [options]\n"
               "       %s --self-test NEW_DIR [--verbose]\n\n"
               "Options: --nx N --ny N --nz N (default 32, each at least 8)\n"
               "         --steps N (default 1537, at least 1)\n"
               "         --poles N (default 11, 0..12) or --mixed-poles\n"
               "         --rigid-every N (default 3; 0: all ADE, 1: all rigid)\n"
               "         --panel-spacing N (default 0: one central panel;\n"
               "                            positive: planes every N x cells)\n"
               "         --verbose --help\n\n"
               "Creates sim_consts.h5, vox_out.h5, comms_out.h5 and sim_mats.h5.\n"
               "The output directory must not exist. Panels cut reciprocal x links\n"
               "and stay clear of absorbing boundary cells; passive RLC branches,\n"
               "two differentiated dyadic impulses and interior receivers are included.\n"
               "Every fixture is verified with the original native HDF5 loader.\n"
               "Cartesian analytical benchmark inputs only; no mesh processing,\n"
               "Python calculations or solver execution. Run the CUDA benchmark\n"
               "from the generated directory on a machine with one visible GPU.\n",
               program,program);
}

int64_t integer(const char *text, const char *option)
{
   require(text && *text && *text!='-',std::string("Invalid integer for ")+option);
   char *end=nullptr;
   errno=0;
   const long long value=std::strtoll(text,&end,10);
   require(!errno && *end=='\0' && value>=0,std::string("Invalid integer for ")+option);
   return static_cast<int64_t>(value);
}

bool parse(int argc, char **argv, Options &options)
{
   bool explicit_poles=false, scene_option=false;
   for (int i=1; i<argc; ++i) {
      const std::string arg(argv[i]);
      if (arg=="--help") { usage(argv[0]); return false; }
      if (arg=="--verbose") options.verbose=true;
      else if (arg=="--mixed-poles") { options.mixed_poles=true; scene_option=true; }
      else if (arg=="--output" || arg=="--self-test") {
         require(++i<argc,"Missing directory for "+arg);
         if (arg=="--output") options.output=argv[i];
         else options.self_test=argv[i];
      }
      else if (arg=="--nx" || arg=="--ny" || arg=="--nz" || arg=="--steps" ||
               arg=="--poles" || arg=="--rigid-every" || arg=="--panel-spacing") {
         require(++i<argc,"Missing integer for "+arg);
         const int64_t value=integer(argv[i],arg.c_str());
         scene_option=true;
         if (arg=="--nx") options.nx=value;
         else if (arg=="--ny") options.ny=value;
         else if (arg=="--nz") options.nz=value;
         else if (arg=="--steps") options.steps=value;
         else if (arg=="--rigid-every") options.rigid_every=value;
         else if (arg=="--panel-spacing") options.panel_spacing=value;
         else {
            require(value<=MMb,"--poles must be in 0..12");
            options.poles=static_cast<int>(value); explicit_poles=true;
         }
      }
      else throw std::runtime_error("Unknown argument: "+arg);
   }
   require(options.output.empty()!=options.self_test.empty(),
           "Specify exactly one of --output or --self-test with a nonempty directory");
   require(!(options.mixed_poles && explicit_poles),"Choose --poles or --mixed-poles");
   require(options.self_test.empty() || !scene_option,"--self-test uses its own scene matrix");
   return true;
}

int64_t product(int64_t a, int64_t b, const char *name)
{
   require(a>=0 && b>=0 && (b==0 || a<=INT64_MAX/b),std::string(name)+" count overflows int64");
   return a*b;
}

void host_count(int64_t count, std::size_t element_size, const char *name)
{
   require(count>=0 && static_cast<uint64_t>(count)<=SIZE_MAX/element_size,
           std::string(name)+" byte count exceeds host address space");
}

struct Fixture {
   Options options;
   const double sample_rate=48000.0, speed=343.0, courant=0.5;
   double ts, spacing;
   int64_t npts;
   std::vector<int64_t> boundary, sources, receivers, reorder;
   std::vector<uint8_t> adjacency;
   std::vector<int8_t> materials, poles;
   std::vector<double> area, input, weights;
   std::vector<std::vector<double>> def;

   int64_t index(int64_t x, int64_t y, int64_t z) const
   {
      return (x*options.ny+y)*options.nz+z;
   }

   bool interior(int64_t i) const
   {
      if (i<0 || i>=npts) return false;
      const int64_t yz=options.ny*options.nz;
      const int64_t x=i/yz, y=(i%yz)/options.nz, z=i%options.nz;
      return x>0 && x<options.nx-1 && y>0 && y<options.ny-1 && z>0 && z<options.nz-1;
   }

   bool away_from_abc(int64_t i) const
   {
      if (!interior(i)) return false;
      const int64_t yz=options.ny*options.nz;
      const int64_t x=i/yz, y=(i%yz)/options.nz, z=i%options.nz;
      return x>1 && x<options.nx-2 && y>1 && y<options.ny-2 && z>1 && z<options.nz-2;
   }

   explicit Fixture(const Options &value) : options(value)
   {
      require(options.nx>=8 && options.ny>=8 && options.nz>=8,"Each grid dimension must be at least 8");
      require(options.steps>=1,"--steps must be at least 1");
      require(options.poles>=0 && options.poles<=MMb,"--poles must be in 0..12");
      npts=product(product(options.nx,options.ny,"grid"),options.nz,"grid");
      host_count(npts,sizeof(bool),"loader mask");
      const int64_t nsamples=product(2,options.steps,"source signal");
      host_count(nsamples,sizeof(double),"source signal");
      host_count(product(6,options.steps,"receiver output"),sizeof(double),"receiver output");
      host_count(options.nx,sizeof(uint8_t),"panel locations");
      ts=1.0/sample_rate;
      spacing=speed*ts/courant;
      std::vector<uint8_t> planes(static_cast<std::size_t>(options.nx),0);
      if (options.panel_spacing==0) planes[static_cast<std::size_t>((options.nx-2)/2)]=1;
      else {
         require(options.panel_spacing>0,"Panel spacing must be nonnegative");
         for (int64_t x=2; x<=options.nx-4;) {
            planes[static_cast<std::size_t>(x)]=1;
            if (options.panel_spacing>options.nx-4-x) break;
            x+=options.panel_spacing;
         }
      }
      // Each removed link represents a plane between x and x+1. Both endpoints
      // carry the same cut; one blocked Cartesian link contributes saf=1.
      for (int64_t x=2; x<options.nx-2; ++x) {
         if (!planes[static_cast<std::size_t>(x)] && !planes[static_cast<std::size_t>(x-1)]) continue;
         for (int64_t y=3; y<options.ny-3; ++y)
            for (int64_t z=3; z<options.nz-3; ++z) {
               uint8_t adj[6]={1,1,1,1,1,1};
               adj[0]=planes[static_cast<std::size_t>(x)] ? 0 : 1;
               adj[1]=planes[static_cast<std::size_t>(x-1)] ? 0 : 1;
               const int removed=2-adj[0]-adj[1];
               if (!removed) continue;
               boundary.push_back(index(x,y,z));
               adjacency.insert(adjacency.end(),adj,adj+6);
               area.push_back(static_cast<double>(removed));
            }
      }
      require(!boundary.empty(),"No analytical panel was generated");
      const int nmat=options.mixed_poles ? 13 : 1;
      poles.resize(nmat); def.resize(nmat);
      for (int k=0; k<nmat; ++k) {
         poles[k]=static_cast<int8_t>(options.mixed_poles ? k : options.poles);
         for (int m=0; m<poles[k]; ++m) {
            // Positive inertance, resistance and stiffness define passive RLC
            // branches. Scaling D/F with Ts gives well-conditioned loader quads.
            def[k].push_back(ts*(m+1));
            def[k].push_back(2.0+0.125*k);
            def[k].push_back((m+1)/(64.0*ts));
         }
      }
      for (std::size_t nb=0; nb<boundary.size(); ++nb) {
         const bool rigid=options.rigid_every>0 && nb%static_cast<uint64_t>(options.rigid_every)==0;
         materials.push_back(rigid ? int8_t(-1) : static_cast<int8_t>(nb%nmat));
      }
      sources={index(2,2,options.nz/2),index(options.nx-3,2,options.nz/2)};
      receivers={sources[0],sources[0],sources[1],boundary.front(),boundary.back(),
                 index(options.nx/2,options.ny/2,options.nz/2)};
      std::sort(receivers.begin(),receivers.end());
      reorder.resize(receivers.size()); weights.assign(receivers.size(),1.0);
      for (std::size_t i=0; i<reorder.size(); ++i) reorder[i]=static_cast<int64_t>(i);
      input.assign(static_cast<std::size_t>(nsamples),0.0);
      input[0]=0.0078125; input[static_cast<std::size_t>(options.steps)]=-0.00390625;
      if (options.steps>1) {
         input[1]=-input[0];
         input[static_cast<std::size_t>(options.steps+1)]=-input[static_cast<std::size_t>(options.steps)];
      }
      validate_analytical();
   }

   void validate_analytical() const
   {
      require(std::is_sorted(boundary.begin(),boundary.end()) &&
              std::adjacent_find(boundary.begin(),boundary.end())==boundary.end(),
              "Boundary indices must be sorted and unique");
      const int64_t offsets[6]={options.ny*options.nz,-options.ny*options.nz,
                               options.nz,-options.nz,1,-1};
      for (std::size_t nb=0; nb<boundary.size(); ++nb) {
         require(interior(boundary[nb]),"Boundary index is outside the interior");
         int removed=0;
         for (int link=0; link<6; ++link) {
            const bool connected=adjacency[nb*6+link]!=0;
            removed+=!connected;
            const int64_t other=boundary[nb]+offsets[link];
            const auto found=std::lower_bound(boundary.begin(),boundary.end(),other);
            if (found!=boundary.end() && *found==other) {
               const std::size_t ordinal=static_cast<std::size_t>(found-boundary.begin());
               require(connected==(adjacency[ordinal*6+(link^1)]!=0),
                       "Panel adjacency is not reciprocal");
            }
            else require(connected,"Removed link has no opposite boundary endpoint");
         }
         require(removed>0 && area[nb]==removed,"Panel surface area does not match removed links");
         require(materials[nb]>=-1 && materials[nb]<static_cast<int>(poles.size()),
                 "Boundary material is outside the material table");
      }
      require(sources[0]<sources[1] && interior(sources[0]) && interior(sources[1]),
              "Sources must be distinct sorted interior indices");
      for (int64_t source : sources)
         require(away_from_abc(source) && !std::binary_search(boundary.begin(),boundary.end(),source),
                 "Sources must be outside rigid/ADE and ABC nodes");
      for (int64_t receiver : receivers) require(interior(receiver),"Receiver is outside the interior");
      for (std::size_t k=0; k<def.size(); ++k)
         for (int m=0; m<poles[k]; ++m) {
            const double d=def[k][m*3], e=def[k][m*3+1], f=def[k][m*3+2];
            const double denominator=2*d/ts+e+0.5*f*ts;
            require(std::isfinite(d) && std::isfinite(e) && std::isfinite(f) &&
                    d>=0 && e>0 && f>=0 && denominator>0 && std::isfinite(denominator),
                    "RLC branch is not finite/passive");
         }
   }
};

class OutputDirectory {
   std::string path;
   std::vector<std::string> owned;
   bool committed=false;
public:
   explicit OutputDirectory(const std::string &name) : path(name)
   {
      if (mkdir(path.c_str(),0755)!=0) throw system_error("Cannot create new directory "+path);
   }
   void track(const std::string &file) { owned.push_back(file); }
   void commit() { committed=true; }
   ~OutputDirectory()
   {
      if (!committed) {
         for (const std::string &file : owned) unlink(file.c_str());
         rmdir(path.c_str());
      }
   }
};

class H5Handle {
   hid_t handle;
   herr_t (*closer)(hid_t);
public:
   H5Handle(hid_t value, herr_t (*close)(hid_t)) : handle(value), closer(close)
   { require(handle>=0,"HDF5 operation failed"); }
   ~H5Handle() { if (handle>=0) closer(handle); }
   operator hid_t() const { return handle; }
   void close()
   {
      require(handle>=0,"HDF5 handle was already closed");
      const hid_t old=handle; handle=-1;
      require(closer(old)>=0,"Cannot close HDF5 object");
   }
   H5Handle(const H5Handle &)=delete;
   H5Handle &operator=(const H5Handle &)=delete;
};

void dataset(hid_t file, const char *name, hid_t stored_type, hid_t memory_type,
             const std::vector<hsize_t> &dimensions, const void *data)
{
   H5Handle space(dimensions.empty() ? H5Screate(H5S_SCALAR) :
         H5Screate_simple(static_cast<int>(dimensions.size()),dimensions.data(),nullptr),H5Sclose);
   H5Handle value(H5Dcreate2(file,name,stored_type,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),H5Dclose);
   const double placeholder=0.0;
   require(H5Dwrite(value,memory_type,H5S_ALL,H5S_ALL,H5P_DEFAULT,data ? data : &placeholder)>=0,
           std::string("Cannot write HDF5 dataset ")+name);
   value.close(); space.close();
}

void scalar_i64(hid_t file, const char *name, int64_t value)
{ dataset(file,name,H5T_STD_I64LE,H5T_NATIVE_INT64,{},&value); }
void scalar_i8(hid_t file, const char *name, int8_t value)
{ dataset(file,name,H5T_STD_I8LE,H5T_NATIVE_INT8,{},&value); }
void scalar_double(hid_t file, const char *name, double value)
{ dataset(file,name,H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{},&value); }

void write_fixture(const Fixture &fixture, OutputDirectory &directory)
{
   const Options &o=fixture.options;
   const char *files[]={"sim_consts.h5","vox_out.h5","comms_out.h5","sim_mats.h5"};
   for (int which=0; which<4; ++which) {
      const std::string filename=o.output+"/"+files[which];
      H5Handle file(H5Fcreate(filename.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
      directory.track(filename);
      if (which==0) {
         scalar_double(file,"l",fixture.courant); scalar_double(file,"l2",fixture.courant*fixture.courant);
         scalar_double(file,"Ts",fixture.ts); scalar_i8(file,"fcc_flag",0);
         scalar_double(file,"c",fixture.speed); scalar_double(file,"h",fixture.spacing);
         scalar_double(file,"SR",fixture.sample_rate); scalar_double(file,"Tc",20); scalar_double(file,"rh",50);
      }
      else if (which==1) {
         scalar_i64(file,"Nx",o.nx); scalar_i64(file,"Ny",o.ny); scalar_i64(file,"Nz",o.nz);
         scalar_i64(file,"Nb",static_cast<int64_t>(fixture.boundary.size()));
         const hsize_t nb=fixture.boundary.size();
         dataset(file,"bn_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{nb},fixture.boundary.data());
         dataset(file,"adj_bn",H5T_STD_I8LE,H5T_NATIVE_INT8,{nb,6},fixture.adjacency.data());
         dataset(file,"mat_bn",H5T_STD_I8LE,H5T_NATIVE_INT8,{nb},fixture.materials.data());
         dataset(file,"saf_bn",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{nb},fixture.area.data());
         scalar_double(file,"h",fixture.spacing);
         const char *names[]={"xv","yv","zv"};
         const int64_t lengths[]={o.nx,o.ny,o.nz};
         for (int axis=0; axis<3; ++axis) {
            std::vector<double> coordinates(static_cast<std::size_t>(lengths[axis]));
            for (int64_t i=0; i<lengths[axis]; ++i) coordinates[static_cast<std::size_t>(i)]=i*fixture.spacing;
            dataset(file,names[axis],H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,
                    {static_cast<hsize_t>(lengths[axis])},coordinates.data());
         }
      }
      else if (which==2) {
         const hsize_t nr=fixture.receivers.size();
         scalar_i64(file,"Ns",fixture.sources.size()); scalar_i64(file,"Nr",nr);
         scalar_i64(file,"Nt",o.steps); scalar_i8(file,"diff",1);
         dataset(file,"in_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{2},fixture.sources.data());
         dataset(file,"out_ixyz",H5T_STD_I64LE,H5T_NATIVE_INT64,{nr},fixture.receivers.data());
         dataset(file,"out_reorder",H5T_STD_I64LE,H5T_NATIVE_INT64,{nr},fixture.reorder.data());
         dataset(file,"out_alpha",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{nr,1},fixture.weights.data());
         dataset(file,"in_sigs",H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,{2,static_cast<hsize_t>(o.steps)},fixture.input.data());
      }
      else {
         scalar_i8(file,"Nmat",static_cast<int8_t>(fixture.poles.size()));
         dataset(file,"Mb",H5T_STD_I8LE,H5T_NATIVE_INT8,
                 {static_cast<hsize_t>(fixture.poles.size())},fixture.poles.data());
         for (std::size_t k=0; k<fixture.poles.size(); ++k) {
            char name[40]; std::snprintf(name,sizeof(name),"mat_%02d_DEF",static_cast<int>(k));
            dataset(file,name,H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE,
                    {static_cast<hsize_t>(fixture.poles[k]),3},fixture.def[k].data());
         }
      }
      file.close();
   }
}

class WorkingDirectory {
   int old;
public:
   explicit WorkingDirectory(const std::string &path) : old(open(".",O_RDONLY|O_DIRECTORY))
   {
      if (old<0) throw system_error("Cannot retain current directory");
      if (chdir(path.c_str())!=0) {
         const int saved=errno; close(old); errno=saved;
         throw system_error("Cannot enter fixture directory");
      }
   }
   ~WorkingDirectory() { fchdir(old); close(old); }
};

class QuietStdout {
   int old=-1;
public:
   explicit QuietStdout(bool quiet)
   {
      if (!quiet) return;
      require(std::fflush(stdout)==0,"Cannot flush stdout");
      old=dup(STDOUT_FILENO);
      if (old<0) throw system_error("Cannot retain stdout");
      const int null_fd=open("/dev/null",O_WRONLY);
      if (null_fd<0 || dup2(null_fd,STDOUT_FILENO)<0) {
         const int saved=errno;
         if (null_fd>=0) close(null_fd);
         close(old); old=-1; errno=saved;
         throw system_error("Cannot quiet loader stdout");
      }
      close(null_fd);
   }
   ~QuietStdout()
   {
      if (old>=0) { std::fflush(stdout); dup2(old,STDOUT_FILENO); close(old); }
   }
};

struct LoadedData {
   SimData data{};
   ~LoadedData() { free_sim_data(&data); }
};

void validate_loaded(const Fixture &fixture)
{
   const Options &o=fixture.options;
   WorkingDirectory current(o.output);
   QuietStdout quiet(!o.verbose);
   LoadedData loaded;
   load_sim_data(&loaded.data);
   const SimData &data=loaded.data;
   require(data.Nx==o.nx && data.Ny==o.ny && data.Nz==o.nz && data.Npts==fixture.npts &&
           data.Nt==o.steps && data.NN==6 && data.fcc_flag==0,"Loaded grid/step metadata differs");
   require(data.Ns==2 && data.Nr==static_cast<int64_t>(fixture.receivers.size()) &&
           data.Nb==static_cast<int64_t>(fixture.boundary.size()) &&
           data.Nm==static_cast<int>(fixture.poles.size()),"Loaded counts differ");
   require(data.l==fixture.courant && data.l2==fixture.courant*fixture.courant,
           "Loaded Courant coefficients differ");
   int64_t lossy=0;
   for (std::size_t nb=0; nb<fixture.boundary.size(); ++nb) {
      require(data.bn_ixyz[nb]==fixture.boundary[nb],"Loaded boundary indices differ");
      int neighbours=0;
      for (int link=0; link<6; ++link) {
         const int connected=fixture.adjacency[nb*6+link];
         neighbours+=connected;
         require(GET_BIT(data.adj_bn[nb],link)==connected,"Loaded adjacency differs");
      }
      require(data.K_bn[nb]==neighbours,"Loaded neighbour count differs");
      if (fixture.materials[nb]>=0) {
         require(lossy<data.Nbl && data.bnl_ixyz[lossy]==fixture.boundary[nb] &&
                 data.mat_bnl[lossy]==fixture.materials[nb] &&
                 data.ssaf_bnl[lossy]==static_cast<Real>(fixture.area[nb]),"Loaded ADE association differs");
         ++lossy;
      }
   }
   require(lossy==data.Nbl,"Loaded lossy count differs");
   std::size_t next=0;
   for (int64_t i=0; i<data.Npts; ++i) {
      const bool expected=next<fixture.boundary.size() && fixture.boundary[next]==i;
      require((GET_BIT(data.bn_mask[i>>3],i%8)!=0)==expected,"Loaded boundary mask differs");
      if (expected) ++next;
   }
   for (int64_t ns=0; ns<data.Ns; ++ns)
      require(data.in_ixyz[ns]==fixture.sources[ns] &&
              !GET_BIT(data.bn_mask[data.in_ixyz[ns]>>3],data.in_ixyz[ns]%8) &&
              !std::binary_search(data.bna_ixyz,data.bna_ixyz+data.Nba,data.in_ixyz[ns]),
              "Loaded source is misplaced or intersects boundary/ABC nodes");
   for (int64_t nr=0; nr<data.Nr; ++nr)
      require(data.out_ixyz[nr]==fixture.receivers[nr] && data.out_reorder[nr]==nr,
              "Loaded receiver/reorder differs");
   require(std::memcmp(data.in_sigs,fixture.input.data(),fixture.input.size()*sizeof(double))==0,
           "Loaded differentiated input differs");
   // These metadata fields are used by output processing but are not attached
   // to SimData by the solver loader. Verify them separately through HDF5.
   {
      H5Handle file(H5Fopen("comms_out.h5",H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
      H5Handle differentiated(H5Dopen2(file,"diff",H5P_DEFAULT),H5Dclose);
      int8_t diff=0;
      require(H5Dread(differentiated,H5T_NATIVE_INT8,H5S_ALL,H5S_ALL,H5P_DEFAULT,&diff)>=0 && diff==1,
              "Differentiated input metadata differs");
      H5Handle weights(H5Dopen2(file,"out_alpha",H5P_DEFAULT),H5Dclose);
      H5Handle space(H5Dget_space(weights),H5Sclose);
      hsize_t dimensions[2]={0,0};
      require(H5Sget_simple_extent_ndims(space)==2 &&
              H5Sget_simple_extent_dims(space,dimensions,nullptr)>=0 &&
              dimensions[0]==fixture.receivers.size() && dimensions[1]==1,
              "Receiver weight metadata has the wrong shape");
      std::vector<double> actual(fixture.weights.size());
      require(H5Dread(weights,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,actual.data())>=0 &&
              actual==fixture.weights,"Receiver weight metadata differs");
   }
   for (int k=0; k<data.Nm; ++k) {
      require(data.Mb[k]==fixture.poles[k],"Loaded pole count differs");
      Real expected_beta=0;
      for (int m=0; m<data.Mb[k]; ++m) {
         const double d=fixture.def[k][m*3]/fixture.ts;
         const double e=fixture.def[k][m*3+1];
         const double f=fixture.def[k][m*3+2]*fixture.ts;
         const double b=1.0/(2*d+e+0.5*f);
         const MatQuad &quad=data.mat_quads[k*MMb+m];
         require(quad.b==static_cast<Real>(b) && quad.bd==static_cast<Real>(b*(2*d-e-0.5*f)) &&
                 quad.bDh==static_cast<Real>(b*d) && quad.bFh==static_cast<Real>(b*f),
                 "Loaded passive material coefficients differ");
         require(std::isfinite(quad.b) && quad.b>0 && std::abs(quad.bd)<=1 &&
                 quad.bDh>=0 && quad.bFh>=0,"Loaded RLC coefficients violate passivity bounds");
         expected_beta+=static_cast<Real>(b);
      }
      require(data.mat_beta[k]==expected_beta,"Loaded material beta differs");
      for (int m=data.Mb[k]; m<MMb; ++m) {
         const MatQuad &quad=data.mat_quads[k*MMb+m];
         require(quad.b==0 && quad.bd==0 && quad.bDh==0 && quad.bFh==0,
                 "Unused material coefficient padding is nonzero");
      }
   }
}

void generate(const Options &options)
{
   Fixture fixture(options);
   OutputDirectory directory(options.output);
   write_fixture(fixture,directory);
   validate_loaded(fixture);
   directory.commit();
}

void self_test(const Options &options)
{
   OutputDirectory root(options.self_test);
   int cases=0;
   for (int poles=0; poles<=MMb; ++poles) {
      Options scene;
      scene.nx=8; scene.ny=9; scene.nz=10; scene.steps=7; scene.poles=poles;
      scene.rigid_every=poles%3==0 ? 0 : 3; scene.verbose=options.verbose;
      scene.output=options.self_test+"/poles_"+std::to_string(poles);
      generate(scene); ++cases;
   }
   const int64_t lengths[]={1,2,6,95,96,97,511,512,513,1536,1537,3073};
   for (std::size_t i=0; i<sizeof(lengths)/sizeof(lengths[0]); ++i) {
      Options scene;
      scene.nx=10; scene.ny=8; scene.nz=11; scene.steps=lengths[i];
      scene.poles=i%2 ? 12 : 11; scene.mixed_poles=i%3==0;
      scene.rigid_every=i%4==0 ? 1 : (i%4==1 ? 0 : 3);
      scene.panel_spacing=i%2 ? 1 : 2; scene.verbose=options.verbose;
      scene.output=options.self_test+"/steps_"+std::to_string(lengths[i]);
      generate(scene); ++cases;
   }
   Options guard;
   guard.nx=8; guard.ny=8; guard.nz=8; guard.steps=2; guard.poles=0;
   guard.output=options.self_test+"/no_overwrite"; guard.verbose=options.verbose;
   generate(guard); ++cases;
   bool refused=false;
   try { Options changed=guard; changed.nx=9; generate(changed); }
   catch (const std::runtime_error &) { refused=true; }
   require(refused,"Generator overwrote an existing directory");
   validate_loaded(Fixture(guard));
   root.commit();
   std::printf("fdtd_fixture: PASS FP%d, %d HDF5 round-trip cases; reciprocal panels, all poles, mixed/rigid/ADE, output tails and overwrite refusal; loader checks only, no solver execution\n",
               static_cast<int>(8*sizeof(Real)),cases);
}

} // namespace

int main(int argc, char **argv)
{
   try {
      Options options;
      if (!parse(argc,argv,options)) return EXIT_SUCCESS;
      if (!options.self_test.empty()) self_test(options);
      else {
         generate(options);
         std::printf("Generated validated Cartesian HDF5 fixture in %s (%lld x %lld x %lld, Nt=%lld, poles=%s, loader FP%d).\n",
                     options.output.c_str(),static_cast<long long>(options.nx),
                     static_cast<long long>(options.ny),static_cast<long long>(options.nz),
                     static_cast<long long>(options.steps),options.mixed_poles ? "mixed 0..12" : std::to_string(options.poles).c_str(),
                     static_cast<int>(8*sizeof(Real)));
      }
      return EXIT_SUCCESS;
   }
   catch (const std::exception &error) {
      std::fprintf(stderr,"fdtd_fixture: %s\n",error.what());
      return EXIT_FAILURE;
   }
}
