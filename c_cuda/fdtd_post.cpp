// Native receiver reconstruction, filtering and resampling of solver HDF5 output.
#include <hdf5.h>
#include <post_pipeline.h>
#ifdef __CUDACC__
#include <post_cuda.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
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

#ifndef PFFDTD_BUILD_REVISION
#define PFFDTD_BUILD_REVISION "unknown"
#endif

namespace {

struct Options {
   std::string directory, output;
#ifdef __CUDACC__
   std::string backend = "cuda";
#else
   std::string backend = "cpu";
#endif
   std::string iir_mode = "serial";
   pffdtd_post::PostOptions processing;
   bool overwrite = false, verify = false, save_raw = false;
};

void require(bool condition, const std::string &message)
{
   if (!condition) throw std::runtime_error(message);
}

std::runtime_error system_error(const std::string &operation)
{
   return std::runtime_error(operation+": "+std::strerror(errno));
}

std::string join(const std::string &directory, const char *name)
{
   return directory+(directory.empty() || directory.back() != '/' ? "/" : "")+name;
}

void usage(const char *program)
{
   std::printf("Usage: %s --data-dir DIR [options]\n\n"
               "  --output FILE          Default DIR/sim_outs_processed.h5\n"
               "  --backend cpu|cuda     Default %s in this build\n"
               "  --iir-mode serial|chunked  CUDA IIR mode; default serial\n"
               "  --lowcut HZ            High-pass cutoff; default 10, 0 disables\n"
               "  --lowcut-order N       Default 8\n"
               "  --lowpass HZ           Low-pass cutoff; default 0 (disabled)\n"
               "  --lowpass-order N      Default 8\n"
               "  --symmetric-lowpass    Apply low-pass forward and backward\n"
               "  --sample-rate HZ       Output rate; default 48000, 0 preserves native rate\n"
               "  --verify               Compare CUDA with CPU: abs 1e-12 + rel 1e-8\n"
               "  --save-raw             Also write reconstructed r_out\n"
               "  --overwrite            Explicitly replace an existing regular output file\n"
               "  --help\n\n"
               "Reads comms_out.h5/out_alpha[Nmic,K], sim_consts.h5/Ts and\n"
               "sim_outs.h5/u_out[Nmic*K,Nt]. u_out is already reordered and rescaled\n"
               "by the solver. diff=1 enables integration before receiver filtering.\n"
               "Writes r_out_f[Nmic,Nt_out] and scalar Fs_f through atomic publication.\n"
               "Air absorption: none. The native Kaiser resampler differs from resampy;\n"
               "the processed waveform is not expected to match its samples bit for bit.\n"
               "No Python calculations, plotting or WAV export. CUDA absence returns 77\n"
               "before creating output files. CPU --verify validates the reference result.\n",
               program,
#ifdef __CUDACC__
               "cuda"
#else
               "cpu"
#endif
   );
}

double number(const char *text, const char *option)
{
   require(text && *text,std::string("Missing number for ")+option);
   char *end = nullptr;
   errno = 0;
   const double value = std::strtod(text,&end);
   require(!errno && *end == '\0' && std::isfinite(value) && value >= 0,
           std::string("Invalid nonnegative finite number for ")+option);
   return value;
}

unsigned order(const char *text, const char *option)
{
   require(text && *text && *text != '-',std::string("Invalid filter order for ")+option);
   char *end = nullptr;
   errno = 0;
   const unsigned long value = std::strtoul(text,&end,10);
   require(!errno && *end == '\0' && value > 0 && value <= std::numeric_limits<unsigned>::max(),
           std::string("Invalid filter order for ")+option);
   return static_cast<unsigned>(value);
}

bool parse(int argc, char **argv, Options &options)
{
   for (int i=1; i<argc; i++) {
      const std::string arg(argv[i]);
      if (arg == "--help") { usage(argv[0]); return false; }
      if (arg == "--overwrite") options.overwrite = true;
      else if (arg == "--verify") options.verify = true;
      else if (arg == "--save-raw") options.save_raw = true;
      else if (arg == "--symmetric-lowpass") options.processing.symmetric_lowpass = true;
      else if (arg == "--data-dir" || arg == "--output" || arg == "--backend" || arg == "--iir-mode" ||
               arg == "--lowcut" || arg == "--lowpass" || arg == "--sample-rate" ||
               arg == "--lowcut-order" || arg == "--lowpass-order") {
         require(++i<argc,"Missing value for "+arg);
         if (arg == "--data-dir") options.directory = argv[i];
         else if (arg == "--output") {
            require(argv[i][0] != '\0',"--output requires a nonempty file path");
            options.output = argv[i];
         }
         else if (arg == "--backend") options.backend = argv[i];
         else if (arg == "--iir-mode") options.iir_mode = argv[i];
         else if (arg == "--lowcut") options.processing.lowcut = number(argv[i],arg.c_str());
         else if (arg == "--lowpass") options.processing.lowpass = number(argv[i],arg.c_str());
         else if (arg == "--sample-rate") options.processing.output_rate = number(argv[i],arg.c_str());
         else if (arg == "--lowcut-order") options.processing.lowcut_order = order(argv[i],arg.c_str());
         else options.processing.lowpass_order = order(argv[i],arg.c_str());
      }
      else throw std::runtime_error("Unknown argument: "+arg);
   }
   require(!options.directory.empty(),"--data-dir DIR is required");
   require(options.backend == "cpu" || options.backend == "cuda","--backend must be cpu or cuda");
   require(options.iir_mode == "serial" || options.iir_mode == "chunked","--iir-mode must be serial or chunked");
   require(options.backend != "cpu" || options.iir_mode == "serial","Chunked IIR requires the CUDA backend");
   if (options.output.empty()) options.output = join(options.directory,"sim_outs_processed.h5");
   require(options.output.back() != '/',"Output must name a file");
   return true;
}

class H5Handle {
   hid_t id;
   herr_t (*closer)(hid_t);
public:
   H5Handle(hid_t value, herr_t (*close_function)(hid_t), const std::string &message)
      : id(value), closer(close_function) { require(id>=0,message); }
   ~H5Handle() { if (id>=0) closer(id); }
   operator hid_t() const { return id; }
   void close()
   {
      require(closer(id)>=0,"Cannot close HDF5 handle");
      id = -1;
   }
   H5Handle(const H5Handle &) = delete;
   H5Handle &operator=(const H5Handle &) = delete;
};

size_t product(size_t first, size_t second, const char *description)
{
   require(second == 0 || first <= std::numeric_limits<size_t>::max()/second,
           std::string(description)+" count overflows size_t");
   const size_t count = first*second;
   require(count <= std::numeric_limits<size_t>::max()/sizeof(double) &&
           count <= std::vector<double>().max_size(),std::string(description)+" exceeds host array limits");
   return count;
}

std::vector<hsize_t> shape(hid_t dataset, int rank, const char *name)
{
   H5Handle space(H5Dget_space(dataset),H5Sclose,"Cannot open dataspace");
   require(H5Sget_simple_extent_ndims(space) == rank,std::string(name)+" has an incorrect rank");
   if (!rank) require(H5Sget_simple_extent_type(space) == H5S_SCALAR,
                      std::string(name)+" must be a scalar dataspace");
   std::vector<hsize_t> dimensions(static_cast<size_t>(rank));
   if (rank) require(H5Sget_simple_extent_dims(space,dimensions.data(),nullptr)>=0,"Cannot read dataset dimensions");
   return dimensions;
}

void numeric_type(hid_t dataset, H5T_class_t expected, const char *name)
{
   H5Handle type(H5Dget_type(dataset),H5Tclose,"Cannot open datatype");
   require(H5Tget_class(type) == expected,std::string(name)+" has an incorrect datatype class");
   if (expected == H5T_FLOAT)
      require(H5Tget_size(type) == sizeof(double),std::string(name)+" must be float64");
   else require(H5Tget_size(type) <= sizeof(int64_t),std::string(name)+" exceeds int64 storage");
}

int64_t scalar_integer(hid_t file, const char *name)
{
   H5Handle dataset(H5Dopen2(file,name,H5P_DEFAULT),H5Dclose,"Cannot open "+std::string(name));
   shape(dataset,0,name); numeric_type(dataset,H5T_INTEGER,name);
   int64_t value = 0;
   require(H5Dread(dataset,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)>=0,"Cannot read "+std::string(name));
   return value;
}

double scalar_double(hid_t file, const char *name)
{
   H5Handle dataset(H5Dopen2(file,name,H5P_DEFAULT),H5Dclose,"Cannot open "+std::string(name));
   shape(dataset,0,name); numeric_type(dataset,H5T_FLOAT,name);
   double value = 0;
   require(H5Dread(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)>=0,"Cannot read "+std::string(name));
   require(std::isfinite(value),std::string(name)+" must be finite");
   return value;
}

std::vector<double> matrix(hid_t file, const char *name, std::vector<hsize_t> &dimensions,
                           size_t expected_count, const std::vector<hsize_t> &expected_shape = {})
{
   H5Handle dataset(H5Dopen2(file,name,H5P_DEFAULT),H5Dclose,"Cannot open "+std::string(name));
   dimensions = shape(dataset,2,name); numeric_type(dataset,H5T_FLOAT,name);
   require(dimensions[0] > 0 && dimensions[1] > 0,"Empty matrix "+std::string(name));
   require(dimensions[0] <= std::numeric_limits<size_t>::max() && dimensions[1] <= std::numeric_limits<size_t>::max(),
           "Matrix dimensions exceed size_t");
   const size_t count = product(static_cast<size_t>(dimensions[0]),static_cast<size_t>(dimensions[1]),name);
   require(count == expected_count,std::string(name)+" count differs from Nr/Nt metadata");
   require(expected_shape.empty() || dimensions == expected_shape,std::string(name)+" shape differs from Nr/Nt metadata");
   std::vector<double> values(count);
   require(H5Dread(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,values.data())>=0,"Cannot read "+std::string(name));
   for (double value : values) require(std::isfinite(value),"Nonfinite value in "+std::string(name));
   return values;
}

struct Inputs {
   std::vector<double> data, weights;
   size_t channels, corners, samples;
   double fs;
   bool differentiated;
};

Inputs load(const Options &options)
{
   Inputs input;
   H5Handle communications(H5Fopen(join(options.directory,"comms_out.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),
                            H5Fclose,"Cannot open comms_out.h5");
   const int64_t nr = scalar_integer(communications,"Nr"), nt = scalar_integer(communications,"Nt");
   const int64_t diff = scalar_integer(communications,"diff");
   require(nr>0 && nt>0,"Nr and Nt must be positive");
   require(static_cast<uint64_t>(nr) <= std::numeric_limits<size_t>::max() &&
           static_cast<uint64_t>(nt) <= std::numeric_limits<size_t>::max(),"Nr or Nt exceeds size_t");
   require(diff == 0 || diff == 1,"diff must be 0 or 1");
   std::vector<hsize_t> weight_shape;
   input.weights = matrix(communications,"out_alpha",weight_shape,static_cast<size_t>(nr));
   input.channels = static_cast<size_t>(weight_shape[0]);
   input.corners = static_cast<size_t>(weight_shape[1]);
   input.samples = static_cast<size_t>(nt);
   require(input.weights.size() == static_cast<size_t>(nr),"Nr must equal Nmic*K from out_alpha");
   input.differentiated = diff == 1;

   H5Handle constants(H5Fopen(join(options.directory,"sim_consts.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),
                       H5Fclose,"Cannot open sim_consts.h5");
   const double ts = scalar_double(constants,"Ts");
   require(ts>0,"Ts must be positive");
   input.fs = 1/ts;
   require(std::isfinite(input.fs) && input.fs>0,"Native sample rate is invalid");

   H5Handle outputs(H5Fopen(join(options.directory,"sim_outs.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),
                     H5Fclose,"Cannot open sim_outs.h5");
   std::vector<hsize_t> output_shape;
   input.data = matrix(outputs,"u_out",output_shape,product(static_cast<size_t>(nr),static_cast<size_t>(nt),"input samples"),
                       {static_cast<hsize_t>(nr),static_cast<hsize_t>(nt)});
   require(output_shape[0] == static_cast<hsize_t>(nr) && output_shape[1] == static_cast<hsize_t>(nt),
           "u_out shape must be [Nr,Nt]");
   return input;
}

void check_destination(const Options &options)
{
   struct stat destination;
   if (lstat(options.output.c_str(),&destination)) {
      if (errno == ENOENT) return;
      throw system_error("Inspect output "+options.output);
   }
   require(!S_ISLNK(destination.st_mode),"Output destination must not be a symlink");
   require(S_ISREG(destination.st_mode),"Output destination must be a regular file");
   const char *names[] = {"sim_consts.h5","comms_out.h5","sim_outs.h5","vox_out.h5","sim_mats.h5","cart_grid.h5"};
   for (const char *name : names) {
      struct stat source;
      if (stat(join(options.directory,name).c_str(),&source) == 0)
         require(destination.st_dev != source.st_dev || destination.st_ino != source.st_ino,
                 "Output destination aliases input "+std::string(name));
   }
   require(options.overwrite,"Output already exists; use --overwrite to replace it");
}

class AtomicOutput {
   std::string temporary;
public:
   explicit AtomicOutput(const Options &options)
   {
      check_destination(options);
      const size_t slash = options.output.find_last_of('/');
      const std::string parent = slash == std::string::npos ? "." :
                                 slash == 0 ? "/" : options.output.substr(0,slash);
      std::string pattern = join(parent,".pffdtd-post-XXXXXX");
      std::vector<char> writable(pattern.begin(),pattern.end()); writable.push_back('\0');
      const int fd = mkstemp(writable.data());
      if (fd<0) throw system_error("Create temporary output");
      temporary = writable.data();
      if (close(fd)) {
         const int error = errno; unlink(temporary.c_str()); temporary.clear(); errno = error;
         throw system_error("Close temporary output");
      }
   }
   ~AtomicOutput() { if (!temporary.empty()) unlink(temporary.c_str()); }
   const char *path() const { return temporary.c_str(); }
   void publish(const Options &options)
   {
      check_destination(options);
      if (options.overwrite) {
         if (rename(temporary.c_str(),options.output.c_str())) throw system_error("Publish output");
      }
      else {
         if (link(temporary.c_str(),options.output.c_str())) throw system_error("Publish exclusive output");
         if (unlink(temporary.c_str())) {
            std::fprintf(stderr,"Output published; temporary link cleanup failed: %s\n",std::strerror(errno));
            return;
         }
      }
      temporary.clear();
   }
};

void write_matrix(hid_t file, const char *name, const std::vector<double> &values, size_t rows, size_t columns)
{
   require(values.size() == product(rows,columns,name),"Incorrect output size for "+std::string(name));
   const hsize_t dimensions[] = {static_cast<hsize_t>(rows),static_cast<hsize_t>(columns)};
   H5Handle space(H5Screate_simple(2,dimensions,nullptr),H5Sclose,"Cannot create output dataspace");
   H5Handle dataset(H5Dcreate2(file,name,H5T_IEEE_F64LE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),
                     H5Dclose,"Cannot create "+std::string(name));
   require(H5Dwrite(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,values.data())>=0,
           "Cannot write "+std::string(name));
   dataset.close(); space.close();
}

void write_scalar(hid_t file, const char *name, double value)
{
   H5Handle space(H5Screate(H5S_SCALAR),H5Sclose,"Cannot create scalar dataspace");
   H5Handle dataset(H5Dcreate2(file,name,H5T_IEEE_F64LE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),
                     H5Dclose,"Cannot create "+std::string(name));
   require(H5Dwrite(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)>=0,"Cannot write "+std::string(name));
   dataset.close(); space.close();
}

void write_integer(hid_t file, const char *name, int64_t value)
{
   H5Handle space(H5Screate(H5S_SCALAR),H5Sclose,"Cannot create integer dataspace");
   H5Handle dataset(H5Dcreate2(file,name,H5T_STD_I64LE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),
                     H5Dclose,"Cannot create "+std::string(name));
   require(H5Dwrite(dataset,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)>=0,"Cannot write "+std::string(name));
   dataset.close(); space.close();
}

void write_string(hid_t file, const char *name, const char *value)
{
   H5Handle type(H5Tcopy(H5T_C_S1),H5Tclose,"Cannot create string datatype");
   require(H5Tset_size(type,H5T_VARIABLE)>=0 && H5Tset_cset(type,H5T_CSET_UTF8)>=0,
           "Cannot configure string datatype");
   H5Handle space(H5Screate(H5S_SCALAR),H5Sclose,"Cannot create string dataspace");
   H5Handle dataset(H5Dcreate2(file,name,type,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),
                     H5Dclose,"Cannot create "+std::string(name));
   require(H5Dwrite(dataset,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,&value)>=0,"Cannot write "+std::string(name));
   dataset.close(); space.close(); type.close();
}

void validate_result(const pffdtd_post::PostResult &result, const Inputs &input)
{
   require(result.channels == input.channels && result.samples == input.samples && result.samples_out > 0,
           "Postprocessing returned inconsistent dimensions");
   require(result.raw.size() == product(input.channels,input.samples,"raw output") &&
           result.filtered.size() == product(input.channels,result.samples_out,"filtered output"),
           "Postprocessing returned inconsistent storage");
   require(std::isfinite(result.fs_out) && result.fs_out>0,"Postprocessing returned an invalid sample rate");
   for (double value : result.raw) require(std::isfinite(value),"Reconstructed output contains a nonfinite sample");
   for (double value : result.filtered) require(std::isfinite(value),"Filtered output contains a nonfinite sample");
}

void compare(const std::vector<double> &actual, const std::vector<double> &expected, const char *stage)
{
   require(actual.size() == expected.size(),"Verification shape mismatch for "+std::string(stage));
   double maximum_error = 0;
   for (size_t i=0; i<actual.size(); i++) {
      const double error = std::fabs(actual[i]-expected[i]);
      const double limit = 1e-12+1e-8*std::fabs(expected[i]);
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error>limit) {
         std::fprintf(stderr,"Verification mismatch: stage=%s index=%zu actual=%.17g reference=%.17g limit=%.17g\n",
                      stage,i,actual[i],expected[i],limit);
         throw std::runtime_error("CPU/CUDA verification failed");
      }
      maximum_error = std::max(maximum_error,error);
   }
   std::printf("Verify %s: PASS, max_abs_error=%.17g, tolerance=1e-12+1e-8*abs(reference)\n",stage,maximum_error);
}

int run(Options options)
{
   if (options.backend == "cuda") {
#ifdef __CUDACC__
      int devices = 0;
      const cudaError_t error = cudaGetDeviceCount(&devices);
      if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
          (error == cudaSuccess && devices == 0)) {
         std::fprintf(stderr,"SKIP: no usable CUDA device (%s).\n",cudaGetErrorString(error));
         return 77;
      }
      require(error == cudaSuccess,std::string("cudaGetDeviceCount: ")+cudaGetErrorString(error));
#else
      std::fprintf(stderr,"SKIP: the CUDA backend is unavailable in this build.\n");
      return 77;
#endif
   }
   Inputs input = load(options);
   options.processing.integrate = input.differentiated;
   check_destination(options);
   const auto start = std::chrono::steady_clock::now();
   pffdtd_post::PostResult result;
   if (options.backend == "cpu")
      result = pffdtd_post::process_cpu(input.data,input.weights,input.channels,input.corners,input.samples,input.fs,options.processing);
#ifdef __CUDACC__
   else result = pffdtd_post::process_cuda(input.data,input.weights,input.channels,input.corners,input.samples,input.fs,
                                         options.processing,options.iir_mode == "chunked",options.verify);
#endif
   const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   validate_result(result,input);
   if (options.verify && options.backend == "cuda") {
      const auto reference = pffdtd_post::process_cpu(input.data,input.weights,input.channels,input.corners,input.samples,input.fs,options.processing);
      validate_result(reference,input);
      require(result.samples_out == reference.samples_out && result.fs_out == reference.fs_out,"Verification metadata mismatch");
      compare(result.raw,reference.raw,"receiver reconstruction");
      compare(result.filtered,reference.filtered,"filtered/resampled output");
   }
   else if (options.verify) std::printf("Verify: CPU reference dimensions and finite samples PASS.\n");
   AtomicOutput output(options);
   {
      H5Handle file(H5Fcreate(output.path(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT),H5Fclose,"Cannot create output HDF5");
      write_matrix(file,"r_out_f",result.filtered,result.channels,result.samples_out);
      write_scalar(file,"Fs_f",result.fs_out);
      write_scalar(file,"Fs_native",input.fs);
      write_scalar(file,"lowcut",options.processing.lowcut);
      write_scalar(file,"lowpass",options.processing.lowpass);
      write_integer(file,"lowcut_order",options.processing.lowcut_order);
      write_integer(file,"lowpass_order",options.processing.lowpass_order);
      write_integer(file,"symmetric_lowpass",options.processing.symmetric_lowpass ? 1 : 0);
      write_integer(file,"diff",input.differentiated ? 1 : 0);
      write_string(file,"post_backend",options.backend.c_str());
      write_string(file,"iir_mode",options.iir_mode.c_str());
      write_string(file,"resampler",result.fs_out == input.fs ? "identity" : "kaiser_sinc_analytic_v1");
      write_string(file,"build_revision",PFFDTD_BUILD_REVISION);
      write_string(file,"air_filter","none");
      if (options.save_raw) {
         write_matrix(file,"r_out",result.raw,result.channels,result.samples);
      }
      require(H5Fflush(file,H5F_SCOPE_GLOBAL)>=0,"Cannot flush output HDF5");
      file.close();
   }
   output.publish(options);
   std::printf("Processed %zu receivers: %zu samples at %.17g Hz -> %zu samples at %.17g Hz; "
               "backend=%s, iir=%s, integrate=%d, air=none, process_wall=%.9fs\nOutput: %s\n",
               input.channels,input.samples,input.fs,result.samples_out,result.fs_out,
               options.backend.c_str(),options.iir_mode.c_str(),input.differentiated ? 1 : 0,seconds,options.output.c_str());
   return 0;
}

} // namespace

int main(int argc, char **argv)
{
   try {
      Options options;
      if (!parse(argc,argv,options)) return 0;
      return run(options);
   }
   catch (const std::exception &error) {
      std::fprintf(stderr,"ERROR: %s\n",error.what());
      return 1;
   }
}
