// Native CUDA benchmark for already prepared PFFDTD HDF5 input directories.
// Run from the directory containing sim_consts.h5, vox_out.h5,
// comms_out.h5 and sim_mats.h5. No output HDF5 files are written.

#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fdtd_data.h>
#include <gpu_engine.h>

#ifndef PFFDTD_BUILD_REVISION
#define PFFDTD_BUILD_REVISION "unknown"
#endif

namespace {

struct Options {
   int warmup = 1;
   int repetitions = 12;
   bool verbose = false;
   bool overwrite_csv = false;
   std::string csv;
};

void usage(const char *program)
{
   std::printf("Usage: %s [--warmup N] [--repetitions N] [--csv PATH]\n"
               "       [--overwrite-csv] [--verbose] [--help]\n\n"
               "Run from an existing HDF5 simulation directory. Select exactly one GPU\n"
               "with CUDA_VISIBLE_DEVICES. Default: one warmup round and twelve measured\n"
               "rounds, each comparing sync/events/graphs with unfused, generic-fused,\n"
               "reload-fused and fixed-fused boundaries (twelve modes).\n"
               "Every result must exactly match a synchronous, unfused reference.\n"
               "CSV creation refuses existing files unless --overwrite-csv is supplied.\n"
               "No Python processing or simulation-output files are produced.\n"
               "Timing is external wall time for run_sim, including allocation, graph\n"
               "construction, transfers, cleanup and context reset; HDF5 loading, input\n"
               "scaling and output verification are excluded. Loop clocks are reported\n"
               "separately and are not used to compare modes.\n",program);
}

int parse_count(const char *text, bool allow_zero, const char *option)
{
   if (!text || !*text || *text == '-')
      throw std::runtime_error(std::string("Invalid count for ")+option);
   char *end = nullptr;
   errno = 0;
   const long value = std::strtol(text,&end,10);
   if (errno || *end || value < (allow_zero ? 0 : 1) ||
       value > std::numeric_limits<int>::max())
      throw std::runtime_error(std::string("Invalid count for ")+option);
   return static_cast<int>(value);
}

bool parse_options(int argc, char **argv, Options &options)
{
   for (int i=1; i<argc; i++) {
      const std::string arg(argv[i]);
      if (arg == "--help") { usage(argv[0]); return false; }
      if (arg == "--verbose") options.verbose = true;
      else if (arg == "--overwrite-csv") options.overwrite_csv = true;
      else if (arg == "--warmup" || arg == "--repetitions" || arg == "--csv") {
         if (++i == argc) throw std::runtime_error("Missing value for "+arg);
         if (arg == "--warmup") options.warmup = parse_count(argv[i],true,argv[i-1]);
         else if (arg == "--repetitions") options.repetitions = parse_count(argv[i],false,argv[i-1]);
         else {
            options.csv = argv[i];
            if (options.csv.empty()) throw std::runtime_error("Empty CSV path");
         }
      }
      else throw std::runtime_error("Unknown argument: "+arg);
   }
   if (options.overwrite_csv && options.csv.empty())
      throw std::runtime_error("--overwrite-csv requires --csv PATH");
   return true;
}

std::runtime_error system_error(const std::string &operation)
{
   return std::runtime_error(operation+": "+std::strerror(errno));
}

class Environment {
   struct Entry { std::string name, value; bool present; };
   std::vector<Entry> entries;
public:
   Environment()
   {
      const char *names[] = {"PFFDTD_ASYNC","PFFDTD_GRAPHS",
                            "PFFDTD_BOUNDARY_FUSED","PFFDTD_ADE_MODE","PFFDTD_PROGRESS"};
      for (const char *name : names) {
         const char *value = std::getenv(name);
         entries.push_back(Entry{name,value ? value : "",value != nullptr});
      }
   }
   ~Environment()
   {
      for (const Entry &entry : entries) {
         const int result = entry.present ? setenv(entry.name.c_str(),entry.value.c_str(),1)
                                          : unsetenv(entry.name.c_str());
         if (result) std::fprintf(stderr,"Cannot restore %s: %s\n",
                                  entry.name.c_str(),std::strerror(errno));
      }
   }
   void set(const char *name, const char *value)
   {
      if (setenv(name,value,1)) throw system_error(std::string("setenv ")+name);
   }
};

// Only stdout is redirected. CUDA errors and benchmark diagnostics retain stderr.
class QuietStdout {
   int saved = -1;
public:
   explicit QuietStdout(bool quiet)
   {
      if (!quiet) return;
      if (std::fflush(stdout)) throw system_error("flush stdout");
      saved = dup(STDOUT_FILENO);
      if (saved == -1) throw system_error("dup stdout");
      const int null_fd = open("/dev/null",O_WRONLY);
      if (null_fd == -1) {
         const int error = errno; close(saved); saved = -1; errno = error;
         throw system_error("open /dev/null");
      }
      const int result = dup2(null_fd,STDOUT_FILENO);
      const int error = errno;
      close(null_fd);
      if (result == -1) {
         close(saved); saved = -1; errno = error;
         throw system_error("redirect stdout");
      }
   }
   void restore()
   {
      if (saved == -1) return;
      const bool failed_flush = std::fflush(stdout) != 0;
      const int flush_error = errno;
      if (dup2(saved,STDOUT_FILENO) == -1) throw system_error("restore stdout");
      close(saved); saved = -1;
      if (failed_flush) { errno = flush_error; throw system_error("flush engine stdout"); }
   }
   ~QuietStdout()
   {
      if (saved != -1) {
         std::fflush(stdout);
         if (dup2(saved,STDOUT_FILENO) == -1)
            std::fprintf(stderr,"Cannot restore stdout: %s\n",std::strerror(errno));
         close(saved);
      }
   }
   QuietStdout(const QuietStdout &) = delete;
   QuietStdout &operator=(const QuietStdout &) = delete;
};

class Simulation {
public:
   SimData data = {};
   bool loaded = false;
   ~Simulation() { if (loaded) free_sim_data(&data); }
};

std::string working_directory()
{
   std::vector<char> buffer(1024);
   while (!getcwd(buffer.data(),buffer.size())) {
      if (errno != ERANGE) throw system_error("getcwd");
      if (buffer.size() > std::numeric_limits<size_t>::max()/2)
         throw std::runtime_error("Working directory is too long");
      buffer.resize(buffer.size()*2);
   }
   return buffer.data();
}

std::string csv_string(const std::string &value)
{
   std::string result = "\"";
   for (char c : value) { if (c == '"') result += '"'; result += c; }
   return result+'"';
}

class Csv {
   FILE *file = nullptr;
public:
   explicit Csv(const Options &options)
   {
      if (options.csv.empty()) return;
      const int flags = O_WRONLY|O_CREAT|O_NOFOLLOW|O_NONBLOCK|
                        (options.overwrite_csv ? 0 : O_EXCL);
      const int fd = open(options.csv.c_str(),flags,0666);
      if (fd == -1) throw system_error("Create CSV "+options.csv);
      struct stat info;
      if (fstat(fd,&info) || !S_ISREG(info.st_mode)) {
         const int error = errno; close(fd); errno = error;
         throw std::runtime_error("CSV destination must be a regular file");
      }
      const char *inputs[] = {"sim_consts.h5","vox_out.h5","comms_out.h5","sim_mats.h5"};
      for (const char *input : inputs) {
         struct stat input_info;
         if (stat(input,&input_info) == 0 && info.st_dev == input_info.st_dev &&
             info.st_ino == input_info.st_ino) {
            close(fd);
            throw std::runtime_error("CSV destination aliases input file "+std::string(input));
         }
      }
      if (options.overwrite_csv && ftruncate(fd,0)) {
         const int error = errno; close(fd); errno = error;
         throw system_error("Truncate CSV");
      }
      file = fdopen(fd,"w");
      if (!file) { const int error = errno; close(fd); errno = error; throw system_error("fdopen CSV"); }
      std::fputs("round,position,scheduler,boundary_fused,ade_mode,precision,revision,gpu,cc_major,cc_minor,"
                 "gpu_total_memory_bytes,cuda_toolkit,cuda_runtime,cuda_driver,"
                 "nx,ny,nz,npts,nb,nbl,nba,nt,ns,nr,nm,fcc_flag,"
                 "wall_seconds,loop_reported_seconds,loop_clock,exact_output,"
                 "baseline_nonzero_samples,input_directory,context_reset_each_run\n",file);
      try { flush(); }
      catch (...) { std::fclose(file); file = nullptr; throw; }
   }
   ~Csv()
   {
      if (file) {
         std::fclose(file);
         std::fprintf(stderr,"CSV incomplete: benchmark stopped before completing all measured samples.\n");
      }
   }
   FILE *get() const { return file; }
   void flush()
   {
      if (file && (std::ferror(file) || std::fflush(file)))
         throw system_error("Write CSV");
   }
   void finish()
   {
      if (!file) return;
      flush();
      FILE *closing = file; file = nullptr;
      if (std::fclose(closing)) throw system_error("Close CSV");
   }
};

struct Mode { const char *scheduler; bool fused; const char *ade_mode; };
const Mode modes[] = {{"sync",false,"generic"},{"events",false,"generic"},{"graphs",false,"generic"},
                      {"sync",true,"generic"},{"events",true,"generic"},{"graphs",true,"generic"},
                      {"sync",true,"reload"},{"events",true,"reload"},{"graphs",true,"reload"},
                      {"sync",true,"fixed"},{"events",true,"fixed"},{"graphs",true,"fixed"}};
const size_t mode_count = sizeof(modes)/sizeof(modes[0]);

std::vector<size_t> run_order(uint64_t round)
{
   // A Williams design for an even number of modes balances positions and directed neighbors in
   // each complete cycle. Reverse only whole cycles, never alternate rows.
   static_assert(mode_count%2 == 0,"The Williams order requires an even mode count");
   std::vector<size_t> order;
   for (size_t position=0; position<mode_count; position++) {
      const size_t offset = round%mode_count;
      const bool reverse = (round/mode_count)&1;
      const size_t base_position = reverse ? mode_count-1-position : position;
      const size_t base = base_position == 0 ? 0 :
                          base_position%2 ? (base_position+1)/2 : mode_count-base_position/2;
      order.push_back((offset+base)%mode_count);
   }
   return order;
}

void select_mode(Environment &environment, const Mode &mode)
{
   environment.set("PFFDTD_ASYNC",std::strcmp(mode.scheduler,"events") == 0 ? "1" : "0");
   environment.set("PFFDTD_GRAPHS",std::strcmp(mode.scheduler,"graphs") == 0 ? "1" : "0");
   environment.set("PFFDTD_BOUNDARY_FUSED",mode.fused ? "1" : "0");
   environment.set("PFFDTD_ADE_MODE",mode.ade_mode);
   environment.set("PFFDTD_PROGRESS","0");
}

size_t validate_data(const SimData &sd)
{
   if (sd.Nt <= 0 || sd.Nr <= 0 || sd.Ns <= 0 || sd.Nx < 3 || sd.Ny < 3 || sd.Nz < 3)
      throw std::runtime_error("Benchmark requires positive Nt/Nr/Ns and grid axes of at least three cells");
   if (sd.fcc_flag != 0 && sd.fcc_flag != 2)
      throw std::runtime_error("CUDA requires Cartesian or folded FCC input data");
   if (sd.Ny > INT64_MAX/sd.Nz || sd.Nx > INT64_MAX/(sd.Ny*sd.Nz) ||
       sd.Npts != sd.Nx*sd.Ny*sd.Nz)
      throw std::runtime_error("Inconsistent grid size or index overflow");
   if (!pffdtd::scheduler_receivers_interior(sd.out_ixyz,sd.Nr,sd.Nx,sd.Ny,sd.Nz))
      throw std::runtime_error("All receivers must be interior; async fallback would invalidate the comparison");
   if (static_cast<uint64_t>(sd.Nt) > std::numeric_limits<size_t>::max()/sizeof(double)/static_cast<uint64_t>(sd.Nr) ||
       static_cast<uint64_t>(sd.Nt) > std::numeric_limits<size_t>::max()/sizeof(double)/static_cast<uint64_t>(sd.Ns))
      throw std::runtime_error("Input/output sample count overflow");
   if (!sd.in_ixyz || !sd.in_sigs || !sd.u_out)
      throw std::runtime_error("Missing input or output storage");
   std::vector<int64_t> source_indices(sd.in_ixyz,sd.in_ixyz+sd.Ns);
   for (int64_t index : source_indices) {
      if (index < 0 || index >= sd.Npts) throw std::runtime_error("Source index is outside the grid");
   }
   std::sort(source_indices.begin(),source_indices.end());
   if (std::adjacent_find(source_indices.begin(),source_indices.end()) != source_indices.end())
      throw std::runtime_error("Duplicate source indices would race in batched CUDA injection");
   double max_input = 0;
   for (int64_t i=0; i<sd.Ns*sd.Nt; i++) {
      if (!std::isfinite(sd.in_sigs[i])) throw std::runtime_error("Input contains a nonfinite sample");
      max_input = std::max(max_input,std::fabs(sd.in_sigs[i]));
   }
   if (!(max_input > 0)) throw std::runtime_error("At least one nonzero input sample is required for scaling");
   return static_cast<size_t>(sd.Nr)*static_cast<size_t>(sd.Nt);
}

struct Sample { double wall, loop; };

Sample run_one(SimData &sd, size_t output_count, Environment &environment,
               const Mode &mode, bool verbose)
{
   select_mode(environment,mode);
   // NaNs reveal any missing output writes; the engine starts device state afresh.
   std::fill(sd.u_out,sd.u_out+output_count,std::numeric_limits<double>::quiet_NaN());
   QuietStdout quiet(!verbose);
   const auto start = std::chrono::steady_clock::now();
   const double loop = run_sim(&sd);
   const auto end = std::chrono::steady_clock::now();
   quiet.restore();
   const double wall = std::chrono::duration<double>(end-start).count();
   if (!std::isfinite(wall) || wall <= 0 || !std::isfinite(loop) || loop < 0)
      throw std::runtime_error("Engine produced an invalid timing value");
   return Sample{wall,loop};
}

uint64_t double_bits(double value)
{
   uint64_t bits;
   std::memcpy(&bits,&value,sizeof(bits));
   return bits;
}

void check_output(const SimData &sd, const std::vector<double> &reference,
                  const Mode &mode, const char *phase, int round)
{
   for (size_t i=0; i<reference.size(); i++) {
      if (!std::isfinite(sd.u_out[i]) || double_bits(sd.u_out[i]) != double_bits(reference[i])) {
         std::fprintf(stderr,"Output mismatch: phase=%s round=%d scheduler=%s fused=%d ade_mode=%s "
                             "receiver=%llu sample=%llu reference=%.17g actual=%.17g "
                             "reference_bits=%016llx actual_bits=%016llx\n",
                      phase,round,mode.scheduler,mode.fused ? 1 : 0,mode.ade_mode,
                      static_cast<unsigned long long>(i/static_cast<size_t>(sd.Nt)),
                      static_cast<unsigned long long>(i%static_cast<size_t>(sd.Nt)),
                      reference[i],sd.u_out[i],
                      static_cast<unsigned long long>(double_bits(reference[i])),
                      static_cast<unsigned long long>(double_bits(sd.u_out[i])));
         throw std::runtime_error("Exact output verification failed; timing comparison aborted");
      }
   }
}

const char *loop_clock(const Mode &mode)
{
   return std::strcmp(mode.scheduler,"sync") == 0 ? "cuda_events_loop" : "host_wall_loop";
}

void write_row(Csv &csv, int round, size_t position, const Mode &mode, const Sample &sample,
               const cudaDeviceProp &gpu, int runtime, int driver, const SimData &sd,
               size_t baseline_nonzero, const std::string &directory)
{
   if (!csv.get()) return;
   const std::string gpu_name = csv_string(gpu.name);
   const std::string revision = csv_string(PFFDTD_BUILD_REVISION);
   const std::string source = csv_string(directory);
   const std::string toolkit = std::to_string(__CUDACC_VER_MAJOR__)+"."+
                               std::to_string(__CUDACC_VER_MINOR__)+"."+
                               std::to_string(__CUDACC_VER_BUILD__);
   std::fprintf(csv.get(),"%d,%zu,%s,%d,%s,FP%d,%s,%s,%d,%d,%llu,%s,%d,%d,"
                         "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%d,"
                         "%.17g,%.17g,%s,1,%zu,%s,1\n",
                round,position,mode.scheduler,mode.fused ? 1 : 0,mode.ade_mode,static_cast<int>(sizeof(Real)*8),
                revision.c_str(),gpu_name.c_str(),gpu.major,gpu.minor,static_cast<unsigned long long>(gpu.totalGlobalMem),
                toolkit.c_str(),runtime,driver,
                static_cast<long long>(sd.Nx),static_cast<long long>(sd.Ny),static_cast<long long>(sd.Nz),
                static_cast<long long>(sd.Npts),static_cast<long long>(sd.Nb),static_cast<long long>(sd.Nbl),
                static_cast<long long>(sd.Nba),static_cast<long long>(sd.Nt),static_cast<long long>(sd.Ns),
                static_cast<long long>(sd.Nr),static_cast<int>(sd.Nm),static_cast<int>(sd.fcc_flag),
                sample.wall,sample.loop,loop_clock(mode),baseline_nonzero,source.c_str());
   csv.flush();
}

void check_cuda(cudaError_t error, const char *operation)
{
   if (error != cudaSuccess)
      throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}

int benchmark(const Options &options)
{
#if defined(SCHEDULER_SYNC) || defined(BOUNDARY_SEPARATE)
   (void)options;
   std::fprintf(stderr,"Benchmark comparison requires a build without SCHEDULER_SYNC or BOUNDARY_SEPARATE.\n");
   return EXIT_FAILURE;
#else
   int devices = 0;
   const cudaError_t count_error = cudaGetDeviceCount(&devices);
   if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
       (count_error == cudaSuccess && devices == 0)) {
      std::fprintf(stderr,"SKIP: no usable CUDA device (%s).\n",cudaGetErrorString(count_error));
      return 77;
   }
   check_cuda(count_error,"cudaGetDeviceCount");
   if (devices != 1) throw std::runtime_error("Select exactly one GPU with CUDA_VISIBLE_DEVICES");
   const char *blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
   if (blocking && blocking[0] == '1')
      throw std::runtime_error("Unset CUDA_LAUNCH_BLOCKING=1 before benchmarking");
   cudaDeviceProp gpu;
   int runtime = 0, driver = 0;
   check_cuda(cudaGetDeviceProperties(&gpu,0),"cudaGetDeviceProperties");
   check_cuda(cudaRuntimeGetVersion(&runtime),"cudaRuntimeGetVersion");
   check_cuda(cudaDriverGetVersion(&driver),"cudaDriverGetVersion");
   check_cuda(cudaSetDevice(0),"cudaSetDevice");
   check_cuda(cudaDeviceReset(),"initial cudaDeviceReset");

   const char *files[] = {"sim_consts.h5","vox_out.h5","comms_out.h5","sim_mats.h5"};
   for (const char *filename : files) {
      if (access(filename,R_OK)) throw system_error(std::string("Read input ")+filename);
   }
   const std::string directory = working_directory();
   Simulation simulation;
   {
      QuietStdout quiet(!options.verbose);
      load_sim_data(&simulation.data);
      simulation.loaded = true;
      quiet.restore();
   }
   SimData &sd = simulation.data;
   const size_t output_count = validate_data(sd);
   {
      QuietStdout quiet(!options.verbose);
      scale_input(&sd);
      quiet.restore();
   }
   if (!std::isfinite(sd.infac) || sd.infac <= 0)
      throw std::runtime_error("Input scaling produced an invalid gain");
   for (int64_t i=0; i<sd.Ns*sd.Nt; i++) {
      if (!std::isfinite(sd.in_sigs[i])) throw std::runtime_error("Input scaling produced a nonfinite sample");
   }

   std::printf("GPU: %s; CC %d.%d; memory %llu bytes; precision FP%zu\n",
               gpu.name,gpu.major,gpu.minor,static_cast<unsigned long long>(gpu.totalGlobalMem),sizeof(Real)*8);
   std::printf("CUDA toolkit: %d.%d.%d; runtime %d; driver %d\n",
               __CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__,runtime,driver);
   std::printf("Build revision: %s\n",PFFDTD_BUILD_REVISION);
   std::printf("Input: %s\nGrid: %lld x %lld x %lld; Npts=%lld; Nb=%lld; Nbl=%lld; "
               "Nba=%lld; Nt=%lld; Ns=%lld; Nr=%lld; Nm=%d; FCC=%d\n",
               directory.c_str(),static_cast<long long>(sd.Nx),static_cast<long long>(sd.Ny),
               static_cast<long long>(sd.Nz),static_cast<long long>(sd.Npts),static_cast<long long>(sd.Nb),
               static_cast<long long>(sd.Nbl),static_cast<long long>(sd.Nba),static_cast<long long>(sd.Nt),
               static_cast<long long>(sd.Ns),static_cast<long long>(sd.Nr),static_cast<int>(sd.Nm),
               static_cast<int>(sd.fcc_flag));
   std::printf("Rounds: warmup=%d, measured=%d; rotating/reversed interleaving of twelve modes.\n"
               "Every complete twelve-round cycle balances positions and adjacent pairs; partial cycles do not.\n"
               "GPU clocks, power and temperature are not sampled or controlled by this harness.\n"
               "Metric: external run_sim wall seconds, including setup/cleanup and graph construction.\n"
               "Each run resets the CUDA context; warmup may warm host/disk caches and hardware,\n"
               "but does not preserve device allocations, graph executables or the CUDA context.\n"
               "Raw outputs share one input scaling; exact comparison excludes rescaling and I/O.\n",
               options.warmup,options.repetitions);
   std::fflush(stdout);
   Csv csv(options);
   Environment environment;
   const Sample baseline = run_one(sd,output_count,environment,modes[0],options.verbose);
   std::vector<double> reference(sd.u_out,sd.u_out+output_count);
   size_t nonzero = 0;
   for (double value : reference) {
      if (!std::isfinite(value)) throw std::runtime_error("Synchronous baseline contains a nonfinite output");
      if (value != 0) nonzero++;
   }
   std::printf("Reference: sync, boundary_fused=0, ade_mode=generic; wall=%.9fs; nonzero outputs=%zu/%zu.\n",
               baseline.wall,nonzero,output_count);
   std::vector<double> timings[mode_count];
   for (int phase=0; phase<2; phase++) {
      const int rounds = phase == 0 ? options.warmup : options.repetitions;
      for (int round=0; round<rounds; round++) {
         const std::vector<size_t> order = run_order(static_cast<uint64_t>(round));
         for (size_t position=0; position<order.size(); position++) {
            const size_t index = order[position];
            const Mode &mode = modes[index];
            const Sample sample = run_one(sd,output_count,environment,mode,options.verbose);
            check_output(sd,reference,mode,phase == 0 ? "warmup" : "measured",round);
            if (phase != 0) {
               timings[index].push_back(sample.wall);
               write_row(csv,round,position,mode,sample,gpu,runtime,driver,sd,nonzero,directory);
            }
            std::printf("%s round=%d mode=%s fused=%d ade_mode=%s wall=%.9fs "
                        "loop_reported=%.9fs loop_clock=%s exact=PASS\n",
                        phase == 0 ? "warmup" : "measured",round,mode.scheduler,mode.fused ? 1 : 0,mode.ade_mode,
                        sample.wall,sample.loop,loop_clock(mode));
            std::fflush(stdout);
         }
      }
   }
   csv.finish();
   std::printf("\nExternal wall-time distribution (seconds; loop clocks are not compared):\n"
               "scheduler boundary_fused ade_mode median min max spread(max-min) samples\n");
   for (size_t index=0; index<mode_count; index++) {
      std::vector<double> values = timings[index];
      std::sort(values.begin(),values.end());
      const size_t middle = values.size()/2;
      const double median = values.size()%2 ? values[middle] : (values[middle-1]+values[middle])*0.5;
      std::printf("%-9s %d %-7s %.9f %.9f %.9f %.9f %zu\n",
                  modes[index].scheduler,modes[index].fused ? 1 : 0,modes[index].ade_mode,
                  median,values.front(),values.back(),values.back()-values.front(),values.size());
   }
   std::printf("PASS: every warmup and measured output exactly matches the synchronous reference.\n");
   if (!options.csv.empty()) std::printf("CSV: %s (measured samples only).\n",options.csv.c_str());
   return EXIT_SUCCESS;
#endif
}

} // namespace

int main(int argc, char **argv)
{
   try {
      Options options;
      if (!parse_options(argc,argv,options)) return EXIT_SUCCESS;
      return benchmark(options);
   }
   catch (const std::exception &error) {
      std::fprintf(stderr,"ERROR: %s\n",error.what());
      return EXIT_FAILURE;
   }
}
