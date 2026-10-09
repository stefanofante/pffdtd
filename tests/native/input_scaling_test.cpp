// Regression against the original normalization arithmetic, plus fatal-input
// checks isolated with fork so an intentional EXIT_FAILURE cannot stop tests.
#include <cpu_engine.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

static std::size_t checks = 0, rejected = 0;

static void check(bool condition, const char *message)
{
   ++checks;
   if (!condition) throw std::runtime_error(message);
}

static bool same_bits(double a, double b)
{
   return std::memcmp(&a,&b,sizeof(double)) == 0;
}

static double legacy_scale(std::vector<double> &input, int64_t sources, int64_t samples)
{
   double maximum = 0.0;
   for (int64_t n = 0; n < samples; ++n)
      for (int64_t source = 0; source < sources; ++source)
         maximum = MAX(maximum,std::fabs(input[static_cast<std::size_t>(source*samples+n)]));
   const double aexp = 0.5;
   const int32_t exponent = static_cast<int32_t>(std::round(aexp*REAL_MAX_EXP+(1-aexp)*REAL_MIN_EXP));
   const double norm = std::pow(2.0,exponent);
   const double gain = norm/maximum;
   const double factor = 1.0/gain;
   for (int64_t source = 0; source < sources; ++source)
      for (int64_t n = 0; n < samples; ++n)
         input[static_cast<std::size_t>(source*samples+n)] *= gain;
   return factor;
}

static void normal_case(const std::vector<double> &input, int64_t sources, int64_t samples)
{
   check(input.size() == static_cast<std::size_t>(sources*samples),"invalid test shape");
   std::vector<double> expected = input, actual = input;
   const double factor = legacy_scale(expected,sources,samples);
   check(std::isfinite(factor) && factor > 0,"test requires representable original gain");
   SimData data = {};
   data.Ns = sources; data.Nt = samples; data.in_sigs = actual.data(); data.infac = -1;
   scale_input(&data);
   check(data.in_sigs == actual.data(),"normalization replaced source storage");
   check(same_bits(data.infac,factor),"normal gain changed from original arithmetic");
   for (std::size_t i = 0; i < input.size(); ++i)
      check(same_bits(actual[i],expected[i]),"normalized sample changed from original arithmetic");
}

static void expect_failure(SimData data, const char *diagnostic)
{
   int descriptor[2];
   check(pipe(descriptor) == 0,"cannot create diagnostic pipe");
   std::fflush(nullptr);
   const pid_t child = fork();
   check(child >= 0,"cannot fork validation case");
   if (child == 0) {
      close(descriptor[0]);
      if (dup2(descriptor[1],STDERR_FILENO) < 0) _exit(125);
      close(descriptor[1]);
      scale_input(&data);
      _exit(0);
   }
   close(descriptor[1]);
   std::string error;
   char buffer[256];
   for (;;) {
      const ssize_t count = read(descriptor[0],buffer,sizeof(buffer));
      if (count < 0 && errno == EINTR) continue;
      check(count >= 0,"cannot read child diagnostic");
      if (!count) break;
      error.append(buffer,static_cast<std::size_t>(count));
   }
   close(descriptor[0]);
   int status = 0;
   pid_t waited;
   do { waited = waitpid(child,&status,0); } while (waited < 0 && errno == EINTR);
   check(waited == child,"cannot reap validation child");
   check(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE,
         "invalid input must exit EXIT_FAILURE rather than crash or pass");
   check(error.find(diagnostic) != std::string::npos,"expected scaling diagnostic missing");
   ++rejected;
}

static void test_normal()
{
   normal_case({0.0,-0.0,0.125,-0.5,0.25,-2.0},2,3);
   normal_case({std::numeric_limits<double>::max(),-1.0,0.0,-0.0},1,4);
   normal_case({std::nextafter(std::numeric_limits<double>::min(),1.0),-0.0},1,2);
   normal_case({0.0,-16.0,8.0,0.0,-4.0,2.0,0.0,-1.0,0.5},3,3);
   std::mt19937_64 random(0x7363616c696e67ULL);
   for (int sample_case = 0; sample_case < 100; ++sample_case) {
      const int64_t sources = 1+static_cast<int64_t>(random()%9);
      const int64_t samples = 1+static_cast<int64_t>(random()%71);
      std::vector<double> input(static_cast<std::size_t>(sources*samples));
      const int exponent = static_cast<int>(random()%1800)-900;
      for (double &value : input) {
         const double mantissa = static_cast<double>(random()%2000001)/1000000.0-1.0;
         value = std::ldexp(mantissa,exponent);
      }
      normal_case(input,sources,samples);
   }
}

static void test_zero()
{
   std::vector<double> input = {0.0,-0.0,-0.0,0.0,-0.0,0.0};
   const auto original = input;
   SimData data = {};
   data.Ns = 2; data.Nt = 3; data.in_sigs = input.data(); data.infac = -1;
   scale_input(&data);
   check(data.infac == 1.0,"silent source scaling must be identity");
   for (std::size_t i = 0; i < input.size(); ++i)
      check(same_bits(input[i],original[i]),"zero source sign or value changed");
   std::vector<double> output(12,0.0);
   data.Nr = 4; data.u_out = output.data();
   rescale_output(&data);
   for (double value : output) check(value == 0.0 && std::isfinite(value),"zero output rescaling produced NaN");
   data.Ns = 0; data.Nt = INT64_MAX; data.in_sigs = nullptr; data.infac = -1;
   scale_input(&data);
   check(data.infac == 1.0,"empty source set must preserve identity gain");
}

static void test_zero_forward()
{
   omp_set_num_threads(2);
   check(setenv("PFFDTD_PROGRESS","0",1) == 0,"cannot suppress forward progress");
   for (int8_t fcc : {int8_t(0),int8_t(1),int8_t(2)}) {
      SimData data = {};
      data.Nx = 8; data.Ny = fcc == 2 ? 5 : 8; data.Nz = 10;
      data.Npts = data.Nx*data.Ny*data.Nz;
      data.Ns = 1; data.Nr = 2; data.Nt = 7;
      data.fcc_flag = fcc; data.NN = fcc ? 12 : 6;
      data.a1 = Real(1.4); data.a2 = fcc ? Real(0.05) : Real(0.1);
      data.sl2 = Real(0.1); data.l = 0.5; data.lo2 = Real(0.25);
      std::vector<double> source(static_cast<std::size_t>(data.Nt),-0.0);
      std::vector<double> output(static_cast<std::size_t>(data.Nr*data.Nt),1.0);
      std::vector<uint8_t> mask(static_cast<std::size_t>((data.Npts+7)/8),0);
      const int64_t source_y = fcc == 2 ? 2 : 3;
      int64_t source_index = 3*data.Ny*data.Nz+source_y*data.Nz+4;
      int64_t receivers[2] = {source_index,source_index+data.Ny*data.Nz+1};
      data.in_sigs = source.data(); data.u_out = output.data(); data.bn_mask = mask.data();
      data.in_ixyz = &source_index; data.out_ixyz = receivers;
      scale_input(&data);
      check(data.infac == 1,"forward zero-source gain must stay identity");
      check(std::isfinite(run_sim(&data)),"zero-source forward timing is invalid");
      rescale_output(&data);
      for (double value : output)
         check(std::isfinite(value) && value == 0.0,"zero source must yield zero forward pressure");
   }
}

static void test_rejections()
{
   for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity()}) {
      for (std::size_t index : {std::size_t(0),std::size_t(3),std::size_t(5)}) {
         std::vector<double> input(6,0.0);
         input[index] = invalid;
         SimData data = {};
         data.Ns = 2; data.Nt = 3; data.in_sigs = input.data();
         expect_failure(data,"nonfinite input");
      }
   }
   for (double tiny : {std::numeric_limits<double>::denorm_min(),
                      std::numeric_limits<double>::min(),
                      -std::numeric_limits<double>::min()}) {
      std::vector<double> input = {0.0,tiny,-0.0};
      SimData data = {};
      data.Ns = 1; data.Nt = 3; data.in_sigs = input.data();
      expect_failure(data,"positive finite gains");
   }
   double signal = 1.0;
   SimData data = {};
   data.in_sigs = &signal;
   for (const auto &counts : {std::pair<int64_t,int64_t>{-1,1}, {1,-1}, {1,0}, {INT64_MAX,2}}) {
      data.Ns = counts.first; data.Nt = counts.second;
      expect_failure(data,"invalid or overflowing");
   }
   data.Ns = 1; data.Nt = INT64_MAX;
   expect_failure(data,"exceeds addressable bytes");
   data.Ns = 1; data.Nt = 1; data.in_sigs = nullptr;
   expect_failure(data,"source storage is missing");
}

int main()
{
   try {
      // Keep the legacy informational gain line out of the regression log.
      std::fflush(stdout);
      const int saved = dup(STDOUT_FILENO);
      check(saved >= 0,"cannot preserve stdout");
      FILE *quiet = std::fopen("/dev/null","w");
      check(quiet && dup2(fileno(quiet),STDOUT_FILENO) >= 0,"cannot suppress gain log");
      std::fclose(quiet);
      test_normal(); test_zero(); test_zero_forward(); test_rejections();
      std::fflush(stdout);
      check(dup2(saved,STDOUT_FILENO) >= 0,"cannot restore stdout");
      close(saved);
      std::printf("input_scaling FP%d: PASS (%zu bit/guard checks, %zu fatal cases, zero/signed-zero identity)\n",
                  PRECISION == 1 ? 32 : 64,checks,rejected);
      return 0;
   }
   catch (const std::exception &error) {
      std::fprintf(stderr,"input_scaling: FAIL: %s\n",error.what());
      return 1;
   }
}
