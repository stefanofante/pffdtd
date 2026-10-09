// CUDA/reference gates for air kernels and the complete resident DSP path.
#include "post_cuda.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace {
std::size_t checks=0;
double worst=0;
void compare(const std::vector<double>& a,const std::vector<double>& b,bool peak_relative=false)
{
   if (a.size()!=b.size()) throw std::runtime_error("Air CUDA/reference shape mismatch");
   double peak=0;
   for (double x:b) peak=std::max(peak,std::fabs(x));
   for (std::size_t i=0;i<a.size();++i) {
      const double error=std::fabs(a[i]-b[i]);
      const double tolerance=2e-11+2e-8*(peak_relative ? peak : std::fabs(b[i]));
      if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || error>tolerance) {
         std::fprintf(stderr,"Air mismatch index=%zu actual=%.17g reference=%.17g error=%.17g limit=%.17g\n",
            i,a[i],b[i],error,tolerance);
         throw std::runtime_error("Air CUDA/reference waveform mismatch");
      }
      worst=std::max(worst,error/(1+peak)); ++checks;
   }
}
std::vector<double> signal(std::size_t channels,std::size_t samples,unsigned pattern)
{
   std::vector<double> data(channels*samples,0);
   for (std::size_t ch=0;ch<channels;++ch) {
      if (pattern==0) continue;
      if (pattern==1) { data[ch*samples+samples/3]=0.75/(ch+1); continue; }
      for (std::size_t n=0;n<samples;++n)
         data[ch*samples+n]=(std::sin(0.073*n)+0.25*std::cos(0.39*n))/(ch+1);
   }
   return data;
}
}
int main()
{
   int devices=0;
   const cudaError_t status=cudaGetDeviceCount(&devices);
   if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !devices)) {
      std::fprintf(stderr,"SKIP: no usable CUDA device (%s).\n",cudaGetErrorString(status)); return 77;
   }
   try {
      pffdtd_post::post_cuda_check(status);
      for (const char* mode:{"none","stokes","ola","modal"}) {
         for (std::size_t samples:{std::size_t(1),std::size_t(17),std::size_t(65),std::size_t(257),std::size_t(1025)}) {
            for (unsigned pattern=0;pattern<3;++pattern) {
               auto data=signal(2,samples,pattern);
               pffdtd_post::AirOptions options; options.mode=mode; options.window=68;
               options.temperature=-5; options.humidity=73; options.pressure=89;
               if (options.mode=="modal") options.modal_pad=9.0/48000;
               const auto reference=pffdtd_post::process_air_cpu(data,2,samples,48000,options);
               const auto actual=pffdtd_post::process_air_cuda(data,2,samples,48000,options);
               if (actual.samples!=reference.samples) throw std::runtime_error("Air sample metadata mismatch");
               compare(actual.data,reference.data);
            }
         }
      }
      // Cross the 128-frame batching boundary with non-power-of-two windows.
      for (std::size_t window:{std::size_t(7),std::size_t(63),std::size_t(258),std::size_t(1024)}) {
         auto data=signal(1,8193,2);
         pffdtd_post::AirOptions air; air.mode="ola"; air.window=window;
         compare(pffdtd_post::process_air_cuda(data,1,8193,44100,air).data,
                 pffdtd_post::process_air_cpu(data,1,8193,44100,air).data);
      }
      // Exercise the algorithmic FFT replacement, including prime lengths,
      // zero signal fallback, tail batches, and its independent recurrence.
      for (std::size_t nt:{std::size_t(1025),std::size_t(4097),std::size_t(8192)}) {
         for (unsigned pattern:{1U,2U}) {
            auto data=signal(2,nt,pattern);
            pffdtd_post::AirOptions air; air.mode="modal"; air.modal_method="fft";
            const auto actual=pffdtd_post::process_air_cuda(data,2,nt,48000,air);
            compare(actual.data,pffdtd_post::process_air_cpu(data,2,nt,48000,air).data);
            air.modal_method="recurrence";
            // The double recurrence accumulates drift near the marginal poles;
            // its native long-double gate separately establishes FFT accuracy.
            compare(actual.data,pffdtd_post::process_air_cpu(data,2,nt,48000,air).data,true);
         }
      }
      for (const char* mode:{"none","stokes","ola","modal"}) {
         for (bool chunked:{false,true}) {
            const std::size_t nt=257;
            auto input=signal(6,nt,2);
            const std::vector<double> weights={0.2,0.3,0.5,0.25,0.5,0.25};
            pffdtd_post::PostOptions options; options.integrate=true;
            options.lowcut=100; options.lowcut_order=4; options.lowpass=5000;
            options.lowpass_order=4; options.symmetric_lowpass=true; options.output_rate=44100;
            pffdtd_post::AirOptions air; air.mode=mode; air.window=64;
            const auto actual=pffdtd_post::process_cuda(input,weights,2,3,nt,48000,options,chunked,true,&air);
            auto reference=pffdtd_post::process_cpu(input,weights,2,3,nt,48000,options);
            const auto attenuated=pffdtd_post::process_air_cpu(reference.filtered,2,reference.samples_out,reference.fs_out,air);
            if (actual.samples_out!=attenuated.samples) throw std::runtime_error("Resident air metadata mismatch");
            compare(actual.raw,reference.raw); compare(actual.filtered,attenuated.data);
         }
      }
      std::printf("CUDA air/resident post: PASS %zu samples, worst scaled absolute error %.17g\n",checks,worst);
      return 0;
   }
   catch(const std::exception& error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
