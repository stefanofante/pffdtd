#include "post_cuda.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
using namespace pffdtd_post;

void compare(const std::vector<double>& reference, const std::vector<double>& actual)
{
   if (reference.size()!=actual.size()) throw std::runtime_error("CUDA post output size mismatch");
   double peak=0,error=0;
   for (std::size_t i=0; i<reference.size(); ++i) {
      if (!std::isfinite(actual[i])) throw std::runtime_error("Non-finite CUDA post sample");
      peak=std::max(peak,std::abs(reference[i]));
      error=std::max(error,std::abs(reference[i]-actual[i]));
   }
   if (error>1e-12+1e-8*peak) {
      std::fprintf(stderr,"CUDA post error %.17g, reference peak %.17g\n",error,peak);
      throw std::runtime_error("CUDA post numerical comparison failed");
   }
}

int main()
{
   int count=0;
   const cudaError_t status=cudaGetDeviceCount(&count);
   if (status!=cudaSuccess || count!=1) {
      std::fprintf(stderr,"SKIP: CUDA post gate requires exactly one visible GPU (%s, count=%d)\n",
                   cudaGetErrorString(status),count); return 77;
   }
   try {
      std::size_t cases=0;
      const std::size_t lengths[]={1,63,64,65,513,3073,100003};
      const double rates[]={24000,44100,48000,96000};
      for (std::size_t samples : lengths) {
         const std::size_t channels=3,corners=8;
         const double corner_weights[]={0.0625,0.1875,0.125,0.125,0.25,0.0625,0.125,0.0625};
         std::vector<double> weights(channels*corners);
         for (std::size_t channel=0; channel<channels; ++channel)
            for (std::size_t corner=0; corner<corners; ++corner)
               weights[channel*corners+corner]=corner_weights[(corner+channel)%corners];
         std::vector<double> input(channels*corners*samples);
         for (std::size_t receiver=0; receiver<channels*corners; ++receiver)
            for (std::size_t n=0; n<samples; ++n)
               input[receiver*samples+n]=(receiver+1)*
                  (std::sin(0.021*n)*0.00390625+(n%67==0 ? 0.001953125 : 0));
         for (double rate : rates) {
            if ((samples==1 && rate<48000) || (samples>10000 && rate!=48000)) continue;
            for (unsigned mode=0; mode<5; ++mode) {
               PostOptions options; options.output_rate=rate;
               options.integrate=mode!=2 && mode!=4;
               options.lowcut=(mode==0 || mode==4) ? 0 : 10;
               options.lowpass=mode==3 ? 3000 : 0;
               options.symmetric_lowpass=mode==3;
               const PostResult reference=process_cpu(input,weights,channels,corners,samples,48000,options);
               for (bool chunked : {false,true}) {
                  const PostResult actual=process_cuda(input,weights,channels,corners,samples,48000,options,chunked,true);
                  if (actual.samples_out!=reference.samples_out || actual.fs_out!=reference.fs_out)
                     throw std::runtime_error("CUDA post shape/rate mismatch");
                  compare(reference.raw,actual.raw); compare(reference.filtered,actual.filtered);
                  ++cases;
               }
            }
         }
      }
      std::printf("PASS: %zu CUDA post comparisons, receiver weights, integrator/HP/LP/reverse, resampling and serial/chunked tails; tolerance 1e-12+1e-8*peak\n",cases);
      return 0;
   }
   catch (const std::exception& error) {
      std::fprintf(stderr,"post_cuda_check: %s\n",error.what()); return 1;
   }
}
