// Native models of CUDA air-filter indexing and cuFFT spectrum conventions.
// These verify algorithm plans; executing the CUDA kernels still needs a GPU.
#include "post_air.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::size_t checks=0,stokes_cases=0,ola_cases=0,dct_cases=0;
std::size_t full_batches=0,tail_batches=0,negative_frames=0,early_contributions=0;
double stokes_error=0,ola_error=0,dct_error=0;
const double pi=std::acos(-1.0);

void require(bool condition,const char* message)
{
   if (!condition) throw std::runtime_error(message);
   ++checks;
}

std::vector<double> input_signal(std::size_t channels,std::size_t samples)
{
   std::vector<double> data(channels*samples);
   for (std::size_t channel=0;channel<channels;++channel)
      for (std::size_t n=0;n<samples;++n)
         data[channel*samples+n]=channel==0 ? (n==0 ? 1 : n+1==samples ? -0.5 : 0) :
            channel==1 ? static_cast<double>((n*17+3)%43)/64-.25 :
                         (n%3==0 ? -0.125 : 0.0625);
   return data;
}

bool close_vectors(const std::vector<double>& actual,const std::vector<double>& expected,
                   double tolerance,double* largest=nullptr)
{
   if (actual.size()!=expected.size()) return false;
   bool close=true;
   for (std::size_t i=0;i<actual.size();++i) {
      const double error=std::abs(actual[i]-expected[i]);
      if (largest) *largest=std::max(*largest,error);
      close=close && std::isfinite(actual[i]) && error<=tolerance*(1+std::abs(expected[i]));
   }
   return close;
}

std::vector<double> stokes_gather(const std::vector<double>& input,std::size_t channels,
      std::size_t samples,const pffdtd_post::AirStokesPlan& plan)
{
   const std::size_t max_width=pffdtd_post::air_stokes_width(plan,samples-1);
   std::vector<double> output(channels*plan.samples,std::numeric_limits<double>::quiet_NaN());
   const double two_ts_gamma=2*plan.ts*plan.gamma;
   for (std::size_t index=0;index<output.size();++index) {
      const std::size_t channel=index/plan.samples,m=index%plan.samples;
      double value=m<plan.first ? input.at(channel*samples+m) : 0;
      const std::size_t begin=std::max(plan.first,m>max_width ? m-max_width : std::size_t(0));
      const std::size_t end=std::min(samples-1,m>samples-1 || max_width>samples-1-m ? samples-1 : m+max_width);
      for (std::size_t n=begin;n<=end;++n) {
         const std::size_t width=static_cast<std::size_t>(std::ceil(std::sqrt(plan.width_factor*static_cast<double>(n))/plan.ts));
         if ((n>m ? n-m : m-n)>width) continue;
         require(n<samples && channel*samples+n<input.size(),"Stokes gather reads beyond input");
         const double dt=(static_cast<double>(n)-static_cast<double>(m))*plan.ts;
         const double gain=plan.ts/std::sqrt(static_cast<double>(n)*two_ts_gamma*pi)*input[channel*samples+n];
         value+=gain*std::exp(-dt*dt/(static_cast<double>(n)*two_ts_gamma));
      }
      output[index]=value;
   }
   return output;
}

void stokes_contracts()
{
   pffdtd_post::AirOptions options; options.mode="stokes";
   for (double fs : {32000.0,48000.0,96000.0})
      for (double temperature : {-20.0,20.0,50.0}) {
         options.temperature=temperature;
         const std::size_t first=pffdtd_post::air_stokes_plan(100000,fs,options).first;
         for (std::size_t samples : {std::size_t(1),first-1,first,first+1,first+97}) {
            const auto input=input_signal(3,samples);
            const auto plan=pffdtd_post::air_stokes_plan(samples,fs,options);
            const auto gathered=stokes_gather(input,3,samples,plan);
            const auto scattered=pffdtd_post::process_air_cpu(input,3,samples,fs,options);
            require(scattered.samples==plan.samples,"Stokes output padding differs");
            require(close_vectors(gathered,scattered.data,1e-14,&stokes_error),"Stokes gather differs from source-ordered CPU scatter");
            ++stokes_cases;
         }
      }
}

// Reproduce frame formation and batched gather. FFT arithmetic is the native
// transform, with cuFFT's unnormalised inverse represented explicitly. The
// comparison below uses the separate full-sequence CPU scatter implementation.
std::vector<double> ola_batches(const std::vector<double>& input,std::size_t channels,
      std::size_t samples,const pffdtd_post::AirOlaPlan& plan,double fs,
      bool omit_tail=false,bool omit_early_bypass=false)
{
   const std::size_t capacity=std::min(std::size_t(128),plan.frames);
   const pffdtd_post::NativeFftPlan fft(plan.fft_size);
   std::vector<double> output(channels*plan.samples,0);
   for (std::size_t channel=0;channel<channels;++channel) {
      std::vector<unsigned> coverage(plan.samples,0);
      for (std::size_t begin=0;begin<plan.frames;begin+=capacity) {
         const std::size_t batch=std::min(capacity,plan.frames-begin);
         require(batch>0 && batch<=128,"OLA frame batch exceeds capacity");
         if (batch==128) ++full_batches;
         else ++tail_batches;
         if (omit_tail && batch<capacity) continue;
         std::vector<double> frames(batch*plan.fft_size+2,123456.0);
         for (std::size_t local=0;local<batch;++local) {
            const std::size_t frame=begin+local,start=frame*plan.hop;
            std::vector<std::complex<double>> work(plan.fft_size);
            for (std::size_t k=0;k<plan.fft_size;++k) {
               double value=0;
               if (k<plan.window && start>=plan.window/2 &&
                   start+k>=plan.window && start+k-plan.window<samples) {
                  const std::size_t n=start+k-plan.window;
                  require(n<samples,"OLA frame reads beyond channel");
                  value=plan.analysis[k]*input.at(channel*samples+n);
               }
               work[k]=value;
            }
            fft.transform(work,false);
            const double distance=plan.c/fs*(static_cast<double>(start)-0.5*plan.window);
            if (distance<0) ++negative_frames;
            for (std::size_t k=0;k<plan.fft_size;++k) {
               const std::size_t bin=std::min(k,plan.fft_size-k);
               require(bin<plan.absorption.size(),"OLA attenuation bin exceeds spectrum");
               const double gain=distance<0 ? 0 : std::exp(-plan.absorption[bin]*distance);
               work[k]*=gain;
            }
            fft.transform(work,true);
            for (std::size_t k=0;k<plan.fft_size;++k)
               frames[1+local*plan.fft_size+k]=work[k].real()*plan.fft_size;
         }
         const std::size_t output_begin=begin*plan.hop>plan.window ? begin*plan.hop-plan.window : 0;
         const std::size_t output_end=std::min(plan.samples,(begin+batch-1)*plan.hop);
         require(output_begin<=output_end && output_end<=plan.samples,"OLA gather range exceeds output");
         for (std::size_t m=output_begin;m<output_end;++m) {
            const std::size_t padded=plan.window+m;
            const std::size_t first=std::max(begin,padded>=plan.window ? (padded-plan.window)/plan.hop+1 : std::size_t(0));
            const std::size_t last=std::min(begin+batch-1,padded/plan.hop);
            double value=output.at(channel*plan.samples+m);
            for (std::size_t frame=first;frame<=last;++frame) {
               const std::size_t start=frame*plan.hop,k=padded-start;
               if (k>=plan.window) continue;
               require(frame>=begin && frame<begin+batch,"OLA gather reads a frame outside its batch");
               const std::size_t offset=(frame-begin)*plan.fft_size+k;
               require(offset<batch*plan.fft_size,"OLA gather uses window stride instead of FFT stride");
               const bool early=static_cast<double>(start)<0.5*plan.window;
               if (early) ++early_contributions;
               const double sample=early && !omit_early_bypass ?
                  (m<samples ? input[channel*samples+m] : 0) : frames[1+offset]/plan.fft_size;
               value+=plan.synthesis[k]*sample;
               ++coverage[m];
            }
            output[channel*plan.samples+m]=value;
         }
         require(frames.front()==123456.0 && frames.back()==123456.0,"OLA frame canary was overwritten");
      }
      if (!omit_tail)
         for (std::size_t m=0;m<plan.samples;++m) {
            // Independent support enumeration, without gather bounds or batch
            // assumptions: a frame covers padded output iff 0<=k<window.
            unsigned expected=0;
            for (std::size_t frame=0;frame<plan.frames;++frame) {
               const std::size_t start=frame*plan.hop,padded=plan.window+m;
               if (start<=padded && padded-start<plan.window) ++expected;
            }
            require(coverage[m]==expected,"OLA batching misses or duplicates a frame contribution");
         }
   }
   return output;
}

void ola_contracts()
{
   pffdtd_post::AirOptions options; options.mode="ola";
   for (std::size_t window : {4u,5u,6u,7u,9u,10u,11u,14u,18u,31u,33u,63u,65u,257u}) {
      options.window=window;
      const std::size_t hop=pffdtd_post::air_ola_plan(1,48000,options).hop;
      std::vector<std::size_t> sample_counts={1,window-1,window};
      for (std::size_t frames : {127u,128u,129u,255u,256u,257u}) {
         sample_counts.push_back((frames-1)*hop-window+1);
         if (hop>1) sample_counts.push_back(frames*hop-window);
      }
      for (std::size_t samples : sample_counts) {
         const auto plan=pffdtd_post::air_ola_plan(samples,48000,options);
         const auto input=input_signal(3,samples);
         const auto gathered=ola_batches(input,3,samples,plan,48000);
         const auto scattered=pffdtd_post::process_air_cpu(input,3,samples,48000,options);
         require(plan.samples==scattered.samples,"OLA batched output length differs");
         require(close_vectors(gathered,scattered.data,2e-12,&ola_error),"OLA batched gather differs from full CPU frame scatter");
         ++ola_cases;
      }
   }
   options.window=18;
   // Two real input samples reach the final frame. A remainder of one would
   // put its sole sample under the Hann zero at k=0, hiding omission errors.
   const std::size_t remainder_samples=129*4-18-2;
   const auto plan=pffdtd_post::air_ola_plan(remainder_samples,48000,options);
   require(plan.frames==129,"remainder mutant scene has wrong frame count");
   const auto input=input_signal(3,remainder_samples);
   const auto expected=pffdtd_post::process_air_cpu(input,3,remainder_samples,48000,options);
   require(!close_vectors(ola_batches(input,3,remainder_samples,plan,48000,true),expected.data,2e-12),
           "OLA remainder omission mutant escaped comparison");
   options.window=5;
   const auto early_plan=pffdtd_post::air_ola_plan(1,1e-7,options);
   const auto early_expected=pffdtd_post::process_air_cpu({1},1,1,1e-7,options);
   require(close_vectors(ola_batches({1},1,1,early_plan,1e-7),early_expected.data,2e-12),"odd-window negative-distance branch differs");
   require(!close_vectors(ola_batches({1},1,1,early_plan,1e-7,false,true),early_expected.data,2e-12),
           "OLA early-bypass omission mutant escaped comparison");
   require(full_batches>0 && tail_batches>0 && negative_frames>0 && early_contributions>0,
           "OLA suite failed to exercise complete/tail batches and negative-distance bypass");
}

std::vector<long double> modal_direct(const std::vector<double>& modes)
{
   const std::size_t n=modes.size();
   const long double pi_long=std::acos(-1.0L);
   std::vector<long double> output(n);
   for (std::size_t t=0;t<n;++t) {
      output[t]=modes[0]/std::sqrt(static_cast<long double>(n));
      for (std::size_t q=1;q<n;++q)
         output[t]+=std::sqrt(2.0L/n)*modes[q]*std::cos(pi_long*q*(t+0.5L)/n);
   }
   return output;
}

std::vector<double> modal_cufft_model(const std::vector<double>& modes,bool wrong_phase=false)
{
   const std::size_t n=modes.size(),extent=2*n;
   std::vector<std::complex<double>> spectrum(extent);
   for (std::size_t q=0;q<n;++q) {
      const double angle=(wrong_phase ? -1 : 1)*pi*(static_cast<double>(q)/(2*n));
      const double value=modes[q]*(q ? std::sqrt(2.0*n) : 2*std::sqrt(static_cast<double>(n)));
      spectrum[q]={value*std::cos(angle),value*std::sin(angle)};
      if (q) spectrum[extent-q]=std::conj(spectrum[q]);
   }
   require(spectrum[n]==std::complex<double>(0,0),"modal Nyquist bin is not zero");
   // Direct unnormalised C2R sum, followed by exactly the kernel's 1/(2*N).
   // This does not use native_fft or idct2_ortho for the expected transform.
   std::vector<double> output(n);
   for (std::size_t t=0;t<n;++t) {
      long double real=spectrum[0].real();
      for (std::size_t q=1;q<n;++q) {
         const long double angle=2*std::acos(-1.0L)*q*t/extent;
         real+=2*(spectrum[q].real()*std::cos(angle)-spectrum[q].imag()*std::sin(angle));
      }
      output[t]=static_cast<double>(real/extent);
   }
   return output;
}

void modal_contracts()
{
   for (std::size_t n : {1u,2u,3u,5u,6u,7u,9u,16u,31u,32u,97u,127u,128u,129u,257u})
      for (unsigned shape=0;shape<4;++shape) {
         std::vector<double> modes(n,0);
         for (std::size_t q=0;q<n;++q)
            modes[q]=shape==0 ? (q==0 ? 1 : 0) : shape==1 ? (q+1==n ? 1 : 0) :
                     shape==2 ? static_cast<double>((q*17+3)%41)/32-.5 : 0;
         const auto expected=modal_direct(modes);
         const auto transformed=modal_cufft_model(modes);
         auto native=modes;
         pffdtd_post::idct2_ortho(native);
         for (std::size_t t=0;t<n;++t) {
            const double error=static_cast<double>(std::abs(transformed[t]-expected[t]));
            dct_error=std::max(dct_error,error);
            require(std::isfinite(transformed[t]) && error<2e-12,"CUDA modal half-spectrum scale/phase differs from orthonormal IDCT");
            require(std::abs(native[t]-expected[t])<2e-12,"native modal IDCT differs from direct sum");
         }
         ++dct_cases;
      }
   for (std::size_t n : {1u,2u,3u,5u,6u,7u,9u,16u,31u,32u,97u,127u,128u,129u,257u}) {
      const std::size_t channels=3;
      const auto modes=input_signal(channels,n);
      std::vector<std::complex<double>> packed(channels*(n+1));
      // Flattened kernel indexing includes one Nyquist bin per channel;
      // modes have stride N, while the cuFFT half-spectrum has stride N+1.
      for (std::size_t index=0;index<packed.size();++index) {
         const std::size_t channel=index/(n+1),q=index%(n+1);
         if (q==n) { packed[index]={0,0}; continue; }
         const double angle=pi*(static_cast<double>(q)/(2*n));
         const double value=modes.at(channel*n+q)*(q ? std::sqrt(2.0*n) : 2*std::sqrt(static_cast<double>(n)));
         packed[index]={value*std::cos(angle),value*std::sin(angle)};
      }
      std::vector<double> transformed(channels*2*n);
      for (std::size_t channel=0;channel<channels;++channel) {
         require(packed[channel*(n+1)+n]==std::complex<double>(0,0),"channel Nyquist bin was overwritten");
         std::vector<std::complex<double>> spectrum(2*n);
         for (std::size_t q=0;q<n;++q) {
            spectrum[q]=packed[channel*(n+1)+q];
            if (q) spectrum[2*n-q]=std::conj(spectrum[q]);
         }
         pffdtd_post::native_fft(spectrum,true);
         for (std::size_t t=0;t<2*n;++t) transformed[channel*2*n+t]=spectrum[t].real()*(2*n);
      }
      std::vector<double> output(channels*n);
      for (std::size_t index=0;index<output.size();++index)
         output[index]=transformed.at((index/n)*2*n+index%n)/(2*n);
      for (std::size_t channel=0;channel<channels;++channel) {
         const std::vector<double> row(modes.begin()+channel*n,modes.begin()+(channel+1)*n);
         const auto expected=modal_direct(row);
         for (std::size_t t=0;t<n;++t)
            require(std::abs(output[channel*n+t]-expected[t])<2e-12,"CUDA modal channel/spectrum/output strides differ");
         ++dct_cases;
      }
   }
   const std::vector<double> modes={0.25,-0.5,0.125,0.75,-0.125};
   require(!close_vectors(modal_cufft_model(modes,true),modal_cufft_model(modes),2e-12),
           "modal conjugate-phase mutant escaped comparison");
}

} // namespace

int main()
{
   try {
      stokes_contracts();
      ola_contracts();
      modal_contracts();
      std::printf("air_cuda_plan: %zu Stokes, %zu OLA, %zu IDCT cases; %zu assertions passed\n"
                  "  errors Stokes %.3g OLA %.3g IDCT %.3g; batches full %zu tail %zu\n",
                  stokes_cases,ola_cases,dct_cases,checks,stokes_error,ola_error,dct_error,full_batches,tail_batches);
      return 0;
   }
   catch (const std::exception& error) {
      std::fprintf(stderr,"air_cuda_plan: %s\n",error.what());
      return 1;
   }
}
