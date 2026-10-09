// FP64 CUDA air attenuation. Geometry-independent plans are shared with the
// native reference; gather kernels avoid floating-point atomic reductions.
#ifndef PFFDTD_POST_AIR_CUDA_H
#define PFFDTD_POST_AIR_CUDA_H
#ifdef __CUDACC__
#include "post_air.h"
#include "post_cuda_common.h"
#include <cufft.h>
#include <climits>
#include <memory>

namespace pffdtd_post {

inline void air_cufft_check(cufftResult status)
{
   if (status!=CUFFT_SUCCESS)
      throw std::runtime_error("cuFFT air processing failed (code "+std::to_string(static_cast<int>(status))+")");
}

class AirCudaFft {
   cufftHandle handle=0;
public:
   AirCudaFft(std::size_t length,cufftType type,std::size_t batch,cudaStream_t stream)
   {
      if (!length || length>INT_MAX || !batch || batch>INT_MAX)
         throw std::invalid_argument("Air FFT length/batch exceeds cuFFT int32 limit");
      try {
         air_cufft_check(cufftCreate(&handle));
         std::size_t workspace=0;
         air_cufft_check(cufftMakePlan1d(handle,static_cast<int>(length),type,static_cast<int>(batch),&workspace));
         air_cufft_check(cufftSetStream(handle,stream));
      }
      catch (...) { if (handle) cufftDestroy(handle); handle=0; throw; }
   }
   ~AirCudaFft() { if (handle) cufftDestroy(handle); }
   operator cufftHandle() const { return handle; }
   AirCudaFft(const AirCudaFft&)=delete;
   AirCudaFft& operator=(const AirCudaFft&)=delete;
};

// Scratch and cuFFT plans must outlive queued work even when a later stage
// throws. This guard is declared after all stage-local resources.
class AirCudaDrain {
   cudaStream_t stream;
   bool finished=false;
public:
   explicit AirCudaDrain(cudaStream_t value):stream(value) {}
   void finish() { post_cuda_check(cudaStreamSynchronize(stream)); finished=true; }
   ~AirCudaDrain() { if (!finished) cudaStreamSynchronize(stream); }
};

inline std::size_t air_output_samples(std::size_t samples,double fs,const AirOptions& options)
{
   const std::string mode=air_mode(options);
   if (mode=="stokes") return air_stokes_plan(samples,fs,options).samples;
   if (mode=="ola") return air_ola_plan(samples,fs,options).samples;
   if (mode=="modal") return air_modal_plan(samples,fs,options).samples;
   return samples;
}

__global__ void AirStokesGather(const double* input,double* output,std::size_t channels,
      std::size_t samples,AirStokesPlan plan,std::size_t max_width)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*plan.samples) return;
   const std::size_t channel=index/plan.samples,m=index%plan.samples;
   double value=m<plan.first ? input[channel*samples+m] : 0;
   const std::size_t begin=max(plan.first,m>max_width ? m-max_width : std::size_t(0));
   const std::size_t end=min(samples-1,m>samples-1 || max_width>samples-1-m ? samples-1 : m+max_width);
   const double two_ts_gamma=2*plan.ts*plan.gamma;
   const double pi=3.1415926535897932384626433832795;
   for (std::size_t n=begin;n<=end;++n) {
      const std::size_t width=static_cast<std::size_t>(ceil(sqrt(plan.width_factor*static_cast<double>(n))/plan.ts));
      if ((n>m ? n-m : m-n)>width) continue;
      const double dt=(static_cast<double>(n)-static_cast<double>(m))*plan.ts;
      const double gain=plan.ts/sqrt(static_cast<double>(n)*two_ts_gamma*pi)*input[channel*samples+n];
      value+=gain*exp(-dt*dt/(static_cast<double>(n)*two_ts_gamma));
   }
   output[index]=value;
}

__global__ void AirOlaFrames(const double* input,double* frames,const double* analysis,
      std::size_t samples,std::size_t window,std::size_t hop,std::size_t fft_size,
      std::size_t begin,std::size_t batch)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=batch*fft_size) return;
   const std::size_t frame=begin+index/fft_size,k=index%fft_size,start=frame*hop;
   double value=0;
   if (k<window && start>=window/2 && start+k>=window && start+k-window<samples)
      value=analysis[k]*input[start+k-window];
   frames[index]=value;
}

__global__ void AirOlaAttenuate(cufftDoubleComplex* spectrum,const double* absorption,
      std::size_t bins,std::size_t hop,std::size_t window,std::size_t begin,
      std::size_t batch,double c_over_fs)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=batch*bins) return;
   const double distance=c_over_fs*(static_cast<double>((begin+index/bins)*hop)-0.5*window);
   const double gain=distance<0 ? 0 : exp(-absorption[index%bins]*distance);
   spectrum[index].x*=gain; spectrum[index].y*=gain;
}

__global__ void AirOlaGather(const double* input,const double* frames,double* output,
      const double* synthesis,std::size_t samples,std::size_t samples_out,
      std::size_t window,std::size_t hop,std::size_t fft_size,
      std::size_t begin,std::size_t batch,std::size_t output_begin,std::size_t count)
{
   const std::size_t offset=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (offset>=count) return;
   const std::size_t m=output_begin+offset,padded=window+m;
   if (m>=samples_out) return;
   const std::size_t first=max(begin,padded>=window ? (padded-window)/hop+1 : std::size_t(0));
   const std::size_t last=min(begin+batch-1,padded/hop);
   double value=output[m];
   for (std::size_t frame=first;frame<=last;++frame) {
      const std::size_t start=frame*hop,k=padded-start;
      if (k>=window) continue;
      // Preserve the legacy early-frame bypass, including its window gain.
      const double sample=static_cast<double>(start)<0.5*window ?
         (m<samples ? input[m] : 0) : frames[(frame-begin)*fft_size+k]/fft_size;
      value+=synthesis[k]*sample;
   }
   output[m]=value;
}

__global__ void AirModalRecurrence(const double* input,double* modes,const double* a1,
      const double* a2,const double* f1,const double* f2,std::size_t channels,
      std::size_t samples,std::size_t samples_out)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*samples_out) return;
   const std::size_t channel=index/samples_out,q=index%samples_out;
   if (!q) {
      double sum=0;
      for (std::size_t n=0;n<samples;++n) sum+=input[channel*samples+n];
      modes[index]=sum/sqrt(static_cast<double>(samples_out)); return;
   }
   double previous=0,current=0,source_previous=0;
   for (std::size_t t=0;t<samples;++t) {
      const double source=input[channel*samples+samples-1-t];
      const double next=a1[q]*current+a2[q]*previous+f1[q]*source-f2[q]*source_previous;
      previous=current; current=next; source_previous=source;
   }
   modes[index]=current;
}

__global__ void AirModalSpectrum(const double* modes,cufftDoubleComplex* spectrum,
      std::size_t channels,std::size_t samples)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*(samples+1)) return;
   const std::size_t channel=index/(samples+1),q=index%(samples+1);
   if (q==samples) { spectrum[index]={0,0}; return; }
   const double angle=3.1415926535897932384626433832795*(static_cast<double>(q)/(2*samples));
   const double value=modes[channel*samples+q]*(q ? sqrt(2.0*samples) : 2*sqrt(static_cast<double>(samples)));
   spectrum[index]={value*cos(angle),value*sin(angle)};
}

__global__ void AirModalOutput(const double* transformed,double* output,
      std::size_t channels,std::size_t samples)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*samples) return;
   output[index]=transformed[(index/samples)*2*samples+index%samples]/(2*samples);
}

__global__ void AirChannelL1(const double* input,double* norms,std::size_t samples)
{
   __shared__ double partial[256];
   double value=0;
   for (std::size_t n=threadIdx.x;n<samples;n+=blockDim.x)
      value+=fabs(input[static_cast<std::size_t>(blockIdx.x)*samples+n]);
   partial[threadIdx.x]=value; __syncthreads();
   for (unsigned stride=128;stride;stride/=2) {
      if (threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
      __syncthreads();
   }
   if (!threadIdx.x) norms[blockIdx.x]=partial[0];
}

__global__ void AirModalWeightedInput(const double* input,double* frames,const double* nodes,
      std::size_t samples,std::size_t extent,std::size_t begin,std::size_t batch)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=batch*extent) return;
   const std::size_t n=index%extent,j=begin+index/extent;
   frames[index]=n<samples ? input[n]*exp(-nodes[j]*static_cast<double>(n)) : 0;
}

__global__ void AirModalInterpolate(const cufftDoubleComplex* spectrum,double* modes,
      double* compensation,const double* sigma,const double* nodes,const double* barycentric,
      const double* normalizers,const double* exact,const double* real_coeff,
      const double* sine_coeff,std::size_t samples,std::size_t begin,std::size_t batch,double sigma_max)
{
   const std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (!q || q>=samples) return;
   double value=modes[q],carry=compensation[q];
   const int exact_node=static_cast<int>(exact[q]);
   const double point=sigma[q]/sigma_max;
   for (std::size_t j=0;j<batch;++j) {
      const std::size_t node=begin+j;
      const double weight=exact_node>=0 ? (static_cast<std::size_t>(exact_node)==node ? 1.0 : 0.0) :
         (barycentric[node]/(point-nodes[node]/sigma_max))/normalizers[q];
      const cufftDoubleComplex z=spectrum[j*(samples+1)+q];
      const double term=weight*(real_coeff[q]*z.x-sine_coeff[q]*z.y);
      const double corrected=term-carry,next=value+corrected;
      carry=(next-value)-corrected; value=next;
   }
   modes[q]=value; compensation[q]=carry;
}

__global__ void AirModalZero(const double* input,double* modes,std::size_t samples,std::size_t samples_out)
{
   double sum=0;
   for (std::size_t n=0;n<samples;++n) sum+=input[n];
   modes[0]=sum/sqrt(static_cast<double>(samples_out));
}

inline void air_copy_plan(PostCudaBuffer& destination,const std::vector<double>& source,cudaStream_t stream)
{
   post_cuda_check(cudaMemcpyAsync(destination.get(),source.data(),source.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
}

// Interpolate the mode-dependent damped Fourier transform at Chebyshev nodes.
// K bounded FFTs replace N*N recurrence work; no N*K coefficient matrix is
// stored. The analytical bound excludes floating-point FFT roundoff.
inline bool air_modal_fft_cuda(const double* input,double* modes,std::size_t channels,
      std::size_t samples,const AirModalPlan& modal,const AirOptions& options,
      const cudaDeviceProp& properties,cudaStream_t stream)
{
   if (options.modal_method!="fft") return false;
   if (channels>static_cast<std::size_t>(properties.maxGridSize[0]))
      throw std::invalid_argument("Modal norm channel grid exceeds device limits");
   PostCudaBuffer norms(channels);
   std::vector<double> host_norms(channels);
   AirModalFftPlan plan;
   {
      AirCudaDrain drain(stream);
      AirChannelL1<<<static_cast<unsigned>(channels),256,0,stream>>>(input,norms.get(),samples);
      post_cuda_check(cudaGetLastError());
      post_cuda_check(cudaMemcpyAsync(host_norms.data(),norms.get(),channels*sizeof(double),cudaMemcpyDeviceToHost,stream));
      drain.finish();
      const double l1=*std::max_element(host_norms.begin(),host_norms.end());
      const double roundoff=8*std::numeric_limits<double>::epsilon()*static_cast<double>(samples);
      const double upper=roundoff<0.5 ? l1/(1-roundoff) : std::numeric_limits<double>::infinity();
      plan=air_modal_fft_plan(modal,samples,upper,options.modal_tolerance);
   }
   if (!plan.usable) return false;
   const std::size_t extent=air_detail::product(modal.samples,2),bins=modal.samples+1;
   const std::size_t capacity=std::min(std::size_t(8),plan.nodes.size()),tail=plan.nodes.size()%capacity;
   PostCudaBuffer sigma(modal.samples),nodes(plan.nodes.size()),barycentric(plan.nodes.size());
   PostCudaBuffer normalizers(modal.samples),real_coeff(modal.samples),sine_coeff(modal.samples),exact(modal.samples);
   PostCudaBuffer frames(post_product(capacity,extent)),spectrum(post_product(post_product(capacity,bins),2));
   PostCudaBuffer compensation(modal.samples);
   AirCudaFft forward(extent,CUFFT_D2Z,capacity,stream);
   std::unique_ptr<AirCudaFft> tail_forward;
   if (tail) tail_forward.reset(new AirCudaFft(extent,CUFFT_D2Z,tail,stream));
   std::vector<double> exact_host(plan.exact_node.begin(),plan.exact_node.end());
   AirCudaDrain drain(stream);
   air_copy_plan(sigma,modal.sigma,stream); air_copy_plan(nodes,plan.nodes,stream);
   air_copy_plan(barycentric,plan.barycentric,stream); air_copy_plan(normalizers,plan.normalizers,stream);
   air_copy_plan(real_coeff,plan.real_coeff,stream); air_copy_plan(sine_coeff,plan.sine_coeff,stream);
   air_copy_plan(exact,exact_host,stream);
   for (std::size_t channel=0;channel<channels;++channel) {
      double* channel_modes=modes+channel*modal.samples;
      post_cuda_check(cudaMemsetAsync(channel_modes,0,modal.samples*sizeof(double),stream));
      post_cuda_check(cudaMemsetAsync(compensation.get(),0,modal.samples*sizeof(double),stream));
      for (std::size_t begin=0;begin<plan.nodes.size();begin+=capacity) {
         const std::size_t batch=std::min(capacity,plan.nodes.size()-begin);
         AirModalWeightedInput<<<post_cuda_grid(post_product(batch,extent),properties),256,0,stream>>>
            (input+channel*samples,frames.get(),nodes.get(),samples,extent,begin,batch);
         post_cuda_check(cudaGetLastError());
         const cufftHandle fwd=batch==capacity ? static_cast<cufftHandle>(forward) : static_cast<cufftHandle>(*tail_forward);
         air_cufft_check(cufftExecD2Z(fwd,frames.get(),reinterpret_cast<cufftDoubleComplex*>(spectrum.get())));
         AirModalInterpolate<<<post_cuda_grid(modal.samples,properties),256,0,stream>>>
            (reinterpret_cast<cufftDoubleComplex*>(spectrum.get()),channel_modes,compensation.get(),sigma.get(),nodes.get(),
             barycentric.get(),normalizers.get(),exact.get(),real_coeff.get(),sine_coeff.get(),modal.samples,begin,batch,plan.sigma_max);
         post_cuda_check(cudaGetLastError());
      }
      AirModalZero<<<1,1,0,stream>>>(input+channel*samples,channel_modes,samples,modal.samples);
      post_cuda_check(cudaGetLastError());
   }
   drain.finish(); return true;
}

// Input and final output stay on the caller's device/stream. Stage-local FFT
// workspace is bounded by the frame batch; the drain protects its lifetime.
inline void air_apply_cuda(const double* input,double* output,std::size_t channels,
      std::size_t samples,double fs,const AirOptions& options,
      const cudaDeviceProp& properties,cudaStream_t stream)
{
   const std::string mode=air_mode(options);
   if (mode=="none") {
      post_cuda_check(cudaMemcpyAsync(output,input,post_product(channels,samples)*sizeof(double),cudaMemcpyDeviceToDevice,stream));
      return;
   }
   if (mode=="stokes") {
      const AirStokesPlan plan=air_stokes_plan(samples,fs,options);
      AirStokesGather<<<post_cuda_grid(post_product(channels,plan.samples),properties),256,0,stream>>>
         (input,output,channels,samples,plan,air_stokes_width(plan,samples-1));
      post_cuda_check(cudaGetLastError()); return;
   }
   if (mode=="ola") {
      const AirOlaPlan plan=air_ola_plan(samples,fs,options);
      const std::size_t capacity=std::min(std::size_t(128),plan.frames),bins=plan.fft_size/2+1;
      const std::size_t tail=plan.frames%capacity;
      PostCudaBuffer analysis(plan.window),synthesis(plan.window),absorption(bins);
      PostCudaBuffer frames(post_product(capacity,plan.fft_size)),spectrum(post_product(post_product(capacity,bins),2));
      AirCudaFft forward(plan.fft_size,CUFFT_D2Z,capacity,stream),inverse(plan.fft_size,CUFFT_Z2D,capacity,stream);
      std::unique_ptr<AirCudaFft> tail_forward,tail_inverse;
      if (tail) {
         tail_forward.reset(new AirCudaFft(plan.fft_size,CUFFT_D2Z,tail,stream));
         tail_inverse.reset(new AirCudaFft(plan.fft_size,CUFFT_Z2D,tail,stream));
      }
      AirCudaDrain drain(stream);
      air_copy_plan(analysis,plan.analysis,stream); air_copy_plan(synthesis,plan.synthesis,stream);
      air_copy_plan(absorption,plan.absorption,stream);
      post_cuda_check(cudaMemsetAsync(output,0,post_product(channels,plan.samples)*sizeof(double),stream));
      for (std::size_t channel=0;channel<channels;++channel) {
         for (std::size_t begin=0;begin<plan.frames;begin+=capacity) {
            const std::size_t batch=std::min(capacity,plan.frames-begin);
            AirOlaFrames<<<post_cuda_grid(post_product(batch,plan.fft_size),properties),256,0,stream>>>
               (input+channel*samples,frames.get(),analysis.get(),samples,plan.window,plan.hop,plan.fft_size,begin,batch);
            post_cuda_check(cudaGetLastError());
            const cufftHandle fwd=batch==capacity ? static_cast<cufftHandle>(forward) : static_cast<cufftHandle>(*tail_forward);
            const cufftHandle inv=batch==capacity ? static_cast<cufftHandle>(inverse) : static_cast<cufftHandle>(*tail_inverse);
            air_cufft_check(cufftExecD2Z(fwd,frames.get(),reinterpret_cast<cufftDoubleComplex*>(spectrum.get())));
            AirOlaAttenuate<<<post_cuda_grid(post_product(batch,bins),properties),256,0,stream>>>
               (reinterpret_cast<cufftDoubleComplex*>(spectrum.get()),absorption.get(),bins,plan.hop,plan.window,begin,batch,plan.c/fs);
            post_cuda_check(cudaGetLastError());
            air_cufft_check(cufftExecZ2D(inv,reinterpret_cast<cufftDoubleComplex*>(spectrum.get()),frames.get()));
            const std::size_t out_begin=begin*plan.hop>plan.window ? begin*plan.hop-plan.window : 0;
            const std::size_t out_end=std::min(plan.samples,(begin+batch-1)*plan.hop);
            if (out_end>out_begin) {
               AirOlaGather<<<post_cuda_grid(out_end-out_begin,properties),256,0,stream>>>
                  (input+channel*samples,frames.get(),output+channel*plan.samples,synthesis.get(),samples,plan.samples,
                   plan.window,plan.hop,plan.fft_size,begin,batch,out_begin,out_end-out_begin);
               post_cuda_check(cudaGetLastError());
            }
         }
      }
      drain.finish(); return;
   }
   const AirModalPlan plan=air_modal_plan(samples,fs,options);
   PostCudaBuffer a1(plan.samples),a2(plan.samples),f1(plan.samples),f2(plan.samples);
   PostCudaBuffer modes(post_product(channels,plan.samples));
   PostCudaBuffer spectrum(post_product(post_product(channels,air_detail::add(plan.samples,1)),2));
   PostCudaBuffer transformed(post_product(post_product(channels,plan.samples),2));
   AirCudaFft inverse(air_detail::product(plan.samples,2),CUFFT_Z2D,channels,stream);
   AirCudaDrain drain(stream);
   if (!air_modal_fft_cuda(input,modes.get(),channels,samples,plan,options,properties,stream)) {
      air_copy_plan(a1,plan.a1,stream); air_copy_plan(a2,plan.a2,stream);
      air_copy_plan(f1,plan.f1,stream); air_copy_plan(f2,plan.f2,stream);
      AirModalRecurrence<<<post_cuda_grid(post_product(channels,plan.samples),properties),256,0,stream>>>
         (input,modes.get(),a1.get(),a2.get(),f1.get(),f2.get(),channels,samples,plan.samples);
      post_cuda_check(cudaGetLastError());
   }
   AirModalSpectrum<<<post_cuda_grid(post_product(channels,air_detail::add(plan.samples,1)),properties),256,0,stream>>>
      (modes.get(),reinterpret_cast<cufftDoubleComplex*>(spectrum.get()),channels,plan.samples);
   post_cuda_check(cudaGetLastError());
   air_cufft_check(cufftExecZ2D(inverse,reinterpret_cast<cufftDoubleComplex*>(spectrum.get()),transformed.get()));
   AirModalOutput<<<post_cuda_grid(post_product(channels,plan.samples),properties),256,0,stream>>>
      (transformed.get(),output,channels,plan.samples);
   post_cuda_check(cudaGetLastError()); drain.finish();
}

inline AirResult process_air_cuda(const std::vector<double>& input,std::size_t channels,
      std::size_t samples,double fs,const AirOptions& options)
{
   air_detail::rate(fs);
   if (!channels || !samples || input.size()!=air_detail::product(channels,samples))
      throw std::invalid_argument("Invalid planar air-filter input shape");
   for (double value:input) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite air-filter input");
   post_cuda_check(cudaSetDevice(0));
   cudaDeviceProp properties{}; post_cuda_check(cudaGetDeviceProperties(&properties,0));
   AirResult result; result.samples=air_output_samples(samples,fs,options);
   result.data.resize(post_product(channels,result.samples));
   PostCudaBuffer data(input.size()),output(result.data.size());
   PostCudaStream stream;
   post_cuda_check(cudaMemcpyAsync(data.get(),input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
   post_cuda_check(cudaMemsetAsync(output.get(),0xff,result.data.size()*sizeof(double),stream));
   air_apply_cuda(data.get(),output.get(),channels,samples,fs,options,properties,stream);
   post_cuda_check(cudaMemcpyAsync(result.data.data(),output.get(),result.data.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
   post_cuda_check(cudaStreamSynchronize(stream)); air_detail::finite_result(result.data);
   return result;
}
} // namespace pffdtd_post
#endif
#endif
