// Resident CUDA postprocessing with a serial reference and optional two-pass
// affine chunk filtering. Chunking changes rounding and requires device gates.
#ifndef PFFDTD_POST_CUDA_H
#define PFFDTD_POST_CUDA_H
#ifdef __CUDACC__
#include "post_pipeline.h"
#include "post_chunked.h"
#include "post_cuda_common.h"
#include "post_air_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <string>

namespace pffdtd_post {

__global__ void PostRecombine(const double* input, const double* weights,
      double* output, std::size_t channels, std::size_t corners, std::size_t samples)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*samples) return;
   const std::size_t channel=index/samples, n=index%samples;
   double sum=0;
   for (std::size_t corner=0; corner<corners; ++corner) {
      const std::size_t receiver=channel*corners+corner;
      sum+=weights[receiver]*input[receiver*samples+n];
   }
   output[index]=sum;
}

__global__ void PostSosSerial(double* data, std::size_t channels,
      std::size_t samples, Sos section, bool reverse)
{
   const std::size_t channel=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (channel>=channels) return;
   double z1=0,z2=0;
   for (std::size_t n=0; n<samples; ++n) {
      const std::size_t index=channel*samples+(reverse ? samples-1-n : n);
      data[index]=sos_sample(section,data[index],z1,z2);
   }
}

static const std::size_t POST_IIR_CHUNK=64;

__global__ void PostSosChunkResponse(const double* data, PostState* response,
      std::size_t channels, std::size_t samples, std::size_t chunks,
      Sos section, bool reverse)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*chunks) return;
   const std::size_t channel=index/chunks, begin=(index%chunks)*POST_IIR_CHUNK;
   double z1=0,z2=0;
   for (std::size_t offset=0; offset<POST_IIR_CHUNK && begin+offset<samples; ++offset) {
      const std::size_t n=begin+offset;
      sos_sample(section,data[channel*samples+(reverse ? samples-1-n : n)],z1,z2);
   }
   response[index]={z1,z2};
}

__global__ void PostSosChunkPrefix(const PostState* response, PostState* initial,
      std::size_t channels, std::size_t chunks, PostMatrix transition)
{
   const std::size_t channel=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (channel>=channels) return;
   PostState state={0,0};
   for (std::size_t chunk=0; chunk<chunks; ++chunk) {
      const std::size_t index=channel*chunks+chunk;
      initial[index]=state;
      const PostState propagated=post_matrix_apply(transition,state);
      state={propagated.z1+response[index].z1,propagated.z2+response[index].z2};
   }
}

__global__ void PostSosChunkApply(double* data, const PostState* initial,
      std::size_t channels, std::size_t samples, std::size_t chunks,
      Sos section, bool reverse)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*chunks) return;
   const std::size_t channel=index/chunks, begin=(index%chunks)*POST_IIR_CHUNK;
   double z1=initial[index].z1,z2=initial[index].z2;
   for (std::size_t offset=0; offset<POST_IIR_CHUNK && begin+offset<samples; ++offset) {
      const std::size_t n=begin+offset;
      const std::size_t position=channel*samples+(reverse ? samples-1-n : n);
      data[position]=sos_sample(section,data[position],z1,z2);
   }
}

inline void filter_cuda(double* data, std::size_t channels, std::size_t samples,
      const std::vector<Sos>& sections, bool reverse, bool chunked,
      const cudaDeviceProp& properties, cudaStream_t stream,
      PostState* response, PostState* initial)
{
   if (sections.empty()) return;
   const std::size_t chunks=(samples-1)/POST_IIR_CHUNK+1;
   const std::size_t count=post_product(channels,chunks);
   const unsigned serial_grid=post_cuda_grid(channels,properties);
   const unsigned chunk_grid=post_cuda_grid(count,properties);
   for (const Sos& section : sections) {
      if (!chunked) {
         PostSosSerial<<<serial_grid,256,0,stream>>>(data,channels,samples,section,reverse);
      }
      else {
         PostSosChunkResponse<<<chunk_grid,256,0,stream>>>(data,
            response,channels,samples,chunks,section,reverse);
         post_cuda_check(cudaGetLastError());
         PostSosChunkPrefix<<<serial_grid,256,0,stream>>>(
            response,initial,channels,chunks,
            post_transition_power(section,POST_IIR_CHUNK));
         post_cuda_check(cudaGetLastError());
         PostSosChunkApply<<<chunk_grid,256,0,stream>>>(data,
            initial,channels,samples,chunks,section,reverse);
      }
      post_cuda_check(cudaGetLastError());
   }
}

__global__ void PostResample(const double* input, double* output,
      std::size_t channels, std::size_t samples, std::size_t samples_out,
      double ratio)
{
   const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
   if (index>=channels*samples_out) return;
   const std::size_t channel=index/samples_out;
   const double position=static_cast<double>(index%samples_out)/ratio;
   const double radius=resample_radius(ratio);
   const int64_t begin=static_cast<int64_t>(fmax(0.0,ceil(position-radius)));
   const int64_t end=static_cast<int64_t>(fmin(static_cast<double>(samples-1),floor(position+radius)));
   double sum=0;
   for (int64_t n=begin; n<=end; ++n)
      sum+=input[channel*samples+static_cast<std::size_t>(n)]*
            resample_weight(position-static_cast<double>(n),ratio);
   output[index]=sum;
}

inline PostResult process_cuda(const std::vector<double>& input,
      const std::vector<double>& weights, std::size_t channels,
      std::size_t corners, std::size_t samples, double fs,
      const PostOptions& options, bool chunked, bool check_writes=false,
      const AirOptions* air=nullptr)
{
   validate_post_input(input,weights,channels,corners,samples,fs,options);
   post_cuda_check(cudaSetDevice(0));
   cudaDeviceProp properties{};
   post_cuda_check(cudaGetDeviceProperties(&properties,0));
   const double fs_out=options.output_rate ? options.output_rate : fs;
   const std::size_t samples_out=resample_length(samples,fs,fs_out);
   const std::size_t count=post_product(channels,samples);
   const std::size_t count_out=post_product(channels,samples_out);
   const bool has_air=air && air_mode(*air)!="none";
   const std::size_t final_samples=has_air ? air_output_samples(samples_out,fs_out,*air) : samples_out;
   const std::size_t final_count=post_product(channels,final_samples);
   const unsigned raw_grid=post_cuda_grid(count,properties);
   const unsigned out_grid=post_cuda_grid(count_out,properties);
   PostResult result;
   result.channels=channels; result.samples=samples; result.samples_out=final_samples; result.fs_out=fs_out;
   result.raw.resize(count); result.filtered.resize(final_count);
   PostCudaBuffer raw(input.size()), alpha(weights.size()), data(count);
   PostCudaBuffer resampled(fs_out!=fs ? count_out : 0);
   PostCudaBuffer air_output(has_air ? final_count : 0);
   const std::size_t max_samples=std::max(samples,samples_out);
   const std::size_t scratch_count=chunked ?
      post_product(post_product(channels,(max_samples-1)/POST_IIR_CHUNK+1),2) : 0;
   // Shared scratch survives every queued filter stage, avoiding stage fences
   // and repeated allocation while processing the resident receiver array.
   PostCudaBuffer response(scratch_count), initial(scratch_count);
   // Declared last: the stream is drained before any buffer is freed on failure.
   PostCudaStream stream;
   post_cuda_check(cudaMemcpyAsync(raw.get(),input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
   post_cuda_check(cudaMemcpyAsync(alpha.get(),weights.data(),weights.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
   if (check_writes) {
      post_cuda_check(cudaMemsetAsync(data.get(),0xff,count*sizeof(double),stream));
      if (resampled.get()) post_cuda_check(cudaMemsetAsync(resampled.get(),0xff,count_out*sizeof(double),stream));
      if (scratch_count) {
         post_cuda_check(cudaMemsetAsync(response.get(),0xff,scratch_count*sizeof(double),stream));
         post_cuda_check(cudaMemsetAsync(initial.get(),0xff,scratch_count*sizeof(double),stream));
      }
   }
   PostRecombine<<<raw_grid,256,0,stream>>>(raw.get(),alpha.get(),data.get(),channels,corners,samples);
   post_cuda_check(cudaGetLastError());
   post_cuda_check(cudaMemcpyAsync(result.raw.data(),data.get(),count*sizeof(double),cudaMemcpyDeviceToHost,stream));
   filter_cuda(data.get(),channels,samples,
      design_highpass(options.lowcut_order,fs,options.lowcut,options.integrate),false,chunked,properties,stream,
      reinterpret_cast<PostState*>(response.get()),reinterpret_cast<PostState*>(initial.get()));
   double* processed=data.get();
   if (fs_out!=fs) {
      PostResample<<<out_grid,256,0,stream>>>(data.get(),resampled.get(),channels,samples,samples_out,fs_out/fs);
      post_cuda_check(cudaGetLastError()); processed=resampled.get();
   }
   const std::vector<Sos> lowpass=post_lowpass_sections(fs_out,options);
   filter_cuda(processed,channels,samples_out,lowpass,false,chunked,properties,stream,
      reinterpret_cast<PostState*>(response.get()),reinterpret_cast<PostState*>(initial.get()));
   if (options.symmetric_lowpass)
      filter_cuda(processed,channels,samples_out,lowpass,true,chunked,properties,stream,
         reinterpret_cast<PostState*>(response.get()),reinterpret_cast<PostState*>(initial.get()));
   if (has_air) {
      air_apply_cuda(processed,air_output.get(),channels,samples_out,fs_out,*air,properties,stream);
      processed=air_output.get();
   }
   post_cuda_check(cudaMemcpyAsync(result.filtered.data(),processed,final_count*sizeof(double),cudaMemcpyDeviceToHost,stream));
   post_cuda_check(cudaStreamSynchronize(stream));
   check_post_finite(result.raw); check_post_finite(result.filtered);
   return result;
}
} // namespace pffdtd_post
#endif // __CUDACC__
#endif
