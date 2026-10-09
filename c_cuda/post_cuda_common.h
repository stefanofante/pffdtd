// Shared ownership and launch checks for resident CUDA postprocessing stages.
#ifndef PFFDTD_POST_CUDA_COMMON_H
#define PFFDTD_POST_CUDA_COMMON_H
#ifdef __CUDACC__
#include "post_pipeline.h"
#include <cuda_runtime.h>
#include <string>
namespace pffdtd_post {
inline void post_cuda_check(cudaError_t status)
{
   if (status!=cudaSuccess)
      throw std::runtime_error(std::string("CUDA postprocessing: ")+cudaGetErrorString(status));
}
class PostCudaBuffer {
   double* pointer=nullptr;
public:
   explicit PostCudaBuffer(std::size_t count)
   { if (count) post_cuda_check(cudaMalloc(&pointer,post_product(count,1)*sizeof(double))); }
   ~PostCudaBuffer() { if (pointer) cudaFree(pointer); }
   double* get() const { return pointer; }
   PostCudaBuffer(const PostCudaBuffer&)=delete;
   PostCudaBuffer& operator=(const PostCudaBuffer&)=delete;
};
class PostCudaStream {
   cudaStream_t stream=nullptr;
public:
   PostCudaStream() { post_cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
   ~PostCudaStream() {
      if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
   }
   operator cudaStream_t() const { return stream; }
   PostCudaStream(const PostCudaStream&)=delete;
   PostCudaStream& operator=(const PostCudaStream&)=delete;
};
inline unsigned post_cuda_grid(std::size_t count,const cudaDeviceProp& properties)
{
   if (!count || properties.maxThreadsPerBlock<256)
      throw std::invalid_argument("Invalid CUDA postprocessing launch size");
   const std::size_t blocks=(count-1)/256+1;
   if (blocks>static_cast<std::size_t>(properties.maxGridSize[0]))
      throw std::invalid_argument("Postprocessing grid exceeds CUDA device limit");
   return static_cast<unsigned>(blocks);
}
} // namespace pffdtd_post
#endif
#endif
