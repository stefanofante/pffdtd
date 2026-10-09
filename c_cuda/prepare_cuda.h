// Resident mesh/axis data and bounded, stable boundary compaction on CUDA.
#ifndef PFFDTD_PREPARE_CUDA_H
#define PFFDTD_PREPARE_CUDA_H
#ifdef __CUDACC__

#include "prepare_scene.h"
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace pffdtd_prepare {
namespace detail {

inline void prepare_cuda_check(cudaError_t status)
{
   if (status!=cudaSuccess)
      throw std::runtime_error(std::string("CUDA mesh preparation: ")+cudaGetErrorString(status));
}

inline std::size_t prepare_cuda_bytes(std::size_t count, std::size_t size)
{
   if (size && count>std::numeric_limits<std::size_t>::max()/size)
      throw std::overflow_error("CUDA preparation byte extent overflows");
   return count*size;
}

inline void prepare_cuda_add(std::size_t& total, std::size_t amount)
{
   if (amount>std::numeric_limits<std::size_t>::max()-total)
      throw std::overflow_error("CUDA preparation memory estimate overflows");
   total+=amount;
}

template<typename T>
class PrepareCudaBuffer {
   T* pointer=nullptr;
public:
   explicit PrepareCudaBuffer(std::size_t count=0) { allocate(count); }
   void allocate(std::size_t count)
   {
      if (pointer) throw std::logic_error("CUDA preparation buffer is already allocated");
      if (count) prepare_cuda_check(cudaMalloc(&pointer,prepare_cuda_bytes(count,sizeof(T))));
   }
   ~PrepareCudaBuffer() { if (pointer) cudaFree(pointer); }
   T* get() const { return pointer; }
   PrepareCudaBuffer(const PrepareCudaBuffer&)=delete;
   PrepareCudaBuffer& operator=(const PrepareCudaBuffer&)=delete;
};

template<typename T>
class PreparePinnedBuffer {
   T* pointer=nullptr;
public:
   explicit PreparePinnedBuffer(std::size_t count)
   { if (count) prepare_cuda_check(cudaMallocHost(&pointer,prepare_cuda_bytes(count,sizeof(T)))); }
   ~PreparePinnedBuffer() { if (pointer) cudaFreeHost(pointer); }
   T* get() const { return pointer; }
   PreparePinnedBuffer(const PreparePinnedBuffer&)=delete;
   PreparePinnedBuffer& operator=(const PreparePinnedBuffer&)=delete;
};

class PrepareCudaStream {
   cudaStream_t stream=nullptr;
public:
   PrepareCudaStream()
   { prepare_cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
   ~PrepareCudaStream()
   {
      // This object is declared after every device/pinned buffer. Fence also
      // exceptional exits before their destruction, including active uploads.
      if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
   }
   operator cudaStream_t() const { return stream; }
   PrepareCudaStream(const PrepareCudaStream&)=delete;
   PrepareCudaStream& operator=(const PrepareCudaStream&)=delete;
};

inline void prepare_cuda_geometry(const FlatBvh& geometry)
{
   const std::size_t limit=std::numeric_limits<std::uint32_t>::max();
   if (geometry.triangles.size()>limit || geometry.nodes.size()>limit || geometry.order.size()>limit ||
       geometry.nodes.empty()!=geometry.order.empty())
      throw std::invalid_argument("CUDA mesh BVH shape is invalid");
   if (!geometry.nodes.empty() && geometry.nodes[0].escape!=geometry.nodes.size())
      throw std::invalid_argument("CUDA mesh BVH root escape is invalid");
   for (std::size_t i=0; i<geometry.nodes.size(); ++i) {
      const BvhNode& node=geometry.nodes[i];
      if (node.escape<=i || node.escape>geometry.nodes.size() ||
          node.begin>geometry.order.size() || node.count>geometry.order.size()-node.begin ||
          (!node.count && node.escape<=i+1))
         throw std::invalid_argument("CUDA mesh BVH traversal indices are invalid");
   }
   for (std::uint32_t index : geometry.order)
      if (index>=geometry.triangles.size() || !(geometry.triangles[index].area>0))
         throw std::invalid_argument("CUDA mesh BVH primitive index is invalid");
}

// Local rank spans only interior coordinates. Odd FCC nodes return before the
// BVH traversal. Records retain full physical indices in ascending rank order.
static __global__ void PrepareClassifyBatch(const Triangle* triangles,
      const BvhNode* nodes, const std::uint32_t* order, std::uint32_t node_count,
      const double* x_axis, const double* y_axis, const double* z_axis,
      std::int64_t ny, std::int64_t nz, std::int64_t interior_y,
      std::int64_t interior_z, std::int64_t begin, int count, double h, bool fcc,
      BoundaryRecord* records, std::uint8_t* flags)
{
   const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
   if (index>=static_cast<unsigned>(count)) return;
   const std::int64_t rank=begin+index;
   const std::int64_t x=rank/(interior_y*interior_z)+1,
                      y=(rank/interior_z)%interior_y+1,z=rank%interior_z+1;
   flags[index]=0;
   if (fcc && ((x^y^z)&1)) return;
   const Vec3 point={x_axis[x],y_axis[y],z_axis[z]};
   const NodeBoundary node=classify_node(triangles,nodes,order,node_count,point,h,fcc);
   const std::uint16_t all=static_cast<std::uint16_t>((1U<<(fcc ? 12 : 6))-1);
   if (node.adjacency==all) return;
   BoundaryRecord& record=records[index];
   record.index=(x*ny+y)*nz+z;
   record.adjacency=node.adjacency; record.triangle=node.nearest_triangle;
   record.material=node.material; record.saf=node.saf;
   record.near_boundary=node.near_boundary!=0;
   flags[index]=1;
}

} // namespace detail

// The thread option belongs to the CPU backend. CUDA uses a fixed 128-thread
// block and at most 262144 interior candidates per batch; no whole-grid masks
// or dense per-grid records are allocated. Geometry and exact axis coordinates
// remain on the device, and only compacted boundary records return to the host.
inline std::vector<BoundaryRecord> voxelize_cuda(const FlatBvh& geometry,
      const Grid& grid, unsigned /*cpu_threads*/=0)
{
   static_assert(std::is_trivially_copyable<Triangle>::value &&
                 std::is_trivially_copyable<BvhNode>::value &&
                 std::is_trivially_copyable<BoundaryRecord>::value,
                 "CUDA mesh records must be trivially copyable");
   validate_grid(grid); detail::prepare_cuda_geometry(geometry);
   if (grid.h>0.125*std::numeric_limits<double>::max())
      throw std::invalid_argument("CUDA mesh link lengths exceed FP64 range");
   const GridDims dims=grid.dims();
   const std::int64_t interior_y=dims[1]-2,interior_z=dims[2]-2;
   const GridDims interior_dims={{dims[0]-2,interior_y,interior_z}};
   const std::int64_t interior_count=grid_volume(interior_dims);
   std::vector<BoundaryRecord> result;
   if (geometry.nodes.empty()) return result;

   detail::prepare_cuda_check(cudaSetDevice(0));
   cudaDeviceProp properties{};
   detail::prepare_cuda_check(cudaGetDeviceProperties(&properties,0));
   const unsigned threads=128;
   if (properties.maxThreadsPerBlock<static_cast<int>(threads))
      throw std::runtime_error("CUDA device cannot run the mesh classification block");
   const int capacity=static_cast<int>(std::min<std::int64_t>(262144,interior_count));
   const unsigned max_blocks=(static_cast<unsigned>(capacity)+threads-1)/threads;
   if (max_blocks>static_cast<unsigned>(properties.maxGridSize[0]))
      throw std::invalid_argument("CUDA mesh batch exceeds the device grid limit");

   std::size_t scratch_bytes=0;
   detail::prepare_cuda_check(cub::DeviceSelect::Flagged(nullptr,scratch_bytes,
      static_cast<BoundaryRecord*>(nullptr),static_cast<std::uint8_t*>(nullptr),
      static_cast<BoundaryRecord*>(nullptr),static_cast<int*>(nullptr),capacity));
   std::size_t required=scratch_bytes;
   detail::prepare_cuda_add(required,detail::prepare_cuda_bytes(geometry.triangles.size(),sizeof(Triangle)));
   detail::prepare_cuda_add(required,detail::prepare_cuda_bytes(geometry.nodes.size(),sizeof(BvhNode)));
   detail::prepare_cuda_add(required,detail::prepare_cuda_bytes(geometry.order.size(),sizeof(std::uint32_t)));
   for (const auto& axis : grid.axes)
      detail::prepare_cuda_add(required,detail::prepare_cuda_bytes(axis.size(),sizeof(double)));
   detail::prepare_cuda_add(required,detail::prepare_cuda_bytes(capacity,2*sizeof(BoundaryRecord)+sizeof(std::uint8_t)));
   detail::prepare_cuda_add(required,sizeof(int));
   std::size_t free_bytes=0,total_bytes=0;
   detail::prepare_cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes));
   if (required>free_bytes)
      throw std::runtime_error("Resident geometry and bounded CUDA batch exceed available device memory");

   detail::PrepareCudaBuffer<Triangle> triangles(geometry.triangles.size());
   detail::PrepareCudaBuffer<BvhNode> nodes(geometry.nodes.size());
   detail::PrepareCudaBuffer<std::uint32_t> order(geometry.order.size());
   detail::PrepareCudaBuffer<double> x_axis(grid.axes[0].size()),
      y_axis(grid.axes[1].size()),z_axis(grid.axes[2].size());
   detail::PrepareCudaBuffer<BoundaryRecord> records(capacity),selected(capacity);
   detail::PrepareCudaBuffer<std::uint8_t> flags(capacity),scratch(scratch_bytes);
   detail::PrepareCudaBuffer<int> selected_count(1);
   detail::PreparePinnedBuffer<BoundaryRecord> staging(capacity);
   detail::PreparePinnedBuffer<int> host_count(1);
   detail::PrepareCudaStream stream; // Last: drains before every buffer dies.

   detail::prepare_cuda_check(cudaMemcpyAsync(triangles.get(),geometry.triangles.data(),
      detail::prepare_cuda_bytes(geometry.triangles.size(),sizeof(Triangle)),cudaMemcpyHostToDevice,stream));
   detail::prepare_cuda_check(cudaMemcpyAsync(nodes.get(),geometry.nodes.data(),
      detail::prepare_cuda_bytes(geometry.nodes.size(),sizeof(BvhNode)),cudaMemcpyHostToDevice,stream));
   detail::prepare_cuda_check(cudaMemcpyAsync(order.get(),geometry.order.data(),
      detail::prepare_cuda_bytes(geometry.order.size(),sizeof(std::uint32_t)),cudaMemcpyHostToDevice,stream));
   detail::prepare_cuda_check(cudaMemcpyAsync(x_axis.get(),grid.axes[0].data(),
      detail::prepare_cuda_bytes(grid.axes[0].size(),sizeof(double)),cudaMemcpyHostToDevice,stream));
   detail::prepare_cuda_check(cudaMemcpyAsync(y_axis.get(),grid.axes[1].data(),
      detail::prepare_cuda_bytes(grid.axes[1].size(),sizeof(double)),cudaMemcpyHostToDevice,stream));
   detail::prepare_cuda_check(cudaMemcpyAsync(z_axis.get(),grid.axes[2].data(),
      detail::prepare_cuda_bytes(grid.axes[2].size(),sizeof(double)),cudaMemcpyHostToDevice,stream));
   // CUB can load an unselected input record before observing its flag. Define
   // every byte, including aggregate padding, once before the first batch.
   detail::prepare_cuda_check(cudaMemsetAsync(records.get(),0,
      detail::prepare_cuda_bytes(capacity,sizeof(BoundaryRecord)),stream));

   const bool fcc=grid.fcc_flag!=0;
   const std::uint16_t all=static_cast<std::uint16_t>((1U<<(fcc ? 12 : 6))-1);
   for (std::int64_t begin=0; begin<interior_count;) {
      const int count=static_cast<int>(std::min<std::int64_t>(capacity,interior_count-begin));
      const unsigned blocks=(static_cast<unsigned>(count)+threads-1)/threads;
      detail::PrepareClassifyBatch<<<blocks,threads,0,stream>>>(triangles.get(),nodes.get(),order.get(),
         static_cast<std::uint32_t>(geometry.nodes.size()),x_axis.get(),y_axis.get(),z_axis.get(),
         dims[1],dims[2],interior_y,interior_z,begin,count,grid.h,fcc,records.get(),flags.get());
      detail::prepare_cuda_check(cudaGetLastError());
      std::size_t available=scratch_bytes;
      detail::prepare_cuda_check(cub::DeviceSelect::Flagged(scratch.get(),available,records.get(),flags.get(),
         selected.get(),selected_count.get(),count,stream));
      detail::prepare_cuda_check(cudaMemcpyAsync(host_count.get(),selected_count.get(),sizeof(int),
                                                cudaMemcpyDeviceToHost,stream));
      detail::prepare_cuda_check(cudaStreamSynchronize(stream));
      const int kept=*host_count.get();
      if (kept<0 || kept>count) throw std::runtime_error("CUDA boundary compaction count is invalid");
      if (static_cast<std::size_t>(kept)>result.max_size()-result.size())
         throw std::length_error("Compacted boundary exceeds host allocation limits");
      if (kept) {
         detail::prepare_cuda_check(cudaMemcpyAsync(staging.get(),selected.get(),
            detail::prepare_cuda_bytes(kept,sizeof(BoundaryRecord)),cudaMemcpyDeviceToHost,stream));
         detail::prepare_cuda_check(cudaStreamSynchronize(stream));
         for (int i=0; i<kept; ++i) {
            const BoundaryRecord& record=staging.get()[i];
            if (record.triangle>=geometry.triangles.size() || record.adjacency>=all ||
                !std::isfinite(record.saf) || record.saf<0 ||
                (i ? record.index<=staging.get()[i-1].index :
                     (!result.empty() && record.index<=result.back().index)))
               throw std::runtime_error("CUDA compacted boundary metadata or stable order is invalid");
         }
         result.insert(result.end(),staging.get(),staging.get()+kept);
      }
      begin+=count;
   }
   return result;
}

} // namespace pffdtd_prepare
#endif // __CUDACC__
#endif
