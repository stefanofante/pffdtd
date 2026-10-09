///////////////////////////////////////////////////////////////////////////////
// Checked host arithmetic for device allocations and CUDA launch dimensions.
// This header has no CUDA dependency, so validation runs without a GPU.
///////////////////////////////////////////////////////////////////////////////

#ifndef PFFDTD_MEMORY_LAUNCH_H
#define PFFDTD_MEMORY_LAUNCH_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace pffdtd {

// Failure leaves the output unchanged, including when it aliases an operand.
inline bool checked_add_size(std::size_t a, std::size_t b, std::size_t &result)
{
   if (b > std::numeric_limits<std::size_t>::max() - a) return false;
   result = a + b;
   return true;
}

inline bool checked_mul_size(std::size_t a, std::size_t b, std::size_t &result)
{
   if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
   result = a * b;
   return true;
}

inline bool checked_count_bytes(int64_t count, std::size_t element,
                                std::size_t &result)
{
   if (count < 0
       || static_cast<uintmax_t>(count)
             > static_cast<uintmax_t>(std::numeric_limits<std::size_t>::max()))
      return false;
   return checked_mul_size(static_cast<std::size_t>(count), element, result);
}

inline bool checked_ceil_count(int64_t count, int64_t divisor, int64_t &result)
{
   if (count < 0 || divisor <= 0) return false;
   // Unlike (count + divisor - 1) / divisor, this also handles INT64_MAX.
   result = count / divisor + (count % divisor != 0 ? 1 : 0);
   return true;
}

struct MemoryCounts {
   int64_t Npts, Nb, Nbl, Nba, Ns, Nr, Nt, Nm;
   std::size_t branches, readout_block;
};

struct DeviceMemoryPlan {
   std::size_t total_bytes, grid_bytes, boundary_bytes, source_bytes;
   std::size_t receiver_bytes, material_bytes, optional_bytes;
};

namespace memory_launch_detail {

// Calculate each allocation's factors before adding it to a category. Empty
// allocations remain zero without overflowing unused per-element factors.
inline bool add_allocation(int64_t count, std::size_t elements,
                           std::size_t element_bytes, std::size_t copies,
                           std::size_t &category)
{
   std::size_t bytes;
   return checked_count_bytes(count, elements, bytes)
          && checked_mul_size(bytes, element_bytes, bytes)
          && checked_mul_size(bytes, copies, bytes)
          && checked_add_size(category, bytes, category);
}

inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t &result)
{
   if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
   result = a * b;
   return true;
}

} // namespace memory_launch_detail

// Exact bytes requested by the solver, excluding allocator/runtime overhead.
// Includes the three boundary carries in both fused and reference modes.
// Map indices are disabled (0), int32_t (4), or int64_t (8). On failure the
// caller's plan stays unchanged; partially checked totals are never exposed.
inline bool device_memory_plan(const MemoryCounts &counts,
                                std::size_t real_bytes,
                                std::size_t matquad_bytes,
                                std::size_t map_index_bytes,
                                bool graph_counter, DeviceMemoryPlan &result)
{
   if (counts.Npts < 0 || counts.Nb < 0 || counts.Nbl < 0 || counts.Nba < 0
       || counts.Ns < 0 || counts.Nr < 0 || counts.Nt < 0 || counts.Nm < 0
       || (map_index_bytes != 0 && map_index_bytes != 4 && map_index_bytes != 8))
      return false;

   DeviceMemoryPlan plan = {};
   int64_t mask_count;
   using memory_launch_detail::add_allocation;
   if (!checked_ceil_count(counts.Npts, 8, mask_count)
       || !add_allocation(counts.Npts, 1, real_bytes, 2, plan.grid_bytes)
       || !add_allocation(mask_count, 1, 1, 1, plan.grid_bytes)
       || !add_allocation(counts.Nb, 1, 11, 1, plan.boundary_bytes)
       || !add_allocation(counts.Nbl, 1, 9, 1, plan.boundary_bytes)
       || !add_allocation(counts.Nbl, 1, real_bytes, 4, plan.boundary_bytes)
       || !add_allocation(counts.Nbl, counts.branches, real_bytes, 2,
                          plan.boundary_bytes)
       || !add_allocation(counts.Nba, 1, 9, 1, plan.boundary_bytes)
       || !add_allocation(counts.Nba, 1, real_bytes, 1, plan.boundary_bytes)
       || !add_allocation(counts.Ns, 1, 8, 1, plan.source_bytes))
      return false;

   // Nt is signed too: convert it with the same checked count operation.
   std::size_t timesteps;
   if (!checked_count_bytes(counts.Nt, 1, timesteps)
       || !add_allocation(counts.Ns, timesteps, 8, 1, plan.source_bytes)
       || !add_allocation(counts.Nr, 1, 8, 1, plan.receiver_bytes)
       || !add_allocation(counts.Nr, counts.readout_block, real_bytes, 1,
                          plan.receiver_bytes)
       || !add_allocation(counts.Nm, 1, real_bytes, 1, plan.material_bytes)
       || !add_allocation(counts.Nm, counts.branches, matquad_bytes, 1,
                          plan.material_bytes)
       || !add_allocation(counts.Nb, 1, map_index_bytes, 1, plan.optional_bytes)
       || !checked_add_size(plan.optional_bytes, graph_counter ? 8 : 0,
                            plan.optional_bytes)
       || !checked_add_size(plan.total_bytes, plan.grid_bytes, plan.total_bytes)
       || !checked_add_size(plan.total_bytes, plan.boundary_bytes, plan.total_bytes)
       || !checked_add_size(plan.total_bytes, plan.source_bytes, plan.total_bytes)
       || !checked_add_size(plan.total_bytes, plan.receiver_bytes, plan.total_bytes)
       || !checked_add_size(plan.total_bytes, plan.material_bytes, plan.total_bytes)
       || !checked_add_size(plan.total_bytes, plan.optional_bytes, plan.total_bytes))
      return false;
   result = plan;
   return true;
}

struct LaunchShape {
   uint64_t x, y, z;
};

struct LaunchLimits {
   uint64_t grid[3], block[3], threads;
};

inline bool valid_launch(const LaunchShape &grid, const LaunchShape &block,
                          const LaunchLimits &limits)
{
   const uint64_t grid_axes[] = {grid.x, grid.y, grid.z};
   const uint64_t block_axes[] = {block.x, block.y, block.z};
   const uint64_t dimension_max = std::numeric_limits<uint32_t>::max();
   for (std::size_t axis = 0; axis < 3; ++axis) {
      if (grid_axes[axis] == 0 || block_axes[axis] == 0
          || grid_axes[axis] > dimension_max || block_axes[axis] > dimension_max
          || grid_axes[axis] > limits.grid[axis]
          || block_axes[axis] > limits.block[axis])
         return false;
   }
   uint64_t threads;
   return memory_launch_detail::checked_mul_u64(block.x, block.y, threads)
          && memory_launch_detail::checked_mul_u64(threads, block.z, threads)
          && threads <= limits.threads;
}

} // namespace pffdtd

#endif
