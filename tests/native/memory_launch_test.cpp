// Allocation and launch validation uses scalar fixtures; no device is needed.
#include "memory_launch.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

int cases = 0;

void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr, "FAIL: %s\n", message);
      std::exit(EXIT_FAILURE);
   }
   ++cases;
}

bool same_plan(const pffdtd::DeviceMemoryPlan &a,
               const pffdtd::DeviceMemoryPlan &b)
{
   return a.total_bytes == b.total_bytes && a.grid_bytes == b.grid_bytes
          && a.boundary_bytes == b.boundary_bytes
          && a.source_bytes == b.source_bytes
          && a.receiver_bytes == b.receiver_bytes
          && a.material_bytes == b.material_bytes
          && a.optional_bytes == b.optional_bytes;
}

void check_arithmetic()
{
   const std::size_t maximum = std::numeric_limits<std::size_t>::max();
   std::size_t bytes = 37;
   require(pffdtd::checked_add_size(maximum, 0, bytes) && bytes == maximum,
           "addition rejected SIZE_MAX + 0");
   require(!pffdtd::checked_add_size(maximum, 1, bytes) && bytes == maximum,
           "addition overflow changed the output");
   require(pffdtd::checked_add_size(0, maximum, bytes) && bytes == maximum,
           "addition is not symmetric at SIZE_MAX");
   require(pffdtd::checked_mul_size(maximum, 1, bytes) && bytes == maximum,
           "multiplication rejected SIZE_MAX * 1");
   require(!pffdtd::checked_mul_size(maximum, 2, bytes) && bytes == maximum,
           "multiplication overflow changed the output");
   require(pffdtd::checked_mul_size(maximum, 0, bytes) && bytes == 0,
           "multiplication rejected a zero second operand");
   require(pffdtd::checked_mul_size(0, maximum, bytes) && bytes == 0,
           "multiplication rejected a zero first operand");
   bytes = 9;
   require(pffdtd::checked_mul_size(bytes, bytes, bytes) && bytes == 81,
           "aliased multiplication lost an operand");
   require(pffdtd::checked_add_size(bytes, bytes, bytes) && bytes == 162,
           "aliased addition lost an operand");
   require(!pffdtd::checked_count_bytes(-1, 0, bytes) && bytes == 162,
           "negative count was accepted with a zero element size");
   require(!pffdtd::checked_count_bytes(std::numeric_limits<int64_t>::min(), 1,
                                       bytes) && bytes == 162,
           "INT64_MIN count was accepted");
   require(pffdtd::checked_count_bytes(0, maximum, bytes) && bytes == 0,
           "empty byte count overflowed");
   require(pffdtd::checked_count_bytes(1, maximum, bytes) && bytes == maximum,
           "SIZE_MAX-byte allocation was rejected");
   require(!pffdtd::checked_count_bytes(2, maximum, bytes) && bytes == maximum,
           "byte product overflow changed the output");
   const int64_t signed_maximum = std::numeric_limits<int64_t>::max();
   if (static_cast<uintmax_t>(maximum) >= static_cast<uintmax_t>(signed_maximum)) {
      require(pffdtd::checked_count_bytes(signed_maximum, 1, bytes)
              && bytes == static_cast<std::size_t>(signed_maximum),
              "INT64_MAX byte count was rejected when representable");
   } else {
      require(!pffdtd::checked_count_bytes(signed_maximum, 1, bytes),
              "count conversion truncated on a narrower size_t");
   }
   require(!pffdtd::checked_count_bytes(signed_maximum, 3, bytes),
           "INT64_MAX byte product overflow was accepted");

   int64_t count = 19;
   require(pffdtd::checked_ceil_count(0, 8, count) && count == 0,
           "empty ceiling count is not zero");
   require(pffdtd::checked_ceil_count(16, 8, count) && count == 2,
           "exact ceiling count is wrong");
   require(pffdtd::checked_ceil_count(17, 8, count) && count == 3,
           "ceiling remainder is wrong");
   require(pffdtd::checked_ceil_count(signed_maximum, 1, count)
           && count == signed_maximum, "ceil(INT64_MAX/1) overflowed");
   require(pffdtd::checked_ceil_count(signed_maximum, 2, count)
           && count == signed_maximum / 2 + 1, "ceil(INT64_MAX/2) overflowed");
   require(pffdtd::checked_ceil_count(signed_maximum, signed_maximum, count)
           && count == 1, "ceil(INT64_MAX/INT64_MAX) overflowed");
   require(!pffdtd::checked_ceil_count(-1, 8, count) && count == 1,
           "negative ceiling count changed the output");
   require(!pffdtd::checked_ceil_count(8, 0, count) && count == 1,
           "zero divisor was accepted");
   require(!pffdtd::checked_ceil_count(8, -1, count) && count == 1,
           "negative divisor was accepted");
}

void check_manual_allocations()
{
   // Hand-counted fixture: 65 grid cells (9 mask bytes), 7 boundary nodes,
   // 3 lossy nodes, 2 ABC nodes, 2 sources with 11 samples, 5 receivers with
   // 13 columns, and 4 materials with 6 four-real coefficient records each.
   const pffdtd::MemoryCounts counts = {65, 7, 3, 2, 2, 5, 11, 4, 6, 13};
   const std::size_t map_sizes[] = {0, 4, 8};
   for (std::size_t precision = 0; precision < 2; ++precision) {
      const std::size_t real_bytes = precision == 0 ? 4 : 8;
      const std::size_t matquad_bytes = precision == 0 ? 16 : 32;
      const pffdtd::DeviceMemoryPlan expected = precision == 0
         ? pffdtd::DeviceMemoryPlan{1743, 529, 322, 192, 300, 400, 0}
         : pffdtd::DeviceMemoryPlan{3123, 1049, 522, 192, 560, 800, 0};
      for (std::size_t map = 0; map < 3; ++map) {
         for (int counter = 0; counter < 2; ++counter) {
            pffdtd::DeviceMemoryPlan actual = {};
            pffdtd::DeviceMemoryPlan with_optional = expected;
            const std::size_t map_allocation = map == 0 ? 0 : (map == 1 ? 28 : 56);
            with_optional.optional_bytes = map_allocation + (counter == 0 ? 0 : 8);
            with_optional.total_bytes += with_optional.optional_bytes;
            require(pffdtd::device_memory_plan(counts, real_bytes, matquad_bytes,
                                               map_sizes[map], counter != 0, actual),
                    "manual allocation fixture was rejected");
            require(same_plan(actual, with_optional),
                    "planner does not match the manually counted allocations");
         }
      }
   }

   const std::size_t maximum = std::numeric_limits<std::size_t>::max();
   const pffdtd::MemoryCounts empty = {0, 0, 0, 0, 0, 0, 0, 0, maximum, maximum};
   pffdtd::DeviceMemoryPlan actual = {1, 2, 3, 4, 5, 6, 7};
   const pffdtd::DeviceMemoryPlan zero = {};
   require(pffdtd::device_memory_plan(empty, maximum, maximum, 0, false, actual)
           && same_plan(actual, zero), "empty allocations overflow unused factors");
   const pffdtd::DeviceMemoryPlan counter_only = {8, 0, 0, 0, 0, 0, 8};
   require(pffdtd::device_memory_plan(empty, 4, 16, 8, true, actual)
           && same_plan(actual, counter_only), "empty plan lost the graph counter");

   pffdtd::MemoryCounts mask_only = {};
   mask_only.Npts = 8;
   require(pffdtd::device_memory_plan(mask_only, 4, 16, 0, false, actual)
           && actual.grid_bytes == 65 && actual.total_bytes == 65,
           "exact multiple of eight allocated an extra mask byte");
   mask_only.Npts = 9;
   require(pffdtd::device_memory_plan(mask_only, 4, 16, 0, false, actual)
           && actual.grid_bytes == 74 && actual.total_bytes == 74,
           "mask remainder byte is missing");
}

void reject_plan(const pffdtd::MemoryCounts &counts, std::size_t real_bytes = 4,
                  std::size_t matquad_bytes = 16, std::size_t map_bytes = 0)
{
   const pffdtd::DeviceMemoryPlan before = {1, 2, 3, 4, 5, 6, 7};
   pffdtd::DeviceMemoryPlan actual = before;
   require(!pffdtd::device_memory_plan(counts, real_bytes, matquad_bytes,
                                      map_bytes, true, actual),
           "invalid or overflowing memory plan was accepted");
   require(same_plan(actual, before), "failed memory plan changed its output");
}

void check_rejected_allocations()
{
   const int64_t signed_maximum = std::numeric_limits<int64_t>::max();
   const std::size_t maximum = std::numeric_limits<std::size_t>::max();
   int64_t pffdtd::MemoryCounts::*const signed_members[] = {
      &pffdtd::MemoryCounts::Npts, &pffdtd::MemoryCounts::Nb,
      &pffdtd::MemoryCounts::Nbl, &pffdtd::MemoryCounts::Nba,
      &pffdtd::MemoryCounts::Ns, &pffdtd::MemoryCounts::Nr,
      &pffdtd::MemoryCounts::Nt, &pffdtd::MemoryCounts::Nm
   };
   for (std::size_t i = 0; i < 8; ++i) {
      pffdtd::MemoryCounts counts = {};
      counts.*signed_members[i] = -1;
      reject_plan(counts);
   }
   pffdtd::MemoryCounts counts = {};
   reject_plan(counts, 4, 16, 1);
   reject_plan(counts, 4, 16, 16);
   counts.Npts = signed_maximum;
   reject_plan(counts);                        // Grid allocation itself overflows.
   reject_plan(counts, 2);                     // One grid fits; two do not.
   reject_plan(counts, 1);                     // Grids fit; adding mask overflows.
   counts = {};
   counts.Npts = 1;
   reject_plan(counts, maximum);               // Real-byte factor exceeds paired grids.
   counts = {};
   counts.Nb = signed_maximum;
   reject_plan(counts);                        // Boundary metadata overflows.
   counts = {};
   counts.Nbl = 1;
   counts.branches = maximum;
   reject_plan(counts, 1);                     // Two branch carries overflow.
   counts.Nbl = 2;
   reject_plan(counts, 1);                     // Node * branch count overflows first.
   counts = {};
   counts.Nba = signed_maximum;
   reject_plan(counts);                        // ABC index/flag allocation overflows.
   counts = {};
   counts.Ns = 2;
   counts.Nt = signed_maximum;
   reject_plan(counts);                        // Source * timestep * double bytes.
   counts = {};
   counts.Nr = 2;
   counts.readout_block = maximum;
   reject_plan(counts);                        // Receiver * column count overflows first.
   counts.Nr = 1;
   reject_plan(counts);                        // Column bytes overflow afterward.
   counts = {};
   counts.Nm = 1;
   counts.branches = maximum;
   reject_plan(counts, 1, 2);                  // Material * branch * record bytes.
   counts.Nm = 2;
   reject_plan(counts, 1, 1);                  // Material * branch count overflows first.
   counts = {};
   counts.Npts = signed_maximum / 2;
   counts.Nb = signed_maximum / 11;
   reject_plan(counts, 1, 1);                  // Each category fits; their sum does not.
}

void check_launches()
{
   const pffdtd::LaunchLimits cuda_limits = {
      {2147483647ULL, 65535, 65535}, {1024, 1024, 64}, 1024
   };
   const pffdtd::LaunchShape one = {1, 1, 1};
   require(pffdtd::valid_launch(one, one, cuda_limits), "single-thread launch rejected");
   require(pffdtd::valid_launch({2147483647ULL, 65535, 65535},
                                {1024, 1, 1}, cuda_limits),
           "launch at grid and thread limits rejected");
   require(pffdtd::valid_launch({1, 1, 1}, {16, 1, 64}, cuda_limits),
           "launch at block-z and thread limits rejected");
   require(!pffdtd::valid_launch(one, {32, 32, 2}, cuda_limits),
           "block thread product exceeded the limit");
   const pffdtd::LaunchShape empty_axes[] = {{0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
   for (const pffdtd::LaunchShape &shape : empty_axes) {
      require(!pffdtd::valid_launch(shape, one, cuda_limits), "zero grid axis accepted");
      require(!pffdtd::valid_launch(one, shape, cuda_limits), "zero block axis accepted");
   }
   require(!pffdtd::valid_launch({2147483648ULL, 1, 1}, one, cuda_limits),
           "grid-x device limit was ignored");
   require(!pffdtd::valid_launch({1, 65536, 1}, one, cuda_limits),
           "grid-y device limit was ignored");
   require(!pffdtd::valid_launch({1, 1, 65536}, one, cuda_limits),
           "grid-z device limit was ignored");
   require(!pffdtd::valid_launch(one, {1025, 1, 1}, cuda_limits),
           "block-x device limit was ignored");
   require(!pffdtd::valid_launch(one, {1, 1025, 1}, cuda_limits),
           "block-y device limit was ignored");
   require(!pffdtd::valid_launch(one, {1, 1, 65}, cuda_limits),
           "block-z device limit was ignored");

   const uint64_t maximum = std::numeric_limits<uint64_t>::max();
   const uint64_t dimension_maximum = std::numeric_limits<uint32_t>::max();
   const pffdtd::LaunchLimits unlimited = {
      {maximum, maximum, maximum}, {maximum, maximum, maximum}, maximum
   };
   require(!pffdtd::valid_launch({dimension_maximum + 1, 1, 1}, one, unlimited),
           "dim3 grid conversion would truncate");
   require(!pffdtd::valid_launch(one, {1, dimension_maximum + 1, 1}, unlimited),
           "dim3 block conversion would truncate");
   require(pffdtd::valid_launch(one, {dimension_maximum, dimension_maximum, 1},
                                unlimited), "representable wide block product rejected");
   require(!pffdtd::valid_launch(one,
                                 {dimension_maximum, dimension_maximum, dimension_maximum},
                                 unlimited), "uint64_t block product overflow accepted");

   // CUDA built-in coordinates are uint32_t; promotion after multiplication
   // is too late. This reproduces the first wrap without executing a kernel.
   const uint32_t block_index = uint32_t(1) << 24;
   const uint32_t block_size = 256;
   const uint64_t late_promotion = static_cast<uint64_t>(block_index * block_size);
   const uint64_t early_promotion = static_cast<uint64_t>(block_index) * block_size;
   require(late_promotion == 0 && early_promotion == (uint64_t(1) << 32),
           "wide-index fixture failed to expose uint32_t multiplication wrap");

   // The air launch maps the x tiles onto grid.z. A thin, valid volume can
   // exceed grid.z even though its grid memory demand is modest.
   const int64_t Nx = 131073, Ny = 3, Nz = 3;
   int64_t x_tiles = 0, y_tiles = 0, z_tiles = 0;
   require(pffdtd::checked_ceil_count(Nx - 2, 2, x_tiles)
           && pffdtd::checked_ceil_count(Ny - 2, 4, y_tiles)
           && pffdtd::checked_ceil_count(Nz - 2, 32, z_tiles),
           "thin volume tile arithmetic failed");
   require(x_tiles == 65536 && y_tiles == 1 && z_tiles == 1,
           "thin volume did not reproduce grid-z overflow");
   require(!pffdtd::valid_launch({static_cast<uint64_t>(z_tiles),
                                  static_cast<uint64_t>(y_tiles),
                                  static_cast<uint64_t>(x_tiles)},
                                 {32, 4, 2}, cuda_limits),
           "thin volume launch exceeded CUDA grid-z limit undetected");
}

} // namespace

int main()
{
   check_arithmetic();
   check_manual_allocations();
   check_rejected_allocations();
   check_launches();
   std::printf("PASS: memory and launch validation (%d checks)\n", cases);
   return EXIT_SUCCESS;
}
