// Standalone host validation of boundary-to-lossy ordinals and failure atomicity.
#include "boundary_map.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {
int allocations_before_failure = -1;
}

#if defined(__GNUC__)
#define PFFDTD_TEST_NOINLINE __attribute__((noinline))
#else
#define PFFDTD_TEST_NOINLINE
#endif

// Inject allocation failures only while the builder runs; fixtures and checks
// use the normal allocator. This covers each temporary, rather than relying
// on an actual memory shortage or an enormous allocation. Keep replacement
// allocators out of line so diagnostics see matched new/delete calls.
PFFDTD_TEST_NOINLINE void *operator new(std::size_t size)
{
   if (allocations_before_failure == 0) throw std::bad_alloc();
   if (allocations_before_failure > 0) --allocations_before_failure;
   void *pointer = std::malloc(size == 0 ? 1 : size);
   if (pointer == nullptr) throw std::bad_alloc();
   return pointer;
}

PFFDTD_TEST_NOINLINE void operator delete(void *pointer) noexcept
{
   std::free(pointer);
}

PFFDTD_TEST_NOINLINE void *operator new[](std::size_t size)
{
   return ::operator new(size);
}

PFFDTD_TEST_NOINLINE void operator delete[](void *pointer) noexcept
{
   ::operator delete(pointer);
}

#if __cplusplus >= 201402L
PFFDTD_TEST_NOINLINE void operator delete(void *pointer, std::size_t) noexcept
{
   ::operator delete(pointer);
}

PFFDTD_TEST_NOINLINE void operator delete[](void *pointer, std::size_t) noexcept
{
   ::operator delete[](pointer);
}
#endif

#undef PFFDTD_TEST_NOINLINE

namespace {

using pffdtd::BoundaryMapStatus;

int cases = 0;

void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr, "FAIL: %s\n", message);
      std::exit(EXIT_FAILURE);
   }
}

template<typename MapIdx>
void check_valid(const std::vector<int64_t> &bn, const std::vector<int64_t> &bnl,
                 int64_t npts)
{
   const std::vector<int64_t> boundary_before = bn;
   const std::vector<int64_t> lossy_before = bnl;
   std::vector<MapIdx> map(bn.size() + 2, static_cast<MapIdx>(-7));
   const BoundaryMapStatus status = pffdtd::build_boundary_lossy_map(
      bn.empty() ? nullptr : bn.data(), static_cast<int64_t>(bn.size()),
      bnl.empty() ? nullptr : bnl.data(), static_cast<int64_t>(bnl.size()),
      npts, map.data() + 1);
   require(status == BoundaryMapStatus::Success, pffdtd::boundary_map_status_string(status));
   require(map.front() == -7 && map.back() == -7, "output write crossed boundary-map extent");
   require(bn == boundary_before && bnl == lossy_before, "builder changed input order");
   for (std::size_t i = 0; i < bn.size(); ++i) {
      const std::vector<int64_t>::const_iterator found = std::find(bnl.begin(), bnl.end(), bn[i]);
      const MapIdx expected = found == bnl.end() ? static_cast<MapIdx>(-1)
                                                : static_cast<MapIdx>(found - bnl.begin());
      require(map[i + 1] == expected, "map lost original boundary or lossy ordinal");
   }
   ++cases;
}

template<typename MapIdx>
void check_error(const int64_t *bn, int64_t nb, const int64_t *bnl, int64_t nbl,
                 int64_t npts, BoundaryMapStatus expected, bool null_output = false)
{
   // Oversized-count checks use these few sentinel cells: the validation must
   // reject infeasible sizes before attempting to dereference the short arrays.
   std::vector<MapIdx> map(9, static_cast<MapIdx>(-7));
   const std::vector<MapIdx> before = map;
   const BoundaryMapStatus status = pffdtd::build_boundary_lossy_map(
      bn, nb, bnl, nbl, npts, null_output ? nullptr : map.data() + 1);
   require(status == expected, pffdtd::boundary_map_status_string(status));
   require(map == before, "failed builder changed output");
   ++cases;
}

template<typename MapIdx>
void check_small()
{
   check_valid<MapIdx>({}, {}, 0);
   check_valid<MapIdx>({}, {}, std::numeric_limits<int64_t>::max());
   check_valid<MapIdx>({0}, {}, 1);
   check_valid<MapIdx>({0}, {0}, 1);
   check_valid<MapIdx>({0, 2, 4, 6, 8}, {}, 10);
   check_valid<MapIdx>({0, 2, 4, 6, 8}, {0, 2, 4, 6, 8}, 10);
   check_valid<MapIdx>({0, 2, 4, 6, 8}, {2, 6}, 10);
   check_valid<MapIdx>({0, 2, 4, 6, 8}, {6, 2}, 10);
   check_valid<MapIdx>({6, 0, 8, 2, 4}, {2, 6}, 10);
   check_valid<MapIdx>({6, 0, 8, 2, 4}, {4, 8, 0, 6, 2}, 10);
   const int64_t maximum = std::numeric_limits<int64_t>::max();
   check_valid<MapIdx>({maximum - 1, 0, (int64_t(1) << 40), maximum - 17},
                       {maximum - 17, maximum - 1, (int64_t(1) << 40)}, maximum);

   const int64_t valid[] = {0, 2, 4, 6};
   const int64_t lossy[] = {2, 6};
   const int64_t boundary_dup_adjacent[] = {0, 2, 2, 6};
   const int64_t boundary_dup_unsorted[] = {6, 2, 0, 2};
   const int64_t lossy_dup_adjacent[] = {2, 2};
   const int64_t lossy_dup_unsorted[] = {6, 2, 6};
   const int64_t below[] = {-1, 2, 4, 6};
   const int64_t above[] = {0, 2, 4, 8};
   const int64_t huge[] = {0, 2, 4, maximum};
   const int64_t lossy_below[] = {-1, 2};
   const int64_t lossy_above[] = {2, 8};
   const int64_t missing_first[] = {1, 2};
   const int64_t missing_middle[] = {2, 3};
   const int64_t missing_last[] = {2, 7};
   const int64_t missing_unsorted[] = {7, 2};
   check_error<MapIdx>(valid, -1, lossy, 0, 8, BoundaryMapStatus::InvalidCounts);
   check_error<MapIdx>(valid, 4, lossy, -1, 8, BoundaryMapStatus::InvalidCounts);
   check_error<MapIdx>(valid, 4, lossy, 2, -1, BoundaryMapStatus::InvalidCounts);
   check_error<MapIdx>(valid, 1, lossy, 2, 8, BoundaryMapStatus::InvalidCounts);
   check_error<MapIdx>(nullptr, 4, lossy, 2, 8, BoundaryMapStatus::NullPointer);
   check_error<MapIdx>(valid, 4, nullptr, 2, 8, BoundaryMapStatus::NullPointer);
   check_error<MapIdx>(valid, 4, lossy, 2, 8, BoundaryMapStatus::NullPointer, true);
   check_error<MapIdx>(boundary_dup_adjacent, 4, lossy, 2, 8, BoundaryMapStatus::DuplicateBoundary);
   check_error<MapIdx>(boundary_dup_unsorted, 4, lossy, 2, 8, BoundaryMapStatus::DuplicateBoundary);
   check_error<MapIdx>(valid, 4, lossy_dup_adjacent, 2, 8, BoundaryMapStatus::DuplicateLossy);
   check_error<MapIdx>(valid, 4, lossy_dup_unsorted, 3, 8, BoundaryMapStatus::DuplicateLossy);
   check_error<MapIdx>(below, 4, lossy, 2, 8, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(above, 4, lossy, 2, 8, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(huge, 4, lossy, 2, maximum, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(valid, 4, lossy_below, 2, 8, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(valid, 4, lossy_above, 2, 8, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(valid, 4, nullptr, 0, 0, BoundaryMapStatus::IndexOutOfRange);
   check_error<MapIdx>(valid, 4, missing_first, 2, 8, BoundaryMapStatus::LossyNotBoundary);
   check_error<MapIdx>(valid, 4, missing_middle, 2, 8, BoundaryMapStatus::LossyNotBoundary);
   check_error<MapIdx>(valid, 4, missing_last, 2, 8, BoundaryMapStatus::LossyNotBoundary);
   check_error<MapIdx>(valid, 4, missing_unsorted, 2, 8, BoundaryMapStatus::LossyNotBoundary);
   check_error<MapIdx>(valid, maximum, nullptr, 0, maximum, BoundaryMapStatus::OutOfMemory);
   require(pffdtd::build_boundary_lossy_map<MapIdx>(nullptr, 0, nullptr, 0, 0, nullptr)
           == BoundaryMapStatus::Success, "empty null output was rejected");
   ++cases;
}

template<typename MapIdx>
void check_random()
{
   std::mt19937_64 random(0x8ba16d5f37ULL);
   for (int trial = 0; trial < 500; ++trial) {
      const int64_t count = static_cast<int64_t>(random() % 257);
      std::vector<int64_t> bn(static_cast<std::size_t>(count));
      for (int64_t i = 0; i < count; ++i) bn[static_cast<std::size_t>(i)] = 5 * i + 3;
      std::vector<int64_t> bnl;
      for (int64_t node : bn)
         if ((random() % 4) != 0) bnl.push_back(node);
      const int64_t npts = 5 * count + 4;
      check_valid<MapIdx>(bn, bnl, npts);
      std::shuffle(bnl.begin(), bnl.end(), random);
      check_valid<MapIdx>(bn, bnl, npts);
      std::sort(bnl.begin(), bnl.end());
      std::shuffle(bn.begin(), bn.end(), random);
      check_valid<MapIdx>(bn, bnl, npts);
      std::shuffle(bnl.begin(), bnl.end(), random);
      check_valid<MapIdx>(bn, bnl, npts);
   }
   // Exercise the sorted linear path and both unsorted inputs at useful sizes.
   std::vector<int64_t> bn(200000);
   std::vector<int64_t> bnl;
   for (std::size_t i = 0; i < bn.size(); ++i) {
      bn[i] = static_cast<int64_t>(4 * i + 1);
      if (i % 8192 == 0) bnl.push_back(bn[i]);
   }
   check_valid<MapIdx>(bn, bnl, 800001);
   std::shuffle(bn.begin(), bn.end(), random);
   std::shuffle(bnl.begin(), bnl.end(), random);
   check_valid<MapIdx>(bn, bnl, 800001);
}

void check_overflow()
{
   const int64_t node = 0;
   const int64_t too_many = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 2;
   check_error<int32_t>(&node, too_many, &node, too_many, too_many,
                        BoundaryMapStatus::MapIndexOverflow);
   // Count=max+1 needs an ordinal of max, which is representable. No actual
   // arrays of that size are needed to distinguish this from count overflow.
   const int64_t largest_count = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
   check_error<int32_t>(nullptr, largest_count, nullptr, largest_count, largest_count,
                        BoundaryMapStatus::NullPointer);
}

void check_aliasing()
{
   std::vector<int64_t> bn = {6, 0, 8, 2, 4};
   const std::vector<int64_t> bnl = {4, 8, 0, 6, 2};
   require(pffdtd::build_boundary_lossy_map(bn.data(), 5, bnl.data(), 5, 10, bn.data())
           == BoundaryMapStatus::Success, "map cannot alias boundary input");
   require(bn == std::vector<int64_t>({3, 2, 1, 4, 0}), "aliased map has wrong ordinals");
   ++cases;
   std::vector<int64_t> shared = {6, 0, 8, 2, 4};
   require(pffdtd::build_boundary_lossy_map(shared.data(), 5, shared.data(), 5, 10,
                                           shared.data()) == BoundaryMapStatus::Success,
           "all input/output arrays cannot alias");
   require(shared == std::vector<int64_t>({0, 1, 2, 3, 4}), "full alias changed ordinal identity");
   ++cases;
   bn = {6, 0, 8, 2, 4};
   const int64_t invalid[] = {8, 3};
   const std::vector<int64_t> before = bn;
   require(pffdtd::build_boundary_lossy_map(bn.data(), 5, invalid, 2, 10, bn.data())
           == BoundaryMapStatus::LossyNotBoundary, "aliased failing map returned wrong status");
   require(bn == before, "aliased failing map changed original input");
   ++cases;
}

template<typename MapIdx>
void check_allocation_failures()
{
   const int64_t sorted[] = {0, 2, 4, 6, 8};
   const int64_t unsorted[] = {6, 0, 8, 2, 4};
   const int64_t sorted_lossy[] = {2, 6, 8};
   const int64_t unsorted_lossy[] = {8, 2, 6};
   for (int boundary_unsorted = 0; boundary_unsorted <= 1; ++boundary_unsorted)
      for (int lossy_unsorted = 0; lossy_unsorted <= 1; ++lossy_unsorted) {
         const int allocations = 1 + boundary_unsorted + lossy_unsorted;
         for (int allowed = 0; allowed <= allocations; ++allowed) {
            MapIdx map[5] = {-7, -7, -7, -7, -7};
            allocations_before_failure = allowed;
            const BoundaryMapStatus status = pffdtd::build_boundary_lossy_map(
               boundary_unsorted ? unsorted : sorted, 5,
               lossy_unsorted ? unsorted_lossy : sorted_lossy, 3, 10, map);
            allocations_before_failure = -1;
            require(status == (allowed < allocations ? BoundaryMapStatus::OutOfMemory
                                                      : BoundaryMapStatus::Success),
                    "allocation failure was not handled at each temporary");
            if (status == BoundaryMapStatus::OutOfMemory)
               for (MapIdx value : map)
                  require(value == -7, "allocation failure changed output");
            ++cases;
         }
      }
}

void check_status_strings()
{
   const BoundaryMapStatus statuses[] = {
      BoundaryMapStatus::Success, BoundaryMapStatus::InvalidCounts, BoundaryMapStatus::NullPointer,
      BoundaryMapStatus::IndexOutOfRange, BoundaryMapStatus::DuplicateBoundary,
      BoundaryMapStatus::DuplicateLossy, BoundaryMapStatus::LossyNotBoundary,
      BoundaryMapStatus::MapIndexOverflow, BoundaryMapStatus::OutOfMemory
   };
   for (BoundaryMapStatus status : statuses)
      require(pffdtd::boundary_map_status_string(status)[0] != '\0', "missing status diagnostic");
}

} // namespace

int main()
{
   check_small<int32_t>();
   check_small<int64_t>();
   check_random<int32_t>();
   check_random<int64_t>();
   check_overflow();
   check_aliasing();
   check_allocation_failures<int32_t>();
   check_allocation_failures<int64_t>();
   check_status_strings();
   std::printf("PASS: %d boundary-map cases (int32/int64, sorted/unsorted, atomic errors)\n", cases);
   return EXIT_SUCCESS;
}
