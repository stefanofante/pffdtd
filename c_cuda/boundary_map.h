///////////////////////////////////////////////////////////////////////////////
// Host-side association between boundary nodes and lossy-boundary state.
///////////////////////////////////////////////////////////////////////////////

#ifndef PFFDTD_BOUNDARY_MAP_H
#define PFFDTD_BOUNDARY_MAP_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pffdtd {

enum class BoundaryMapStatus {
   Success,
   InvalidCounts,
   NullPointer,
   IndexOutOfRange,
   DuplicateBoundary,
   DuplicateLossy,
   LossyNotBoundary,
   MapIndexOverflow,
   OutOfMemory
};

inline const char *boundary_map_status_string(BoundaryMapStatus status)
{
   switch (status) {
      case BoundaryMapStatus::Success: return "success";
      case BoundaryMapStatus::InvalidCounts: return "invalid boundary counts";
      case BoundaryMapStatus::NullPointer: return "null boundary array or map";
      case BoundaryMapStatus::IndexOutOfRange: return "boundary index outside grid";
      case BoundaryMapStatus::DuplicateBoundary: return "duplicate boundary index";
      case BoundaryMapStatus::DuplicateLossy: return "duplicate lossy-boundary index";
      case BoundaryMapStatus::LossyNotBoundary: return "lossy node absent from boundary";
      case BoundaryMapStatus::MapIndexOverflow: return "lossy ordinal exceeds map index range";
      case BoundaryMapStatus::OutOfMemory: return "boundary-map allocation failed";
   }
   return "unknown boundary-map status";
}

namespace boundary_map_detail {

typedef std::pair<int64_t, int64_t> IndexedNode;

// An empty ordered vector means that the original array was already sorted.
// Only unsorted inputs need additional node/ordinal pairs.
inline BoundaryMapStatus validate_nodes(const int64_t *nodes, int64_t count,
                                       int64_t npts, bool lossy,
                                       std::vector<IndexedNode> &ordered)
{
   bool sorted = true;
   bool adjacent_duplicate = false;
   for (int64_t i = 0; i < count; ++i) {
      if (nodes[i] < 0 || nodes[i] >= npts)
         return BoundaryMapStatus::IndexOutOfRange;
      if (i != 0) {
         if (nodes[i] < nodes[i - 1]) sorted = false;
         if (nodes[i] == nodes[i - 1]) adjacent_duplicate = true;
      }
   }
   const BoundaryMapStatus duplicate = lossy ? BoundaryMapStatus::DuplicateLossy
                                             : BoundaryMapStatus::DuplicateBoundary;
   if (adjacent_duplicate) return duplicate;
   if (sorted) return BoundaryMapStatus::Success;

   if (static_cast<uint64_t>(count) > static_cast<uint64_t>(ordered.max_size()))
      return BoundaryMapStatus::OutOfMemory;
   ordered.resize(static_cast<std::size_t>(count));
   for (int64_t i = 0; i < count; ++i)
      ordered[static_cast<std::size_t>(i)] = IndexedNode(nodes[i], i);
   std::sort(ordered.begin(), ordered.end());
   for (std::size_t i = 1; i < ordered.size(); ++i)
      if (ordered[i].first == ordered[i - 1].first) return duplicate;
   return BoundaryMapStatus::Success;
}

inline IndexedNode ordered_node(const int64_t *nodes,
                                const std::vector<IndexedNode> &ordered,
                                int64_t ordinal)
{
   return ordered.empty() ? IndexedNode(nodes[ordinal], ordinal)
                          : ordered[static_cast<std::size_t>(ordinal)];
}

} // namespace boundary_map_detail

// map[i] is -1 for a rigid boundary node, or the original ordinal of bn[i]
// in bnl. Both input orders are preserved and duplicates are rejected. Indices
// are grid indices in [0, npts), independent of the selected map integer width.
// Zero counts allow null arrays; otherwise each array and map must be present.
// The caller provides nb output elements. On any failure map stays unchanged,
// including when MapIdx is int64_t and the output aliases an input array.
// Sorted inputs take O(nb+nbl); unsorted inputs are sorted as node/ordinal pairs.
template<typename MapIdx>
BoundaryMapStatus build_boundary_lossy_map(const int64_t *bn, int64_t nb,
                                           const int64_t *bnl, int64_t nbl,
                                           int64_t npts, MapIdx *map)
{
   static_assert(std::numeric_limits<MapIdx>::is_integer
                 && std::numeric_limits<MapIdx>::is_signed
                 && (std::numeric_limits<MapIdx>::digits == 31
                     || std::numeric_limits<MapIdx>::digits == 63),
                 "boundary map requires signed 32-bit or 64-bit indices");
   if (nb < 0 || nbl < 0 || npts < 0 || nbl > nb)
      return BoundaryMapStatus::InvalidCounts;
   // The stored value is an ordinal: nbl=max(MapIdx)+1 still fits.
   if (nbl > 0 && nbl - 1 > static_cast<int64_t>(std::numeric_limits<MapIdx>::max()))
      return BoundaryMapStatus::MapIndexOverflow;
   if ((nb > 0 && (bn == nullptr || map == nullptr))
       || (nbl > 0 && bnl == nullptr))
      return BoundaryMapStatus::NullPointer;
   if (nb == 0) return BoundaryMapStatus::Success;

   try {
      std::vector<MapIdx> staged;
      if (static_cast<uint64_t>(nb) > static_cast<uint64_t>(staged.max_size()))
         return BoundaryMapStatus::OutOfMemory;
      std::vector<boundary_map_detail::IndexedNode> boundary_order;
      std::vector<boundary_map_detail::IndexedNode> lossy_order;
      BoundaryMapStatus status = boundary_map_detail::validate_nodes(
         bn, nb, npts, false, boundary_order);
      if (status != BoundaryMapStatus::Success) return status;
      status = boundary_map_detail::validate_nodes(bnl, nbl, npts, true, lossy_order);
      if (status != BoundaryMapStatus::Success) return status;

      staged.assign(static_cast<std::size_t>(nb), static_cast<MapIdx>(-1));
      int64_t boundary_position = 0;
      for (int64_t lossy_position = 0; lossy_position < nbl; ++lossy_position) {
         const boundary_map_detail::IndexedNode lossy =
            boundary_map_detail::ordered_node(bnl, lossy_order, lossy_position);
         while (boundary_position < nb
                && boundary_map_detail::ordered_node(bn, boundary_order,
                                                      boundary_position).first < lossy.first)
            ++boundary_position;
         if (boundary_position == nb) return BoundaryMapStatus::LossyNotBoundary;
         const boundary_map_detail::IndexedNode boundary =
            boundary_map_detail::ordered_node(bn, boundary_order, boundary_position);
         if (boundary.first != lossy.first) return BoundaryMapStatus::LossyNotBoundary;
         staged[static_cast<std::size_t>(boundary.second)] = static_cast<MapIdx>(lossy.second);
         ++boundary_position;
      }
      std::copy(staged.begin(), staged.end(), map);
      return BoundaryMapStatus::Success;
   }
   catch (const std::bad_alloc &) {
      return BoundaryMapStatus::OutOfMemory;
   }
   catch (const std::length_error &) {
      return BoundaryMapStatus::OutOfMemory;
   }
}

} // namespace pffdtd

#endif
