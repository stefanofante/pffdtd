///////////////////////////////////////////////////////////////////////////////
// Native grid/comms preparation and the exact Cartesian/FCC solver layout.
// Physical coordinates stay unfolded; folded FCC dimensions describe storage.
///////////////////////////////////////////////////////////////////////////////
#ifndef PFFDTD_PREPARE_GRID_H
#define PFFDTD_PREPARE_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace pffdtd_prepare {

using GridDims = std::array<int64_t, 3>;
using GridPoint = std::array<double, 3>;
using GridCoord = std::array<int64_t, 3>;
using AxisPermutation = std::array<int, 3>;

inline int64_t grid_volume(const GridDims &dims)
{
   int64_t count = 1;
   for (int64_t axis : dims) {
      if (axis <= 0) throw std::invalid_argument("grid dimensions must be positive");
      if (axis > std::numeric_limits<int64_t>::max() / count)
         throw std::overflow_error("grid volume exceeds int64 indexing");
      count *= axis;
   }
   return count;
}

inline int64_t grid_index(const GridDims &dims, const GridCoord &coord)
{
   (void)grid_volume(dims);
   for (std::size_t j = 0; j < 3; ++j)
      if (coord[j] < 0 || coord[j] >= dims[j])
         throw std::out_of_range("grid coordinate outside dimensions");
   return (coord[0] * dims[1] + coord[1]) * dims[2] + coord[2];
}

inline GridCoord grid_coord(const GridDims &dims, int64_t index)
{
   const int64_t count = grid_volume(dims);
   if (index < 0 || index >= count)
      throw std::out_of_range("grid index outside dimensions");
   const int64_t yz = dims[1] * dims[2];
   return {{index / yz, (index / dims[2]) % dims[1], index % dims[2]}};
}

// Check parity without adding coordinates (which could overflow int64).
inline bool even_fcc_coord(const GridCoord &coord)
{
   return ((coord[0] ^ coord[1] ^ coord[2]) & 1) == 0;
}

struct Grid {
   std::array<std::vector<double>, 3> axes;
   double h = 0.0;
   int fcc_flag = 0; // 0 Cartesian; 1 physical, unfolded FCC.

   GridDims dims() const
   {
      GridDims result;
      for (std::size_t j = 0; j < 3; ++j) {
         if (axes[j].size() > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()))
            throw std::overflow_error("coordinate axis exceeds int64 indexing");
         result[j] = static_cast<int64_t>(axes[j].size());
      }
      (void)grid_volume(result);
      return result;
   }
};

inline void validate_grid_shape(const Grid &grid)
{
   if (!std::isfinite(grid.h) || grid.h <= 0.0)
      throw std::invalid_argument("grid spacing must be finite and positive");
   if (grid.fcc_flag != 0 && grid.fcc_flag != 1)
      throw std::invalid_argument("interpolation needs physical, unfolded grid coordinates");
   const GridDims dims = grid.dims();
   for (int64_t dim : dims) {
      if (dim < 3) throw std::invalid_argument("grid axis needs at least three coordinates");
      if (grid.fcc_flag == 1 && dim % 2 != 0)
         throw std::invalid_argument("unfolded FCC dimensions must all be even");
   }
}

// Call once for externally supplied axes; build_grid already makes valid axes.
inline void validate_grid(const Grid &grid)
{
   validate_grid_shape(grid);
   for (const auto &axis : grid.axes) {
      const double origin = axis.front();
      if (!std::isfinite(origin)) throw std::invalid_argument("nonfinite grid coordinate");
      for (std::size_t i = 0; i < axis.size(); ++i) {
         const double expected = origin + static_cast<double>(i) * grid.h;
         const double tolerance = 16.0 * std::numeric_limits<double>::epsilon()
                                  * std::max({1.0, std::abs(expected), grid.h});
         if (!std::isfinite(axis[i]) || std::abs(axis[i] - expected) > tolerance
             || (i > 0 && axis[i] <= axis[i - 1]))
            throw std::invalid_argument("coordinate axes must be finite, increasing and uniform");
      }
   }
}

// Offset follows CartGrid's three-layer halo contract and is measured in h.
inline Grid build_grid(const GridPoint &bmin, const GridPoint &bmax, double h,
                       bool fcc, double offset = 3.5)
{
   if (!std::isfinite(h) || h <= 0.0 || !std::isfinite(offset) || offset <= 2.0)
      throw std::invalid_argument("grid requires h > 0 and finite halo offset > 2");
   Grid grid;
   grid.h = h;
   grid.fcc_flag = fcc ? 1 : 0;
   GridDims dims;
   GridPoint origin;
   for (std::size_t j = 0; j < 3; ++j) {
      if (!std::isfinite(bmin[j]) || !std::isfinite(bmax[j]) || bmax[j] < bmin[j])
         throw std::invalid_argument("grid bounds must be finite and ordered");
      origin[j] = bmin[j] - offset * h;
      const double end = bmax[j] + offset * h;
      const double cells = std::ceil((end - origin[j]) / h);
      if (!std::isfinite(origin[j]) || !std::isfinite(end) || !std::isfinite(cells)
          || cells < 0.0 || cells >= static_cast<double>(std::numeric_limits<int64_t>::max()))
         throw std::overflow_error("grid coordinate extent exceeds int64 indexing");
      dims[j] = static_cast<int64_t>(cells) + 1;
      if (fcc && dims[j] % 2 != 0) {
         if (dims[j] == std::numeric_limits<int64_t>::max())
            throw std::overflow_error("FCC even dimension exceeds int64 indexing");
         ++dims[j];
      }
   }
   (void)grid_volume(dims); // Reject products before allocating coordinate axes.
   for (std::size_t j = 0; j < 3; ++j) {
      if (static_cast<uintmax_t>(dims[j]) > grid.axes[j].max_size())
         throw std::length_error("grid coordinate axis exceeds host allocation limits");
      grid.axes[j].resize(static_cast<std::size_t>(dims[j]));
      for (std::size_t i = 0; i < grid.axes[j].size(); ++i)
         grid.axes[j][i] = static_cast<double>(i) * h + origin[j];
   }
   validate_grid(grid);
   return grid;
}

struct Interp8 {
   std::array<int64_t, 8> indices;
   std::array<double, 8> weights;
};

inline bool grid_coord_on_abc(const GridDims &dims, const GridCoord &coord)
{
   (void)grid_index(dims, coord);
   for (std::size_t j = 0; j < 3; ++j)
      if (coord[j] <= 1 || coord[j] >= dims[j] - 2) return true;
   return false;
}

// Exact legacy corner order. FCC selects an even-parity cube of spacing 2h;
// when the high corner is odd, increment the first axis with minimal alpha.
// Check every stored corner, including zero-weight corners, against halo/ABC.
inline Interp8 interpolate8(const Grid &grid, const GridPoint &point,
                            bool reject_abc = true)
{
   validate_grid_shape(grid);
   const GridDims dims = grid.dims();
   GridCoord hi;
   GridPoint alpha;
   for (std::size_t j = 0; j < 3; ++j) {
      if (!std::isfinite(point[j])) throw std::invalid_argument("nonfinite communication point");
      const auto &axis = grid.axes[j];
      const auto found = std::lower_bound(axis.begin(), axis.end(), point[j]);
      if (found == axis.end()) throw std::out_of_range("communication point beyond grid bounds");
      hi[j] = static_cast<int64_t>(found - axis.begin());
      alpha[j] = (*found - point[j]) / grid.h;
   }
   const int64_t stride = grid.fcc_flag == 1 ? 2 : 1;
   if (grid.fcc_flag == 1 && !even_fcc_coord(hi)) {
      std::size_t selected = 0;
      for (std::size_t j = 1; j < 3; ++j)
         if (alpha[j] < alpha[selected]) selected = j;
      if (hi[selected] >= dims[selected] - 1)
         throw std::out_of_range("FCC interpolation parity adjustment exceeds grid bounds");
      ++hi[selected];
   }
   for (std::size_t j = 0; j < 3; ++j) {
      if (hi[j] < stride) throw std::out_of_range("interpolation stencil crosses grid bounds");
      alpha[j] = (grid.axes[j][static_cast<std::size_t>(hi[j])] - point[j])
                 / (static_cast<double>(stride) * grid.h);
      if (!std::isfinite(alpha[j]) || alpha[j] < 0.0 || alpha[j] > 1.0)
         throw std::invalid_argument("point is outside its uniform interpolation cell");
   }
   static const int corners[8][3] = {
      {0,0,0}, {-1,0,0}, {0,-1,0}, {0,0,-1},
      {-1,-1,0}, {-1,0,-1}, {0,-1,-1}, {-1,-1,-1}
   };
   Interp8 result;
   for (std::size_t i = 0; i < 8; ++i) {
      GridCoord coord;
      double weight = 1.0;
      for (std::size_t j = 0; j < 3; ++j) {
         coord[j] = hi[j] + corners[i][j] * stride;
         weight *= corners[i][j] == 0 ? 1.0 - alpha[j] : alpha[j];
      }
      if (reject_abc && grid_coord_on_abc(dims, coord))
         throw std::out_of_range("communication stencil touches halo or absorbing boundary");
      result.indices[i] = grid_index(dims, coord);
      result.weights[i] = weight;
   }
   return result;
}

inline AxisPermutation longest_axis_permutation(const GridDims &dims)
{
   (void)grid_volume(dims);
   AxisPermutation permutation = {{0, 1, 2}};
   // Match reverse argsort's descending-axis tie order for equal dimensions.
   std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
      return dims[a] != dims[b] ? dims[a] > dims[b] : a > b;
   });
   return permutation;
}

struct IndexTransform {
   GridDims full_dims;
   AxisPermutation permutation;
   bool fold_fcc;

   // Neutral staging value for PreparedScene before its physical grid is built.
   IndexTransform() : full_dims{{1, 1, 1}}, permutation{{0, 1, 2}}, fold_fcc(false) {}

   IndexTransform(const GridDims &dimensions,
                  const AxisPermutation &axes = {{0, 1, 2}}, bool fold = false)
      : full_dims(dimensions), permutation(axes), fold_fcc(fold)
   {
      (void)grid_volume(full_dims);
      AxisPermutation sorted = permutation;
      std::sort(sorted.begin(), sorted.end());
      if (sorted != AxisPermutation{{0, 1, 2}})
         throw std::invalid_argument("axis transform must be a permutation of 0,1,2");
      if (fold_fcc)
         for (int64_t dim : full_dims)
            if (dim % 2 != 0)
               throw std::invalid_argument("FCC folding requires even full dimensions");
   }

   GridDims permuted_dims() const
   {
      return {{full_dims[permutation[0]], full_dims[permutation[1]], full_dims[permutation[2]]}};
   }

   GridDims output_dims() const
   {
      GridDims result = permuted_dims();
      if (fold_fcc) result[1] = result[1] / 2 + 1; // Last row is the fold ghost.
      (void)grid_volume(result);
      return result;
   }

   GridCoord permuted_coord(int64_t index) const
   {
      const GridCoord old = grid_coord(full_dims, index);
      return {{old[permutation[0]], old[permutation[1]], old[permutation[2]]}};
   }

   int64_t map_index(int64_t index) const
   {
      GridCoord coord = permuted_coord(index);
      if (fold_fcc) {
         if (!even_fcc_coord(coord))
            throw std::invalid_argument("only even-parity FCC sites can be folded");
         const int64_t full_y = permuted_dims()[1];
         if (coord[1] >= full_y / 2) coord[1] = full_y - coord[1] - 1;
      }
      return grid_index(output_dims(), coord);
   }

   int64_t unmap_index(int64_t index) const
   {
      GridCoord coord = grid_coord(output_dims(), index);
      if (fold_fcc) {
         const int64_t full_y = permuted_dims()[1];
         if (coord[1] == full_y / 2)
            throw std::invalid_argument("FCC fold ghost row has no physical grid site");
         if (!even_fcc_coord(coord)) coord[1] = full_y - coord[1] - 1;
      }
      GridCoord original;
      for (std::size_t j = 0; j < 3; ++j) original[permutation[j]] = coord[j];
      return grid_index(full_dims, original);
   }
};

// Exact neighbor ordering used by voxelizer/CPU/CUDA boundary kernels.
inline const int (*grid_directions(int count))[3]
{
   static const int cart[6][3] = {
      {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}
   };
   static const int fcc[12][3] = {
      {1,1,0}, {-1,-1,0}, {0,1,1}, {0,-1,-1}, {1,0,1}, {-1,0,-1},
      {1,-1,0}, {-1,1,0}, {0,1,-1}, {0,-1,1}, {1,0,-1}, {-1,0,1}
   };
   if (count == 6) return cart;
   if (count == 12) return fcc;
   throw std::invalid_argument("boundary stencil must have 6 or 12 directions");
}

inline uint16_t transform_adjacency(const IndexTransform &transform,
                                     int64_t original_index, uint16_t adjacency,
                                     int directions)
{
   const auto stencil = grid_directions(directions);
   if (transform.fold_fcc && directions != 12)
      throw std::invalid_argument("FCC fold requires twelve boundary directions");
   if (adjacency >= (uint16_t(1) << directions))
      throw std::invalid_argument("adjacency mask contains bits outside stencil");
   const GridCoord coord = transform.permuted_coord(original_index);
   // Validate physical parity and the complete transform even for mask zero.
   (void)transform.map_index(original_index);
   const bool reflect = transform.fold_fcc && coord[1] >= transform.permuted_dims()[1] / 2;
   uint16_t result = 0;
   for (int old = 0; old < directions; ++old) {
      if ((adjacency & (uint16_t(1) << old)) == 0) continue;
      int delta[3];
      for (int j = 0; j < 3; ++j) delta[j] = stencil[old][transform.permutation[j]];
      if (reflect) delta[1] = -delta[1];
      for (int mapped = 0; mapped < directions; ++mapped) {
         if (delta[0] == stencil[mapped][0] && delta[1] == stencil[mapped][1]
             && delta[2] == stencil[mapped][2]) {
            result |= uint16_t(1) << mapped;
            break;
         }
      }
   }
   return result;
}

inline std::vector<int64_t> transform_indices(const std::vector<int64_t> &indices,
                                               const IndexTransform &transform)
{
   std::vector<int64_t> result;
   result.reserve(indices.size());
   for (int64_t index : indices) result.push_back(transform.map_index(index));
   return result; // Each entry retained, including repeated source/receiver sites.
}

struct BoundaryTransform {
   std::vector<int64_t> indices;
   std::vector<uint16_t> adjacency;
};

inline BoundaryTransform transform_boundary(const std::vector<int64_t> &indices,
                                             const std::vector<uint16_t> &adjacency,
                                             const IndexTransform &transform,
                                             int directions)
{
   if (indices.size() != adjacency.size())
      throw std::invalid_argument("boundary indices and masks must have equal rows");
   BoundaryTransform result;
   result.indices.reserve(indices.size());
   result.adjacency.reserve(indices.size());
   for (std::size_t i = 0; i < indices.size(); ++i) {
      result.indices.push_back(transform.map_index(indices[i]));
      result.adjacency.push_back(transform_adjacency(transform, indices[i], adjacency[i], directions));
   }
   return result;
}

struct IndexSort {
   std::vector<int64_t> indices;
   std::vector<int64_t> order;   // sorted row -> original row, for mat/saf/signals.
   std::vector<int64_t> reorder; // original receiver corner -> sorted readout row.
};

inline IndexSort stable_index_order(const std::vector<int64_t> &indices)
{
   if (indices.size() > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("index row count exceeds int64 ordering");
   IndexSort result;
   result.order.resize(indices.size());
   std::iota(result.order.begin(), result.order.end(), int64_t(0));
   std::stable_sort(result.order.begin(), result.order.end(), [&](int64_t a, int64_t b) {
      return indices[static_cast<std::size_t>(a)] < indices[static_cast<std::size_t>(b)];
   });
   result.indices.resize(indices.size());
   result.reorder.resize(indices.size());
   for (std::size_t i = 0; i < indices.size(); ++i) {
      const std::size_t original = static_cast<std::size_t>(result.order[i]);
      result.indices[i] = indices[original];
      result.reorder[original] = static_cast<int64_t>(i);
   }
   return result;
}

template<class T>
inline std::vector<T> reorder_rows(const std::vector<T> &rows,
                                  const std::vector<int64_t> &order,
                                  std::size_t width = 1)
{
   if (width == 0 || order.size() > std::numeric_limits<std::size_t>::max() / width
       || rows.size() != order.size() * width)
      throw std::invalid_argument("row permutation has incompatible dimensions");
   std::vector<T> result(rows.size());
   std::vector<uint8_t> seen(order.size(), 0);
   for (std::size_t row = 0; row < order.size(); ++row) {
      const int64_t old = order[row];
      if (old < 0 || static_cast<uintmax_t>(old) >= order.size()
          || seen[static_cast<std::size_t>(old)]++)
         throw std::invalid_argument("row order must be a complete permutation");
      for (std::size_t j = 0; j < width; ++j)
         result[row * width + j] = rows[static_cast<std::size_t>(old) * width + j];
   }
   return result;
}

// If receivers were already reordered, compose instead of replacing the map.
inline std::vector<int64_t> compose_output_reorder(const std::vector<int64_t> &previous,
                                                   const IndexSort &sorted)
{
   if (previous.size() != sorted.reorder.size())
      throw std::invalid_argument("output reorder has incompatible receiver count");
   std::vector<int64_t> result(previous.size());
   std::vector<uint8_t> seen(previous.size(), 0);
   for (std::size_t i = 0; i < previous.size(); ++i) {
      const int64_t old = previous[i];
      if (old < 0 || static_cast<uintmax_t>(old) >= previous.size()
          || seen[static_cast<std::size_t>(old)]++)
         throw std::invalid_argument("output reorder must be a complete permutation");
      result[i] = sorted.reorder[static_cast<std::size_t>(old)];
   }
   return result;
}

} // namespace pffdtd_prepare
#endif
