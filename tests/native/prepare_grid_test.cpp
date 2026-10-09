#include "prepare_grid.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

using namespace pffdtd_prepare;

static std::size_t checks = 0;

static void check(bool condition)
{
   ++checks;
   if (!condition) throw std::runtime_error("prepare_grid assertion failed");
}

template<class Exception, class Function>
static void rejects(Function operation)
{
   bool caught = false;
   try { operation(); }
   catch (const Exception &) { caught = true; }
   check(caught);
}

static bool close(double a, double b, double tolerance = 2.0e-12)
{
   return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

static Grid simple_grid(bool fcc, const GridDims &dims = {{16, 18, 20}})
{
   Grid grid;
   grid.h = 1.0;
   grid.fcc_flag = fcc ? 1 : 0;
   for (std::size_t j = 0; j < 3; ++j)
      for (int64_t i = 0; i < dims[j]; ++i) grid.axes[j].push_back(static_cast<double>(i));
   validate_grid(grid);
   return grid;
}

static std::vector<AxisPermutation> permutations()
{
   std::vector<AxisPermutation> result;
   AxisPermutation permutation = {{0, 1, 2}};
   do { result.push_back(permutation); }
   while (std::next_permutation(permutation.begin(), permutation.end()));
   return result;
}

static void check_interpolation(const Grid &grid, const GridPoint &point)
{
   const Interp8 interp = interpolate8(grid, point);
   double sum = 0.0, plane = 0.0;
   const GridDims dims = grid.dims();
   std::set<int64_t> sites;
   for (std::size_t i = 0; i < 8; ++i) {
      check(interp.weights[i] >= 0.0 && interp.weights[i] <= 1.0);
      const GridCoord coord = grid_coord(dims, interp.indices[i]);
      check(!grid_coord_on_abc(dims, coord));
      if (grid.fcc_flag == 1) check(even_fcc_coord(coord));
      sites.insert(interp.indices[i]);
      sum += interp.weights[i];
      const double value = 3.0 + 0.75 * grid.axes[0][static_cast<std::size_t>(coord[0])]
                               - 2.0 * grid.axes[1][static_cast<std::size_t>(coord[1])]
                               + 1.25 * grid.axes[2][static_cast<std::size_t>(coord[2])];
      plane += interp.weights[i] * value;
   }
   check(sites.size() == 8);
   check(close(sum, 1.0));
   check(close(plane, 3.0 + 0.75 * point[0] - 2.0 * point[1] + 1.25 * point[2]));
}

static void test_grid_build()
{
   const GridPoint low = {{-1.0, 2.0, 4.0}}, high = {{1.0, 5.0, 4.0}};
   const Grid cart = build_grid(low, high, 0.5, false);
   check(cart.dims() == GridDims{{12, 14, 8}});
   check(cart.axes[0].front() == -2.75 && cart.axes[0].back() == 2.75);
   const Grid fcc = build_grid(low, high, 0.4, true);
   for (std::size_t j = 0; j < 3; ++j) {
      check(fcc.dims()[j] % 2 == 0);
      check(fcc.axes[j].front() == low[j] - 3.5 * fcc.h);
      check(fcc.axes[j].back() >= high[j] + 3.5 * fcc.h);
   }
   rejects<std::invalid_argument>([&] { build_grid(low, high, 0.0, false); });
   rejects<std::invalid_argument>([&] { build_grid(low, high, 0.5, false, 2.0); });
   rejects<std::invalid_argument>([&] { build_grid(high, low, 0.5, false); });
   rejects<std::invalid_argument>([&] {
      build_grid(low, high, std::numeric_limits<double>::quiet_NaN(), false);
   });
   rejects<std::overflow_error>([&] { build_grid(low, high, 1.0e-200, false); });
   rejects<std::overflow_error>([] {
      build_grid({{0, 0, 0}}, {{1.0e9, 1.0e9, 1.0e9}}, 1.0, false);
   });
   rejects<std::invalid_argument>([] {
      build_grid({{1.0e20, 1.0e20, 1.0e20}}, {{1.0e20, 1.0e20, 1.0e20}}, 1.0, false);
   });
   Grid malformed = simple_grid(false);
   malformed.axes[0][4] += 0.5;
   rejects<std::invalid_argument>([&] { validate_grid(malformed); });
   malformed = simple_grid(false);
   malformed.axes[1][8] = malformed.axes[1][7];
   rejects<std::invalid_argument>([&] { validate_grid(malformed); });
}

static void test_legacy_interpolation()
{
   const Grid cart = simple_grid(false), fcc = simple_grid(true);
   const GridPoint point = {{5.75, 6.75, 5.75}};
   const Interp8 cart_interp = interpolate8(cart, point);
   const Interp8 fcc_interp = interpolate8(fcc, point);
   const GridCoord cart_high = {{6, 7, 6}}, fcc_high = {{7, 7, 6}};
   static const int corners[8][3] = {
      {0,0,0}, {-1,0,0}, {0,-1,0}, {0,0,-1},
      {-1,-1,0}, {-1,0,-1}, {0,-1,-1}, {-1,-1,-1}
   };
   const double fcc_alpha[3] = {0.625, 0.125, 0.125};
   for (std::size_t i = 0; i < 8; ++i) {
      GridCoord cart_coord, fcc_coord;
      double cart_weight = 1.0, fcc_weight = 1.0;
      for (std::size_t j = 0; j < 3; ++j) {
         cart_coord[j] = cart_high[j] + corners[i][j];
         fcc_coord[j] = fcc_high[j] + 2 * corners[i][j];
         cart_weight *= corners[i][j] == 0 ? 0.75 : 0.25;
         fcc_weight *= corners[i][j] == 0 ? 1.0 - fcc_alpha[j] : fcc_alpha[j];
      }
      check(cart_interp.indices[i] == grid_index(cart.dims(), cart_coord));
      check(fcc_interp.indices[i] == grid_index(fcc.dims(), fcc_coord));
      check(cart_interp.weights[i] == cart_weight);
      check(fcc_interp.weights[i] == fcc_weight);
   }
   // Equal-alpha FCC tie chooses x, including exact-coordinate alpha == 0.
   const Interp8 tied = interpolate8(fcc, {{5.0, 5.0, 5.0}});
   check(grid_coord(fcc.dims(), tied.indices[0]) == GridCoord{{6, 5, 5}});
   check(tied.weights[0] == 0.5 && tied.weights[1] == 0.5);
   const Interp8 choose_y = interpolate8(fcc, {{5.3, 6.9, 5.5}});
   check(grid_coord(fcc.dims(), choose_y.indices[0]) == GridCoord{{6, 8, 6}});
   const Interp8 choose_z = interpolate8(fcc, {{5.3, 6.4, 5.9}});
   check(grid_coord(fcc.dims(), choose_z.indices[0]) == GridCoord{{6, 7, 7}});
   uint64_t random = 0x4a61736f6eULL;
   for (int case_index = 0; case_index < 700; ++case_index) {
      GridPoint sample;
      for (std::size_t j = 0; j < 3; ++j) {
         random = random * 6364136223846793005ULL + 1442695040888963407ULL;
         sample[j] = 4.0 + static_cast<double>((random >> 16) % 700000) / 100000.0;
      }
      check_interpolation(cart, sample);
      check_interpolation(fcc, sample);
   }
   check_interpolation(cart, {{5.0, 6.0, 7.0}});
   check_interpolation(fcc, {{5.0, 6.0, 7.0}});
   const Grid fractional = build_grid({{-2, -1, -3}}, {{3, 4, 2}}, 0.125, true);
   check_interpolation(fractional, {{0.04, 1.23, -0.17}});
}

static void test_abc_and_bounds()
{
   const Grid cart = simple_grid(false), fcc = simple_grid(true);
   rejects<std::out_of_range>([&] { interpolate8(cart, {{1.9, 5.2, 5.4}}); });
   rejects<std::out_of_range>([&] { interpolate8(cart, {{2.0, 5.0, 5.0}}); });
   // Stored corners with zero weights must still avoid the ABC shell.
   const Interp8 zero_corner = interpolate8(cart, {{2.0, 5.0, 5.0}}, false);
   check(zero_corner.weights[1] == 0.0);
   check(grid_coord(cart.dims(), zero_corner.indices[1])[0] == 1);
   rejects<std::out_of_range>([&] { interpolate8(cart, {{14.0, 5.0, 5.0}}); });
   rejects<std::out_of_range>([&] { interpolate8(fcc, {{2.9, 5.9, 6.9}}); });
   rejects<std::out_of_range>([&] { interpolate8(fcc, {{14.9, 14.9, 14.9}}, false); });
   rejects<std::out_of_range>([&] { interpolate8(cart, {{-0.1, 6, 6}}, false); });
   rejects<std::out_of_range>([&] { interpolate8(cart, {{100, 6, 6}}, false); });
   rejects<std::invalid_argument>([&] {
      interpolate8(cart, {{std::numeric_limits<double>::infinity(), 6, 6}});
   });
   Grid folded = fcc;
   folded.fcc_flag = 2;
   rejects<std::invalid_argument>([&] { interpolate8(folded, {{6, 6, 6}}); });
   Grid odd = fcc;
   odd.axes[0].pop_back();
   rejects<std::invalid_argument>([&] { interpolate8(odd, {{6, 6, 6}}); });
   check_interpolation(cart, {{2.1, 5.2, 5.4}});
   check_interpolation(fcc, {{4.1, 5.2, 5.4}});
}

static void test_cartesian_permutations()
{
   const Grid grid = simple_grid(false, {{6, 8, 10}});
   const GridDims dims = grid.dims();
   for (const auto &permutation : permutations()) {
      const IndexTransform transform(dims, permutation);
      std::set<int64_t> mapped;
      for (int64_t index = 0; index < grid_volume(dims); ++index) {
         const int64_t transformed = transform.map_index(index);
         mapped.insert(transformed);
         check(transform.unmap_index(transformed) == index);
         const GridCoord old = grid_coord(dims, index);
         check(grid_coord(transform.output_dims(), transformed)
               == GridCoord{{old[permutation[0]], old[permutation[1]], old[permutation[2]]}});
      }
      check(mapped.size() == static_cast<std::size_t>(grid_volume(dims)));
      for (int direction = 0; direction < 6; ++direction) {
         const int64_t node = grid_index(dims, {{2, 3, 4}});
         const uint16_t mask = transform_adjacency(transform, node, uint16_t(1) << direction, 6);
         check(mask != 0 && (mask & (mask - 1)) == 0);
         const auto stencil = grid_directions(6);
         GridCoord neighbor = {{2, 3, 4}};
         for (std::size_t j = 0; j < 3; ++j) neighbor[j] += stencil[direction][j];
         int target = 0;
         while ((mask & (uint16_t(1) << target)) == 0) ++target;
         const GridCoord center_new = grid_coord(transform.output_dims(), transform.map_index(node));
         GridCoord neighbor_new = center_new;
         for (std::size_t j = 0; j < 3; ++j) neighbor_new[j] += stencil[target][j];
         check(transform.map_index(grid_index(dims, neighbor))
               == grid_index(transform.output_dims(), neighbor_new));
      }
   }
   check(longest_axis_permutation({{6, 12, 8}}) == AxisPermutation{{1, 2, 0}});
   check(longest_axis_permutation({{8, 8, 8}}) == AxisPermutation{{2, 1, 0}});
}

static void test_fcc_fold()
{
   const GridDims dims = {{6, 8, 10}};
   static const int reflected[12] = {6, 7, 9, 8, 4, 5, 0, 1, 3, 2, 10, 11};
   const IndexTransform identity(dims, {{0, 1, 2}}, true);
   const int64_t lower = grid_index(dims, {{2, 2, 2}});
   const int64_t upper = grid_index(dims, {{2, 6, 2}});
   check(identity.output_dims() == GridDims{{6, 5, 10}});
   check(identity.map_index(lower) == grid_index(identity.output_dims(), {{2, 2, 2}}));
   check(identity.map_index(upper) == grid_index(identity.output_dims(), {{2, 1, 2}}));
   for (int direction = 0; direction < 12; ++direction) {
      check(transform_adjacency(identity, lower, uint16_t(1) << direction, 12)
            == (uint16_t(1) << direction));
      check(transform_adjacency(identity, upper, uint16_t(1) << direction, 12)
            == (uint16_t(1) << reflected[direction]));
   }
   for (const auto &permutation : permutations()) {
      const IndexTransform transform(dims, permutation, true);
      const GridDims compact = transform.output_dims();
      std::set<int64_t> mapped;
      for (int64_t index = 0; index < grid_volume(dims); ++index) {
         const GridCoord coord = grid_coord(dims, index);
         if (!even_fcc_coord(coord)) continue;
         const int64_t storage = transform.map_index(index);
         check(mapped.insert(storage).second);
         check(transform.unmap_index(storage) == index);
         check(grid_coord(compact, storage)[1] < compact[1] - 1);
         bool interior = true;
         for (std::size_t j = 0; j < 3; ++j)
            interior = interior && coord[j] > 0 && coord[j] < dims[j] - 1;
         if (!interior) continue;
         // Verify the physical FCC stencil after permutation/reflection,
         // including the seam ghost row copied from the final physical row.
         const auto stencil = grid_directions(12);
         for (int direction = 0; direction < 12; ++direction) {
            GridCoord physical_neighbor = coord;
            for (std::size_t j = 0; j < 3; ++j)
               physical_neighbor[j] += stencil[direction][j];
            const uint16_t mask = transform_adjacency(transform, index,
                                                       uint16_t(1) << direction, 12);
            check(mask != 0 && (mask & (mask - 1)) == 0);
            int target = 0;
            while ((mask & (uint16_t(1) << target)) == 0) ++target;
            GridCoord storage_neighbor = grid_coord(compact, storage);
            for (std::size_t j = 0; j < 3; ++j)
               storage_neighbor[j] += stencil[target][j];
            if (storage_neighbor[1] == compact[1] - 1) --storage_neighbor[1];
            check(grid_index(compact, storage_neighbor)
                  == transform.map_index(grid_index(dims, physical_neighbor)));
         }
      }
      check(mapped.size() == static_cast<std::size_t>(grid_volume(dims) / 2));
      for (int64_t i = 0; i < grid_volume(compact); ++i) {
         const GridCoord coord = grid_coord(compact, i);
         if (coord[1] == compact[1] - 1)
            rejects<std::invalid_argument>([&] { transform.unmap_index(i); });
         else check(mapped.count(i) == 1);
      }
   }
}

static void test_interpolation_permutations()
{
   for (bool fcc : {false, true}) {
      const Grid grid = simple_grid(fcc);
      const GridPoint point = {{5.31, 6.47, 7.89}};
      const Interp8 original = interpolate8(grid, point);
      for (const auto &permutation : permutations()) {
         Grid rotated;
         rotated.h = grid.h;
         rotated.fcc_flag = grid.fcc_flag;
         GridPoint rotated_point;
         for (std::size_t j = 0; j < 3; ++j) {
            rotated.axes[j] = grid.axes[permutation[j]];
            rotated_point[j] = point[permutation[j]];
         }
         const IndexTransform transform(grid.dims(), permutation);
         const Interp8 result = interpolate8(rotated, rotated_point);
         for (std::size_t i = 0; i < 8; ++i) {
            bool found = false;
            for (std::size_t j = 0; j < 8; ++j)
               if (transform.map_index(original.indices[i]) == result.indices[j]) {
                  check(close(original.weights[i], result.weights[j]));
                  found = true;
               }
            check(found);
         }
      }
   }
}

static void test_rows_and_duplicates()
{
   const std::vector<int64_t> sites = {17, 4, 17, 9, 4, 4, 22};
   const IndexSort sorted = stable_index_order(sites);
   check(sorted.indices == std::vector<int64_t>({4, 4, 4, 9, 17, 17, 22}));
   check(sorted.order == std::vector<int64_t>({1, 4, 5, 3, 0, 2, 6}));
   check(sorted.reorder == std::vector<int64_t>({4, 0, 5, 3, 1, 2, 6}));
   const std::vector<double> signals = {0, 0.1, 1, 1.1, 2, 2.1, 3, 3.1,
                                        4, 4.1, 5, 5.1, 6, 6.1};
   const auto ordered_signals = reorder_rows(signals, sorted.order, 2);
   for (std::size_t original = 0; original < sites.size(); ++original) {
      const std::size_t row = static_cast<std::size_t>(sorted.reorder[original]);
      check(sorted.indices[row] == sites[original]);
      check(ordered_signals[row * 2] == signals[original * 2]);
      check(ordered_signals[row * 2 + 1] == signals[original * 2 + 1]);
   }
   const std::vector<int64_t> existing = {6, 2, 3, 4, 0, 5, 1};
   const auto composed = compose_output_reorder(existing, sorted);
   for (std::size_t i = 0; i < existing.size(); ++i)
      check(composed[i] == sorted.reorder[static_cast<std::size_t>(existing[i])]);
   const GridDims dims = {{6, 8, 10}};
   const IndexTransform transform(dims, {{2, 0, 1}}, true);
   const int64_t a = grid_index(dims, {{2, 2, 2}});
   const int64_t b = grid_index(dims, {{2, 6, 2}});
   const std::vector<int64_t> duplicates = {b, a, b, a, a};
   const auto output = transform_indices(duplicates, transform);
   check(output.size() == duplicates.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      check(transform.unmap_index(output[i]) == duplicates[i]);
   const std::vector<uint16_t> masks = {1, 2, 4, 8, 16};
   const auto boundary = transform_boundary(duplicates, masks, transform, 12);
   check(boundary.indices == output && boundary.adjacency.size() == masks.size());
   for (std::size_t i = 0; i < masks.size(); ++i)
      check(boundary.adjacency[i] == transform_adjacency(transform, duplicates[i], masks[i], 12));
   const auto boundary_sort = stable_index_order(boundary.indices);
   const auto sorted_masks = reorder_rows(boundary.adjacency, boundary_sort.order);
   for (std::size_t i = 0; i < masks.size(); ++i)
      check(sorted_masks[static_cast<std::size_t>(boundary_sort.reorder[i])] == boundary.adjacency[i]);
   check(stable_index_order({}).indices.empty());
   check(reorder_rows(std::vector<double>{}, std::vector<int64_t>{}, 8).empty());
   rejects<std::invalid_argument>([&] { reorder_rows(signals, sorted.order, 3); });
   rejects<std::invalid_argument>([] { reorder_rows(std::vector<int>{1, 2}, {0, 0}); });
   rejects<std::invalid_argument>([] { reorder_rows(std::vector<int>{1, 2}, {0, -1}); });
   rejects<std::invalid_argument>([] { reorder_rows(std::vector<int>{1}, {0}, 0); });
   rejects<std::invalid_argument>([] {
      reorder_rows(std::vector<int>{1, 2}, {0, 1}, std::numeric_limits<std::size_t>::max());
   });
   rejects<std::invalid_argument>([&] { compose_output_reorder({0, 0, 1, 2, 3, 4, 5}, sorted); });
   rejects<std::invalid_argument>([&] { transform_boundary(duplicates, {1}, transform, 12); });
}

static void test_checked_indices()
{
   const GridDims wide = {{65536, 65536, 8}};
   const GridCoord coord = {{65535, 65535, 7}};
   check(grid_index(wide, coord) == int64_t(34359738367));
   check(grid_coord(wide, grid_index(wide, coord)) == coord);
   const IndexTransform transform(wide, {{2, 0, 1}}, true);
   const GridCoord even = {{65535, 65535, 6}};
   const int64_t original = grid_index(wide, even);
   check(original > int64_t(UINT32_MAX));
   check(transform.unmap_index(transform.map_index(original)) == original);
   rejects<std::overflow_error>([] { grid_volume({{INT64_MAX, 2, 2}}); });
   rejects<std::invalid_argument>([] { grid_volume({{6, 0, 8}}); });
   rejects<std::out_of_range>([&] { grid_index(wide, {{65536, 0, 0}}); });
   rejects<std::out_of_range>([&] { grid_index(wide, {{-1, 0, 0}}); });
   rejects<std::out_of_range>([&] { grid_coord(wide, -1); });
   rejects<std::out_of_range>([&] { grid_coord(wide, grid_volume(wide)); });
   rejects<std::invalid_argument>([] { IndexTransform({{6, 8, 10}}, {{0, 0, 1}}); });
   rejects<std::invalid_argument>([] { IndexTransform({{6, 9, 10}}, {{0, 1, 2}}, true); });
   rejects<std::invalid_argument>([] {
      IndexTransform({{6, 8, 10}}, {{0, 1, 2}}, true).map_index(1);
   });
   rejects<std::invalid_argument>([] {
      transform_adjacency(IndexTransform(GridDims{{6, 8, 10}}), 0, 4096, 12);
   });
   rejects<std::invalid_argument>([] {
      transform_adjacency(IndexTransform({{6, 8, 10}}, {{0, 1, 2}}, true), 0, 1, 6);
   });
   rejects<std::invalid_argument>([] { grid_directions(8); });
}

int main()
{
   test_grid_build();
   test_legacy_interpolation();
   test_abc_and_bounds();
   test_cartesian_permutations();
   test_fcc_fold();
   test_interpolation_permutations();
   test_rows_and_duplicates();
   test_checked_indices();
   std::printf("prepare_grid: PASS (%zu checks; Cart/FCC interpolation, 6 permutations, fold bijection/stencil, duplicates/overflow/ABC)\n", checks);
}
