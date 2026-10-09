// Device gate against the actual OpenMP voxelizer in fdtd_prepare.cpp. Link
// that translation unit with PFFDTD_PREPARE_NO_MAIN; a build is not a runtime pass.
#include "prepare_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pffdtd_prepare;
std::size_t cases=0,compared_records=0;

void require(bool condition,const std::string &message)
{
   if (!condition) throw std::runtime_error(message);
}

Triangle make_triangle(Vec3 a,Vec3 b,Vec3 c,std::uint32_t ordinal,
                       std::int32_t material=0,std::uint8_t sides=3)
{
   Triangle triangle;
   require(precompute_triangle(triangle,a,b,c,material,sides,ordinal),"fixture triangle is degenerate");
   return triangle;
}

void quad(std::vector<Triangle> &triangles,Vec3 a,Vec3 b,Vec3 c,Vec3 d,
          std::int32_t material,std::uint8_t sides)
{
   const std::uint32_t first=static_cast<std::uint32_t>(triangles.size());
   triangles.push_back(make_triangle(a,b,c,first,material,sides));
   triangles.push_back(make_triangle(a,c,d,first+1,material,sides));
}

std::vector<Triangle> horizontal(double lower,double upper,double height,
                                 std::uint8_t sides=3)
{
   std::vector<Triangle> result;
   quad(result,{lower,lower,height},{upper,lower,height},{upper,upper,height},
        {lower,upper,height},2,sides);
   return result;
}

std::vector<Triangle> box(double a)
{
   std::vector<Triangle> result;
   quad(result,{-a,-a,-a},{-a,-a,a},{-a,a,a},{-a,a,-a},0,0);
   quad(result,{a,-a,-a},{a,a,-a},{a,a,a},{a,-a,a},1,1);
   quad(result,{-a,-a,-a},{a,-a,-a},{a,-a,a},{-a,-a,a},2,2);
   quad(result,{-a,a,-a},{-a,a,a},{a,a,a},{a,a,-a},3,3);
   quad(result,{-a,-a,-a},{-a,a,-a},{a,a,-a},{a,-a,-a},4,1);
   quad(result,{-a,-a,a},{a,-a,a},{a,a,a},{-a,a,a},5,2);
   return result;
}

std::size_t physical_interior_count(const Grid &grid)
{
   const GridDims dims=grid.dims();
   std::size_t count=0;
   for (int64_t x=1;x<dims[0]-1;++x)
      for (int64_t y=1;y<dims[1]-1;++y)
         for (int64_t z=1;z<dims[2]-1;++z)
            if (!grid.fcc_flag || !((x+y+z)&1)) ++count;
   return count;
}

void validate_sparse(const char *label,const FlatBvh &geometry,const Grid &grid,
                     const std::vector<BoundaryRecord> &records)
{
   const GridDims dims=grid.dims();
   int64_t previous=-1;
   for (const BoundaryRecord &record : records) {
      require(record.index>previous,std::string(label)+": compaction must be strictly index ordered");
      previous=record.index;
      const GridCoord point=grid_coord(dims,record.index);
      for (unsigned axis=0;axis<3;++axis)
         require(point[axis]>0 && point[axis]<dims[axis]-1,
                 std::string(label)+": compaction includes a halo point");
      require(!grid.fcc_flag || even_fcc_coord(point),std::string(label)+": odd-parity FCC point emitted");
      require(record.triangle<geometry.triangles.size(),std::string(label)+": nearest triangle index invalid");
      const unsigned directions=grid.fcc_flag ? 12 : 6;
      require(record.adjacency<((1U<<directions)-1),std::string(label)+": fully connected point emitted");
      require(record.material>=-1 && std::isfinite(record.saf) && record.saf>=0,
              std::string(label)+": nonfinite or invalid boundary metadata");
   }
}

void compare(const char *label,const FlatBvh &geometry,const Grid &grid,
             const std::vector<BoundaryRecord> &reference,const std::vector<BoundaryRecord> &actual)
{
   validate_sparse(label,geometry,grid,reference);
   validate_sparse(label,geometry,grid,actual);
   require(reference.size()==actual.size(),std::string(label)+": CUDA boundary count differs from true CPU voxelizer");
   for (std::size_t row=0;row<reference.size();++row) {
      const BoundaryRecord &a=reference[row],&b=actual[row];
      if (a.index!=b.index || a.adjacency!=b.adjacency || a.triangle!=b.triangle ||
          a.material!=b.material || a.near_boundary!=b.near_boundary ||
          std::abs(a.saf-b.saf)>1e-13*(1+std::abs(a.saf))) {
         std::fprintf(stderr,"FAIL %s row=%zu: CPU index=%lld adj=%u triangle=%u mat=%d near=%d saf=%.17g; "
                      "CUDA index=%lld adj=%u triangle=%u mat=%d near=%d saf=%.17g\n",label,row,
                      static_cast<long long>(a.index),static_cast<unsigned>(a.adjacency),a.triangle,
                      a.material,a.near_boundary,a.saf,static_cast<long long>(b.index),
                      static_cast<unsigned>(b.adjacency),b.triangle,b.material,b.near_boundary,b.saf);
         throw std::runtime_error("CUDA preparation metadata differs from actual CPU voxelizer");
      }
      // Equality of triangle indices also verifies the original ordinal mapping
      // after BVH permutation; ordinal is carried by the unchanged input array.
      require(geometry.triangles[a.triangle].ordinal==geometry.triangles[b.triangle].ordinal,
              std::string(label)+": nearest original triangle ordinal differs");
   }
   compared_records+=reference.size();
   ++cases;
}

void check_case(const char *label,const FlatBvh &geometry,const Grid &grid,
                bool expect_empty=false,bool expect_all_near=false,bool crosses_batch=false,
                int expected_ordinal=-1)
{
   validate_grid(grid);
   const std::vector<BoundaryRecord> reference=voxelize_cpu(geometry,grid,2);
   if (expect_empty) require(reference.empty(),std::string(label)+": empty fixture CPU oracle emitted nodes");
   else require(!reference.empty(),std::string(label)+": fixture must intersect the physical grid");
   if (expected_ordinal>=0)
      for (const BoundaryRecord &record : reference)
         require(geometry.triangles[record.triangle].ordinal==static_cast<std::uint32_t>(expected_ordinal),
                 std::string(label)+": equal-distance fixture did not choose the smallest original ordinal");
   if (expect_all_near) {
      require(reference.size()==physical_interior_count(grid),std::string(label)+": all-near fixture misses interior cells");
      for (const BoundaryRecord &record : reference)
         require(record.near_boundary && record.adjacency==0 && record.material==-1,
                 std::string(label)+": all-near fixture must isolate rigid nodes");
   }
   if (crosses_batch) {
      require(physical_interior_count(grid)>262144,std::string(label)+": grid must cross a CUDA candidate batch");
      const GridDims dims=grid.dims();
      const GridCoord first=grid_coord(dims,reference.front().index),
                      last=grid_coord(dims,reference.back().index);
      const int64_t first_dense=((first[0]-1)*(dims[1]-2)+(first[1]-1))*(dims[2]-2)+(first[2]-1);
      const int64_t last_dense=((last[0]-1)*(dims[1]-2)+(last[1]-1))*(dims[2]-2)+(last[2]-1);
      require(first_dense<262144 && last_dense>(grid.fcc_flag ? 524288 : 262144),
              std::string(label)+": fixture must emit boundary records on both sides of the batch split");
   }
   compare(label,geometry,grid,reference,voxelize_cuda(geometry,grid,2));
}

void check_small_fixtures()
{
   for (double h : {0.125,1.0,8.0})
      for (bool fcc : {false,true}) {
         const Grid grid=build_grid({{-2*h,-2*h,-2*h}},{{2*h,2*h,2*h}},h,fcc);
         check_case("empty geometry",build_bvh({}),grid,true);
         check_case("nonempty geometry outside grid",build_bvh(horizontal(-2*h,2*h,20*h)),grid,true);
         for (unsigned sides=0;sides<=3;++sides)
            check_case("finite sided plane",build_bvh(horizontal(-2*h,2*h,0,static_cast<std::uint8_t>(sides))),grid);
         check_case("closed box with mixed sides",build_bvh(box(2*h)),grid);
         const double layer=grid.axes[2][grid.axes[2].size()/2];
         for (double offset : {-0.75e-6,0.0,0.75e-6})
            check_case("near plane",build_bvh(horizontal(-10*h,10*h,layer+offset*h)),grid);

         std::vector<Triangle> all_near;
         for (std::size_t z=1;z+1<grid.axes[2].size();++z)
            quad(all_near,{-10*h,-10*h,grid.axes[2][z]},{10*h,-10*h,grid.axes[2][z]},
                 {10*h,10*h,grid.axes[2][z]},{-10*h,10*h,grid.axes[2][z]},
                 static_cast<std::int32_t>(z%4),static_cast<std::uint8_t>(z%4));
         check_case("all interior nodes near",build_bvh(all_near),grid,false,true);

         std::vector<Triangle> tied;
         Triangle removed;
         precompute_triangle(removed,{0,0,0},{0,0,0},{0,0,0},-1,0,999);
         tied.push_back(removed); // Preserve a hole in original source indices.
         for (unsigned i=0;i<25;++i)
            tied.push_back(make_triangle({-10*h,-10*h,0},{10*h,-10*h,0},{0,10*h,0},
                                         100-i,static_cast<std::int32_t>(i%5),3));
         for (unsigned i=0;i<20;++i)
            tied.push_back(make_triangle({(20.0+i)*h,20*h,20*h},{(21.0+i)*h,20*h,20*h},
                                         {(20.0+i)*h,21*h,20*h},200+i,0,3));
         std::mt19937 random(0x42a12u);
         std::shuffle(tied.begin(),tied.end(),random);
         check_case("equal-distance original ordinal across BVH leaves",build_bvh(tied),grid,
                    false,false,false,76);

         std::vector<Triangle> mixed=box(2*h);
         std::mt19937_64 angles(0x4112acU);
         std::uniform_real_distribution<double> coordinate(-1.75,1.75),edge(-0.75,0.75);
         for (unsigned i=0;i<41;++i) {
            const Vec3 a={coordinate(angles)*h,coordinate(angles)*h,coordinate(angles)*h};
            const Vec3 b={a.x+edge(angles)*h,a.y+edge(angles)*h,a.z+edge(angles)*h};
            const Vec3 c={a.x+edge(angles)*h,a.y+edge(angles)*h,a.z+edge(angles)*h};
            mixed.push_back(make_triangle(a,b,c,500+i,static_cast<std::int32_t>(i%7)-1,
                                          static_cast<std::uint8_t>(i%4)));
         }
         std::shuffle(mixed.begin(),mixed.end(),random);
         check_case("mixed slanted facets",build_bvh(mixed),grid);
      }
}

void check_multiple_batches()
{
   for (bool fcc : {false,true}) {
      const Grid grid=build_grid({{0,0,0}},{{76,76,76}},1,fcc);
      const double plane=grid.axes[2][grid.axes[2].size()/2];
      check_case("candidate batch boundary and tail",build_bvh(horizontal(-10,90,plane)),grid,
                  false,false,true);
   }
}

} // namespace

int main()
{
   int devices=0;
   const cudaError_t status=cudaGetDeviceCount(&devices);
   if (status!=cudaSuccess || devices<1) {
      std::fprintf(stderr,"prepare_cuda_check: SKIP (no usable CUDA device, %s)\n",cudaGetErrorString(status));
      return 77;
   }
   try {
      const cudaError_t selected=cudaSetDevice(0);
      require(selected==cudaSuccess,std::string("Cannot select CUDA device: ")+cudaGetErrorString(selected));
      check_small_fixtures();
      check_multiple_batches();
      std::printf("PASS: %zu CUDA preparation comparisons, %zu boundary records; true OpenMP CPU oracle, "
                  "Cart/FCC, sided planes/box/mixed facets, near isolation, original ordinal ties, "
                  "sorted compaction/empty geometry and candidate batch tails; SAF tolerance 1e-13*(1+abs(ref))\n",
                  cases,compared_records);
      return EXIT_SUCCESS;
   }
   catch (const std::exception &error) {
      std::fprintf(stderr,"prepare_cuda_check: %s\n",error.what());
      return EXIT_FAILURE;
   }
}
