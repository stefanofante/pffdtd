// Device comparison of the ordered rigid+FDGrid kernels and complete fusion.
// Seed every field/history, compare inactive poles and padding, rotate 2/3 slots.
// Compilation is not a runtime pass: absent CUDA hardware exits with status 77.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <fdtd_data.h>
#include <gpu_engine.h>

enum { TEST_NX=11, TEST_NY=9, TEST_NZ=9, TEST_N=TEST_NX*TEST_NY*TEST_NZ,
       TEST_NB=257, TEST_MATERIALS=13, TEST_STEPS=37, TEST_PADDING=4 };

template<typename T>
static T *upload(const std::vector<T> &values)
{
   T *device=nullptr;
   const std::size_t count=std::max<std::size_t>(values.size(),1);
   gpuErrchk(cudaMalloc(&device,count*sizeof(T)));
   if (!values.empty())
      gpuErrchk(cudaMemcpy(device,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
   return device;
}

static void equal_device(const char *name, const Real *ref, const Real *fused,
                         std::size_t count, int step, int ade_mode)
{
   if (count==0) return;
   std::vector<Real> expected(count),actual(count);
   gpuErrchk(cudaMemcpy(expected.data(),ref,count*sizeof(Real),cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(actual.data(),fused,count*sizeof(Real),cudaMemcpyDeviceToHost));
   for (std::size_t i=0; i<count; ++i)
      if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
          std::memcmp(&expected[i],&actual[i],sizeof(Real))!=0) {
         std::fprintf(stderr,"FAIL %s FP%d ADEmode=%d step=%d index=%zu: %.17g != %.17g\n",name,
                      static_cast<int>(8*sizeof(Real)),ade_mode,step,i,
                      static_cast<double>(expected[i]),static_cast<double>(actual[i]));
         std::exit(EXIT_FAILURE);
      }
}

__global__ void EvolveNonBoundary(Real *write, const Real *read,
                                  const uint8_t *boundary, int step)
{
   const int i=blockIdx.x*blockDim.x+threadIdx.x;
   if (i<TEST_N && !boundary[i]) {
      const Real forcing=Real((step%7)-3)*Real(0.0009765625);
      write[i]=Real(1.125)*read[i]-write[i]+forcing;
   }
}

template<bool FCC, typename MapIdx, int ADEMode>
static void check_case(int lossy_mode, int material_mode)
{
   const int branches=FCC ? 12 : 6;
   std::mt19937 random(0xb0ad32u+lossy_mode*17+material_mode+1);
   std::vector<int64_t> boundary;
   for (int z=1; z<TEST_NZ-1; ++z)
      for (int y=1; y<TEST_NY-1; ++y)
         for (int x=1; x<TEST_NX-1; ++x)
            boundary.push_back((static_cast<int64_t>(z)*TEST_NY+y)*TEST_NX+x);
   std::shuffle(boundary.begin(),boundary.end(),random);
   boundary.resize(TEST_NB);
   std::vector<int64_t> lossy;
   for (std::size_t i=0; i<boundary.size(); ++i)
      if (lossy_mode==1 || (lossy_mode==2 && i%2==0)) lossy.push_back(boundary[i]);
   std::shuffle(lossy.begin(),lossy.end(),random);
   const int64_t nbl=static_cast<int64_t>(lossy.size());
   std::vector<MapIdx> map(boundary.size());
   const pffdtd::BoundaryMapStatus status=pffdtd::build_boundary_lossy_map(
      boundary.data(),TEST_NB,lossy.data(),nbl,TEST_N,map.data());
   if (status!=pffdtd::BoundaryMapStatus::Success) {
      std::fprintf(stderr,"FAIL fixture map: %s\n",pffdtd::boundary_map_status_string(status));
      std::exit(EXIT_FAILURE);
   }
   std::vector<uint16_t> adjacency(TEST_NB);
   std::vector<int8_t> neighbours(TEST_NB);
   std::vector<uint8_t> mask(TEST_N,0);
   for (int i=0; i<TEST_NB; ++i) {
      mask[static_cast<std::size_t>(boundary[i])]=1;
      const uint16_t all=static_cast<uint16_t>((1u<<branches)-1u);
      uint16_t bits=static_cast<uint16_t>(random()&all);
      if (bits==all) bits^=static_cast<uint16_t>(1u<<(i%branches));
      if (map[i]>=0 && bits==0) bits=1;
      if (map[i]<0 && i%7==0) bits=0;
      adjacency[i]=bits;
      int active=0;
      for (int b=0; b<branches; ++b) active+=(bits>>b)&1;
      neighbours[i]=static_cast<int8_t>(active);
   }
   int8_t poles[TEST_MATERIALS];
   std::vector<Real> beta(TEST_MATERIALS);
   for (int k=0; k<TEST_MATERIALS; ++k) {
      poles[k]=static_cast<int8_t>(k);
      beta[k]=Real(k+1)*Real(0.0625);
   }
   std::vector<MatQuad> quads(TEST_MATERIALS*MMb);
   for (int k=0; k<TEST_MATERIALS; ++k)
      for (int m=0; m<MMb; ++m) {
         MatQuad &q=quads[k*MMb+m];
         q.b=Real(m+1)*Real(0.0009765625); q.bd=Real(0.5);
         q.bDh=Real(m+1)*Real(0.001953125); q.bFh=Real(k+1)*Real(0.0009765625);
      }
   std::vector<Real> surface(static_cast<std::size_t>(nbl));
   std::vector<int8_t> materials(static_cast<std::size_t>(nbl));
   for (int64_t i=0; i<nbl; ++i) {
      surface[static_cast<std::size_t>(i)]=Real(i%5+1)*Real(0.125);
      materials[static_cast<std::size_t>(i)]=static_cast<int8_t>(material_mode<0 ? i%TEST_MATERIALS : material_mode);
   }
   const std::size_t state_count=static_cast<std::size_t>(nbl)*MMb+TEST_PADDING;
   const std::size_t carry_count=static_cast<std::size_t>(nbl)+TEST_PADDING;
   std::vector<Real> vh(state_count),gh(state_count);
   for (std::size_t i=0; i<state_count; ++i) {
      vh[i]=Real(i+1)*Real(0.0009765625);
      gh[i]=-Real(i+2)*Real(0.00048828125);
   }
   const int64_t nx=TEST_NX,ny=TEST_NY,nz=TEST_NZ,plane=nx*ny,nb=TEST_NB;
   const Real sl2=FCC ? Real(0.015625) : Real(0.0625);
   const Real a2=sl2,lo2=Real(0.125);
   gpuErrchk(cudaMemcpyToSymbol(cuNx,&nx,sizeof(nx)));
   gpuErrchk(cudaMemcpyToSymbol(cuNy,&ny,sizeof(ny)));
   gpuErrchk(cudaMemcpyToSymbol(cuNz,&nz,sizeof(nz)));
   gpuErrchk(cudaMemcpyToSymbol(cuNxNy,&plane,sizeof(plane)));
   gpuErrchk(cudaMemcpyToSymbol(cuNb,&nb,sizeof(nb)));
   gpuErrchk(cudaMemcpyToSymbol(cuNbl,&nbl,sizeof(nbl)));
   gpuErrchk(cudaMemcpyToSymbol(cuMb,poles,sizeof(poles)));
   gpuErrchk(cudaMemcpyToSymbol(csl2,&sl2,sizeof(sl2)));
   gpuErrchk(cudaMemcpyToSymbol(c2,&a2,sizeof(a2)));
   gpuErrchk(cudaMemcpyToSymbol(clo2,&lo2,sizeof(lo2)));

   int64_t *d_boundary=upload(boundary),*d_lossy=upload(lossy);
   MapIdx *d_map=upload(map);
   uint16_t *d_adjacency=upload(adjacency);
   int8_t *d_neighbours=upload(neighbours),*d_materials=upload(materials);
   uint8_t *d_mask=upload(mask);
   Real *d_surface=upload(surface),*d_beta=upload(beta);
   MatQuad *d_quads=upload(quads);
   Real *ref_vh=upload(vh),*ref_gh=upload(gh),*fused_vh=upload(vh),*fused_gh=upload(gh);
   Real *ref_grid[2],*fused_grid[2],*ref_carry[3],*fused_carry[3];
   for (int bank=0; bank<2; ++bank) {
      std::vector<Real> grid(TEST_N);
      for (Real &value : grid) value=Real(static_cast<int>(random()%4096)-2048)*Real(0.00390625);
      ref_grid[bank]=upload(grid); fused_grid[bank]=upload(grid);
   }
   for (int bank=0; bank<3; ++bank) {
      std::vector<Real> carry(carry_count);
      for (std::size_t i=0; i<carry_count; ++i) carry[i]=Real((i+1)*(bank+1))*Real(0.0078125);
      ref_carry[bank]=upload(carry); fused_carry[bank]=upload(carry);
   }
   int read=1,write=0,carry0=0,carry1=1,carry2=2;
   const int blocks=(TEST_NB+cuBb-1)/cuBb;
   const int lossy_blocks=static_cast<int>(CU_DIV_CEIL(nbl,cuBb));
   const dim3 fold_block(cuBx2,cuBy2,1);
   const dim3 fold_grid((TEST_NX+cuBx2-1)/cuBx2,(TEST_NZ+cuBy2-1)/cuBy2,1);
   for (int step=0; step<TEST_STEPS; ++step) {
      if (FCC) {
         KernelFoldFCC<int64_t><<<fold_grid,fold_block>>>(ref_grid[read]);
         KernelFoldFCC<int64_t><<<fold_grid,fold_block>>>(fused_grid[read]);
      }
      EvolveNonBoundary<<<(TEST_N+127)/128,128>>>(ref_grid[write],ref_grid[read],d_mask,step);
      EvolveNonBoundary<<<(TEST_N+127)/128,128>>>(fused_grid[write],fused_grid[read],d_mask,step);
      if (FCC)
         KernelBoundaryRigidFCC<<<blocks,cuBb>>>(ref_grid[write],ref_grid[read],d_adjacency,d_boundary,d_neighbours);
      else
         KernelBoundaryRigidCart<<<blocks,cuBb>>>(ref_grid[write],ref_grid[read],d_adjacency,d_boundary,d_neighbours);
      KernelBoundaryFDGrid<<<lossy_blocks,cuBb>>>(ref_grid[write],ref_carry[carry0],ref_carry[carry2],
         ref_vh,ref_gh,d_lossy,d_surface,d_materials,d_beta,d_quads);
      KernelBoundaryStencil<FCC,MapIdx,ADEMode><<<blocks,cuBb>>>(fused_grid[write],fused_grid[read],
         fused_carry[carry0],fused_carry[carry2],fused_vh,fused_gh,d_boundary,d_adjacency,
         d_neighbours,d_map,d_surface,d_materials,d_beta,d_quads);
      gpuErrchk(cudaPeekAtLastError());
      for (int bank=0; bank<2; ++bank) equal_device("grid",ref_grid[bank],fused_grid[bank],TEST_N,step,ADEMode);
      for (int bank=0; bank<3; ++bank) equal_device("carry and padding",ref_carry[bank],fused_carry[bank],carry_count,step,ADEMode);
      equal_device("vh including inactive poles and padding",ref_vh,fused_vh,state_count,step,ADEMode);
      equal_device("gh including inactive poles and padding",ref_gh,fused_gh,state_count,step,ADEMode);
      std::swap(read,write);
      const int oldest=carry2; carry2=carry1; carry1=carry0; carry0=oldest;
   }
   for (int bank=0; bank<2; ++bank) {
      gpuErrchk(cudaFree(ref_grid[bank])); gpuErrchk(cudaFree(fused_grid[bank]));
   }
   for (int bank=0; bank<3; ++bank) {
      gpuErrchk(cudaFree(ref_carry[bank])); gpuErrchk(cudaFree(fused_carry[bank]));
   }
   gpuErrchk(cudaFree(d_boundary)); gpuErrchk(cudaFree(d_lossy)); gpuErrchk(cudaFree(d_map));
   gpuErrchk(cudaFree(d_adjacency)); gpuErrchk(cudaFree(d_neighbours)); gpuErrchk(cudaFree(d_materials));
   gpuErrchk(cudaFree(d_mask)); gpuErrchk(cudaFree(d_surface)); gpuErrchk(cudaFree(d_beta));
   gpuErrchk(cudaFree(d_quads)); gpuErrchk(cudaFree(ref_vh)); gpuErrchk(cudaFree(ref_gh));
   gpuErrchk(cudaFree(fused_vh)); gpuErrchk(cudaFree(fused_gh));
}

int main()
{
   int devices=0;
   const cudaError_t status=cudaGetDeviceCount(&devices);
   if (status!=cudaSuccess || devices==0) {
      std::fprintf(stderr,"boundary_stencil_cuda_check: SKIP (no usable CUDA device)\n");
      return 77;
   }
   gpuErrchk(cudaSetDevice(0));
   int cases=0;
   for (int mode=0; mode<3; ++mode)
      for (int material=-1; material<TEST_MATERIALS; ++material) {
         if (mode==0 && material!=-1) continue;
         check_case<false,int32_t,0>(mode,material); check_case<false,int64_t,0>(mode,material);
         check_case<true,int32_t,0>(mode,material); check_case<true,int64_t,0>(mode,material);
         check_case<false,int32_t,1>(mode,material); check_case<false,int64_t,1>(mode,material);
         check_case<true,int32_t,1>(mode,material); check_case<true,int64_t,1>(mode,material);
         check_case<false,int32_t,2>(mode,material); check_case<false,int64_t,2>(mode,material);
         check_case<true,int32_t,2>(mode,material); check_case<true,int64_t,2>(mode,material);
         cases+=12;
      }
   std::printf("boundary_stencil_cuda_check: PASS FP%d, %d cases x37 steps, 257 boundaries, "
               "Cart/FCC, int32/int64 maps, rigid/lossy/mixed, all 0..12 poles, "
               "generic/reload/fixed ADE, full states\n",
               static_cast<int>(8*sizeof(Real)),cases);
   return EXIT_SUCCESS;
}
