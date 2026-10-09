// GPU regression: original gather/ADE/scatter versus the fused CUDA kernel.
// Exit 77 means CUDA hardware/driver is unavailable; compilation alone is not a pass.
#include <vector>
#include <cstring>
#include <cmath>
#include <fdtd_data.h>
#include <gpu_engine.h>

enum { CHECK_GRID = 519, CHECK_NODES = 257, CHECK_MATERIALS = 4, CHECK_STEPS = 37 };

template<typename T>
static T *check_upload(const std::vector<T>& values)
{
   T *p;
   gpuErrchk(cudaMalloc(&p,MAX(values.size(),(size_t)1)*sizeof(T)));
   if (!values.empty())
      gpuErrchk(cudaMemcpy(p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
   return p;
}

static void check_device_equal(const char *name, const Real *ref, const Real *fused,
                               size_t count, int step)
{
   if (count == 0) return;
   std::vector<Real> expected(count),actual(count);
   gpuErrchk(cudaMemcpy(expected.data(),ref,count*sizeof(Real),cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(actual.data(),fused,count*sizeof(Real),cudaMemcpyDeviceToHost));
   for (size_t i=0; i<count; i++) {
      if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
          std::memcmp(&expected[i],&actual[i],sizeof(Real))) {
         fprintf(stderr,"%s differs at step %d index %zu: %.17g != %.17g\n",
                 name,step,i,(double)expected[i],(double)actual[i]);
         exit(EXIT_FAILURE);
      }
   }
}

__global__ void CheckRigidInput(Real *write_grid, const Real *read_grid, int step)
{
   int i = blockIdx.x*blockDim.x+threadIdx.x;
   if (i<CHECK_GRID) {
      Real input = (Real)((step%7)-3)*0.0009765625;
      write_grid[i] = 1.5*read_grid[i] - write_grid[i] + input;
   }
}

static void check_cuda_case(int64_t Nbl, int material)
{
   const int8_t Mb[CHECK_MATERIALS] = {0,1,11,12};
   const Real lo2 = 0.25;
   gpuErrchk(cudaMemcpyToSymbol(cuMb,Mb,sizeof(Mb)));
   gpuErrchk(cudaMemcpyToSymbol(cuNbl,&Nbl,sizeof(Nbl)));
   gpuErrchk(cudaMemcpyToSymbol(clo2,&lo2,sizeof(lo2)));

   std::vector<int64_t> locs(Nbl);
   std::vector<int8_t> mats(Nbl);
   std::vector<Real> ssaf(Nbl),beta = {0.0,0.125,0.25,0.5};
   std::vector<MatQuad> quads(CHECK_MATERIALS*MMb);
   std::vector<Real> vh(Nbl*MMb),gh(Nbl*MMb);
   for (int k=0; k<CHECK_MATERIALS; k++) {
      for (int m=0; m<MMb; m++) {
         MatQuad& tm = quads[k*MMb+m];
         tm.b = (Real)(m+1)*0.0009765625;
         tm.bd = 0.5;
         tm.bDh = (Real)(m+1)*0.001953125;
         tm.bFh = (Real)(k+1)*0.0009765625;
      }
   }
   for (int64_t nb=0; nb<Nbl; nb++) {
      locs[nb] = 2 + 2*nb;
      mats[nb] = (int8_t)((material<0) ? nb%CHECK_MATERIALS : material);
      ssaf[nb] = (Real)(nb%5+1)*0.125;
      for (int m=0; m<MMb; m++) {
         vh[m*Nbl+nb] = (Real)((nb+1)*(m+1))*0.0009765625;
         gh[m*Nbl+nb] = -(Real)((nb+2)*(m+1))*0.00048828125;
      }
   }
   int64_t *d_locs = check_upload(locs);
   int8_t *d_mats = check_upload(mats);
   Real *d_ssaf = check_upload(ssaf), *d_beta = check_upload(beta);
   MatQuad *d_quads = check_upload(quads);
   Real *ref_vh = check_upload(vh), *fused_vh = check_upload(vh);
   Real *ref_gh = check_upload(gh), *fused_gh = check_upload(gh);
   Real *ref_grid[2], *fused_grid[2], *ref_carry[3], *fused_carry[3];
   for (int bank=0; bank<2; bank++) {
      std::vector<Real> grid(CHECK_GRID);
      for (int i=0; i<CHECK_GRID; i++) grid[i] = (Real)((i+1)*(bank+1))*0.015625;
      ref_grid[bank] = check_upload(grid);
      fused_grid[bank] = check_upload(grid);
   }
   for (int bank=0; bank<3; bank++) {
      std::vector<Real> carry(CHECK_NODES);
      for (int nb=0; nb<CHECK_NODES; nb++) carry[nb] = (Real)((nb+1)*(bank+1))*0.0078125;
      ref_carry[bank] = check_upload(carry);
      fused_carry[bank] = check_upload(carry);
   }
   int write_grid=0,read_grid=1,carry0=0,carry1=1,carry2=2;
   const int blocks = (int)CU_DIV_CEIL(Nbl,cuBb);
   for (int step=0; step<CHECK_STEPS; step++) {
      CheckRigidInput<<<CU_DIV_CEIL(CHECK_GRID,128),128>>>(ref_grid[write_grid],ref_grid[read_grid],step);
      CheckRigidInput<<<CU_DIV_CEIL(CHECK_GRID,128),128>>>(fused_grid[write_grid],fused_grid[read_grid],step);
      CopyFromGridKernel<<<blocks,cuBb>>>(ref_carry[carry0],ref_grid[write_grid],d_locs,Nbl);
      KernelBoundaryFD<<<blocks,cuBb>>>(ref_carry[carry0],ref_carry[carry2],ref_vh,ref_gh,
                                      d_ssaf,d_mats,d_beta,d_quads);
      CopyToGridKernel<<<blocks,cuBb>>>(ref_grid[write_grid],ref_carry[carry0],d_locs,Nbl);
      KernelBoundaryFDGrid<<<blocks,cuBb>>>(fused_grid[write_grid],fused_carry[carry0],
            fused_carry[carry2],fused_vh,fused_gh,d_locs,d_ssaf,d_mats,d_beta,d_quads);
      gpuErrchk(cudaPeekAtLastError());
      for (int bank=0; bank<2; bank++)
         check_device_equal("grid",ref_grid[bank],fused_grid[bank],CHECK_GRID,step);
      for (int bank=0; bank<3; bank++)
         check_device_equal("carry",ref_carry[bank],fused_carry[bank],CHECK_NODES,step);
      check_device_equal("vh1 including unused poles",ref_vh,fused_vh,Nbl*MMb,step);
      check_device_equal("gh1 including unused poles",ref_gh,fused_gh,Nbl*MMb,step);
      int tmp = read_grid; read_grid = write_grid; write_grid = tmp;
      tmp = carry2; carry2 = carry1; carry1 = carry0; carry0 = tmp;
   }
   for (int bank=0; bank<2; bank++) {
      gpuErrchk(cudaFree(ref_grid[bank])); gpuErrchk(cudaFree(fused_grid[bank]));
   }
   for (int bank=0; bank<3; bank++) {
      gpuErrchk(cudaFree(ref_carry[bank])); gpuErrchk(cudaFree(fused_carry[bank]));
   }
   gpuErrchk(cudaFree(d_locs)); gpuErrchk(cudaFree(d_mats));
   gpuErrchk(cudaFree(d_ssaf)); gpuErrchk(cudaFree(d_beta)); gpuErrchk(cudaFree(d_quads));
   gpuErrchk(cudaFree(ref_vh)); gpuErrchk(cudaFree(fused_vh));
   gpuErrchk(cudaFree(ref_gh)); gpuErrchk(cudaFree(fused_gh));
}

int main(void)
{
   int devices = 0;
   cudaError_t status = cudaGetDeviceCount(&devices);
   if (status != cudaSuccess || devices == 0) {
      fprintf(stderr,"boundary_fd_cuda_check: SKIP (no usable CUDA device: %s)\n",
              cudaGetErrorString(status));
      return 77;
   }
   gpuErrchk(cudaSetDevice(0));
   check_cuda_case(0,-1);
   for (int material=0; material<CHECK_MATERIALS; material++) {
      check_cuda_case(1,material);
      check_cuda_case(CHECK_NODES,material);
   }
   check_cuda_case(CHECK_NODES,-1);
   puts("boundary_fd_cuda_check: PASS (exact CUDA equality, 10 cases x 37 steps)");
   return EXIT_SUCCESS;
}
