// Full-engine regression for single-device scheduling, using native fixtures.
// Select one device with CUDA_VISIBLE_DEVICES; hardware absence exits with 77.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fdtd_data.h>
#include <gpu_engine.h>

struct SchedulerFixture {
   SimData data;
   std::vector<int64_t> boundary, lossy, abc, sources, receivers;
   std::vector<int8_t> abc_order, material, neighbours, poles;
   std::vector<uint16_t> adjacency;
   std::vector<uint8_t> mask;
   std::vector<Real> surface, beta;
   std::vector<MatQuad> quads;
   std::vector<double> input, output;

   int64_t index(int x, int y, int z) const {
      return (x*data.Ny+y)*data.Nz+z;
   }

   SchedulerFixture(bool fcc, bool with_ade, int64_t steps) : data{} {
      data.Nx=10; data.Ny=8; data.Nz=10;
      data.Npts=data.Nx*data.Ny*data.Nz;
      data.Nt=steps; data.fcc_flag=fcc ? 2 : 0; data.NN=fcc ? 12 : 6;
      data.l=fcc ? 0.9 : 0.5;
      data.l2=data.l*data.l;
      const double factor=fcc ? 0.25 : 1.0;
      const double scaled=(1.0+EPS)*factor*data.l2;
      data.a1=(Real)(2.0-scaled*data.NN);
      data.a2=(Real)(factor*data.l2);
      data.sl2=(Real)scaled;
      data.lo2=(Real)(0.5*data.l);
      data.Nm=1;
      poles.push_back(11);
      beta.push_back((Real)0.125);
      quads.resize(MMb);
      for (int m=0; m<MMb; ++m) {
         quads[m].b=(Real)((m+1)*0.0009765625);
         quads[m].bd=(Real)0.5;
         quads[m].bDh=(Real)((m+1)*0.001953125);
         quads[m].bFh=(Real)0.0009765625;
      }
      mask.assign((data.Npts+7)/8,0);
      if (with_ade) {
         const int cart[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
         const int fcc_links[12][3]={{1,1,0},{-1,-1,0},{0,1,1},{0,-1,-1},
               {1,0,1},{-1,0,-1},{1,-1,0},{-1,1,0},{0,1,-1},{0,-1,1},
               {1,0,-1},{-1,0,1}};
         // A finite panel at x=4.5: cut both directions of each crossing link.
         // Reciprocal adjacency avoids an artificial unstable directed stencil.
         for (int x=1; x<data.Nx-1; ++x)
            for (int y=1; y<data.Ny-1; ++y)
               for (int z=1; z<data.Nz-1; ++z) {
                  uint16_t adj=(uint16_t)((1u<<data.NN)-1u);
                  for (int k=0; k<data.NN; ++k) {
                     const int *d=fcc ? fcc_links[k] : cart[k];
                     const int tx=x+d[0],ty=y+d[1],tz=z+d[2];
                     if (((x==4 && tx==5)||(x==5 && tx==4)) &&
                         y>=2 && y<=data.Ny-3 && ty>=2 && ty<=data.Ny-3 &&
                         z>=2 && z<=data.Nz-3 && tz>=2 && tz<=data.Nz-3)
                        adj &= (uint16_t)~(1u<<k);
                  }
                  if (adj != (1u<<data.NN)-1u) {
                     boundary.push_back(index(x,y,z));
                     adjacency.push_back(adj);
                     neighbours.push_back((int8_t)__builtin_popcount((unsigned)adj));
                  }
               }
         lossy=boundary;
         surface.assign(boundary.size(),(Real)0.75);
         material.assign(boundary.size(),0);
         for (int64_t i : boundary) mask[i>>3] |= (uint8_t)(1u<<(i%8));
      }
      data.Nb=boundary.size(); data.Nbl=lossy.size();
      // Include ABC edges/corners to exercise halo updates before air/ABC.
      for (int x=1; x<data.Nx-1; ++x)
         for (int y=1; y<data.Ny-1; ++y)
            for (int z=1; z<data.Nz-1; ++z) {
               int q=(x==1 || x==data.Nx-2)+(y==1 || y==data.Ny-2)
                     +(z==1 || z==data.Nz-2);
               if (q) { abc.push_back(index(x,y,z)); abc_order.push_back((int8_t)q); }
            }
      data.Nba=abc.size();
      sources={index(3,3,3),index(6,3,4)};
      receivers={sources[0],sources[0],sources[1],index(7,4,5)};
      data.Ns=sources.size(); data.Nr=receivers.size();
      input.resize(data.Ns*steps);
      for (int64_t n=0; n<steps; ++n) {
         input[n]=(n%13==0) ? 0.0078125 : 0.0;
         input[steps+n]=(n%7==0) ? -0.00390625 : 0.0;
      }
      output.resize(data.Nr*steps);
      // Non-null backing storage is required even for zero-length upload paths.
      if (boundary.empty()) boundary.resize(1);
      if (lossy.empty()) lossy.resize(1);
      if (surface.empty()) surface.resize(1);
      if (material.empty()) material.resize(1);
      if (neighbours.empty()) neighbours.resize(1);
      if (adjacency.empty()) adjacency.resize(1);
      data.bn_ixyz=boundary.data(); data.bnl_ixyz=lossy.data();
      data.bna_ixyz=abc.data(); data.Q_bna=abc_order.data();
      data.in_ixyz=sources.data(); data.out_ixyz=receivers.data();
      data.adj_bn=adjacency.data(); data.K_bn=neighbours.data();
      data.mat_bnl=material.data(); data.Mb=poles.data();
      data.bn_mask=mask.data(); data.ssaf_bnl=surface.data();
      data.mat_beta=beta.data(); data.mat_quads=quads.data();
      data.in_sigs=input.data(); data.u_out=output.data();
   }
};

static void compare_case(bool fcc, bool ade, int64_t steps) {
   SchedulerFixture fixture(fcc,ade,steps);
   if (!pffdtd::scheduler_receivers_interior(fixture.data.out_ixyz,fixture.data.Nr,
          fixture.data.Nx,fixture.data.Ny,fixture.data.Nz)) {
      std::fprintf(stderr,"FAIL: fixture receivers would bypass the async scheduler\n");
      std::exit(EXIT_FAILURE);
   }
   setenv("PFFDTD_ASYNC","0",1);
   run_sim(&fixture.data);
   const std::vector<double> expected=fixture.output;
   std::fill(fixture.output.begin(),fixture.output.end(),0.0);
   setenv("PFFDTD_ASYNC","1",1);
   run_sim(&fixture.data);
   bool nonzero=false;
   for (size_t i=0; i<expected.size(); ++i) {
      if (!std::isfinite(expected[i]) || !std::isfinite(fixture.output[i]) ||
          std::memcmp(&expected[i],&fixture.output[i],sizeof(double)) != 0) {
         std::fprintf(stderr,"FAIL scheduler FP%d FCC=%d ADE=%d Nt=%lld sample=%zu\n",
                      (int)(8*sizeof(Real)),fcc,ade,(long long)steps,i);
         std::exit(EXIT_FAILURE);
      }
      nonzero |= expected[i] != 0.0;
   }
   if (steps>1 && !nonzero) {
      std::fprintf(stderr,"FAIL: full-engine fixture produced only zeros\n");
      std::exit(EXIT_FAILURE);
   }
}

int main() {
#ifdef SCHEDULER_SYNC
   std::fprintf(stderr,"scheduler_cuda_check: SKIP (compiled with SCHEDULER_SYNC)\n");
   return 77;
#endif
   int devices=0;
   const cudaError_t status=cudaGetDeviceCount(&devices);
   if (status != cudaSuccess || devices==0) {
      std::fprintf(stderr,"scheduler_cuda_check: SKIP (no usable CUDA device)\n");
      return 77;
   }
   if (devices != 1) {
      std::fprintf(stderr,"Select exactly one GPU with CUDA_VISIBLE_DEVICES\n");
      return EXIT_FAILURE;
   }
   const char *old_async=getenv("PFFDTD_ASYNC");
   const bool had_async=old_async != nullptr;
   const std::string previous_async=had_async ? old_async : "";
   const char *old_progress=getenv("PFFDTD_PROGRESS");
   const bool had_progress=old_progress != nullptr;
   const std::string previous_progress=had_progress ? old_progress : "";
   setenv("PFFDTD_PROGRESS","0",1);
   const int64_t lengths[]={1,2,3,5,6,7,511,512,513,1025};
   int cases=0;
   for (bool fcc : {false,true})
      for (bool ade : {false,true})
         for (int64_t steps : lengths) { compare_case(fcc,ade,steps); ++cases; }
   if (had_async) setenv("PFFDTD_ASYNC",previous_async.c_str(),1);
   else unsetenv("PFFDTD_ASYNC");
   if (had_progress) setenv("PFFDTD_PROGRESS",previous_progress.c_str(),1);
   else unsetenv("PFFDTD_PROGRESS");
   std::printf("scheduler_cuda_check: PASS FP%d, %d full-engine cases, exact outputs\n",
               (int)(8*sizeof(Real)),cases);
   return EXIT_SUCCESS;
}
