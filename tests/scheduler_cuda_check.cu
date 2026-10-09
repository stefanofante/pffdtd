// Full-engine regression for single-device scheduling, using native fixtures.
// Select one device with CUDA_VISIBLE_DEVICES; hardware absence exits with 77.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

   SchedulerFixture(bool fcc, int boundary_case, int64_t steps) : data{} {
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
      data.Nm=MMb+1;
      for (int k=0; k<data.Nm; ++k) poles.push_back((int8_t)k);
      beta.assign(data.Nm,(Real)0.125);
      quads.resize(data.Nm*MMb);
      for (int k=0; k<data.Nm; ++k)
         for (int m=0; m<MMb; ++m) {
            MatQuad &quad=quads[k*MMb+m];
            quad.b=(Real)((m+1)*0.0009765625);
            quad.bd=(Real)0.5;
            quad.bDh=(Real)((m+1)*(k%3+1)*0.001953125);
            quad.bFh=(Real)0.0009765625;
         }
      mask.assign((data.Npts+7)/8,0);
      if (boundary_case != 0) {
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
         for (size_t i=0; i<boundary.size(); ++i)
            if (boundary_case==1 || i%3!=0) {
               lossy.push_back(boundary[i]);
               surface.push_back((Real)0.75);
               material.push_back((int8_t)(i%data.Nm));
            }
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
      receivers={sources[0],sources[0],sources[1],index(4,3,3),index(5,3,3),index(7,4,5)};
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

struct SavedEnvironment {
   const char *name;
   bool present;
   std::string value;
   explicit SavedEnvironment(const char *key) : name(key), present(getenv(key)!=nullptr),
         value(present ? getenv(key) : "") {}
   ~SavedEnvironment() {
      if (present) setenv(name,value.c_str(),1);
      else unsetenv(name);
   }
};

struct SolverMode { const char *name, *async, *graphs, *boundary, *ade; };

static int compare_case(bool fcc, int boundary_case, int64_t steps) {
   SchedulerFixture fixture(fcc,boundary_case,steps);
   if (!pffdtd::scheduler_receivers_interior(fixture.data.out_ixyz,fixture.data.Nr,
          fixture.data.Nx,fixture.data.Ny,fixture.data.Nz)) {
      std::fprintf(stderr,"FAIL: fixture receivers would bypass the async scheduler\n");
      std::exit(EXIT_FAILURE);
   }
   setenv("PFFDTD_ASYNC","0",1);
   setenv("PFFDTD_GRAPHS","0",1);
   setenv("PFFDTD_BOUNDARY_FUSED","0",1);
   setenv("PFFDTD_ADE_MODE","generic",1);
   std::fill(fixture.output.begin(),fixture.output.end(),std::numeric_limits<double>::quiet_NaN());
   run_sim(&fixture.data);
   const std::vector<double> expected=fixture.output;
   bool nonzero=false;
   for (size_t i=0; i<expected.size(); ++i) {
      if (!std::isfinite(expected[i])) {
         std::fprintf(stderr,"FAIL reference FP%d FCC=%d boundary=%d Nt=%lld sample=%zu\n",
                      (int)(8*sizeof(Real)),fcc,boundary_case,(long long)steps,i);
         std::exit(EXIT_FAILURE);
      }
      nonzero |= expected[i] != 0.0;
   }
   if (steps>1 && !nonzero) {
      std::fprintf(stderr,"FAIL: full-engine fixture produced only zeros\n");
      std::exit(EXIT_FAILURE);
   }
   const SolverMode modes[]={
      {"events","1","0","0","generic"}, {"graphs","0","1","0","generic"},
      {"fused-sync","0","0","1","generic"}, {"fused-events","1","0","1","generic"},
      {"fused-graphs","0","1","1","generic"},
      {"reload-sync","0","0","1","reload"}, {"reload-events","1","0","1","reload"},
      {"reload-graphs","0","1","1","reload"},
      {"fixed-sync","0","0","1","fixed"}, {"fixed-events","1","0","1","fixed"},
      {"fixed-graphs","0","1","1","fixed"}
   };
   int comparisons=0;
   for (const SolverMode &mode : modes) {
#ifdef BOUNDARY_SEPARATE
      if (strcmp(mode.boundary,"1")==0) continue;
#endif
      std::fill(fixture.output.begin(),fixture.output.end(),std::numeric_limits<double>::quiet_NaN());
      setenv("PFFDTD_ASYNC",mode.async,1);
      setenv("PFFDTD_GRAPHS",mode.graphs,1);
      setenv("PFFDTD_BOUNDARY_FUSED",mode.boundary,1);
      setenv("PFFDTD_ADE_MODE",mode.ade,1);
      run_sim(&fixture.data);
      for (size_t i=0; i<expected.size(); ++i)
         if (!std::isfinite(fixture.output[i]) ||
             std::memcmp(&expected[i],&fixture.output[i],sizeof(double)) != 0) {
            std::fprintf(stderr,"FAIL %s FP%d FCC=%d boundary=%d Nt=%lld sample=%zu\n",
                         mode.name,(int)(8*sizeof(Real)),fcc,boundary_case,
                         (long long)steps,i);
            std::exit(EXIT_FAILURE);
         }
      ++comparisons;
   }
   return comparisons;
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
   SavedEnvironment saved_async("PFFDTD_ASYNC"), saved_graphs("PFFDTD_GRAPHS"),
      saved_boundary("PFFDTD_BOUNDARY_FUSED"), saved_ade("PFFDTD_ADE_MODE"), saved_progress("PFFDTD_PROGRESS");
   setenv("PFFDTD_PROGRESS","0",1);
   const int64_t lengths[]={1,2,3,5,6,7,95,96,97,511,512,513,1025,1536,1537,3073};
   int cases=0,comparisons=0;
   for (bool fcc : {false,true})
      for (int boundary_case : {0,1,2})
         for (int64_t steps : lengths) {
            comparisons+=compare_case(fcc,boundary_case,steps);
            ++cases;
         }
   std::printf("scheduler_cuda_check: PASS FP%d, %d full-engine cases, %d exact comparisons\n",
               (int)(8*sizeof(Real)),cases,comparisons);
   return EXIT_SUCCESS;
}
