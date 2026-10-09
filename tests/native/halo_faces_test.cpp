// Native regression for fused mirror halos. Also builds with nvcc -x cu to
// validate the same index mapping on a CUDA device without a Python fixture.
#include "halo_faces.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace {

struct Shape { int nx, ny, nz; };
struct Copy { std::size_t dst, src; unsigned face; };

void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr, "FAIL: %s\n", message);
      std::exit(EXIT_FAILURE);
   }
}

std::size_t index(const Shape &s, int x, int y, int z)
{
   return (static_cast<std::size_t>(z)*s.ny + y)*s.nx + x;
}

// Independent reference: execute whole mirror planes in original launch order.
template<typename Real>
void mirror(std::vector<Real> &u, const Shape &s, unsigned axis, bool end)
{
   const int sizes[] = {s.nx, s.ny, s.nz};
   const int dst = end ? sizes[axis]-1 : 0;
   const int src = end ? sizes[axis]-3 : 2;
   for (int z=0; z<s.nz; ++z)
      for (int y=0; y<s.ny; ++y)
         for (int x=0; x<s.nx; ++x) {
            int p[] = {x, y, z};
            if (p[axis] != dst) continue;
            p[axis] = src;
            u[index(s,x,y,z)] = u[index(s,p[0],p[1],p[2])];
         }
}

template<typename Real>
void prefix(std::vector<Real> &u, const Shape &s, bool fcc, unsigned ends)
{
   if (fcc)
      for (int z=0; z<s.nz; ++z)
         for (int x=0; x<s.nx; ++x)
            u[index(s,x,s.ny-1,z)] = u[index(s,x,s.ny-2,z)];
   if (ends & 1) mirror(u,s,2,false);
   if (ends & 2) mirror(u,s,2,true);
}

template<typename Real>
void ordered_faces(std::vector<Real> &u, const Shape &s, bool fcc)
{
   mirror(u,s,1,false);
   if (!fcc) mirror(u,s,1,true);
   mirror(u,s,0,false);
   mirror(u,s,0,true);
}

template<typename Real>
void same_bits(const std::vector<Real> &expected, const std::vector<Real> &actual)
{
   require(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(Real)) == 0,
           "fused halos differ from ordered faces");
}

template<typename Idx>
std::vector<Copy> copies(const Shape &s)
{
   const Idx nx=s.nx, ny=s.ny, nz=s.nz, nxny=nx*ny;
   std::vector<Copy> ops;
   const int amax=std::max(s.nx,s.ny);
   // Include inactive threads in partially occupied 16x8 CUDA blocks.
   for (unsigned face=0; face<3; ++face)
      for (int b=0; b<((s.nz+7)/8)*8; ++b)
         for (int a=0; a<((amax+15)/16)*16; ++a) {
            Idx dst=0, src=0;
            if (pffdtd::halo_face_indices(nx,ny,nz,nxny,static_cast<Idx>(a),
                                         static_cast<Idx>(b),face,dst,src))
               ops.push_back(Copy{static_cast<std::size_t>(dst),
                                  static_cast<std::size_t>(src),face});
         }
   const std::size_t count=static_cast<std::size_t>(s.nx)*s.ny*s.nz;
   std::vector<unsigned char> written(count,0);
   for (const Copy &op : ops) {
      require(op.dst < count && op.src < count,"halo index outside grid");
      require(!written[op.dst],"multiple threads write a fused halo cell");
      written[op.dst]=1;
   }
   for (const Copy &op : ops)
      require(!written[op.src],"fused halo source is written by another thread");
   require(ops.size() == static_cast<std::size_t>(s.nz)*(s.nx-2+2*s.ny),
           "fused launch does not cover all owned cells");
   return ops;
}

#ifdef __CUDACC__
void cuda_check(cudaError_t error)
{
   if (error != cudaSuccess) {
      std::fprintf(stderr,"CUDA failure: %s\n",cudaGetErrorString(error));
      std::exit(EXIT_FAILURE);
   }
}

template<typename Real, typename Idx>
__global__ void fused_device(Real *u, Idx nx, Idx ny, Idx nz)
{
   const Idx a=blockIdx.x*blockDim.x+threadIdx.x;
   const Idx b=blockIdx.y*blockDim.y+threadIdx.y;
   Idx dst, src;
   if (pffdtd::halo_face_indices(nx,ny,nz,static_cast<Idx>(nx*ny),a,b,
                                blockIdx.z,dst,src))
      u[dst]=u[src];
}

template<typename Real, typename Idx>
void device_compare(const std::vector<Real> &base, const std::vector<Real> &expected,
                    const Shape &s)
{
   Real *device=nullptr;
   const std::size_t bytes=base.size()*sizeof(Real);
   cuda_check(cudaMalloc(reinterpret_cast<void **>(&device),bytes));
   cuda_check(cudaMemcpy(device,base.data(),bytes,cudaMemcpyHostToDevice));
   const dim3 block(16,8,1);
   const dim3 grid((std::max(s.nx,s.ny)+15)/16,(s.nz+7)/8,3);
   fused_device<Real,Idx><<<grid,block>>>(device,s.nx,s.ny,s.nz);
   cuda_check(cudaGetLastError());
   std::vector<Real> actual(base.size());
   cuda_check(cudaMemcpy(actual.data(),device,bytes,cudaMemcpyDeviceToHost));
   cuda_check(cudaFree(device));
   same_bits(expected,actual);
}
#endif

template<typename Real, typename Idx>
void check(const Shape &s, bool fcc, unsigned ends, unsigned seed)
{
   const std::size_t count=static_cast<std::size_t>(s.nx)*s.ny*s.nz;
   std::vector<Real> original(count);
   std::mt19937 random(seed);
   for (Real &value : original)
      value=static_cast<Real>(static_cast<int>(random()%1048576)-524288)/Real(32);
   prefix(original,s,fcc,ends);
   std::vector<Real> expected=original;
   ordered_faces(expected,s,fcc);

   const bool fusable=pffdtd::halo_faces_fusable(s.nx,s.ny);
   require(fusable == (s.nx>=4 && s.ny>=4),"incorrect small-axis fallback");
   if (!fusable) {
      // The runtime must keep Ybeg before Yend, and Xbeg before Xend, here.
      ordered_faces(original,s,fcc);
      same_bits(expected,original);
      return;
   }
   if (!fcc) mirror(original,s,1,true); // conditional face preceding fusion
   std::vector<Copy> ops=copies<Idx>(s);
   std::array<unsigned,3> faces={{0,1,2}};
   do {
      std::vector<Real> actual=original;
      for (unsigned face : faces)
         for (const Copy &op : ops)
            if (op.face == face) actual[op.dst]=actual[op.src];
      same_bits(expected,actual);
   } while (std::next_permutation(faces.begin(),faces.end()));
   // Interleave individual threads, in addition to all six complete-face orders.
   for (unsigned attempt=0; attempt<4; ++attempt) {
      std::shuffle(ops.begin(),ops.end(),random);
      std::vector<Real> actual=original;
      for (const Copy &op : ops) actual[op.dst]=actual[op.src];
      same_bits(expected,actual);
   }
#ifdef __CUDACC__
   device_compare<Real,Idx>(original,expected,s);
#endif
}

// Demonstrate sensitivity: the former y=0 source races with the Ybeg face.
void check_former_race()
{
   const Shape s={8,8,4};
   std::vector<double> expected(8*8*4);
   for (std::size_t i=0; i<expected.size(); ++i) expected[i]=static_cast<double>(i+1);
   std::vector<double> reversed=expected;
   mirror(expected,s,1,false);
   mirror(expected,s,0,false);
   mirror(expected,s,0,true);
   mirror(reversed,s,0,false);
   mirror(reversed,s,0,true);
   // Old fused Ybeg skipped X edges: preserve their earlier, stale copy.
   for (int z=0; z<s.nz; ++z)
      for (int x=1; x<s.nx-1; ++x)
         reversed[index(s,x,0,z)]=reversed[index(s,x,2,z)];
   require(expected[0] != reversed[0],"fixture must expose former RAW dependency");
}

void check_large_indices()
{
   // Exercise address arithmetic near/beyond the 32-bit dispatch limit without
   // allocating multi-gigabyte fields. The int64-only grid exceeds INT32_MAX.
   const Shape shapes[]={{65535,17,1927},{65535,17,3000}};
   for (const Shape &s : shapes) {
      const int64_t nx=s.nx, ny=s.ny, nz=s.nz, nxny=nx*ny, count=nxny*nz;
      const int64_t as[]={0,1,nx-2,nx-1,ny-1,std::max(nx,ny)};
      const int64_t bs[]={0,nz-1,nz};
      for (int64_t a : as)
         for (int64_t b : bs)
            for (unsigned face=0; face<4; ++face) {
               int64_t dst64=0,src64=0;
               const bool active=pffdtd::halo_face_indices(nx,ny,nz,nxny,a,b,
                                                           face,dst64,src64);
               if (active) {
                  require(dst64>=0 && dst64<count && src64>=0 && src64<count,
                          "large-grid index outside grid");
                  const int64_t x=src64%nx, y=(src64/nx)%ny;
                  require(x>0 && x<nx-1 && y>0,
                          "large-grid source lies in fused write set");
               }
               if (count<INT32_MAX) {
                  int32_t dst32=0,src32=0;
                  const bool active32=pffdtd::halo_face_indices(
                     static_cast<int32_t>(nx),static_cast<int32_t>(ny),
                     static_cast<int32_t>(nz),static_cast<int32_t>(nxny),
                     static_cast<int32_t>(a),static_cast<int32_t>(b),
                     face,dst32,src32);
                  require(active==active32 && (!active || (dst64==dst32 && src64==src32)),
                          "int32 and int64 index paths disagree");
               }
            }
   }
}

} // namespace

int main()
{
#ifdef __CUDACC__
   int devices=0;
   const cudaError_t status=cudaGetDeviceCount(&devices);
   if (status != cudaSuccess || devices == 0) {
      std::fprintf(stderr,"SKIP: CUDA regression requires a visible NVIDIA device\n");
      return 77;
   }
#endif
   check_former_race();
   check_large_indices();
   const Shape shapes[]={{3,3,3},{3,5,7},{5,3,7},{4,4,3},{5,7,3},
                         {8,8,4},{17,19,9},{33,5,17},{5,33,17},{65,31,9}};
   unsigned cases=0;
   for (const Shape &shape : shapes)
      for (unsigned fcc=0; fcc<2; ++fcc)
         for (unsigned ends=0; ends<4; ++ends)
            for (unsigned seed=0; seed<2; ++seed) {
               check<float,int32_t>(shape,fcc!=0,ends,seed);
               check<float,int64_t>(shape,fcc!=0,ends,seed);
               check<double,int32_t>(shape,fcc!=0,ends,seed);
               check<double,int64_t>(shape,fcc!=0,ends,seed);
               cases+=4;
            }
   std::printf("PASS: %u halo cases, FP32/FP64, int32/int64, Cart/FCC, "
               "slab ends, face permutations, shuffled threads, disjoint sources\n",cases);
   return EXIT_SUCCESS;
}
