// Rigid stencil and ADE in one node update. Real/MatQuad and ADD_O/FMA_D are
// supplied by fdtd_data.h/fdtd_common.h; CUDA keeps their exact rounding policy.
#ifndef PFFDTD_BOUNDARY_STENCIL_H
#define PFFDTD_BOUNDARY_STENCIL_H

#include <stdint.h>
#include <boundary_fd.h>
#include <boundary_fd_specialized.h>

#ifdef __CUDACC__
// ADD_O/FMA_D are CUDA device intrinsics, so do not emit a host overload.
#define PFFDTD_STENCIL_INLINE static __device__ __forceinline__
#else
#define PFFDTD_STENCIL_INLINE static inline
#endif

PFFDTD_STENCIL_INLINE Real boundary_rigid_cart_value(
      const Real *u1, Real previous, int64_t ii, uint16_t adj, int8_t neighbours,
      Real sl2, Real a2, int64_t Nx, int64_t NxNy)
{
   Real K = neighbours;
   Real _2 = 2.0;
   Real b1 = (_2-sl2*K);
   Real b2 = a2;
   Real tmp1,tmp2;
   tmp1 = ADD_O((Real)GET_BIT(adj,0)*u1[ii + NxNy],(Real)GET_BIT(adj,1)*u1[ii - NxNy]);
   tmp2 = ADD_O((Real)GET_BIT(adj,2)*u1[ii + Nx],  (Real)GET_BIT(adj,3)*u1[ii - Nx]);
   tmp1 = ADD_O(tmp1,tmp2);
   tmp2 = ADD_O((Real)GET_BIT(adj,4)*u1[ii + 1],   (Real)GET_BIT(adj,5)*u1[ii - 1]);
   tmp1 = ADD_O(tmp1,tmp2);
   return FMA_D(b1,u1[ii],FMA_D(b2,tmp1,-previous));
}

PFFDTD_STENCIL_INLINE Real boundary_rigid_fcc_value(
      const Real *u1, Real previous, int64_t ii, uint16_t adj, int8_t neighbours,
      Real sl2, Real a2, int64_t Nx, int64_t NxNy)
{
   Real K = neighbours;
   Real _2 = 2.0;
   Real b1 = (_2-sl2*K);
   Real b2 = a2;
   Real tmp1,tmp2,tmp3,tmp4;
   tmp1 = ADD_O((Real)GET_BIT(adj,0)*u1[ii + NxNy + Nx],(Real)GET_BIT(adj,1)*u1[ii - NxNy - Nx]);
   tmp2 = ADD_O((Real)GET_BIT(adj,2)*u1[ii + Nx + 1],   (Real)GET_BIT(adj,3)*u1[ii - Nx - 1]);
   tmp1 = ADD_O(tmp1,tmp2);
   tmp3 = ADD_O((Real)GET_BIT(adj,4)*u1[ii + NxNy + 1], (Real)GET_BIT(adj,5)*u1[ii - NxNy - 1]);
   tmp4 = ADD_O((Real)GET_BIT(adj,6)*u1[ii + NxNy - Nx],(Real)GET_BIT(adj,7)*u1[ii - NxNy + Nx]);
   tmp3 = ADD_O(tmp3,tmp4);
   tmp2 = ADD_O((Real)GET_BIT(adj,8)*u1[ii + Nx - 1],   (Real)GET_BIT(adj,9)*u1[ii - Nx + 1]);
   tmp1 = ADD_O(tmp1,tmp2);
   tmp4 = ADD_O((Real)GET_BIT(adj,10)*u1[ii + NxNy - 1],(Real)GET_BIT(adj,11)*u1[ii - NxNy + 1]);
   tmp3 = ADD_O(tmp3,tmp4);
   tmp1 = ADD_O(tmp1,tmp3);
   return FMA_D(b1,u1[ii],FMA_D(b2,tmp1,-previous));
}

// The validated map holds -1 for rigid nodes, or the original lossy ordinal.
// u1 is immutable throughout this launch; every thread owns a unique u0 index,
// and lossy threads also own unique carry/ADE slots. No inter-block order is used.
template<bool FCC, typename LossyIdx, int ADEMode = 0>
PFFDTD_STENCIL_INLINE void boundary_stencil_node(
      Real *u0, const Real *u1, Real *u0b, const Real *u2b, Real *vh1, Real *gh1,
      const int64_t *bn_ixyz, const uint16_t *adj_bn, const int8_t *K_bn,
      const LossyIdx *lossy_map, const Real *ssaf_bnl, const int8_t *mat_bnl,
      const Real *mat_beta, const struct MatQuad *mat_quads, const int8_t *Mb,
      Real sl2, Real a2, Real lo2, int64_t Nx, int64_t NxNy, int64_t Nbl, int64_t nb)
{
   static_assert(ADEMode>=0 && ADEMode<=2,"unknown ADE implementation");
   const int64_t ii = bn_ixyz[nb];
   Real pressure;
   if (FCC)
      pressure = boundary_rigid_fcc_value(u1,u0[ii],ii,adj_bn[nb],K_bn[nb],sl2,a2,Nx,NxNy);
   else
      pressure = boundary_rigid_cart_value(u1,u0[ii],ii,adj_bn[nb],K_bn[nb],sl2,a2,Nx,NxNy);
   const int64_t lossy = lossy_map[nb];
   if (lossy >= 0) {
      if (ADEMode==1)
         pressure = boundary_fd_scalar_value_node(pressure,u0b,u2b,vh1,gh1,ssaf_bnl,mat_bnl,
                                                   mat_beta,mat_quads,Mb,lo2,Nbl,lossy);
      else if (ADEMode==2)
         pressure = boundary_fd_specialized_value_node(pressure,u0b,u2b,vh1,gh1,ssaf_bnl,mat_bnl,
                                                        mat_beta,mat_quads,Mb,lo2,Nbl,lossy);
      else
         pressure = boundary_fd_value_node(pressure,u0b,u2b,vh1,gh1,ssaf_bnl,mat_bnl,
                                           mat_beta,mat_quads,Mb,lo2,Nbl,lossy);
   }
   u0[ii] = pressure;
}

#undef PFFDTD_STENCIL_INLINE
#endif
