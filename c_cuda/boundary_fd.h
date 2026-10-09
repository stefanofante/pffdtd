// Frequency-dependent boundary update shared by the CUDA path and native checks.
// Real, MMb and struct MatQuad are provided by fdtd_data.h.
#ifndef _BOUNDARY_FD_H
#define _BOUNDARY_FD_H

#include <stdint.h>

#ifdef __CUDACC__
#define PFFDTD_BOUNDARY_INLINE static __host__ __device__ __forceinline__
#else
#define PFFDTD_BOUNDARY_INLINE static inline
#endif

// Accept the rigid-stencil pressure directly so it can stay in a register.
// Each lossy node owns its carry slot and pole-major ADE states. Keep the three
// carry buffers: u2b is needed after the previous grid pressure is overwritten.
PFFDTD_BOUNDARY_INLINE Real boundary_fd_value_node(
      Real u0bint, Real *u0b, const Real *u2b, Real *vh1, Real *gh1,
      const Real *ssaf_bnl, const int8_t *mat_bnl,
      const Real *mat_beta, const struct MatQuad *mat_quads,
      const int8_t *Mb, Real lo2, int64_t Nbl, int64_t nb)
{
   Real _1 = 1.0;
   Real _2 = 2.0;
   int32_t k = mat_bnl[nb];
   Real ssaf = ssaf_bnl[nb];
   Real lo2Kbg = lo2*ssaf*mat_beta[k];
   Real fac = _2*lo2*ssaf / (_1 + lo2Kbg);

   Real u2bint = u2b[nb];
   u0bint = (u0bint + lo2Kbg*u2bint) / (_1 + lo2Kbg);

   Real vh1int[MMb];
   Real gh1int[MMb];
   for (int8_t m=0; m<Mb[k]; m++) {
      int64_t nbm = m*Nbl + nb;
      int32_t mbk = k*MMb+m;
      const struct MatQuad *tm = &(mat_quads[mbk]);
      vh1int[m] = vh1[nbm];
      gh1int[m] = gh1[nbm];
      u0bint -= fac*( _2*(tm->bDh)*vh1int[m] - (tm->bFh)*gh1int[m] );
   }

   Real du = u0bint-u2bint;
   for (int8_t m=0; m<Mb[k]; m++) {
      int64_t nbm = m*Nbl + nb;
      int32_t mbk = k*MMb+m;
      const struct MatQuad *tm = &(mat_quads[mbk]);
      Real vh0m = (tm->b)*du + (tm->bd)*vh1int[m] - _2*(tm->bFh)*gh1int[m];
      gh1[nbm] = gh1int[m] + (vh0m + vh1int[m])/_2;
      vh1[nbm] = vh0m;
   }
   u0b[nb] = u0bint;
   return u0bint;
}

// Existing gather/ADE/scatter fusion: pressure is read from the completed grid.
PFFDTD_BOUNDARY_INLINE void boundary_fd_grid_node(
      Real *u0, Real *u0b, const Real *u2b, Real *vh1, Real *gh1,
      const int64_t *bnl_ixyz, const Real *ssaf_bnl, const int8_t *mat_bnl,
      const Real *mat_beta, const struct MatQuad *mat_quads,
      const int8_t *Mb, Real lo2, int64_t Nbl, int64_t nb)
{
   const int64_t ii = bnl_ixyz[nb];
   u0[ii] = boundary_fd_value_node(u0[ii],u0b,u2b,vh1,gh1,ssaf_bnl,mat_bnl,
                                  mat_beta,mat_quads,Mb,lo2,Nbl,nb);
}

#undef PFFDTD_BOUNDARY_INLINE
#endif
