// Fixed-pole ADE variants retain the original reduction/update order. Their
// compile-time bounds let nvcc scalarize the pressure-history arrays; callers
// must measure the resulting register/occupancy tradeoff on each target.
#ifndef PFFDTD_BOUNDARY_FD_SPECIALIZED_H
#define PFFDTD_BOUNDARY_FD_SPECIALIZED_H

#include <boundary_fd.h>

#ifdef __CUDACC__
#define PFFDTD_FIXED_INLINE static __host__ __device__ __forceinline__
#else
#define PFFDTD_FIXED_INLINE static inline
#endif

template<int Poles>
PFFDTD_FIXED_INLINE Real boundary_fd_fixed_value_node(
      Real u0bint, Real *u0b, const Real *u2b, Real *vh1, Real *gh1,
      const Real *ssaf_bnl, const int8_t *mat_bnl,
      const Real *mat_beta, const struct MatQuad *mat_quads,
      Real lo2, int64_t Nbl, int64_t nb)
{
   static_assert(Poles >= 0 && Poles <= MMb, "fixed ADE pole count is outside its state layout");
   Real _1 = 1.0;
   Real _2 = 2.0;
   int32_t k = mat_bnl[nb];
   Real ssaf = ssaf_bnl[nb];
   Real lo2Kbg = lo2*ssaf*mat_beta[k];
   Real fac = _2*lo2*ssaf / (_1 + lo2Kbg);
   Real u2bint = u2b[nb];
   u0bint = (u0bint + lo2Kbg*u2bint) / (_1 + lo2Kbg);

   // Keep only the used poles. A one-element placeholder makes Poles=0 valid
   // C++; both loops are empty and its arrays disappear from generated code.
   Real vh1int[Poles ? Poles : 1];
   Real gh1int[Poles ? Poles : 1];
#ifdef __CUDA_ARCH__
   #pragma unroll
#endif
   for (int m=0; m<Poles; ++m) {
      int64_t nbm = m*Nbl + nb;
      int32_t mbk = k*MMb+m;
      const struct MatQuad *tm = &(mat_quads[mbk]);
      vh1int[m] = vh1[nbm];
      gh1int[m] = gh1[nbm];
      u0bint -= fac*( _2*(tm->bDh)*vh1int[m] - (tm->bFh)*gh1int[m] );
   }

   Real du = u0bint-u2bint;
#ifdef __CUDA_ARCH__
   #pragma unroll
#endif
   for (int m=0; m<Poles; ++m) {
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

// Dynamic two-pass fallback keeps only scalar histories. The first pass never
// modifies ADE states, so rereading the same pole in the update pass retrieves
// the original value. Each node owns distinct pole-major slots. This trades an
// extra global read per history for removing both dynamically indexed arrays.
PFFDTD_FIXED_INLINE Real boundary_fd_scalar_value_node(
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
   for (int8_t m=0; m<Mb[k]; ++m) {
      int64_t nbm = m*Nbl + nb;
      int32_t mbk = k*MMb+m;
      const struct MatQuad *tm = &(mat_quads[mbk]);
      Real vh1int = vh1[nbm];
      Real gh1int = gh1[nbm];
      u0bint -= fac*( _2*(tm->bDh)*vh1int - (tm->bFh)*gh1int );
   }
   Real du = u0bint-u2bint;
   for (int8_t m=0; m<Mb[k]; ++m) {
      int64_t nbm = m*Nbl + nb;
      int32_t mbk = k*MMb+m;
      const struct MatQuad *tm = &(mat_quads[mbk]);
      Real vh1int = vh1[nbm];
      Real gh1int = gh1[nbm];
      Real vh0m = (tm->b)*du + (tm->bd)*vh1int - _2*(tm->bFh)*gh1int;
      gh1[nbm] = gh1int + (vh0m + vh1int)/_2;
      vh1[nbm] = vh0m;
   }
   u0b[nb] = u0bint;
   return u0bint;
}

// This matches boundary_fd_value_node's API. Fixed branches preserve global
// read counts; all other valid pole counts use the scalar two-pass fallback.
// Inspect the combined kernel's register count as well as its stack/spills.
PFFDTD_FIXED_INLINE Real boundary_fd_specialized_value_node(
      Real pressure, Real *u0b, const Real *u2b, Real *vh1, Real *gh1,
      const Real *ssaf_bnl, const int8_t *mat_bnl,
      const Real *mat_beta, const struct MatQuad *mat_quads,
      const int8_t *Mb, Real lo2, int64_t Nbl, int64_t nb)
{
   switch (Mb[mat_bnl[nb]]) {
      case 0: return boundary_fd_fixed_value_node<0>(pressure,u0b,u2b,vh1,gh1,
                     ssaf_bnl,mat_bnl,mat_beta,mat_quads,lo2,Nbl,nb);
      case 1: return boundary_fd_fixed_value_node<1>(pressure,u0b,u2b,vh1,gh1,
                     ssaf_bnl,mat_bnl,mat_beta,mat_quads,lo2,Nbl,nb);
      case 11: return boundary_fd_fixed_value_node<11>(pressure,u0b,u2b,vh1,gh1,
                     ssaf_bnl,mat_bnl,mat_beta,mat_quads,lo2,Nbl,nb);
      case 12: return boundary_fd_fixed_value_node<12>(pressure,u0b,u2b,vh1,gh1,
                     ssaf_bnl,mat_bnl,mat_beta,mat_quads,lo2,Nbl,nb);
      default: return boundary_fd_scalar_value_node(pressure,u0b,u2b,vh1,gh1,
                     ssaf_bnl,mat_bnl,mat_beta,mat_quads,Mb,lo2,Nbl,nb);
   }
}

#undef PFFDTD_FIXED_INLINE
#endif
