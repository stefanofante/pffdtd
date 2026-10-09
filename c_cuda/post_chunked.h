#ifndef PFFDTD_POST_CHUNKED_H
#define PFFDTD_POST_CHUNKED_H
#include "post_dsp.h"
#ifdef __CUDACC__
#define PFFDTD_CHUNK_HD __host__ __device__
#else
#define PFFDTD_CHUNK_HD
#endif

namespace pffdtd_post {
struct PostState { double z1,z2; };
struct PostMatrix { double a,b,c,d; };

PFFDTD_CHUNK_HD inline PostMatrix post_matrix_product(PostMatrix x, PostMatrix y)
{
   return {x.a*y.a+x.b*y.c,x.a*y.b+x.b*y.d,
           x.c*y.a+x.d*y.c,x.c*y.b+x.d*y.d};
}

PFFDTD_CHUNK_HD inline PostState post_matrix_apply(PostMatrix x, PostState y)
{ return {x.a*y.z1+x.b*y.z2,x.c*y.z1+x.d*y.z2}; }

inline PostMatrix post_transition_power(const Sos& section, std::size_t length)
{
   PostMatrix power={-section.a1,1.0,-section.a2,0.0};
   PostMatrix result={1.0,0.0,0.0,1.0};
   while (length) {
      if (length&1) result=post_matrix_product(result,power);
      length>>=1;
      if (length) power=post_matrix_product(power,power);
   }
   return result;
}
} // namespace pffdtd_post
#undef PFFDTD_CHUNK_HD
#endif
