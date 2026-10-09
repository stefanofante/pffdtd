///////////////////////////////////////////////////////////////////////////////
// Fused mirror-halo index mapping, shared by the CUDA kernel and native tests.
///////////////////////////////////////////////////////////////////////////////

#ifndef PFFDTD_HALO_FACES_H
#define PFFDTD_HALO_FACES_H

#ifdef __CUDACC__
#define PFFDTD_HALO_HD __host__ __device__
#else
#define PFFDTD_HALO_HD
#endif

namespace pffdtd {

// For a three-cell axis the mirror source is the opposite halo. Keep the
// ordered launches for those grids: their source values depend on earlier faces.
template<typename Idx>
PFFDTD_HALO_HD inline bool halo_faces_fusable(Idx Nx, Idx Ny)
{
   return Nx >= 4 && Ny >= 4;
}

// Map one thread in the fused (Ybeg, Xbeg, Xend) launch to a single copy.
// All destination indices are unique and every source is outside that write
// set. At y=0, compose the Y and X mirrors instead of reading the Y halo being
// written by another block. The caller uses this mapping only for fusable axes.
template<typename Idx>
PFFDTD_HALO_HD inline bool halo_face_indices(Idx Nx, Idx Ny, Idx Nz, Idx NxNy,
                                            Idx a, Idx b, unsigned face,
                                            Idx &dst, Idx &src)
{
   if (b >= Nz) return false;
   if (face == 0) {
      if (a < 1 || a >= Nx-1) return false;
      dst = b*NxNy + a;
      src = dst + 2*Nx;
   }
   else if (face == 1) {
      if (a >= Ny) return false;
      dst = b*NxNy + a*Nx;
      src = dst + 2 + ((a == 0) ? 2*Nx : 0);
   }
   else if (face == 2) {
      if (a >= Ny) return false;
      dst = b*NxNy + a*Nx + (Nx-1);
      src = dst - 2 + ((a == 0) ? 2*Nx : 0);
   }
   else return false;
   return true;
}

} // namespace pffdtd

#undef PFFDTD_HALO_HD
#endif
