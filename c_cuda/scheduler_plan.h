#ifndef _SCHEDULER_PLAN_H
#define _SCHEDULER_PLAN_H

#include <stdint.h>

namespace pffdtd {

struct SchedulePlan {
   int64_t step, col, ncol, base;
   bool checkpoint, wait_previous_air;
};

static inline SchedulePlan schedule_plan(int64_t n, int64_t Nt, int64_t readout_block)
{
   SchedulePlan plan;
   plan.step = n;
   plan.col = n % readout_block;
   plan.ncol = plan.col + 1;
   plan.base = n - plan.col;
   plan.checkpoint = plan.col == readout_block-1 || n == Nt-1;
   plan.wait_previous_air = n > 0;
   return plan;
}

// Keep the synchronous reference as the default until the CUDA runtime gate is run.
static inline bool scheduler_async_requested(int ngpus, const char *mode)
{
#ifdef SCHEDULER_SYNC
   (void)ngpus;
   (void)mode;
   return false;
#else
   return ngpus == 1 && mode && mode[0] == '1' && mode[1] == '\0';
#endif
}

// Readout may overlap the air stream, which mutates only u1's ghost faces.
// The loader permits arbitrary receiver indices; use the reference for halos.
static inline bool scheduler_receivers_interior(const int64_t *indices, int64_t Nr,
                                                int64_t Nx, int64_t Ny, int64_t Nz)
{
   if (Nr < 0 || (Nr > 0 && !indices) || Nx < 3 || Ny < 3 || Nz < 3 ||
       Ny > INT64_MAX / Nz) return false;
   const int64_t yz_size = Ny*Nz;
   for (int64_t i=0; i<Nr; i++) {
      const int64_t idx = indices[i];
      if (idx < 0) return false;
      const int64_t x = idx / yz_size;
      const int64_t y = (idx % yz_size) / Nz;
      const int64_t z = idx % Nz;
      if (x < 1 || x >= Nx-1 || y < 1 || y >= Ny-1 || z < 1 || z >= Nz-1)
         return false;
   }
   return true;
}

} // namespace pffdtd
#endif
