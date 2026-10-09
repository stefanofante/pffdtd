#ifndef _GRAPH_PLAN_H
#define _GRAPH_PLAN_H

#include <stdint.h>

namespace pffdtd {

struct GraphChunkPlan {
   int64_t steps;
   int phase;
   bool reusable;
};

// Bulk pointers repeat every two steps and ADE carry pointers every three.
// Replaying a fixed-address graph therefore requires a multiple of six. The
// dynamic device counter supplies its timestep and readout column, but no
// graph may cross a host readout drain or the simulation's final timestep.
// A scalar tail closes an otherwise incomplete six-step rotation.
static inline GraphChunkPlan graph_chunk_plan(int64_t n, int64_t Nt,
                                              int64_t readout_block,
                                              int64_t max_graph_steps = 96)
{
   GraphChunkPlan plan = {0, 0, false};
   if (n < 0 || Nt <= n || readout_block <= 0 || max_graph_steps <= 0)
      return plan;

   plan.phase = (int)(n % 6);
   const int64_t block_remaining = readout_block - n % readout_block;
   const int64_t total_remaining = Nt - n;
   const int64_t available = total_remaining < block_remaining
      ? total_remaining : block_remaining;
   const int64_t large_steps = max_graph_steps - max_graph_steps % 6;

   if (large_steps >= 6 && available >= large_steps) {
      plan.steps = large_steps;
      plan.reusable = true;
   }
   else if (max_graph_steps >= 6 && available >= 6) {
      plan.steps = 6;
      plan.reusable = true;
   }
   else {
      plan.steps = 1;
   }
   return plan;
}

} // namespace pffdtd
#endif
