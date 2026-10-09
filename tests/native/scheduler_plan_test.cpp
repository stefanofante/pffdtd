// Native checks for the asynchronous single-GPU scheduler contract.
// The tagged-cell DAG below checks ordering and buffer reuse, not CUDA event
// semantics or floating-point solver equivalence; those require device tests.
#include "scheduler_plan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

void require(bool condition, const char *message)
{
   if (!condition) {
      std::fprintf(stderr, "FAIL: %s\n", message);
      std::exit(EXIT_FAILURE);
   }
}

void check_plan(int64_t nt, int64_t block)
{
   std::vector<unsigned char> copied(static_cast<std::size_t>(nt), 0);
   int64_t next_base = 0;
   int64_t checkpoints = 0;
   for (int64_t n = 0; n < nt; ++n) {
      const pffdtd::SchedulePlan plan = pffdtd::schedule_plan(n, nt, block);
      require(plan.step == n, "plan changed timestep");
      require(plan.col >= 0 && plan.col < block, "readout column outside block");
      require(plan.ncol == plan.col + 1, "filled-column count is incorrect");
      require(plan.base + plan.col == n, "base does not identify column zero");
      require(plan.base % block == 0, "readout base is not block aligned");
      require(plan.wait_previous_air == (n != 0), "previous-air wait is incorrect");
      const bool expected_checkpoint = (n + 1 == nt) || ((n + 1) % block == 0);
      require(plan.checkpoint == expected_checkpoint, "incorrect drain checkpoint");
      if (!plan.checkpoint) continue;
      ++checkpoints;
      require(plan.base == next_base, "drained batches overlap or leave a gap");
      for (int64_t col = 0; col < plan.ncol; ++col) {
         const int64_t step = plan.base + col;
         require(step >= 0 && step < nt, "drain includes an uncomputed timestep");
         require(!copied[static_cast<std::size_t>(step)], "timestep drained twice");
         copied[static_cast<std::size_t>(step)] = 1;
      }
      next_base = n + 1;
   }
   require(next_base == nt, "final partial batch was not drained");
   require(checkpoints == nt / block + (nt % block != 0), "incorrect number of drains");
   require(std::all_of(copied.begin(), copied.end(), [](unsigned char c) { return c == 1; }),
           "readout coverage is incomplete");
}

void check_large_steps()
{
   const int64_t nt = std::numeric_limits<int64_t>::max();
   const int64_t steps[] = {nt - 1, nt - 2, nt - 511, nt - 512, nt - 513};
   for (int64_t n : steps) {
      const pffdtd::SchedulePlan plan = pffdtd::schedule_plan(n, nt, 512);
      require(plan.step == n && plan.col == n % 512, "64-bit step was truncated");
      require(plan.base == n - plan.col && plan.ncol == plan.col + 1,
              "64-bit readout range was truncated");
      require(plan.checkpoint == (plan.col == 511 || n == nt - 1),
              "64-bit final checkpoint is incorrect");
   }
}

void check_mode()
{
   const char *modes[] = {nullptr, "", "0", "1", "01", "true", "on", "yes", "1 ", " 1", "1\n"};
   const int devices[] = {-1, 0, 1, 2, 16};
   for (int ngpus : devices)
      for (const char *mode : modes) {
         bool expected = ngpus == 1 && mode != nullptr && mode[0] == '1' && mode[1] == '\0';
#ifdef SCHEDULER_SYNC
         expected = false;
#endif
         require(pffdtd::scheduler_async_requested(ngpus, mode) == expected,
                 "async opt-in or single-device guard is incorrect");
      }
}

int64_t index(int64_t x, int64_t y, int64_t z, int64_t ny, int64_t nz)
{
   return (x * ny + y) * nz + z;
}

void check_receivers()
{
   const int64_t nx = 7, ny = 9, nz = 11;
   const int64_t inside[] = {index(1, 1, 1, ny, nz), index(nx - 2, ny - 2, nz - 2, ny, nz),
                             index(3, 4, 5, ny, nz), index(1, 1, 1, ny, nz)};
   require(pffdtd::scheduler_receivers_interior(inside, 4, nx, ny, nz),
           "valid interior or duplicate receivers rejected");
   require(pffdtd::scheduler_receivers_interior(nullptr, 0, nx, ny, nz),
           "empty receiver set rejected");
   require(!pffdtd::scheduler_receivers_interior(nullptr, 1, nx, ny, nz),
           "nonempty null receiver set accepted");
   require(!pffdtd::scheduler_receivers_interior(inside, -1, nx, ny, nz),
           "negative receiver count accepted");
   const int64_t outside[] = {
      index(0, 4, 5, ny, nz), index(nx - 1, 4, 5, ny, nz),
      index(3, 0, 5, ny, nz), index(3, ny - 1, 5, ny, nz),
      index(3, 4, 0, ny, nz), index(3, 4, nz - 1, ny, nz),
      index(0, 0, 0, ny, nz), index(nx - 1, ny - 1, nz - 1, ny, nz),
      -1, nx * ny * nz, nx * ny * nz + 1, std::numeric_limits<int64_t>::max()
   };
   for (int64_t invalid : outside) {
      const int64_t receivers[] = {inside[0], invalid, inside[1]};
      require(!pffdtd::scheduler_receivers_interior(receivers, 3, nx, ny, nz),
              "halo or out-of-bounds receiver accepted for async readout");
   }
   for (int64_t size : {-1, 0, 1, 2}) {
      require(!pffdtd::scheduler_receivers_interior(nullptr, 0, size, ny, nz),
              "invalid x dimension accepted");
      require(!pffdtd::scheduler_receivers_interior(nullptr, 0, nx, size, nz),
              "invalid y dimension accepted");
      require(!pffdtd::scheduler_receivers_interior(nullptr, 0, nx, ny, size),
              "invalid z dimension accepted");
   }
   const int64_t maximum = std::numeric_limits<int64_t>::max();
   require(!pffdtd::scheduler_receivers_interior(inside, 1, nx, maximum, nz),
           "y-z stride overflow accepted");
   // Large dimensions with a representable stride and linear index remain
   // meaningful: decomposing the index must not require Nx*Ny*Nz to fit.
   const int64_t huge_receiver = index(1, 1, 1, 3, 3);
   require(pffdtd::scheduler_receivers_interior(&huge_receiver, 1, maximum, 3, 3),
           "valid 64-bit receiver rejected because total grid product overflows");
}

struct Rotation {
   std::array<int, 2> bulk;
   std::array<int, 3> carry;
};

std::vector<Rotation> rotations(int64_t nt)
{
   Rotation next = {{{0, 1}}, {{0, 1, 2}}};
   std::vector<Rotation> result;
   for (int64_t n = 0; n < nt; ++n) {
      result.push_back(next);
      require(next.bulk[0] != next.bulk[1], "bulk input aliases bulk output");
      require(next.carry[0] != next.carry[1] && next.carry[0] != next.carry[2]
              && next.carry[1] != next.carry[2], "boundary carry slots alias");
      if (n >= 2)
         require(next.carry[2] == result[static_cast<std::size_t>(n - 2)].carry[0],
                 "boundary carry does not retain the two-step-old pressure");
      if (n >= 6) {
         require(next.bulk == result[static_cast<std::size_t>(n - 6)].bulk,
                 "two-buffer rotation period is incorrect");
         require(next.carry == result[static_cast<std::size_t>(n - 6)].carry,
                 "three-buffer rotation period is incorrect");
      }
      std::swap(next.bulk[0], next.bulk[1]);
      const int previous_oldest = next.carry[2];
      next.carry[2] = next.carry[1];
      next.carry[1] = next.carry[0];
      next.carry[0] = previous_oldest;
   }
   return result;
}

enum class Kind { Boundary, Air, Read, Drain };

struct Node {
   Kind kind;
   int64_t step;
   std::vector<std::size_t> successors;
   unsigned predecessors;
};

struct Graph {
   std::vector<Node> nodes;
   std::vector<std::array<std::size_t, 3>> step_nodes;
};

void edge(Graph &graph, std::size_t before, std::size_t after)
{
   graph.nodes[before].successors.push_back(after);
   ++graph.nodes[after].predecessors;
}

Graph dependency_graph(int64_t nt, int64_t block, bool previous_air = true,
                       bool previous_read = true)
{
   Graph graph;
   for (int64_t n = 0; n < nt; ++n) {
      const std::size_t base = graph.nodes.size();
      graph.step_nodes.push_back({{base, base + 1, base + 2}});
      graph.nodes.push_back(Node{Kind::Boundary, n, {}, 0});
      graph.nodes.push_back(Node{Kind::Air, n, {}, 0});
      graph.nodes.push_back(Node{Kind::Read, n, {}, 0});
   }
   std::size_t previous_drain = graph.nodes.size();
   for (int64_t n = 0; n < nt; ++n) {
      const auto ids = graph.step_nodes[static_cast<std::size_t>(n)];
      edge(graph, ids[0], ids[1]); // AIR(n) waits for boundary completion.
      edge(graph, ids[0], ids[2]); // Readout follows BN(n) on the boundary stream.
      if (n > 0) {
         const auto previous = graph.step_nodes[static_cast<std::size_t>(n - 1)];
         if (previous_air) edge(graph, previous[1], ids[0]);
         if (previous_read) edge(graph, previous[2], ids[0]);
         edge(graph, previous[0], ids[0]); // Boundary stream order even in mutant DAGs.
         edge(graph, previous[1], ids[1]); // Air stream order.
         if (previous_drain < graph.nodes.size()) {
            edge(graph, previous_drain, ids[0]);
            previous_drain = graph.nodes.size();
         }
      }
      if (pffdtd::schedule_plan(n, nt, block).checkpoint) {
         const std::size_t drain = graph.nodes.size();
         graph.nodes.push_back(Node{Kind::Drain, n, {}, 0});
         edge(graph, ids[2], drain); // D2H follows the final column's readout.
         edge(graph, ids[1], drain); // Checkpoint also waits for the final AIR event.
         previous_drain = drain;
      }
   }
   return graph;
}

// Versions describe three strict-interior receiver cells: an air node, a
// boundary node, and a source/receiver colocated on an air node. Mirror/fold
// halo cells are excluded because AIR(n) may write those in the read buffer.
struct TaggedState {
   std::array<std::array<int64_t, 3>, 2> fields;
   std::array<int64_t, 3> carry;
   std::vector<std::array<int64_t, 3>> readout;
   std::vector<int64_t> host;
   std::vector<unsigned char> done_air;
   std::vector<unsigned char> done_read;
   uint64_t read_before_air;
   uint64_t read_after_air;

   TaggedState(int64_t nt, int64_t block)
      : fields{{{{-1, -1, -1}}, {{0, 0, 0}}}}, carry{{-3, -1, -2}},
        readout(static_cast<std::size_t>(block), {{-99, -99, -99}}),
        host(static_cast<std::size_t>(nt), -99),
        done_air(static_cast<std::size_t>(nt), 0), done_read(static_cast<std::size_t>(nt), 0),
        read_before_air(0), read_after_air(0)
   {}
};

bool execute(const Node &node, TaggedState &state, const std::vector<Rotation> &rotation,
             int64_t nt, int64_t block)
{
   const int64_t n = node.step;
   const Rotation &slots = rotation[static_cast<std::size_t>(n)];
   const int output = slots.bulk[0], input = slots.bulk[1];
   const pffdtd::SchedulePlan plan = pffdtd::schedule_plan(n, nt, block);
   switch (node.kind) {
   case Kind::Boundary:
      if (state.fields[input] != std::array<int64_t, 3>{{n, n, n}} ||
          state.fields[output] != std::array<int64_t, 3>{{n - 1, n - 1, n - 1}} ||
          state.carry[slots.carry[2]] != n - 2) return false;
      state.fields[output][1] = n + 1;
      state.carry[slots.carry[0]] = n;
      return true;
   case Kind::Air:
      if (state.fields[input] != std::array<int64_t, 3>{{n, n, n}} ||
          state.fields[output] != std::array<int64_t, 3>{{n - 1, n + 1, n - 1}})
         return false;
      // Source injection and air/ABC updates are ordered within this node on
      // the air stream. All three physical receiver cells now represent n+1.
      state.fields[output][0] = n + 1;
      state.fields[output][2] = n + 1;
      state.done_air[static_cast<std::size_t>(n)] = 1;
      return true;
   case Kind::Read:
      if (state.fields[input] != std::array<int64_t, 3>{{n, n, n}}) return false;
      state.readout[static_cast<std::size_t>(plan.col)] = state.fields[input];
      state.done_read[static_cast<std::size_t>(n)] = 1;
      if (state.done_air[static_cast<std::size_t>(n)]) ++state.read_after_air;
      else ++state.read_before_air;
      return true;
   case Kind::Drain:
      if (!plan.checkpoint || !state.done_air[static_cast<std::size_t>(n)]) return false;
      for (int64_t col = 0; col < plan.ncol; ++col) {
         const int64_t step = plan.base + col;
         if (state.readout[static_cast<std::size_t>(col)] !=
             std::array<int64_t, 3>{{step, step, step}} ||
             state.host[static_cast<std::size_t>(step)] != -99) return false;
         state.host[static_cast<std::size_t>(step)] = step;
      }
      return true;
   }
   return false;
}

bool run_schedule(Graph graph, int64_t nt, int64_t block, unsigned seed,
                  uint64_t &read_before_air, uint64_t &read_after_air,
                  int preference = -1)
{
   const std::vector<Rotation> rotation = rotations(nt);
   TaggedState state(nt, block);
   std::vector<std::size_t> ready;
   for (std::size_t i = 0; i < graph.nodes.size(); ++i)
      if (graph.nodes[i].predecessors == 0) ready.push_back(i);
   std::mt19937 random(seed);
   std::size_t executed = 0;
   while (!ready.empty()) {
      std::size_t chosen = random() % ready.size();
      // Mutant DAGs use a deterministic preference to expose the omitted edge:
      // finish all available reads/BN work before AIR, or AIR/BN before reads.
      if (preference >= 0)
         for (std::size_t i = 0; i < ready.size(); ++i) {
            const Kind kind = graph.nodes[ready[i]].kind;
            if ((preference == 0 && kind == Kind::Boundary) ||
                (preference == 1 && kind == Kind::Air)) { chosen = i; break; }
            if ((preference == 0 && kind == Kind::Read) ||
                (preference == 1 && kind == Kind::Boundary)) chosen = i;
         }
      const std::size_t id = ready[chosen];
      ready[chosen] = ready.back();
      ready.pop_back();
      if (!execute(graph.nodes[id], state, rotation, nt, block)) return false;
      ++executed;
      for (std::size_t successor : graph.nodes[id].successors)
         if (--graph.nodes[successor].predecessors == 0) ready.push_back(successor);
   }
   if (executed != graph.nodes.size()) return false;
   for (int64_t n = 0; n < nt; ++n)
      if (state.host[static_cast<std::size_t>(n)] != n ||
          !state.done_air[static_cast<std::size_t>(n)] ||
          !state.done_read[static_cast<std::size_t>(n)]) return false;
   read_before_air += state.read_before_air;
   read_after_air += state.read_after_air;
   return true;
}

void check_dag()
{
   uint64_t before = 0, after = 0;
   const int64_t timesteps[] = {513, 1025, 2049};
   const int64_t blocks[] = {1, 7, 512};
   for (int64_t nt : timesteps)
      for (int64_t block : blocks) {
         const Graph graph = dependency_graph(nt, block);
         for (unsigned attempt = 0; attempt < 64; ++attempt)
            require(run_schedule(graph, nt, block, 0x5c4ed001u + attempt, before, after),
                    "permissible DAG schedule violated field/carry/readout ordering");
      }
   require(before != 0 && after != 0, "random schedules did not exercise both readout/air orders");
   require(!run_schedule(dependency_graph(513, 512, false, true), 513, 512, 1,
                         before, after, 0),
           "model failed to detect missing previous-air dependency");
   require(!run_schedule(dependency_graph(513, 512, true, false), 513, 512, 1,
                         before, after, 1),
           "model failed to detect bulk readout overwritten by the next timestep");
   std::printf("PASS: 576 randomized DAG schedules; read-before-air=%llu, read-after-air=%llu; "
               "missing-edge sensitivity checks\n",
               static_cast<unsigned long long>(before), static_cast<unsigned long long>(after));
}

} // namespace

int main()
{
   const int64_t timesteps[] = {1, 2, 5, 6, 511, 512, 513, 1023, 1024, 1025};
   const int64_t blocks[] = {1, 2, 3, 7, 511, 512, 513};
   for (int64_t nt : timesteps)
      for (int64_t block : blocks) check_plan(nt, block);
   check_large_steps();
   check_mode();
   check_receivers();
   check_dag();
   std::puts("PASS: checkpoint coverage, 64-bit steps, async opt-in, interior receivers, "
             "two/three-buffer rotations; native contracts only, no GPU execution");
   return EXIT_SUCCESS;
}
