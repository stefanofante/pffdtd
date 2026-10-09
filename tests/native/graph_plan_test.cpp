// Native contracts for fixed-address CUDA Graph packets. This model checks
// coverage, pointer periods, device counters and output-buffer reuse under
// independently progressing host/device queues. It does not execute CUDA.
#include "graph_plan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

struct Rotation {
   std::array<int, 2> bulk;
   std::array<int, 3> carry;
};

void rotate(Rotation &slots)
{
   std::swap(slots.bulk[0], slots.bulk[1]);
   const int oldest = slots.carry[2];
   slots.carry[2] = slots.carry[1];
   slots.carry[1] = slots.carry[0];
   slots.carry[0] = oldest;
}

Rotation phase_rotation(int phase)
{
   Rotation slots = {{{0, 1}}, {{0, 1, 2}}};
   for (int i = 0; i < phase; ++i) rotate(slots);
   return slots;
}

bool same_rotation(const Rotation &a, const Rotation &b)
{
   return a.bulk == b.bulk && a.carry == b.carry;
}

void check_coverage(int64_t nt, int64_t block, int64_t maximum)
{
   Rotation slots = phase_rotation(0);
   int64_t next_drain = 0;
   int64_t n = 0;
   while (n < nt) {
      const pffdtd::GraphChunkPlan plan = pffdtd::graph_chunk_plan(n, nt, block, maximum);
      require(plan.steps > 0 && plan.steps <= nt - n, "packet exceeds simulation tail");
      require(plan.steps <= block - n % block, "packet crosses output drain");
      require(plan.phase == n % 6, "packet pointer phase is incorrect");
      require(same_rotation(slots, phase_rotation(plan.phase)), "host pointer phase drifted");
      require(plan.reusable == (plan.steps % 6 == 0), "nonperiodic packet declared reusable");
      require(plan.reusable || plan.steps == 1, "scalar tail is not one timestep");
      require(plan.steps <= maximum || !plan.reusable, "graph exceeds capture limit");
      const Rotation before = slots;
      for (int64_t offset = 0; offset < plan.steps; ++offset) {
         require(same_rotation(slots, phase_rotation(static_cast<int>((n + offset) % 6))),
                 "captured slot does not match independent timestep rotation");
         const int64_t column = (n + offset) % block;
         require(column == n % block + offset, "dynamic columns wrapped inside a packet");
         rotate(slots);
      }
      if (plan.reusable)
         require(same_rotation(slots, before), "graph does not restore fixed pointer slots");
      n += plan.steps;
      if (n == nt || n % block == 0) {
         require(n > next_drain && n - next_drain <= block,
                 "drain left a gap or exceeded output buffer capacity");
         next_drain = n;
      }
   }
   require(next_drain == nt, "final partial output batch was not drained");
}

void check_explicit_policy()
{
   struct Case { int64_t n, nt, block, steps; bool reusable; };
   const Case cases[] = {
      {0, 1, 512, 1, false}, {0, 5, 512, 1, false}, {0, 6, 512, 6, true},
      {0, 95, 512, 6, true}, {0, 96, 512, 96, true}, {0, 512, 512, 96, true},
      {480, 512, 512, 6, true}, {504, 512, 512, 6, true},
      {510, 512, 512, 1, false}, {511, 513, 512, 1, false},
      {512, 1025, 512, 96, true}, {1024, 1025, 512, 1, false}
   };
   for (const Case &test : cases) {
      const pffdtd::GraphChunkPlan plan = pffdtd::graph_chunk_plan(test.n, test.nt, test.block);
      require(plan.steps == test.steps && plan.reusable == test.reusable,
              "96/6/scalar capture policy changed");
   }
   const int64_t limits[] = {0, -1, std::numeric_limits<int64_t>::min()};
   for (int64_t invalid : limits) {
      require(pffdtd::graph_chunk_plan(0, 5, invalid).steps == 0,
              "invalid output capacity accepted");
      require(pffdtd::graph_chunk_plan(0, 5, 512, invalid).steps == 0,
              "invalid capture limit accepted");
   }
   require(pffdtd::graph_chunk_plan(-1, 5, 512).steps == 0, "negative timestep accepted");
   require(pffdtd::graph_chunk_plan(5, 5, 512).steps == 0, "finished simulation accepted");
   require(pffdtd::graph_chunk_plan(6, 5, 512).steps == 0, "timestep beyond Nt accepted");
   require(pffdtd::graph_chunk_plan(0, -1, 512).steps == 0, "negative Nt accepted");
}

void check_large_steps()
{
   const int64_t largest = std::numeric_limits<int64_t>::max();
   for (int64_t remaining = 1; remaining <= 1025; ++remaining) {
      int64_t n = largest - remaining;
      while (n < largest) {
         const pffdtd::GraphChunkPlan plan = pffdtd::graph_chunk_plan(n, largest, 512);
         require(plan.steps > 0 && plan.steps <= largest - n,
                 "64-bit final packet overflows or truncates");
         require(plan.phase == n % 6 && plan.steps <= 512 - n % 512,
                 "64-bit phase or output boundary is incorrect");
         n += plan.steps;
      }
      require(n == largest, "64-bit final timestep was skipped");
   }
   const pffdtd::GraphChunkPlan enormous = pffdtd::graph_chunk_plan(0, largest, largest, largest);
   require(enormous.steps == largest - largest % 6 && enormous.reusable,
           "large capture count was truncated");
}

enum class Kind { Boundary, Air, Read, Advance };
struct Operation {
   Kind kind;
   int64_t step, offset, packet_base, packet_steps;
   Rotation slots;
};

enum class Mutation { None, WrongPhase, EarlyAdvance, NoAdvance };

struct Model {
   std::array<std::array<int64_t, 3>, 2> fields;
   std::array<int64_t, 3> carry;
   std::vector<std::array<int64_t, 3>> output;
   std::vector<int64_t> host;
   int64_t counter;

   Model(int64_t nt, int64_t block)
      : fields{{{{-1, -1, -1}}, {{0, 0, 0}}}}, carry{{-3, -1, -2}},
        output(static_cast<std::size_t>(block), {{-99, -99, -99}}),
        host(static_cast<std::size_t>(nt), -99), counter(0)
   {}
};

bool execute(const Operation &op, Model &model, int64_t block)
{
   const int64_t n = op.step;
   const int output = op.slots.bulk[0], input = op.slots.bulk[1];
   switch (op.kind) {
   case Kind::Boundary:
      if (model.fields[input] != std::array<int64_t, 3>{{n, n, n}} ||
          model.fields[output] != std::array<int64_t, 3>{{n - 1, n - 1, n - 1}} ||
          model.carry[op.slots.carry[2]] != n - 2) return false;
      model.fields[output][1] = n + 1;
      model.carry[op.slots.carry[0]] = n;
      return true;
   case Kind::Air:
      // Source selection reads the device counter at execution time. Reading
      // it at submission/capture time would reuse stale source samples.
      if (model.counter + op.offset != n ||
          model.fields[input] != std::array<int64_t, 3>{{n, n, n}} ||
          model.fields[output] != std::array<int64_t, 3>{{n - 1, n + 1, n - 1}})
         return false;
      model.fields[output][0] = n + 1;
      model.fields[output][2] = n + 1;
      return true;
   case Kind::Read: {
      const int64_t dynamic_step = model.counter + op.offset;
      if (dynamic_step != n || model.fields[input] != std::array<int64_t, 3>{{n, n, n}})
         return false;
      // AIR writes the other bulk bank. Interior receivers can be read after
      // AIR without changing the reference's one-step-old sample value.
      const std::size_t column = static_cast<std::size_t>(dynamic_step % block);
      if (model.output[column] != std::array<int64_t, 3>{{-99, -99, -99}}) return false;
      model.output[column] = model.fields[input];
      return true;
   }
   case Kind::Advance:
      if (model.counter != op.packet_base) return false;
      model.counter += op.packet_steps;
      return true;
   }
   return false;
}

// Host submission can run ahead of device execution. Every queued operation
// retains fixed captured pointer addresses while its counter is read only
// during execution. A drain stops host submission before the output wraps.
bool model_queues(int64_t nt, int64_t block, uint64_t seed, Mutation mutation,
                  uint64_t &queued_ahead, uint64_t &cache_replays)
{
   std::mt19937_64 random(seed);
   Model model(nt, block);
   std::deque<Operation> queue;
   std::array<std::array<bool, 6>, 2> cached = {};
   std::array<std::array<Rotation, 6>, 2> cache_slots = {};
   Rotation host_slots = phase_rotation(0);
   int64_t submitted = 0, drained = 0;
   bool checkpoint = false;

   while (drained < nt) {
      if (checkpoint && queue.empty()) {
         if (model.counter != submitted) return false;
         const int64_t count = submitted - drained;
         if (count <= 0 || count > block) return false;
         for (int64_t col = 0; col < count; ++col) {
            const int64_t n = drained + col;
            const std::array<int64_t, 3> expected = {{n, n, n}};
            if (model.output[static_cast<std::size_t>(col)] != expected ||
                model.host[static_cast<std::size_t>(n)] != -99) return false;
            model.host[static_cast<std::size_t>(n)] = n;
         }
         std::fill(model.output.begin(), model.output.end(), std::array<int64_t, 3>{{-99, -99, -99}});
         drained = submitted;
         checkpoint = false;
         continue;
      }

      const bool can_submit = submitted < nt && !checkpoint;
      if (can_submit && (queue.empty() || (random() & 1))) {
         const pffdtd::GraphChunkPlan plan = pffdtd::graph_chunk_plan(submitted, nt, block);
         if (plan.steps <= 0 || !same_rotation(host_slots, phase_rotation(plan.phase))) return false;
         Rotation captured = host_slots;
         if (plan.reusable) {
            const std::size_t length = plan.steps == 96 ? 0 : 1;
            const int phase = mutation == Mutation::WrongPhase ? 0 : plan.phase;
            if (cached[length][static_cast<std::size_t>(phase)]) ++cache_replays;
            else {
               cached[length][static_cast<std::size_t>(phase)] = true;
               cache_slots[length][static_cast<std::size_t>(phase)] = phase_rotation(phase);
            }
            captured = cache_slots[length][static_cast<std::size_t>(phase)];
         }
         const Operation advance = {Kind::Advance, submitted, 0, submitted, plan.steps, captured};
         if (mutation == Mutation::EarlyAdvance) queue.push_back(advance);
         for (int64_t offset = 0; offset < plan.steps; ++offset) {
            const int64_t n = submitted + offset;
            for (Kind kind : {Kind::Boundary, Kind::Air, Kind::Read})
               queue.push_back(Operation{kind, n, offset, submitted, plan.steps, captured});
            rotate(captured);
            rotate(host_slots);
         }
         if (mutation != Mutation::EarlyAdvance && mutation != Mutation::NoAdvance)
            queue.push_back(advance);
         submitted += plan.steps;
         checkpoint = submitted == nt || submitted % block == 0;
         if (submitted > model.counter) ++queued_ahead;
      }
      else if (!queue.empty()) {
         const Operation next = queue.front();
         queue.pop_front();
         if (!execute(next, model, block)) return false;
      }
      else return false;
   }

   if (model.counter != nt || !queue.empty()) return false;
   for (int64_t n = 0; n < nt; ++n)
      if (model.host[static_cast<std::size_t>(n)] != n) return false;
   return true;
}

} // namespace

int main()
{
   check_explicit_policy();
   check_large_steps();
   const int64_t lengths[] = {1, 2, 5, 6, 7, 95, 96, 97, 511, 512, 513, 1025, 1536, 1537};
   const int64_t blocks[] = {1, 2, 3, 5, 6, 7, 17, 95, 96, 97, 511, 512, 513, 1536};
   const int64_t maxima[] = {1, 5, 6, 7, 18, 95, 96, 97, 600};
   for (int64_t nt : lengths)
      for (int64_t block : blocks)
         for (int64_t maximum : maxima) check_coverage(nt, block, maximum);

   uint64_t runs = 0, queued_ahead = 0, cache_replays = 0;
   for (int64_t nt : {1, 5, 6, 7, 95, 96, 97, 511, 512, 513, 1025, 1537})
      for (int64_t block : {7, 96, 511, 512, 513})
         for (uint64_t seed = 0; seed < 24; ++seed) {
            require(model_queues(nt, block, seed, Mutation::None, queued_ahead, cache_replays),
                    "queued graph model changed samples, pointers, counters or output reuse");
            ++runs;
         }
   for (Mutation mutation : {Mutation::WrongPhase, Mutation::EarlyAdvance, Mutation::NoAdvance}) {
      uint64_t ignored_ahead = 0, ignored_replay = 0;
      require(!model_queues(1025, 512, 37, mutation, ignored_ahead, ignored_replay),
              "model failed to detect an invalid phase or counter mutation");
   }
   require(cache_replays > 0 && queued_ahead > 0, "model did not exercise cached graphs and host runahead");
   std::printf("PASS: %llu randomized host/device queue models; queued-ahead=%llu, cache-replays=%llu; phase/counter mutation checks\n",
               static_cast<unsigned long long>(runs), static_cast<unsigned long long>(queued_ahead),
               static_cast<unsigned long long>(cache_replays));
   std::puts("PASS: 96/6/scalar coverage, six-step pointer periods, 64-bit tails, dynamic counters and output drains; native contracts only, no GPU execution");
   return EXIT_SUCCESS;
}
