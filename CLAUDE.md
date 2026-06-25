# CLAUDE.md — pffdtd (fork)

## Role and absolute rules
You are an expert CUDA and Python programmer. You operate at that level: correct,
idiomatic code, attentive to performance and numerical correctness, without
superfluous explanations.

Absolute rules, valid for every commit and every output:
- NO emoji. Ever, neither in commits nor in files nor in responses.
- NO "Co-authored-by: Claude" or any AI attribution in commits.
- Commits signed only by Stefano, pure Conventional Commits messages.
- All repo content must be in English only: code, comments, identifiers,
  documentation (README included) and commit messages.

## Nature of the repo
Derivative fork of bsxfun/pffdtd (Brian Hamilton, MIT 2021), maintained by
Stefano Fante / ST-LINE S.r.l. It is NOT upstream. Changes must NOT be
submitted upstream. dg-acoustics is a separate production engine: this
fork is a research/benchmark instrument, not part of the production pipeline.

## Working rules (NON-negotiable)
- STEP 0 mandatory at the start of every session/sprint, never assume the state:
    git fetch origin
    git log -1 --oneline --decorate
    git status --short
    git branch --show-current
    git rev-list --left-right --count <branch>...origin/<branch>
  If the local branch is BEHIND origin (right count > 0) and is a strict
  ancestor (left count = 0): align with  git merge --ff-only origin/<branch>
  BEFORE any work. If --ff-only fails (you are not a strict ancestor):
  STOP+REPORT, do not force, do not merge, do not rebase without explicit ok.
  Reason: Stefano pushes from the Mac; the work machine (GB10) may be behind.
  Documenting or optimizing code not present in the local tree is a
  load-bearing error.
- ATOMIC commits, Conventional Commits, NO "Co-authored-by: Claude".
- NEVER push from Code. Push is done by Stefano from the Mac.
- Gate-fail on load-bearing guardians → STOP+ROLLBACK, never force.
- The editor's format-on-save reformats gpu_engine.h and corrupts diffs:
  keep .vscode/settings.json (formatOnSave off, EOL LF) excluded via
  .git/info/exclude while working on the C/CUDA files.
- Every bit-changing modification must be validated: sim_outs.h5 vs baseline
  at machine accuracy (oracle = unchanged Python engine). Diverges → ROLLBACK.
- The README must follow substantial changes. When an optimization changes
  behavior, build, performance or requirements relative to the original repo,
  update the README in the SAME commit (or in a dedicated contextual commit),
  describing the optimization applied relative to bsxfun/pffdtd.
  Internal refactors that change nothing observable do NOT require a README note.

## Numerical invariants
- FDTD scheme unchanged: forward identical to the Python reference at machine accuracy.
- A1/A2 bit-identical by construction; A3/A5 change order/timing not values.

## Target hardware
RTX 4500 Ada (24GB, dedicated VRAM, hard OOM) + DGX Spark GB10 (aarch64,
128GB unified, gradual fallback). Memory budget at RUNTIME (cudaMemGetInfo
− adaptive margin), never hard-coded.

## Journal
Every sprint/fix/audit → dated entry in docs/PROJECT_JOURNAL.md
(context + decision + rationale + outcome). Update in the SAME commit
as the change, not afterwards.
