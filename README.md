# PFFDTD — fork (RaySound)

![PFFDTD Screenshot](https://github.com/bsxfun/pffdtd/raw/main/screenshot.png)

> **This is a derivative fork** (`stefanofante/pffdtd`), maintained by
> **Stefano Fante (ST-LINE S.r.l., Treviso, Italy)**.
> It is **not** the upstream project. The original PFFDTD was written by **Brian Hamilton**
> (University of Edinburgh, 2021) and is released under the MIT License — see
> [`bsxfun/pffdtd`](https://github.com/bsxfun/pffdtd) and the `LICENSE` file. All credit for the
> original simulator, its algorithms and its published references belongs to him. If this code
> contributes to academic work, please cite the original software:
>
> ```
> @misc{hamilton2021pffdtd,
>   title  = {PFFDTD Software},
>   author = {Brian Hamilton},
>   note   = {https://github.com/bsxfun/pffdtd},
>   year   = {2021}
> }
> ```

## What PFFDTD does

PFFDTD is a finite-difference time-domain (FDTD) simulator for 3D room acoustics: it
computes room impulse responses on a regular grid, using a 7-point Cartesian or a
13-point face-centred-cubic (FCC) stencil, with frequency-dependent impedance
boundaries. It runs multi-GPU on CUDA, conserves energy to machine precision in
double, and ships single-precision stability safeguards plus a staircase
surface-area correction for more consistent decay-time estimates. It is fast,
well-validated, and a clean wave-based reference — which is exactly why it is the
base for this fork.

## Why this fork exists

This fork was created, and continues to be maintained, as part of the development of
**RaySound**, an indoor room-acoustics simulation stack built by ST-LINE. PFFDTD
served as a fast, independently-founded wave-based reference solver during that
development: a method-against-method cross-check for RaySound's own engines, and a
benchmarking instrument in the wave-based regime. The performance work here was done
to make that role practical on modern GPU hardware.

It remains a standalone derivative of Brian Hamilton's original work, maintained
independently. It is **not** part of the RaySound production pipeline — it sits
alongside it as a research and validation tool.

## What RaySound is

RaySound is a hybrid indoor room-acoustics simulator that combines two solvers across
the frequency range:

- a **wave-based** solver in the low-frequency (modal) regime, where the wave nature
  of sound dominates and geometric methods are blind;
- a **GPU ray-tracer** in the high-frequency regime, modelling specular reflections
  and diffuse scattering.

Beyond forward simulation, RaySound is built around **differentiable inverse design**:
a certified adjoint over both materials and geometry, letting the stack *invert* for
room parameters (impedances, dimensions, panel placement) rather than only simulate a
fixed configuration. The wave engine (`dg-acoustics`) is described below; the
ray-tracer is a separate component.

PFFDTD's role in this picture was the wave-side reference: a second solver on a
different numerical foundation (finite differences) to cross-validate the
discontinuous-Galerkin wave engine method-against-method.

## Optimizations applied vs upstream

The following changes have been made to the GPU engine relative to
[bsxfun/pffdtd](https://github.com/bsxfun/pffdtd). The numerical scheme is
unchanged; forward output matches the reference Python engine to machine accuracy.

- **Build target.** Upstream compiled for `sm_35` (Kepler), which runs in
  PTX-JIT compatibility mode on modern GPUs. Now `-arch=native` plus
  `-Xptxas -O3`, producing a real binary for the host architecture.
- **Multi-GPU peer access.** Upstream issued `cudaMemcpyPeerAsync` for halo
  exchange without ever enabling P2P, silently staging transfers through host
  RAM. Now `cudaDeviceEnablePeerAccess` is set up for every accessible device
  pair, with an explicit host-staging fallback when P2P is unavailable.
- **Batched source injection.** Upstream launched one `<<<1,1>>>` kernel per
  source per timestep (Ns x Nt micro-launches). Now a single batched kernel
  per timestep over all sources, with source signals and indices uploaded to
  the device once at init.
- **Blocked read-out.** Upstream did a per-timestep device-to-host copy plus
  synchronisation for the receiver outputs. Now outputs accumulate in a device
  buffer and drain in blocks, cutting per-step PCIe transfers and sync points.

Performance work is tuned on the hardware it runs on:

- **NVIDIA RTX 4500 Ada Generation** (24 GB, dedicated VRAM) — the workstation baseline.
- **NVIDIA DGX Spark (GB10 Grace-Blackwell)** — aarch64, 128 GB unified memory.

The two targets differ in memory behaviour: dedicated VRAM fails hard on
over-allocation, whereas the GB10's unified memory degrades gradually. Memory
budgeting is therefore queried at **runtime** (free VRAM via `cudaMemGetInfo` minus
an adaptive safety margin) rather than hard-coded — the engine discovers the GPU it
is running on and adapts, instead of assuming a fixed device at build time.

## Relationship to `dg-acoustics`

`dg-acoustics` is RaySound's wave-based production engine, developed independently of
this fork. The two solvers serve different roles:

- **PFFDTD (this fork)** is a *forward* FDTD solver on Cartesian / FCC grids. It
  computes room impulse responses for a given geometry and set of boundary
  impedances. It is a research and benchmarking instrument.

- **`dg-acoustics`** is a **Discontinuous-Galerkin time-domain** wave solver built
  for **differentiable inverse design**. Beyond the forward solution it provides a
  **locally-reacting impedance boundary with complex frequency-dependent
  admittance**, a **certified adjoint**, and the machinery to **invert for geometry
  and materials** rather than only simulate them. This is a capability a pure forward
  FDTD code does not have by construction — a different class of tool, not a faster
  version of the same one.

### Comparison

| Dimension | PFFDTD (this fork) | dg-acoustics |
|---|---|---|
| Numerical method | Explicit FDTD, 7-pt Cartesian / 13-pt FCC stencil | Nodal Discontinuous Galerkin (Hesthaven-Warburton), LSERK4 |
| Discretization | Structured voxel grid | Unstructured, body-conforming mesh |
| Geometry | Staircase + surface-area correction | Conforming boundary; curved/angled walls without staircasing |
| Boundary model | Frequency-dependent impedance (octave-band passive fit) | Locally-reacting one-pole ADE + multi-pole complex reflection fit; validated vs analytic R(theta) |
| Modal observable | RIR only; descriptors extracted downstream | Damped complex modes s_m = -alpha_m + j*omega_m, native from the solver |
| Inverse design | None, by construction (pure forward) | Certified adjoint: materials, matrix-free geometry, FWI, frequency continuation |
| Differentiability | No | Yes; full forward + boundary chain differentiable |
| Uncertainty | No | Conformal prediction + UQ module |
| Role in stack | Cross-validation / benchmarking instrument | Production wave engine + design-inversion |

The two solvers rest on different numerical foundations (finite differences vs
discontinuous Galerkin), which is precisely what makes cross-validating one against
the other worthwhile.

## Build & run

PFFDTD runs on Linux with the CUDA toolkit and HDF5. To build the engines, run
`make all` in the `c_cuda` folder (see the Makefile for HDF5 paths). The Python side
needs Python 3.9+ with the packages in `pip_requirements.txt` (or the conda env).

The typical flow: build a model in Sketchup and export it (with source/receiver CSVs)
to JSON via the provided plugin; fit absorption/impedance data; run a setup script
that voxelizes the scene and writes `.h5` inputs; run the CUDA engine over those `.h5`
files; post-process `sim_outs.h5` into final RIRs. Single-precision GPU execution is
generally the fastest. For the original tool's full documentation, examples and
references, see [bsxfun/pffdtd](https://github.com/bsxfun/pffdtd).

## Contact

RaySound is developed by **ST-LINE S.r.l.** (Treviso, Italy). For questions about the
fork, the RaySound stack, or collaboration:

- **Stefano Fante** — ST-LINE S.r.l.
- Email: stefano.fante@stline.it
- Web: https://www.stline.it

## License

MIT — see the `LICENSE` file. Original work Copyright 2021 Brian Hamilton; fork
modifications by ST-LINE S.r.l. The Sketchup models under `data/models` are released
under their own licenses; see the README in each folder.