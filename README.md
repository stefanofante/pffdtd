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
boundaries. CPU and CUDA engines support single and double precision, with
multi-GPU forward execution, single-precision source safeguards and staircase
surface-area correction. The native pipeline covers mesh preparation through
final RIR and WAV output. Current numerical evidence and hardware checks are
recorded in [the CUDA audit](docs/CUDA_AUDIT.md).

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

The following changes extend the simulator relative to
[bsxfun/pffdtd](https://github.com/bsxfun/pffdtd). Forward optimizations retain
the Cartesian/FCC update scheme. The latest native preparation also corrects
sound speed, reciprocal links and surface factors; see the compatibility rules
in [the native guide](docs/NATIVE_PIPELINE.md) and the evidence in
[the CUDA audit](docs/CUDA_AUDIT.md). CUDA runtime and performance gates for the
new variants require execution on each target GPU.

- **Build targets.** `CUDA_ARCH` selects GPU device code, with explicit
  `sm_89` for Ada and `sm_121` for GB10; the default is `native`.
  The host compiler separately determines whether the executable runs on
  x86-64 or ARM64. CUDA preparation uses C++17 and CUB.
- **Multi-GPU peer access.** Available peer access is enabled with
  `cudaDeviceEnablePeerAccess`. Halo exchange uses `cudaMemcpyPeerAsync`;
  the application has no explicit pinned-host staging route for devices
  without peer access. Multi-GPU behavior requires runtime checks on that
  topology.
- **Batched source injection.** Upstream launched one `<<<1,1>>>` kernel per
  source per timestep (Ns x Nt micro-launches). Now a single batched kernel
  per timestep over all sources, with source signals and indices uploaded to
  the device once at init.
- **Blocked read-out.** Upstream did a per-timestep device-to-host copy plus
  synchronisation for the receiver outputs. Now outputs accumulate in a device
  buffer and drain in blocks, cutting per-step PCIe transfers and sync points.
- **Halo, scheduler and boundary execution.** Composed halo reads fix a fused
  read/write dependency. Optional event scheduling, reusable CUDA Graphs and
  fused stencil/ADE variants retain ordered references for comparison.
- **Native mesh preparation.** A shared FP64 BVH classifies Cartesian/FCC
  links; CUDA retains geometry and axes on the device and compacts boundary
  records in bounded batches. Preparation imports existing passive DEF
  materials and corrects Kelvin sound speed, reciprocal links and surface
  factors.
- **Native output processing.** CPU/CUDA receiver reconstruction, filters,
  resampling and ISO9613 air attenuation feed HDF5 and float32 WAV export.
  An optional Chebyshev/FFT modal algorithm replaces quadratic recurrence
  work with `O(K*N*log N)` work and a guarded recurrence fallback.

The hardware targets are:

- **NVIDIA RTX 4500 Ada Generation** (24 GB, dedicated VRAM) — the workstation baseline.
- **NVIDIA DGX Spark (GB10 Grace-Blackwell)** — aarch64, 128 GB unified memory.

The two targets differ in memory behaviour: Ada has dedicated VRAM, whereas GB10
shares physical memory with the host. The engine uses `cudaMalloc` and checks its
explicit allocation budget against `cudaMemGetInfo` before allocating fields.
Runtime/Graph overhead and other host allocations need additional room,
especially on GB10; the check does not reserve memory or adapt partitions.
Shared physical memory does not imply managed-memory paging.

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

The native pipeline imports JSONRoomExport meshes and existing passive HDF5
materials, prepares Cartesian/FCC solver inputs, runs the forward engine, and
processes RIRs through filters, resampling, air attenuation and WAV export.
See [the native pipeline guide](docs/NATIVE_PIPELINE.md) for complete commands.
Numerical stages use C++/CUDA; the legacy Python tools remain available separately.

Build on Linux with a C++17 compiler, OpenMP, HDF5 development files and
nlohmann/json headers. GPU preparation and processing also require CUDA with
CUB and cuFFT; CUDA 13 builds have been checked for both device targets.
Override `NVCC`, `HDF5_INC`, `HDF5_LIB`, `JSON_INC` and `CUFFT_LIB` when needed.

Use `CUDA_ARCH=sm_89` for RTX 4500 Ada or `CUDA_ARCH=sm_121` for GB10 with a
compatible toolkit. The default remains `native`. GPU code for GB10 can be
compiled on x86_64, but the complete executable also needs an aarch64 host build
and matching HDF5 libraries to run on DGX Spark. `BUILD_DIR` keeps generated files
outside the checkout, for example:

```sh
make -C c_cuda -j2 cpu gpu prepare post benchmark fixture CUDA_ARCH=sm_89 BUILD_DIR=/tmp/pffdtd-ada
# CPU preparation, processing and regression checks without CUDA:
make -C c_cuda -j2 cpu prepare_cpu post_cpu test-native BUILD_DIR=/tmp/pffdtd-native
make -C c_cuda test-cuda CUDA_ARCH=sm_89 BUILD_DIR=/tmp/pffdtd-ada-tests
```

For a single GPU, `PFFDTD_ASYNC=1` enables an event-based scheduler that queues
timesteps without per-step host barriers and drains output every 512 samples or
at the final partial block. The synchronous scheduler remains the default and
is used for multiple GPUs or receivers on ghost faces. `SCHEDULER_SYNC` forces
the reference at compile time; `PFFDTD_PROGRESS=0` disables progress output.
`PFFDTD_GRAPHS=1` selects reusable single-stream CUDA Graphs instead: packets of
96 or 6 steps preserve the two/three-buffer rotations, a device counter advances
source samples and output columns, and scalar tails stop at each output drain.
Graphs imply the asynchronous mode and have the same device/receiver guards.
Capture and instantiation time is included in the reported wall time.

`PFFDTD_BOUNDARY_FUSED=1` combines the rigid stencil and ADE correction in one
kernel. It validates a boundary-to-lossy map while retaining the three pressure
carry buffers and pole state layout. This removes an intermediate grid write/read
and a launch, but adds a map lookup for every boundary node. The default keeps
the ordered reference; `BOUNDARY_SEPARATE` overrides the runtime fusion option.
With boundary fusion enabled, `PFFDTD_ADE_MODE=reload` uses two scalar passes
that reread unchanged ADE states and eliminate the temporary history arrays.
`PFFDTD_ADE_MODE=fixed` unrolls pole counts 0/1/11/12, using the scalar path for
other counts; it increases register use substantially. The default is `generic`.
Select either variant only after comparing numerical results and timings on the
target GPU; removing local arrays alone does not establish a speedup.

CUDA regression tests include full-engine comparisons of the runtime modes;
select one device with `CUDA_VISIBLE_DEVICES` before running them. GPU correctness
and speedup must be checked on each target before adopting the asynchronous mode.

The native C/C++ tests cover halo and scheduler dependencies, boundary/ADE
states, memory checks, mesh/BVH queries, Cart/FCC layout, HDF5 preparation,
DSP/resampling, air/FFT algorithms, WAV export and source normalization.
CUDA tests compare device results; they return status 77 when no device is
available, which is a skipped check. `HALO_SEPARATE`
and `BOUNDARY_SEPARATE` compiler defines retain the ordered-halo and three-pass
boundary implementations for comparison. Define them through `NVCCFLAGS` when
building a reference executable, retaining the normal includes and build flags.
See [the CUDA audit](docs/CUDA_AUDIT.md) for the optimization plan and validation
limits, and [the native guide](docs/NATIVE_PIPELINE.md) for complete CTK and
Musikverein material mappings and preparation commands. `fdtd_prepare_cpu.x`
and `fdtd_prepare_gpu.x` write five HDF5 files into a new directory and check
them with the original solver loader. FCC folds by default for CUDA; CPU
also accepts unfolded inputs prepared with `--fcc --no-fold`. The pipeline
uses existing passive DEF materials; fitting new absorption data is a
separate material-model workflow.

Build the native comparison harness with `make -C c_cuda benchmark` and the same
architecture/library overrides as the engines. Run it from a prepared simulation
directory containing the four input HDF5 files, for example:

```sh
CUDA_VISIBLE_DEVICES=0 /tmp/pffdtd-ada/fdtd_bench_gpu_single.x --repetitions 12 --csv /tmp/pffdtd-ada-results.csv
```

The harness interleaves twelve scheduler/boundary/ADE variants, checks every output bit,
and reports median/min/max wall time with GPU, CUDA and build revision metadata.
It includes `run_sim` setup/cleanup and graph construction, excludes HDF5 loading
and verification, and keeps the engine's differently clocked loop times separate.
Each run resets the CUDA context. Existing CSV files require `--overwrite-csv`;
simulation output files are not written. Run the CUDA regressions before using
benchmark results to choose a variant for either target.

Build the native HDF5 fixture generator with `make -C c_cuda fixture`, using
the same HDF5 library overrides as the CPU engines. For example:

```sh
make -C c_cuda fixture BUILD_DIR=/tmp/pffdtd-native
/tmp/pffdtd-native/fdtd_fixture.x --output /tmp/pffdtd-panel --nx 128 --ny 128 --nz 128 --steps 3073 --mixed-poles
cd /tmp/pffdtd-panel
CUDA_VISIBLE_DEVICES=0 /tmp/pffdtd-ada/fdtd_bench_gpu_single.x --repetitions 12 --csv /tmp/pffdtd-panel-results.csv
```

The generator writes all four solver input files, checks them with the original
loader, and refuses an existing output directory. It creates Cartesian finite
panels with reciprocal links, passive RLC materials, two differentiated sources
and six receivers; it does not voxelize triangle meshes. Each axis needs at
least eight cells. `--poles 0..12`, `--mixed-poles`, `--rigid-every N` and
`--panel-spacing N` vary the ADE workload and panel density. Defaults are 32³,
1537 steps, 11 poles, every third boundary rigid and one central panel.
`test-native` includes 26 HDF5 round trips in each precision. A 3073-step mixed
fixture also passed integration with the original CPU solver in FP32/FP64;
CUDA execution and acoustic validation still require the hardware gates.

Native output processing reads the original solver's `sim_outs.h5`, rebuilds
physical receivers with `out_alpha`, applies integration/Butterworth high-pass,
resampling, optional low-pass and air attenuation, and writes `r_out_f`/`Fs_f`
in a new HDF5 file. WAV export is optional:

```sh
make -C c_cuda post_gpu CUDA_ARCH=sm_89 BUILD_DIR=/tmp/pffdtd-ada
# Run the engine from the prepared directory to produce sim_outs.h5 first.
/tmp/pffdtd-ada/fdtd_post_gpu.x --data-dir /tmp/pffdtd-panel --verify
# Choose a different output when processing the same input again:
/tmp/pffdtd-ada/fdtd_post_gpu.x --data-dir /tmp/pffdtd-panel --verify \
  --air-filter modal --air-modal-method fft \
  --output /tmp/pffdtd-panel/sim_outs_modal.h5 --save-wav
```

Use `post_cpu` for a reference build without CUDA. Defaults match the original
CLI's stages: 10 Hz high-pass of order 8, integration when `diff=1`, output at
48 kHz, low-pass and air absorption disabled. `--sample-rate 0` retains the native
rate; `--lowpass`, `--lowpass-order` and `--symmetric-lowpass` control the optional
forward/reverse stage. Orders 1..16 are supported. SOS pairing can differ from
SciPy and the analytic Kaiser sinc resampler is a new implementation: processed
samples are compared with tolerances, rather than promised bit-identical.

Air attenuation supports `--air-filter none|stokes|ola|modal`. Temperature and
humidity come from `sim_consts.h5`; `--air-pressure` sets ambient pressure in
kPa, and `--air-window` selects the OLA window. Modal processing defaults to
`--air-modal-method recurrence`; `fft` uses a Chebyshev approximation with
automatic fallback when its cost or range is unsuitable.
`--air-modal-tolerance` controls the interpolation error bound; FFT roundoff
and acoustic accuracy require separate checks.

The CUDA backend retains receiver data and scratch through DSP and air
processing on one stream. Its default IIR mode is serial;
`--iir-mode chunked` computes zero-state responses for
64-sample chunks, propagates their affine carries, then filters chunks in
parallel. It changes rounding and requires a device comparison before use.
`--verify` compares every final CUDA sample, including air processing, against
the C++ reference with `1e-12 + 1e-8*abs(reference)` tolerance.
The default output refuses overwrites;
`--overwrite` replaces an existing regular file while protecting input aliases
and symlinks. `--save-raw` adds reconstructed receivers, and metadata records
filters, backend, resampler, atmosphere and build revision. `--save-wav` writes
globally normalized float32 mono WAVs and adds native-amplitude WAVs when the
global peak is below one. Silence remains finite. `--wav-dir` overrides the
data directory. HDF5 is published before WAVs; a WAV failure reports that the
processed HDF5 already exists.

The complete flow is JSONRoomExport mesh plus passive DEF materials → native
preparation → C/CUDA forward engine → native filters/resampling/air → HDF5/WAV.
Sketchup and the provided export plugin can produce the input JSON with source
and receiver positions. For the original tool's documentation and references,
see [bsxfun/pffdtd](https://github.com/bsxfun/pffdtd).

## Verification status

Native regression and sanitizer checks passed, including real CTK preparation,
1537-step CPU simulation, all three air filters, HDF5 and twelve WAV outputs.
CTK and Musikverein meshes with repository materials also passed the original
solver loader. These checks establish integration and numerical contracts;
measured RIR accuracy and long-time stability require further validation.

CUDA builds for Ada and GB10 passed on the cloud's x86-64 host. Its 28 CUDA
test/benchmark/preparation/processing executions returned **77 (SKIP)** because
no usable GPU was available. Target speedups and CUDA runtime correctness
remain unmeasured; the GB10 executable also needs an ARM64 host build.

In the CTK FCC folded/unfolded comparison at `h=0.15 m` and 1537 steps, FP64
forward outputs passed the pointwise gate. Filtered FP32 outputs differed by
**0.135% normalized RMS** and failed the stricter pointwise gate as the layout changed addition
order. Select precision using the accuracy requirements and target benchmark.
Exact conditions and error measurements are in the native guide and audit.

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
