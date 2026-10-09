# Native preparation, simulation and output processing

The native pipeline reads the existing JSONRoomExport meshes and passive HDF5
materials, prepares solver inputs, runs the C/CUDA solver, and processes its
outputs in C++/CUDA. None of these commands needs Python calculations.

## Build

Install a C++ compiler, OpenMP, HDF5 development files, and the header-only
`nlohmann-json` library. CUDA preparation uses CUB from the CUDA toolkit and
requires C++17. CUDA output processing also links cuFFT. CUDA 13 supports both
RTX 4500 Ada (`sm_89`) and DGX Spark GB10 (`sm_121`):

```sh
make -C c_cuda prepare post cpu gpu CUDA_ARCH=sm_89
# Build the GPU binaries for GB10 with its toolkit:
make -C c_cuda prepare_gpu post_gpu gpu CUDA_ARCH=sm_121
```

The architecture flag selects device code. Build the GB10 executables on the
ARM64 Spark host, or with a complete ARM64 cross toolchain: an x86-64 host
executable compiled with `sm_121` still cannot run on ARM64.

Use the existing checkout. Override `BUILD_DIR`, `HDF5_INC`, `HDF5_LIB`,
`JSON_INC`, `NVCC` and `CUFFT_LIB` when dependencies live outside system paths.
For the configured cloud sysroot, for example:

```sh
make -C c_cuda prepare_cpu \
  BUILD_DIR=/workspace/.pffdtd-setup/native-checks \
  HDF5_INC=/workspace/.pffdtd-setup/sysroot/usr/include/hdf5/serial \
  JSON_INC=/workspace/.pffdtd-setup/sysroot/usr/include \
  LDFLAGS='-fopenmp -lgomp -lm -l:libhdf5_serial.so.310'
```

The CPU and CUDA preparers share mesh, interpolation and HDF5 rules. The CUDA
backend keeps triangle/BVH/axis arrays resident, classifies nodes in bounded
batches, and returns only boundary records through stable CUB compaction. It
does not allocate a dense boundary record for every grid node. A missing CUDA
device or driver returns exit code 77 before reading inputs or writing outputs.

## Prepare real meshes

Run preparation from the repository root. `--output` must name a directory
that does not exist. Every non-`_RIGID` model label requires exactly one explicit
`--material label=FILE`; material filenames are not inferred from model names.
Material HDF5 files contain float64 `DEF[M,3]`, with at most 12 branches.

The following CTK configuration retains all sources and receivers and reuses
the repository materials:

```sh
c_cuda/fdtd_prepare_gpu.x \
  --model data/models/CTK_Church/model_export.json \
  --output /tmp/pffdtd-ctk-cart \
  --fmax 1400 --ppw 10.5 --duration 3 --source-num 1 \
  --material AcousticPanel=data/materials/ctk_acoustic_panel.h5 \
  --material Altar=data/materials/ctk_altar.h5 \
  --material Carpet=data/materials/ctk_carpet.h5 \
  --material Ceiling=data/materials/ctk_ceiling.h5 \
  --material Glass=data/materials/ctk_window.h5 \
  --material PlushChair=data/materials/ctk_chair.h5 \
  --material Tile=data/materials/ctk_tile.h5 \
  --material Walls=data/materials/ctk_walls.h5
```

The existing Musikverein FCC configuration uses source 3; the native CLI also
accepts source 1 and defaults to it:

```sh
c_cuda/fdtd_prepare_gpu.x \
  --model data/models/Musikverein_ConcertHall/model_export.json \
  --output /tmp/pffdtd-mv-fcc \
  --fcc --fmax 2500 --ppw 7.7 --duration 3 --source-num 3 \
  --material Chairs=data/materials/mv_chairs.h5 \
  --material Floor=data/materials/mv_floor.h5 \
  --material Plasterboard=data/materials/mv_plasterboard.h5 \
  --material Window=data/materials/mv_window.h5 \
  --material Wood=data/materials/mv_wood.h5
```

These are full-resolution configurations; grid memory and runtime grow with
the volume divided by `h^3`. For smaller integration checks, replace
`--fmax/--ppw/--duration` with `--spacing 0.15 --steps 1537` for CTK or
`--spacing 0.05 --steps 7` for Musikverein FCC. All original Musikverein
communication stencils passed the native geometry preflight at `h=0.05 m`;
coarser tested grids at `0.075`, `0.10` and `0.15 m` intersected surfaces.
The preparer reports the affected receiver and corner and rejects the scene;
it does not move receivers, discard channels or relax the clash check.
Coarse configurations check integration and do not establish acoustic accuracy.

CPU preparation uses `fdtd_prepare_cpu.x --backend cpu` with the same arguments.
Use `--threads N` to select its OpenMP workers. For unfolded FCC inputs accepted
by the CPU solver, add `--fcc --no-fold`; the physical grid has `fcc_flag=1`.
The default FCC output is folded (`fcc_flag=2`) and works with CPU or CUDA
solvers. CUDA solvers require this folded form. `--no-rotate` preserves the
original axis order; otherwise the longest dimension becomes the x axis.

Defaults are Cartesian, `fmax=1000 Hz`, `PPW=10`, one-second duration, source 1,
20 °C and 50% relative humidity. `--spacing` overrides frequency-based grid
selection, while `--steps` overrides duration. Each pair of explicit alternatives
is mutually exclusive. The default target is `--precision single`, with source
differentiation enabled. `--precision double` defaults to an undifferentiated
impulse; `--diff-source` enables it explicitly. Single precision rejects
`--no-diff-source`.

Preparation creates `sim_consts.h5`, `vox_out.h5`, `comms_out.h5`,
`sim_mats.h5` and `cart_grid.h5`. All five files retain compatible datasets.
The original native solver loader checks every completed preparation. Failed
writes or failed loader validation remove files created by that call. Existing
output directories are refused.

## Run the solver

The solver reads its input files from its working directory. Use an absolute
binary path when changing directories:

```sh
pffdtd_solver="$PWD/c_cuda/fdtd_main_gpu_single.x"
(
  cd /tmp/pffdtd-ctk-cart
  CUDA_VISIBLE_DEVICES=0 \
    PFFDTD_ASYNC=1 PFFDTD_BOUNDARY_FUSED=1 \
    "$pffdtd_solver"
)
```

Replace the directory with `/tmp/pffdtd-mv-fcc` for Musikverein. The forward
solver writes `sim_outs.h5` with `u_out[Nr,Nt]`, already reordered and rescaled.
Do not apply `out_reorder` again during output processing.

CPU reference runs use `fdtd_main_cpu_single.x` or
`fdtd_main_cpu_double.x`, for example with `OMP_NUM_THREADS=2`. Select a prepared
source suitable for the chosen solver precision. Optional CUDA schedulers and
ADE implementations must be compared on the target hardware using the native
benchmark's correctness gate; selecting an implementation alone does not
demonstrate a speedup.

## Process outputs and export WAV

The default processing order is receiver interpolation, optional trapezoidal
integration of differentiated output, high-pass filtering, resampling,
low-pass filtering, optional air attenuation, then optional WAV export.
Default high-pass is 10 Hz/order 8, resampling is 48000 Hz, low-pass is disabled,
and air attenuation is `none`.

```sh
c_cuda/fdtd_post_gpu.x \
  --data-dir /tmp/pffdtd-ctk-cart --backend cuda --verify \
  --lowcut 10 --lowcut-order 8 \
  --sample-rate 48000 --lowpass 1400 --lowpass-order 8 \
  --symmetric-lowpass --air-filter none --save-raw --save-wav
```

For a CPU reference use `fdtd_post_cpu.x --backend cpu` with those arguments.
`--verify` on CUDA compares native CPU results and checks that device kernels
write all expected samples. Serial IIR is the default; `--iir-mode chunked`
selects the CUDA affine-prefix implementation. `--sample-rate 0` keeps the
native solver rate. The Kaiser sinc resampler is new and is not expected to be
bitwise equal to `resampy`.

Air attenuation supports `--air-filter stokes`, `ola` or `modal`. Temperature
and humidity come from `sim_consts.h5`; pressure defaults to 101.325 kPa and can
be set with `--air-pressure`. OLA uses `--air-window 1024` by default. Modal
processing supports `--air-modal-method recurrence|fft`,
`--air-modal-pad SECONDS` and `--air-modal-tolerance` for the FFT approximation.
Changing the air algorithm changes its approximation; compare the selected
algorithm against its native CPU reference using `--verify`.

The processed HDF5 output defaults to `sim_outs_processed.h5` and contains
`r_out_f[channels,samples_out]` and scalar `Fs_f`. `Fs_native` and processing
metadata are also stored; `--save-raw` adds recombined native-rate `r_out`.
`--save-wav` exports native-amplitude and normalized float32 receiver WAV files.
The WAV directory defaults to the data directory; override it with `--wav-dir`.
Outputs require new filenames unless `--overwrite` is explicit, and protected
input aliases are refused. Original solver outputs remain input files.

## Geometry, source and compatibility rules

JSON vertices are metres, triangle indices are zero-based, and material labels
are sorted alphabetically with `_RIGID` last. Side codes are 0 rigid, 1 back,
2 front, and 3 both; vertex winding determines front/back. `_RIGID` requires
side 0. Degenerate triangles and triangles smaller than `1e-6 m²` are removed;
`--min-triangle-area` changes that threshold. Open meshes are supported: the
graph represents blocked links between air nodes, not an inside/outside solid
occupancy classification.

The nearest intersecting triangle selects one material and normal per boundary
node, preserving the existing approximation at material edges. Equal distances
use the stable triangle ordinal, independent of BVH traversal. Geometry uses
float64, a conservative BVH, `1e-3*h` fuzzy edge tolerance, and a near-node
tolerance of `1e-6` times link length.

Three intentional corrections affect legacy numerical comparisons:

* Sound speed uses Kelvin: `343.2*sqrt((Tc+273.15)/293.15)`, replacing
  `343.2*sqrt(Tc/20)`. Humidity is retained for air post-processing.
* Surface correction counts every blocked link. Adding NumPy boolean arrays
  in the legacy pair expression could count two opposite blocked links once.
  FCC surface factors are stored unscaled and scaled once by the solver loader.
* Near-node hits isolate all incident links. Conservative reciprocal union
  blocks both endpoints if either endpoint blocks a link. Newly added endpoints
  stay rigid, and isolation is not propagated beyond that edge. Exact equality
  of opposite links is a required gate before and after the storage transform.

Source and receiver stencils use eight trilinear corners. FCC uses an even-parity
cube of spacing `2h`; Cartesian uses spacing `h`. Weights sum to one and
reproduce the physical coordinate. Every stored corner, including zero-weight
corners, must avoid mesh boundaries and absorbing boundary nodes. Receiver
duplicates are preserved; sources must have distinct grid indices.

The impulse is scaled by `l2/h` for Cartesian or `0.5*l2/h` for FCC and its
interpolation weight. Differentiation uses the actual bilinear recurrence
`y[n]=(2/Ts)*(x[n]-x[n-1])-y[n-1]`, with zero initial state. Its alternating
tail is retained, and `diff=1` tells post-processing to integrate the output.
It is not replaced by a two-sample difference pulse.

Additional HDF5 metadata records preparation backend, build revision,
`geometry_policy=double_bvh_fuzzy_edges_reciprocal_union_v1`,
`surface_factor_policy=sum_each_blocked_link_v1`, triangle counts, reciprocal
corrections, added rigid endpoints, and target precision. Material labels are
saved beside their DEF tables. These corrections do not promise bitwise
compatibility with the former Python preparation pipeline.

## Verification status

`make -C c_cuda test-grid test-mesh test-prepare` runs native tests for grid
geometry, BVH queries and complete HDF5 preparation. The preparation suite
checks Cartesian/FCC folded/unfolded and rigid-only inputs, true bilinear source
signals, interpolation moments, reciprocal links, negative JSON/DEF cases,
and refusal to overwrite outputs. Its round-trips use the original solver
loader. Strict native address/undefined-behavior sanitizer checks passed.

Native CPU preparation also loaded the real CTK model on Cartesian and FCC
grids and the Musikverein model on folded FCC, using their committed DEF files.
For CTK FCC at `h=0.15 m`, source 1 and 1537 samples, original CPU solvers ran
both folded and unfolded inputs. Both FP32 outputs contained 73776 finite
samples, including 72536 nonzero samples. Their normalized RMS difference was
`3.13e-6`; the layout changes floating-point addition order, so they were not
bitwise equal. FP64 confirmed the transform with maximum absolute difference
`5.13e-11` and normalized RMS difference `7.80e-15`; every sample passed the
pointwise comparison `abs(error) <= 1e-10 + 1e-11*abs(reference)`. Both FP32
outputs also completed native receiver recombination, integration, filtering
and native-rate output processing. After 10 Hz/order-8 high-pass and symmetric
500 Hz low-pass filtering, FP32 folded/unfolded outputs differed by at most
`3.54e-6` with normalized RMS difference `1.35e-3` (0.135%); a pointwise
`1e-6 + 1e-3*abs(reference)` comparison failed 1675 of 9222 samples. This is
a precision limitation, not a claim of acoustic equivalence. The corresponding
FP64 processed outputs had maximum difference `9.38e-15` and normalized RMS
difference `4.25e-12`, with all 9222 samples finite and nonzero.
CUDA builds target Ada and GB10, but a cloud instance without a usable GPU can
verify compilation and device-absence behavior only. It cannot establish CUDA
runtime correctness, CUDA acoustic equivalence or a measured GPU speedup.
