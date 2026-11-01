# Warp SOL prototype

`misc_scripts/warp_sol_prototype.py` runs the SOL field-line workflow with
float64 Warp kernels on CPU or CUDA. It accepts the existing SOL JSON inputs,
uses the saved Poincare LCFS to generate seeds, and writes the same compact
per-field-line metadata and plane-sharded crossings consumed by SOL
regularization. The production `illiad-sol-trace` command remains unchanged.

From the repository root, in an environment with the optional Warp dependency:

```bash
conda run --no-capture-output -n warpenv python misc_scripts/warp_sol_prototype.py \
  input_files/sol_trace_inputs.json --device cuda --no-plots
```

Omit `--device` to use the JSON `DEVICE` (`auto` chooses CUDA when Warp can use
it, otherwise CPU). Omit `--no-plots` to respect `GENERATE_PLOTS`. Both positional
JSON paths and `--inputs PATH` are supported. With no input argument, the script
loads `input_files/sol_trace_inputs.json` relative to the repository.

The default output subdirectory is the input's `ANLYS_SUBDIR` plus `_Warp`, so
the current input writes to `output/IOTA3_1000sp_atol1e-9/data/Oct24SOLTrace_Warp`.
Use `--output-subdir NAME` to choose another subdirectory. Rerunning replaces
the trace artifacts in that chosen destination. Logs use `warpSolTrace.log`;
plots use the corresponding plots subdirectory.

## Numerical and execution scope

- Midpoint and RK4 integrate `dx/ds = +/- B/|B|`. Each direction stops at the
  circular toroidal wall, an invalid field sample, or the existing length cap
  `2*pi*MAJOR_RADIUS_M*SPINS`. A final partial step respects that cap.
- A wall-crossing step uses the existing straight-segment bisection convention;
  the directional length gains the fractional step. If either direction fails,
  the combined connection length is NaN. Length-capped directions remain valid
  and carry `reached_max_length`, as in the current SOL tracer.
- The existing Boris Warp volume-weighted interpolator supplies field values,
  including sector rotations and the Cartesian error-field addend. PyTorch is
  still required for the existing field loader and shared workflow imports.
  Field loading/combining occurs on CPU once; integration, wall refinement,
  and crossing capture use Warp arrays and kernels exclusively.
- `STEP_CHUNK_SIZE` is the maximum number of physical steps per launch.
  `COMPILE_STEP_CHUNKS` is accepted for JSON compatibility; Warp always JIT
  compiles its kernel. This flag does not select Torch compilation.
- This prototype sets its module's `max_unroll=0`. With Warp 1.17.0 on an
  RTX 4080 SUPER, the default unrolled release kernel reproducibly failed on
  the first chunk with CUDA error 700. Debug mode, optimization level zero,
  and release mode without unrolling passed; disabling floating-point fusion
  did not fix the failure. Keeping the interpolation loops as loops is the
  local workaround, while preserving release optimization and Boris settings.
  Crossing writes also have a bounds guard and an explicit overflow error.
- Directional batches are capped by both `BATCH_SIZE` and
  `CROSSING_BUFFER_SIZE`. The effective chunk is further capped so
  `batch_count * chunk_steps <= CROSSING_BUFFER_SIZE`. The shared conservative
  `STEP_SIZE_M` guard permits at most one crossing per step, so every event fits
  before the host flushes the buffer. A tiny buffer is valid but causes more
  launches. Memory usage is bounded independently of total trace duration.
- State retains original field-line IDs. Crossings are atomically appended;
  row order within a plane can differ on CUDA. Seed samples have source
  direction zero, and traced crossings have direction +1 or -1.
- Plane IDs, seam wrapping, epsilon tests, and linear crossing reconstruction
  follow the Torch tracer. Warp keeps plane-angle targets in float64. The Torch
  expression multiplying integer plane IDs by a Python float uses Torch's
  default floating dtype (usually float32); this can cause small crossing
  coordinate differences. This prototype does not change the production code.
- The prototype checks active counts and downloads crossings after each chunk.
  It skips inactive threads and stops an empty batch early; it does not compact
  the active row list or use CUDA graphs. Progress renders at most every five
  seconds, with log updates controlled by `PROGRESS_INTERVAL_STEPS`;
  `PROGRESS_REFRESH_STEPS` is accepted but unused.

The low-level `integrate_warp_batch` accepts an existing Warp `VectorGrid`,
NumPy seeds, IDs, direction signs, and a sink implementing
`append_xyz(plane_index, xyz, fieldline_ids, source_directions)`.
`trace_warp_volume` combines both directions, while `WarpSOLTracer` supplies
the saved-data/plotting workflow. Warp array and kernel usage follows the
[NVIDIA Warp runtime documentation](https://nvidia.github.io/warp/v1.17/user_guide/runtime.html).

## Validation

```bash
python3 -m py_compile misc_scripts/warp_sol_prototype.py tests/test_warp_sol_prototype.py
MPLCONFIGDIR=/tmp/illiad-mplconfig MPLBACKEND=Agg \
  conda run -n warpenv python -m unittest discover -s tests -p test_warp_sol_prototype.py -v
```

The CPU tests cover both integrators, chunks of one and eight, positive and
negative directions across full-torus and field-period seams, wall bisection
against an analytic vertical field, partial final steps, zero-field failure,
small crossing buffers, error-field addition, failed combined lengths,
persistent IDs, seed records, and reopening saved shards. Nine CPU tests and
two CUDA tests passed with Warp 1.17.0. CUDA tests compare both integrators with
Torch using chunks of one/eight and exercise a 32,768-direction first chunk.
The CUDA tests skip when no GPU is accessible.
Directional lengths and wall coordinates are compared to Torch with
`2e-10 m` absolute tolerance. Crossing identity matches exactly in these tests;
coordinates use `3e-7 m` tolerance against the default-float32 plane targets and
`2e-10 m` against float64 targets. These tolerances describe the test workloads,
not error bounds for long chaotic trajectories.

A separate CPU CLI smoke run used the saved IOTA3 LCFS40 and configured
magnetic fields/error field, with one seed plane, a 2-by-4 polar grid, and one
spin per direction. It traced 8 seeds/16 directions, saved 1,144 samples, and
returned 14 wall hits and 2 capped directions. A Torch rerun of those same seeds
matched status flags and sample counts; maximum directional-length difference
was `2.79e-13 m`. Smoke output was isolated under `/tmp` and plots were disabled.

After the unrolling workaround, CUDA release-mode runs on the RTX 4080 SUPER
completed 64 steps for all 30,240 directions from the current IOTA3 seed grid
with both midpoint and RK4. Both recorded 15,562 wall hits; midpoint captured
186,658 crossings and RK4 captured 186,661. These short runs test GPU execution,
not full production-length convergence or performance. First-use JIT time is
included in the script's reported trace/output elapsed time; no speedup claim
is made.

If an earlier run hit CUDA error 700, restart the Python process/debugger
session before retrying. The failure may surface at `counters.numpy()` because
the copy waits for the preceding kernel. CUDA leaves the failed process in an
unusable state; see the
[CUDA error documentation](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__TYPES.html).
