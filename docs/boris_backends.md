# Boris execution methods

The production `illiad-boris` workflow supports two `BORIS_METHOD` values:

```json
"BORIS_METHOD": "warp",
"WARP_STEP_CHUNK_SIZE": 16,
"WARP_COMPACTION_INTERVAL": 256
```

Use `"torch"` for the existing PyTorch implementation (also the default when
omitted). Add the key to your existing Boris input JSON and run the same command:

```bash
illiad-boris --inputs input_files/your_boris_inputs.json
```

Warp is an optional dependency, available through the `illiad-fieldlines[warp]`
extra. An environment with Warp already installed needs no extra installation.
Selecting Torch does not import Warp. Selecting Warp without it installed raises
an actionable error before field loading or output setup. Both methods still use
PyTorch for the existing field-loading and output interfaces.

## Shared workflow and physics

Both methods use the same production LCFS emitter distribution and initial
velocity generation, input plasma conditions, solved B/E/density fields, field
scaling, collision selectors, and output postprocessing. Changing the backend
requires no regeneration, transposition, or manual wrapping of saved fields.

Supported collisions are `viscous_drag` and `langevin` for ion-neutral collisions,
and `linear_fp` and `fokker_planck` for ion-ion collisions; use JSON `null` to
disable either category. Neutral collisions precede ion-ion collisions at both
half-step sites. The spatial density field is used for enabled ion-ion collisions;
the Python API also allows the existing constant-density fallback. This preserves
the current collision models and mass conventions, including Li-He full FP.

Both methods save `Wallpt_OUTPUT.npy` and `Ion_traces.npy` through the existing
output handler and use the existing plots. Saved `Ion_traces.npy` arrays use
float32; solver state, in-memory traces, and wall outputs retain float64.
Trace selection and `STRIDE` work for
Warp: samples include the initial state, stride samples, and the final state,
including early termination. Wall positions remain the first outside-wall sample,
not an interpolated surface crossing. The wall output's timestep row retains the
existing last-inside-step convention. Runs without wall hits save a `(7, 0)` wall
array and skip wall plots; runs without selected traces skip the trace plot.

Stochastic trajectories are not identical between methods: Torch and Warp use
different random-number generators. Compare ensembles, wall distributions, and
survival curves. Deterministic results should agree to floating-point tolerance.

## Python API

```python
from illiad.boris import Boris

solver = Boris(simIO, method="warp")
solver.setConditions(ions, cond_string, dt=dt, tmax=tmax)
output = solver.run(Bfield, Efield, nfield,
                    ion_neutral_collisions="langevin",
                    ion_ion_collisions="fokker_planck",
                    trace_IDs=[0, 10], trace_stride=100,
                    warp_step_chunk_size=16, warp_compaction_interval=256)
```

`run(..., method="torch")` or `parallel_solver(..., method="torch")` overrides the
constructor choice for that call. `parallel_solver` retains its four-tensor output
contract and optional `freq_corr` argument for both methods.

The Warp implementation lives in `illiad.boris_warp` and
`illiad.collisions_warp`. The old prototype modules are compatibility imports;
the benchmark now imports the package implementation directly.

## Execution, fields, and memory

Warp uses float64 arithmetic and the device of the loaded B field (CUDA when
available in the normal workflow, otherwise CPU). It shares the Torch field
storage rather than uploading a second field copy. Supported meshes are uniform
toroidal `TorchMesh` grids with B periodicity `[0, 1, P]` and aligned full-torus
E/density grids. The adapter validates layout and spacing and handles the existing
one-spacing angular grid origin, sector wrapping, vector rotations, density
layout, and additive magnetic error field.

A kernel fuses interpolation, collision half-steps, pushing, and wall recording.
`WARP_STEP_CHUNK_SIZE` controls physical timesteps per launch (default 16; a
positive int32 integer). Set it to 1 for one timestep per launch. Torch ignores
this setting. `DT`, `TMAX`, and collision half-step durations do not change.
The final chunk is shortened to the remaining number of steps.

Each original particle retains one thread. Position, velocity, and RNG state
stay local across substeps, while fields and density are interpolated anew at
every physical step. A thread stops immediately on a wall hit.

`WARP_COMPACTION_INTERVAL` sets the physical steps between active-particle ID
filtering passes (default 256; nonnegative int32 integer). The pass occurs at the
first chunk boundary at or beyond that interval since the previous pass, and at
the final chunk. For example, chunk size 16 and interval 100 filter every 112
steps. Subsequent pushes launch only the retained particles. A zero survivor
count stops the host loop, and traces are trimmed to the exact last hit.

Set the interval to 0 to disable compaction. This keeps the original full-particle
launch size and checks for complete termination at chunk boundaries after at
least 256 steps since the previous check, or at the final chunk. Torch ignores
both Warp settings. There is no CUDA graph capture.

Compaction uses two reusable int32 ID buffers and an atomic append per survivor.
Particle state, RNG state, wall output, and traces remain indexed by permanent
particle IDs. GPU scheduling may reorder the active list, but it does not change
particle identity or the random stream assigned to that particle. Selected
traces, including duplicate selections and terminated particles, retain their
existing order and format.

Filtering costs a kernel launch and a survivor-count download. Very frequent
filtering can outweigh the benefit, particularly while most particles remain
active. This reduces scheduled work, not retained particle/output memory: the
two ID buffers add approximately 8 bytes per original particle. They are omitted
when compaction is disabled. The default interval is a starting point, not a
measured GPU optimum.

Selected trace samples are written inside the kernel using global timestep
indices, including every substep with `STRIDE: 1`. Initial and final samples
remain present. A finishing kernel fills post-termination samples with each
tracked particle's frozen wall position, preserving the rectangular trace format.
Chunk size does not multiply trace storage; selected particle count and `STRIDE`
still determine it. Selected traces require an additional int32 lookup per particle
and per trace selection; wall-only runs omit those lookups.

Chunking reduces launch overhead and repeated state loads/stores. Larger chunks
are not necessarily faster: workload imbalance, register use, and slower progress
updates can offset the benefit. The default of 16 is a starting point, not a
measured optimum for your GPU. Benchmark, for example, sizes 1, 8, 16, and 32:

```bash
python misc_scripts/benchmark_warp_boris.py --inputs input_files/your_boris_inputs.json --warp-step-chunk-size 16
```

Without that override, the benchmark reads `WARP_STEP_CHUNK_SIZE` from the JSON
(default 16). `--warp-compaction-interval 0` disables compaction for comparison;
otherwise it reads `WARP_COMPACTION_INTERVAL` (default 256). Both effective values
are recorded in the timing report. Keep chunk size fixed when comparing
compaction intervals. These wall-only measurements do not measure full-trace
recording overhead.

Warp compilation is lazy. The first use of a changed kernel can take appreciable
time; its persistent cache normally avoids recompilation on subsequent runs.
The production solve includes initialization/compilation overhead. The benchmark
continues to warm up before reporting steady-state timing.

Fusing operations avoids many batch-sized Torch temporaries, but retained fields,
particle state, wall data, and selected traces still consume memory. Trace memory
scales with saved steps times selected particles. PyTorch allocator statistics
exclude Warp-owned allocations; the CLI labels those statistics accordingly.
No GPU peak-memory reduction is guaranteed by the backend choice alone.
