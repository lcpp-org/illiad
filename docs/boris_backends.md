# Boris execution methods

The production `illiad-boris` workflow supports two `BORIS_METHOD` values:

```json
"BORIS_METHOD": "warp"
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
output handler and use the existing plots. Trace selection and `STRIDE` work for
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
                    trace_IDs=[0, 10], trace_stride=100)
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
Each original particle retains one thread; inactive threads return immediately.
The host checks for complete termination every 256 steps and trims traces to the
actual last hit. There is no active-list compaction or CUDA graph capture.

Warp compilation is lazy. The first use of a changed kernel can take appreciable
time; its persistent cache normally avoids recompilation on subsequent runs.
The production solve includes initialization/compilation overhead. The benchmark
continues to warm up before reporting steady-state timing.

Fusing operations avoids many batch-sized Torch temporaries, but retained fields,
particle state, wall data, and selected traces still consume memory. Trace memory
scales with saved steps times selected particles. PyTorch allocator statistics
exclude Warp-owned allocations; the CLI labels those statistics accordingly.
No GPU peak-memory reduction is guaranteed by the backend choice alone.
