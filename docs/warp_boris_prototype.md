# Warp Boris prototype

The prototype has been integrated into the production package as the optional
`BORIS_METHOD="warp"` backend. See [Boris backends](boris_backends.md).
The original `misc_scripts/warp_boris_prototype.py` and collision module now
re-export package functions for compatibility. The benchmark uses the package.

The notes below describe the original wall-only prototype interface. Production
Warp additionally supports selected traces, frequency correction, timestep
chunking (`WARP_STEP_CHUNK_SIZE`, default 16), active-ID compaction
(`WARP_COMPACTION_INTERVAL`, default 256), and early termination; use the backend guide for current production behavior.

## What is implemented

- `VectorGrid` and `DensityGrid` structs for calculated vector/scalar fields.
- Device helpers for geometry, vector interpolation, and scalar density sampling.
- Two `@wp.kernel` entry points: `initialize_velocity`, `push_and_record_wall`.
- Host functions: `make_grid`, `make_density_grid`, `run_wall_only`.
- Optional neutral/ion-ion operators, configuration, and a random-state
  initialization kernel in `misc_scripts/warp_collision_operators.py`.

Each timestep launches one logical thread per original particle. Its thread
interpolates B and optional E/density, applies optional collision half-steps around
the ordinary Boris update, advances the position, and records the first
outside-wall state. Inactive threads return immediately. There is no frequency correction,
particle trace arrays, device-side timestep chunks, CUDA graphs, or active-list
compaction. The Python loop runs the requested number of steps even if all
particles have hit the wall. This keeps launch and state-management syntax visible.

The interpolation and push are compiled together; the device helpers do not each
require a separate launch. Local three-vectors replace batch-sized intermediate
tensors. This is where launch and temporary-memory savings may come from; no
speedup has been measured.

## Using already calculated fields

From the repository root, after Warp becomes available:

```python
import numpy as np
from misc_scripts.warp_boris_prototype import make_grid, run_wall_only

# These are illustrative paths to your already combined/scaled field arrays.
# Existing loadCartesianField files contain (3, nr, ntheta, nphi).
B_nodes = np.moveaxis(np.load("B_combined_scaled.npy"), 0, -1)
E_nodes = np.moveaxis(np.load("E_scaled.npy"), 0, -1)

b_grid = make_grid(B_nodes, R0=0.72, a=0.19, periods=5, device="cuda:0")
e_grid = make_grid(E_nodes, R0=0.72, a=0.19, periods=1, device="cuda:0")

# Arrays you already construct when initializing ions:
x0 = np.asarray([ion.pos0_XYZ for ion in ion_list])
v0 = np.asarray([ion.vel0_XYZ for ion in ion_list])
q_over_m = np.asarray([ion.charge_mass_ratio for ion in ion_list])

result = run_wall_only(x0, v0, q_over_m, b_grid,
                       e_grid=e_grid, dt=dt, steps=int(tmax // dt))
hit = result["hit_step"] >= 0
wall_xyz = result["wall_position_xyz"][hit]
wall_velocity = result["wall_velocity_xyz"][hit]
np.savez("warp_wall_output.npz", **result)
```

`ion_list`, `dt`, and `tmax` above are inputs from your existing setup, not objects
created by this module. For B-only tracking, omit `e_grid`. `device="cpu"` selects
Warp's CPU backend when later checking the implementation without a GPU.

Alternatively, an existing `TorchMesh.B` already has `(nr, ntheta, nphi, 3)` layout:

```python
B_nodes = b_hidra.B.detach().cpu().numpy()
error = (b_hidra.err_adder.detach().cpu().numpy()
         if bool(b_hidra.errField) else (0.0, 0.0, 0.0))
b_grid = make_grid(B_nodes, R0=b_hidra.R0, a=b_hidra.a,
                   periods=int(b_hidra.periodicity[2]),
                   device="cuda:0", addend=error)
```

That adapter copies through CPU memory once; zero-copy Torch/Warp wrapping is a
later integration optimization. Coil-current scaling, attenuation, helical-field
addition, and potential scaling must already be applied. `addend` represents the
global Cartesian magnetic error field; it is added after periodic vector rotation.

## Array and wrapping contract

| Quantity | Prototype convention |
| --- | --- |
| Host vector field | Float64 `(nr, ntheta, nphi, 3)` |
| Warp vector field | Three-dimensional `wp.array3d(dtype=wp.vec3d)` |
| Radial nodes | `r[j] = j*a/(nr-1)`, including axis and wall |
| Poloidal nodes | `theta[j] = (j+1)*2*pi/ntheta`, ending at `2*pi` |
| Toroidal nodes | `phi[j] = (j+1)*(2*pi/periods)/nphi` |
| Cartesian coordinate convention | `phi = -atan2(y, x)` wrapped into `[0, 2*pi)` |
| Particle data | Float64 `(N, 3)` position in metres and velocity in m/s; signed `q/m` in C/kg |

Supported mesh periodicity is `[0, 1, P]`. The prototype does not support the
other grid-start conventions that `TorchMesh` can represent. B and E must have
matching radial/poloidal nodes, E must cover the whole torus, and
`E.nphi == B.nphi * B.periods`. Array shape cannot establish node alignment by
itself: the supplied fields must follow the coordinate contract above.

`sample_vector` retains the eight toroidal sub-volume weights from
`TorchMesh.get_weights`, rather than substituting trilinear Cartesian weights.
It also retains the axis weighting and the actual rotation matrix/signs from
`TorchMesh.return_vecs`. Two indexing points matter:

1. The angular high index is `floor(angle/spacing)` and the low index is one
   less, because the first stored node is at one spacing. Every negative angular
   index is wrapped explicitly with a positive modulo. No new ghost nodes or
   zero-angle nodes are required.
2. At a magnetic sector seam, the low-phi corner vectors are rotated by
   `+phi_max`; the interpolated vector is then rotated by `-sector*phi_max`.
   Merely wrapping the integer index is insufficient for Cartesian B components.

For readability, E is sampled independently on its full-torus grid, duplicating
coordinate/weight arithmetic. On aligned grids this corresponds mathematically
to the production solver's shared B weights and full-torus E indexing, subject
to floating-point differences. In a subsequent fused B/E sampler, offset the
**unwrapped** B phi indices by `sector*B.nphi`, then wrap against `E.nphi`.
Wrapping the B low index first would select the wrong E cell at a sector seam.

## Startup and wall semantics

`initialize_velocity` deliberately transcribes the existing startup expression
in `Boris.parallel_solver`, including its minus-cross-product/2 term and electric
kicks. This is a compatibility choice, not a new derivation of a backward
half-step. Any scientific revision to that initialization belongs in a separate
change. The main kernel uses the ordinary collisionless Boris update with
`freq_corr=False`.

Initial positions must be strictly inside the wall. Each thread checks the wall
after moving and never interpolates its outside position. The position saved at
a hit is the first point with minor radius `>= a`, matching the present solver's
overshoot convention; it is not projected onto the torus. The radius expression
has an additional nonnegative guard against roundoff at the magnetic axis.

Returned arrays retain one row per input particle:

- `wall_position_xyz`, `wall_velocity_xyz`: `(N, 3)`, zero for survivors.
- `hit_step`: `(N,)`, actual first outside step (starting at 1), or -1 for survivors.
- `last_inside_step`: `(N,)`, the current solver's `maxStep` convention, initially
  zero and updated only after steps ending inside. For a hit at step k it is k-1.

Use `hit_step` to filter hits; a zero wall velocity is not a reliable hit mask.
These NumPy arrays are not the existing seven-row `Wallpt_OUTPUT.npy` format.
Converting positions to RTP and stacking velocity and `last_inside_step` would
reuse the corresponding part of `Boris.post_solver`. There is no trace return.

## How much production code would change?

The prototype uses two Python files plus the separate benchmark and this note;
the production solver is unchanged. A production collisionless backend would replace the numerical logic in these
existing functions, preferably by leaving Torch implementations available and
adding Warp counterparts:

| Existing file/functions | Scope |
| --- | --- |
| `illiad/boris.py`: `parallel_solver` | Substantial: tensor timestep loop becomes buffer setup and kernel launches |
| `illiad/mesh/torch_mesh.py`: `get_weights`, `return_vecs` | Substantial: numerical work becomes typed per-particle device functions |
| `illiad/mesh/torch_mesh.py`: `rot_vecXYZ_byPHI`; `illiad/utilities/coordtrans.py`: `XYZ_to_RTP2` | Small mathematical translations, included in this prototype; originals need not change |
| `illiad/boris.py`: `run`, `post_solver`, `save_output` | Small adapters for backend selection and absent traces; preserve wall output format |
| `illiad/cli/boris.py`: `main` | Small dispatch/configuration and plotting guards for wall-only output |

Thus **two existing computational areas / three principal functions** require
substantial porting. File loading, field generation, ion initialization, and
wall plotting can remain in their existing Python workflows. Trace-dependent
plotting and saving need to be skipped when integrating this wall-only path.
The collision-enabled extension additionally ports the operators from
`illiad/collisions.py` into `misc_scripts/warp_collision_operators.py` and the
scalar density lookup from `TorchMesh` into the prototype.

## Collision operators and density field

`run_wall_only` remains collisionless by default. Pass a `CollisionConfig` made
by `make_collision_config()` to enable the same named models as the Torch solver:

| Selector | Accepted models |
| --- | --- |
| `ion_neutral_collisions` | `None`, `viscous_drag`, `langevin` |
| `ion_ion_collisions` | `None`, `linear_fp`, `fokker_planck` |

As in the existing selector contract, `False` and strings `none`, `false`, `off`
disable a model. Invalid names and `True` are rejected.

For example, after constructing `b_grid`, `e_grid`, and initial particle arrays:

```python
from misc_scripts.warp_boris_prototype import (
    make_collision_config, make_density_grid, run_wall_only,
)

# The saved scalar file is (phi, theta, r), unlike Cartesian vector-field files.
# Apply PLASMA_DENSITY once, as TorchMesh.loadScalarField does. If the file is
# already physical density in m^-3, use scale=1.0 instead.
n_grid = make_density_grid(
    np.load(params["FIELD_FILE_DENSITY"]), b_grid,
    scale=params["PLASMA_DENSITY"],
)
collisions = make_collision_config(
    ion_neutral_collisions=params["ION_NEUTRAL_COLLISIONS"],
    ion_ion_collisions=params["ION_ION_COLLISIONS"],
    n_gas=params["NEUTRAL_GAS_DENSITY"],
    T_gas_eV=params["NEUTRAL_GAS_TEMP_EV"],
    Ti_eV=params["ION_TEMP"],
    m_gas_amu=4.002603,  # He; supply the mass of the configured background gas
    n_e=params["PLASMA_DENSITY"],
)
result = run_wall_only(
    x0, v0, q_over_m, b_grid, e_grid=e_grid, dt=dt, steps=steps,
    collisions=collisions, density_grid=n_grid,
)
```

`make_density_grid` transposes to internal `(r, theta, phi)` order, scales and
clamps negative values to zero, matching `loadScalarField`. Density must cover
the full torus on nodes aligned with B: its saved shape must be
`(B.nphi*B.periods, B.ntheta, B.nr)`. Sampling uses B's toroidal volume weights
and offsets the unwrapped phi indices into the full-torus scalar array before
wrapping. Scalar values are never vector-rotated. Omitting `density_grid` uses
the configured constant `n_e`; neutral collisions use `n_gas`, not that field.

The timestep ordering reproduces `Boris.parallel_solver`:

1. Sample B/E and ion-ion density at the current position.
2. Apply the neutral half-step, then the ion-ion half-step.
3. Apply Boris and advance position.
4. Apply neutral and ion-ion half-steps again using the same density sample.
5. Record wall position and the velocity after those second half-steps.

The startup velocity adjustment stays collisionless, as in the current solver.
No density evaluation is attempted at an outside-wall position.

The port preserves existing physics choices: neutral cross section `1e-19 m^2`,
gas-mass-based thermal variance for Langevin and linear FP (the current Boris
driver sets `m_ion_amu = m_gas_amu`), and fixed Li/He masses, unit charge numbers,
zero background drift, and `lnLambda=10` for full FP. It retains the small-x
Chandrasekhar series and small-speed floors. This is a backend port, not a change
to those model assumptions.

Each stochastic particle has its own persistent Warp random state indexed by
its original row. Neutral/ion-ion draws and both half-steps consume successive
samples; state advances across timesteps. `seed=None` chooses a fresh seed per
call; an integer `seed` in `[0, 2**31)` gives repeatable Warp runs. Warp and Torch
do not produce identical random streams from an equal seed. Warp 1.17's native
`randn` draws are float32; they are promoted to float64 before all collision
arithmetic. Deterministic drag needs no random state.

The benchmark script resolves the JSON collision selectors and passes the same
models, density, and plasma settings to both backends. Setting both selectors to
`null` retains collisionless benchmarking.

Validation in `warpenv` used actual compiled CPU kernels: all four operators and
the Coulomb rates matched Torch with supplied identical Gaussian samples;
density interpolation matched random points and all five magnetic sector seams;
collisionless, drag, and both combined stochastic models matched the full Torch
one-step wall outputs. Identical Gaussian draws were supplied to Torch for these
checks, so they test the formulas rather than comparing unrelated realizations.
Two-step combined-model checks also matched, exercising RNG-state advancement
and density resampling between timesteps. Constant-density fallback, selector
validation, repeatable seeds, and distinct-seed behavior were checked separately.
GPU validation and performance measurements have not been performed.

## Validation boundary

The original prototype was prepared without Warp installed. The arithmetic
checks below predate the compiled benchmark described in the following section.
Python syntax checks alone do not verify Warp's type checking, generated code,
or GPU behavior. Production adoption still needs GPU validation and representative
field-sampling/trajectory comparisons.

Checks performed while preparing the prototype:

- `python3 -m py_compile misc_scripts/warp_boris_prototype.py` passed.
- The five device-function/kernel bodies were extracted and evaluated as scalar
  NumPy arithmetic in a temporary check script, then compared with the current
  Torch implementation. This did **not** execute Warp or its host adapter.
- B and E interpolation agreed at 559 random/axis/angular-boundary/near-wall
  samples, with maximum absolute differences of `6.66e-14` and `5.85e-13` on the
  synthetic fields. Independent E interpolation also matched the existing
  shared-weight lookup within the asserted `2e-10` absolute/relative tolerance.
- B-only and B+E cases each compared 32 particles over 50 timesteps against
  `Boris.parallel_solver`: 31 hits and one survivor, matching wall states within
  numerical tolerances and matching hit masks and last-inside steps exactly.
- Three additional exact-axis samples caused a negative squared radius through
  cancellation in the existing coordinate conversion. The prototype's zero
  clamp produced finite samples; those three were excluded from parity checks.

These are arithmetic checks on small synthetic fields, not Warp compilation,
GPU tests, production-field validation, or performance measurements.

Syntax references: NVIDIA's [Warp basics](https://nvidia.github.io/warp/latest/user_guide/basics.html)
and [runtime/types documentation](https://nvidia.github.io/warp/latest/user_guide/runtime.html).

## Minimal benchmark with solved B and E fields

[`misc_scripts/benchmark_warp_boris.py`](../misc_scripts/benchmark_warp_boris.py)
compares this prototype with the actual eager `Boris.parallel_solver`. In an
environment containing both Torch and Warp, run from the repository root:

```bash
python misc_scripts/benchmark_warp_boris.py \
  --inputs input_files/boris_iota3_newsoltrace500_25v.json \
  --particles 40960 --steps 1000 --repeats 3 \
  --output output/warp_boris_benchmark.json
```

The default device is `cuda:0`; use `--device cpu` for an explicit CPU comparison.
There is no automatic CPU fallback that could be mistaken for GPU timing. Both
libraries must be able to use the selected device.

The script uses the existing toroidal/helical B component files and combines
them using the supplied JSON's currents and attenuation settings, exactly as
the Boris CLI does. It loads `FIELD_FILE_ELECTRIC` and applies the resolved
`PLASMA_POTENTIAL` multiplier. When ion-ion collisions are enabled, it also loads
`FIELD_FILE_DENSITY`, applies `PLASMA_DENSITY` once, and shares the resulting
physical density values between backends. No field solver is run. The
default JSON points to the existing IOTA3 NewSOLTrace500 NoIslands2 E field at
25 V; other Boris JSONs can be passed with `--inputs`. Relative field paths are
resolved from the repository root.

Particles now come from the production `ionInitializer()`: the JSON's
`LCFS_INDEX`, `NPHI`, `NTHETA`, and `DELTRS` select the same LCFS emitter positions
from `output/<OUTPUT_DIRECTORY_NAME>/data/Poincare`. The production initializer
also supplies its E-field normals (with the existing geometric fallback),
Maxwellian speeds, and cosine-weighted hemispherical emission directions using
`ION_TEMP`, `ION_MASS`, and `CHARGE_NUM`.

The default `--particles` is **40,960**. This overrides
`NPARTICLES_PER_EMITTER`: the initializer makes `ceil(particles/emitter_count)`
copies per configured emitter, then a uniformly sampled subset without
replacement gives the exact requested count. The selected rows retain their
emitter order. With the default 16,200 emitters, it generates three copies each
and selects 40,960 of the 48,600 particles. This preserves the production emitter
grid and emission distribution while controlling the benchmark size. Counts
smaller than the emitter grid sample only some emitters.

A new ensemble is generated once per invocation; both backends and all repeats
share the exact same selected positions and velocities. Initializer data/figure
saves are disabled through a read-only IO adapter, protecting existing production
artifacts. The initializer still builds its diagnostic figures in memory, so
startup may take some time. That work is outside the timing. The former circular
shell option has been removed. Collision selectors follow the input JSON;
tracing remains disabled. `DT` is retained unless overridden with `--dt`, and `--steps` replaces
the long production `TMAX`.

`ION_NEUTRAL_COLLISIONS` accepts `viscous_drag`, `langevin`, or `null`;
`ION_ION_COLLISIONS` accepts `linear_fp`, `fokker_planck`, or `null`.
Disabled aliases are normalized before deciding whether to load density. The
default input enables `langevin` plus `fokker_planck`. Both backends receive
`NEUTRAL_GAS_DENSITY`, `NEUTRAL_GAS_TEMP_EV`, `ION_TEMP`, `PLASMA_DENSITY`, and the
mass resolved from `BACKGROUND_GAS_SPECIES`. With ion-ion collisions disabled,
no density file is opened. Neutral collisions use the constant neutral density.

Stochastic collisions use independent random draws across backends and repeats;
the initial particle arrays are still shared. Wall/step differences therefore
describe independent realizations, not deterministic numerical parity. Compare
the distributions and survival curves rather than expecting matching individual
trajectories. The final-repeat plots retain their existing behavior.

The integrated JSON-driven benchmark was smoke-tested in `warpenv` on CPU with
the solved IOTA3 fields and a reduced production LCFS grid (2 toroidal planes,
8 poloidal emitters, 32 particles, 10 steps). Langevin + full FP, drag + linear
FP, and disabled selectors all completed and saved the reports/plots/arrays.
The disabled case used an intentionally missing density path to verify that
density loading is skipped. These short runs had no wall hits and are workflow
checks, not stochastic distribution validation or GPU benchmarks.

Timing includes particle allocations/uploads, startup velocity adjustment,
stepping, and NumPy wall-result download for both backends. Field loading/upload,
initial-condition generation, first-use compilation, and warmup are excluded.
Both devices are synchronized around each measured call. Runs alternate backend
order, and the script prints individual times, medians, and `Torch/Warp` (greater
than one means Warp was faster). Torch's progress rendering is disabled within
this standalone process. No production source is monkey-patched on disk.

The optional JSON report records the times, workload, library versions, device,
resolved collision models and plasma settings, density-field path, and wall
comparison. Hits and last-inside steps are compared by original
particle row; maximum wall position/velocity differences use particles that hit
in both backends. Differences are reported, not treated as benchmark failures.
If neither backend records wall hits, wall-vector comparisons are null rather
than presented as validated. Nonfinite wall results abort the benchmark.

Plots and raw wall results are now saved automatically **after all timed runs**,
using the final measured result from each backend:

- `Wall_comparison.png`: Torch and Warp panels with shared production angular
  coordinates, 1-degree toroidal/2-degree poloidal bins, and the same logarithmic
  color scale. Density is normalized by each backend's total wall hits (as in
  `boris_plotWallHist`); the titles show absolute hit counts. Empty maps are
  explicitly labeled. The comparison does not overlay port outlines.
- `Ions_vs_time_comparison.png`: both active-ion fractions on one plot, with
  counts in the legend. Counts are calculated directly from actual hit steps:
  an ion hitting at step k is absent at time k*dt; survivors remain through the
  final timestep. No trajectory arrays or residence-time extrapolation are needed.
- `wall_results.npz`: each backend's wall positions/velocities, hit steps,
  last-inside steps, density map, and active counts, plus time/bin/geometry data
  for later replotting without repeating the simulation.

The directory defaults to `output/warp_boris_benchmark/`. With `--output`, it
defaults to that report path with its suffix removed. Override it explicitly:

```bash
python misc_scripts/benchmark_warp_boris.py --steps 10000 \
  --plot-dir output/warp_boris_comparison
```

Existing files with these names in the chosen directory are replaced on rerun.
The JSON report, when requested, includes paths to all three artifacts.

This measures the current implementations: Torch compacts active particles and
can stop early; Warp still launches its inactive threads through the requested
step count. No `torch.compile`, CUDA graph capture, or timestep chunking is used.
The printed ratio is a solver-call comparison, not an isolated kernel throughput
or general GPU speedup claim.

In `warpenv` (Warp 1.17.0, Torch 2.5.1.post303), the actual Warp kernels compiled
and executed on CPU against the default solved fields. GPU access was unavailable
in the validation session; this does not establish GPU compilation or performance.
A previous circular-shell smoke run with 64 particles, 200 steps, and two repeats
returned 39 hits in both backends, identical hit masks
and last-inside steps, maximum position difference `1.73e-18 m`, and maximum
velocity difference `4.55e-12 m/s`. This small CPU check validates the benchmark
path and is not a representative GPU performance measurement.

After switching to the production initializer, a CPU smoke run used all 16,200
configured LCFS emitters, selected the new default 40,960 particles from 48,600,
and completed two timesteps in each backend with matching hit masks and
last-inside steps. No particles hit the wall in that short run, so it checks the
full initialization/compiled execution path, not wall-position agreement.
