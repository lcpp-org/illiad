# ILLIAD Public API

This document defines the public interface of ILLIAD version 1.0.0. Interfaces
not listed here are implementation details even when they are importable from
a source checkout.

ILLIAD is research software for a staged field-line, flux-surface, SOL, and
impurity-ion transport workflow. Its primary interface is the installed
command set and JSON configuration files; selected analysis classes are also
available for programmatic use.

## Versioning Policy

ILLIAD follows [Semantic Versioning 2.0.0](https://semver.org/). The installed
version is available as:

```python
from illiad import __version__
```

For the 1.x series, incompatible changes to active command names, documented
JSON keys, or stable Python utilities require a major release. Minor releases
may add backward-compatible commands, keys, and functionality; patch releases
contain backward-compatible fixes and documentation changes. Interfaces
explicitly labeled provisional are excluded from these stability guarantees.

Numerical results also depend on scientific inputs, dependency versions,
stochastic sampling, and hardware; the version contract does not promise
bitwise-identical output.

## Installed Commands

| Command | Status | Purpose | Configuration |
| --- | --- | --- | --- |
| `illiad-fieldsolver` | Active | Generate Cartesian magnetic-field arrays from coil geometry. | `input_files/fieldsolver_inputs.example.json` |
| `illiad-poincare` | Active | Trace field lines and reconstruct Poincare surfaces. | `input_files/poincare_inputs.example.json` |
| `illiad-flux-calc` | Active | Integrate toroidal flux and diagnose island chains. | `input_files/flux_calc_inputs.example.json` |
| `illiad-flux-grad` | Active | Interpolate a scalar profile and/or generate its Cartesian electric field. | `input_files/flux_grad_inputs.example.json` |
| `illiad-sol-trace` | Active | Trace open SOL field lines with the PyTorch solver. | `input_files/sol_trace_inputs.example.json` |
| `illiad-sol-regularize` | Active | Regularize saved SOL crossings onto a scalar-field mesh. | `input_files/sol_regularize_inputs.example.json` |
| `illiad-sol-connection-length` | Active | Select retained tracing, existing-raw regularization, or direct bounded-memory trace regularization. | `input_files/sol_connection_length_inputs.example.json` |
| `illiad-sol-density` | Active | Construct a piecewise core/SOL plasma-density field. | `input_files/sol_density_inputs.example.json` |
| `illiad-sol-potential` | Active | Construct a piecewise core/SOL electrostatic-potential field. | `input_files/sol_potential_inputs.example.json` |
| `illiad-boris` | Active | Run full-orbit lithium-ion transport. | `input_files/boris_inputs.example.json` |

The ten active commands accept:

```text
--inputs PATH
```

`PATH` is a UTF-8 JSON file whose top-level value is an object. Supplied values
override built-in defaults. Omitting `--inputs` uses those defaults. The
unified connection-length command first requires one of `trace`, `regularize`,
or `trace_regularize`; its input object contains separate `TRACE` and
`REGULARIZE` sections. For the other active commands, the same path may instead
be supplied as the optional positional `INPUTS` argument. Supplying both forms
is an input error; neither form takes precedence.
Relative input paths and `output/` are resolved from the process working
directory.

Root `run*.py` launchers mirror the command modules for source-checkout use.
Installed commands are canonical; `illiad.cli`, launcher modules, and their
`main()` functions are not public Python API.

Large generated scientific inputs and outputs are not bundled. See
[Release Contents](RELEASE_CONTENTS.md).

## JSON Configuration Interface

Tracked `input_files/*.example.json` files are complete templates for their
commands. Copy a template to a filename ending in `.json` before customizing
it; ordinary JSON files are ignored so run-specific inputs are not committed
accidentally. Values use JSON types, array-like values use arrays, and optional
values use `null`.

### `fieldsolver_inputs.example.json`

| Key | Meaning |
| --- | --- |
| `OUTPUT_NAME` | Basename for the generated magnetic-field array. |
| `MESH_SIZE` | Three integer mesh dimensions. |
| `I_TORO`, `I_HELI`, `I_VERT` | Toroidal, helical, and vertical coil currents in amperes. |
| `COILFILE` | Coil-geometry input file. |
| `RMAJOR`, `RMINOR` | Major and minor radii in meters. |
| `MESH_PERIODICITY` | Three-element mesh periodicity descriptor. |

### `poincare_inputs.example.json`

| Group | Keys |
| --- | --- |
| Magnetic configuration | `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD` |
| Initial field lines | `IC_PHI_DEG`, `IC_THETA_DEG`, `START_RADIUS`, `END_RADIUS`, `NLINES` |
| Trace and solver controls | `SPINS`, `NPLANES`, `SOLVER`, `RTOL`, `ATOL`, `NTHREADS`, `DOUBLE_LINE` |
| Output | `OUTPUT_DIR` |

Currents are in kiloamperes, radii in meters, and angles in degrees.

### `flux_calc_inputs.example.json`

| Group | Keys |
| --- | --- |
| Input/output location | `ANLYS_DIR`, `ANLYS_SUBDIR`, `FIELD_FILE_TOR`, `FIELD_FILE_HEL` |
| Magnetic configuration | `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD` |
| Surface sampling | `LCFS_INDEX`, `NPHI`, `NTHETA`, optional `PHI_GENs` |
| Flux integration | `MAX_SUBSETS`, `SMOOTH_FCTR`, `INTEGRATE_EPSABS`, `INTEGRATE_EPSREL` |
| Retained legacy inputs | `ISLAND_ALGORITHM`, `HIST_BINS` |
| Diagnostics | `PLOT_ALL`, `BIG_MESH` |

When `PHI_GENs` is absent or `null`, positive `NPHI` produces
`numpy.linspace(360.0 / NPHI, 360.0, NPHI)` in degrees. Explicit values are
normalized to a floating-point NumPy array.

The active island detector measures rotational transform from each ordered
Poincare surface, matches low-order rationals with denominators no greater
than `MAX_SUBSETS`, and splits matched chains by striding the crossing array.
This permits distinct island-chain subset counts in one run.
`ISLAND_ALGORITHM` and `HIST_BINS` remain accepted and logged for existing
inputs but do not control the active detector.

### `flux_grad_inputs.example.json`

| Group | Keys |
| --- | --- |
| Shared analysis and magnetic inputs | `ANLYS_DIR`, `ANLYS_SUBDIR`, `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD`, `LCFS_INDEX`, `NPHI`, `NTHETA`, optional `PHI_GENs` |
| Prior flux selection | `SMALLEST_ISLAND_INDEX`, `MAX_SUBSETS` |
| Interpolation control | `RUN_INTERPOLATOR`, `INPUT_FIELD_NAME`, `ALPHA`, `DEBUG`, `INV_SURF_INDICES`, `GUESS_PHI_INDEX`, `OUTPUT_FILE_NAME`, `FLUX_INTERPOLATION_MODE` |
| RBF interpolation | `RBF_KERNEL`, `RBF_NEIGHBORS`, `RBF_SMOOTHING`, `RBF_EPSILON` |
| Periodic 3-D interpolation | `RBF_PHI_HALF_WINDOW`, `RBF_PHI_SCALE`, `RBF_POINTS_PER_SURFACE_PER_PHI` |
| Gradient construction | `LEGACY_FILTER_GRADIENTS_OUTSIDE_LCFS`, `GRADIENT_FILTER_BUFFER` |

With `RUN_INTERPOLATOR` set to `true`, `FLUX_INTERPOLATION_MODE` accepts
exactly `2d` or `3d`:

- `2d` fits each output plane from that plane's Poincare samples.
- `3d` fits from a periodically wrapped local toroidal window.
  `RBF_PHI_HALF_WINDOW` selects adjacent source planes,
  `RBF_PHI_SCALE` supplies the angular length scale, and
  `RBF_POINTS_PER_SURFACE_PER_PHI` limits each surface's contribution.

Both modes retain float64 source, query, interpolated, and saved arrays. Source
labels run from 1 at the magnetic axis to 0 at the LCFS, and `rho=0` is filled
from the poloidal average of the innermost repaired radial shell.

With `RUN_INTERPOLATOR` set to `false`, interpolation is skipped and
`INPUT_FIELD_NAME` names an existing scalar array relative to
`output/<ANLYS_DIR>/data/`. The gradient stage still writes
`Efield_<OUTPUT_FILE_NAME>.npy` and its plots.

Gradients use radian angular coordinates and periodic centered differences in
toroidal and poloidal directions. `LEGACY_FILTER_GRADIENTS_OUTSIDE_LCFS`
enables the historical exterior mask; `GRADIENT_FILTER_BUFFER` is its
nonnegative radial buffer in meters.

### `sol_trace_inputs.example.json`

| Group | Keys |
| --- | --- |
| Output location | `ANLYS_DIR`, `ANLYS_SUBDIR` |
| Magnetic configuration | `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD`, `MAJOR_RADIUS_M`, `VESSEL_RADIUS_M` |
| LCFS and seeds | `LCFS_INDEX`, `N_PLANES`, `N_SEED_PLANES`, `SEED_PHI_DEG`, `N_RHO`, `N_THETA`, `RHO_MIN`, `RHO_MAX`, `LCFS_CLEARANCE_M`, `LCFS_SPLINE_SMOOTHING`, `LCFS_BOUNDARY_POINTS` |
| Trace length | `SPINS` |
| Device and integration | `DEVICE`, `INTEGRATOR`, `STEP_SIZE_M`, `BATCH_SIZE`, `CROSSING_BUFFER_SIZE`, `STEP_CHUNK_SIZE`, `COMPILE_STEP_CHUNKS`, `WALL_BISECTION_STEPS`, `MIN_FIELD_MAGNITUDE` |
| Progress | `PROGRESS_INTERVAL_STEPS`, `PROGRESS_REFRESH_STEPS`, `SHOW_PROGRESS` |
| Plots | `GENERATE_PLOTS`, `COLOR_SCALE`, `COLORMAP`, `N_LEVELS`, `VMIN`, `VMAX`, `DPI`, `PLOT_MAX_SAMPLES`, `PLOT_SAMPLE_SEED`, `PHYSICAL_PHI_OFFSET_DEG` |

`DEVICE` accepts `auto`, `cpu`, `cuda`, or an explicit CUDA device.
`INTEGRATOR` accepts `euler`, `midpoint`, or `rk4`. Tracing uses float64
values, integrates both directions from each valid exterior seed, and
terminates each solve at the wall or numerical length limit
`2*pi*MAJOR_RADIUS_M*SPINS`.

Compact raw output is appended directly to packed per-plane files under
`raw_crossings/`. Its `manifest.json` records the plane coordinates and sample
counts, while each crossing record stores RTP, field-line ID, and source
direction. `fieldline_connection_length_m.npy` resolves the value for each
field-line ID. Directional lengths, wall intersections, masks, seed
coordinates, and plane coordinates are saved alongside the shards. The
crossing reader continues to accept the earlier plane-sorted monolithic NumPy
files for existing runs.

### `sol_regularize_inputs.example.json`

| Group | Keys |
| --- | --- |
| Input/output location | `ANLYS_DIR`, `ANLYS_SUBDIR`, `TRACE_SUBDIR`, `OUTPUT_FIELD_FILENAME` |
| Surface and grid | `LCFS_INDEX`, `N_RHO`, `N_THETA`, `RHO_MIN`, `RHO_MAX`, `VESSEL_RADIUS_M` |
| Accumulation and fill | `INTERPOLATION_SPACE`, `FILL_METHOD`, `IDW_NEIGHBORS`, `IDW_POWER`, `TREE_WORKERS`, `RAW_CHUNK_SIZE` |
| Plots | `GENERATE_PLOTS`, `SHOW_PROGRESS`, `COLOR_SCALE`, `COLORMAP`, `N_LEVELS`, `VMIN`, `VMAX`, `CONTOUR_EXTEND`, `DPI`, `PHYSICAL_PHI_OFFSET_DEG` |

The regularizer consumes either compact field-line-indexed trace output or
the legacy expanded connection-length array through a plane/chunk source
interface. It preserves the nearest-node accumulation, optional linear or
logarithmic averaging, and seam-free exterior fill used by the research
interpolator. Output is a float64 `(phi, theta, rho)` field plus matching
coordinate arrays.

### `sol_connection_length_inputs.example.json`

The unified input contains independent `TRACE` and `REGULARIZE` objects so the
sparse seed grid and final regular grid retain separate `N_RHO`, `N_THETA`,
`RHO_MIN`, and `RHO_MAX` settings. The required CLI mode defines the products:

- `trace` retains raw plane shards and compact trace metadata.
- `regularize` reads existing raw crossings and does not remove them.
- `trace_regularize` retains the regular field and compact trace metadata but
  does not create a complete raw crossing dataset.

Direct mode pairs both directions for each field-line batch. Its temporary
records contain only regular-cell and field-line IDs and are consumed after
the batch connection lengths resolve. The accumulator size is fixed by the
configured regular field shape.

### `sol_density_inputs.example.json`

| Group | Keys |
| --- | --- |
| Input/output location | `ANLYS_DIR`, `ANLYS_SUBDIR`, `SOL_SUBDIR`, `SOL_FIELD_FILENAME`, `NFIELD_SUBDIR`, `NFIELD_FILENAME` |
| Surface selection | Optional `LCFS_INDEX`; `null` infers `LCFS<number>` from `NFIELD_FILENAME`, then falls back to the Poincare log |
| Density model | `N_AXIS`, `N_LCFS`, `N_WALL`, `ALPHA`, `SOL_BETA`, optional `L_PARALLEL_0_M` |
| Diagnostics | `GENERATE_PLOTS`, `SHOW_LCFS`, `COLOR_SCALE`, `SHOW_PROGRESS` |

The density model requires `N_AXIS > N_LCFS > N_WALL >= 0`. The output is a
float64 `(phi, theta, rho)` field normalized by `N_AXIS` when the default
`N_AXIS=1` is retained. The official defaults select the generic
`DEFAULT/LCFS19` workflow. Before calculation, the command validates the
regular grid, all selected LCFS planes, the vessel radius, and propagated
trace geometry metadata. With `L_PARALLEL_0_M=null`, trace-length metadata is
preferred; legacy artifacts fall back to the positive `SPINS` value recorded
by the Poincare workflow.

### `sol_potential_inputs.example.json`

| Group | Keys |
| --- | --- |
| Input/output location | `ANLYS_DIR`, `ANLYS_SUBDIR`, `SOL_SUBDIR`, `SOL_FIELD_FILENAME`, `NFIELD_SUBDIR`, `NFIELD_FILENAME` |
| Surface selection | Optional `LCFS_INDEX`, with the same inference rule as the density command |
| Potential model | `PHI_WALL`, `DELTA_PHI_0W`, `DELTA_PHI_SOL`, `ALPHA`, `SOL_BETA`, optional `L_PARALLEL_0_M` |
| Diagnostics | `GENERATE_PLOTS`, `SHOW_LCFS`, `COLOR_SCALE`, `SHOW_PROGRESS` |

The potential model requires
`0 < DELTA_PHI_SOL < DELTA_PHI_0W`. It preserves the gradient-compatible
float64 `(phi, theta, rho)` layout, enforces `PHI_WALL` at the vessel wall,
and writes coordinate arrays and compressed model metadata beside the field.
It uses the same generic defaults, artifact preflight, and
`L_PARALLEL_0_M` provenance order as the density command.

### `boris_inputs.example.json`

| Group | Keys |
| --- | --- |
| Magnetic configuration | `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD`, `TOROIDAL_CURRENT`, `HELICAL_CURRENT` |
| Upstream fields | `FIELD_FILE_DENSITY`, `FIELD_FILE_ELECTRIC` |
| Collision selection | `ION_NEUTRAL_COLLISIONS`, `ION_ION_COLLISIONS` |
| Background plasma | `ELECTRON_TEMP_EV`, `BACKGROUND_GAS_SPECIES`, `NEUTRAL_GAS_TEMP_EV`, `NEUTRAL_GAS_DENSITY`, `PLASMA_DENSITY`, `ION_ELECTRON_SAT_CURRENT_RATIO` |
| Ion properties | `ION_MASS`, `ION_TEMP`, `CHARGE_NUM` |
| Plasma potential | Optional `PLASMA_POTENTIAL`; otherwise derived from background inputs. |
| Particle initialization | `LCFS_INDEX`, `DELTRS`, `NPHI`, `NTHETA`, `NPARTICLES_PER_EMITTER` |
| Time integration | `DT`, `TMAX` |
| Trace selection | `TRACK_NPHI`, `TRACK_NTHETA`, `TRACK_NPARTICLES_PER_EMITTER`, `STRIDE` |
| Output | `OUTPUT_DIRECTORY_NAME`, `TAG` |

`ION_NEUTRAL_COLLISIONS` accepts `viscous_drag`, `langevin`, or `null`.
`ION_ION_COLLISIONS` accepts `linear_fp`, `fokker_planck`, or `null`. Enabled
models are applied as half-steps before and after each Boris push.

Particle initialization uses an emitter-major layout. Initial speeds follow
the configured Maxwellian, and launch directions are cosine-weighted over a
hemisphere about the normalized local electric field, with the outward
geometric LCFS normal as fallback. Every run writes `E0_Dist.png` and the
four-panel `V0_Dist.png`. `STRIDE` must be a positive integer.

Unknown configuration keys are not part of the API. Current runners do not
validate every key up front, so begin with the supplied example.

## Python API

The preferred import root is `illiad`. Former `classes.*`, `utility.*`, and
`plot_funcs.*` packages have been removed.

### Stable Utilities

```python
from illiad import __version__
from illiad.utilities.run_config import (
    load_inputs_json,
    merge_input_params,
    normalize_phi_gens,
)
```

| Import | Contract |
| --- | --- |
| `__version__` | Installed version string. |
| `load_inputs_json(path, label="Inputs")` | Load a top-level JSON object or exit with a readable input error. |
| `merge_input_params(defaults, overrides=None)` | Return a shallow defaults copy updated by supplied overrides. |
| `normalize_phi_gens(input_params)` | Mutate and return the mapping after deriving or normalizing `PHI_GENs`. |

### Provisional Research Interfaces

The following analysis objects and helper functions are documented for
programmatic research use, but their constructors, methods, and detailed
return formats are not part of the stable 1.x compatibility contract. The
installed commands and JSON interfaces above are the stable workflow surface.

```python
from illiad.io import IOHandler
from illiad.mesh import Mesh, TorchMesh
from illiad.particle import Particle, FieldLine, Ion
from illiad.poincare import Poincare
from illiad.flux import FluxCalculator, FluxInterpolator, FluxGradientor
from illiad.sol import (
    CrossingChunk,
    NpyPlaneCrossingSource,
    PlaneCrossingSource,
    SOLDensity,
    SOLPotential,
    SOLRegularizer,
    SOLTracer,
    open_plane_crossing_source,
)
from illiad.boris import Boris
from illiad.collisions import Collisions
from illiad.utilities.coordtrans import RTP_to_XYZ, XYZ_to_RTP
from illiad.utilities.point_generators import (
    generateSeedShells,
    generate_MB_velocities,
    ionInitializer,
)
from illiad import plotting
```

| Object | Constructor and documented methods |
| --- | --- |
| `IOHandler` | `IOHandler(run_name)`; logging, subdirectory, NumPy, CSV, figure, and input-boilerplate methods |
| `Mesh` | `Mesh(R0=0.72, a=0.19)`; field/scalar loading, perturbation, rotation, and interpolation methods |
| `TorchMesh` | `TorchMesh(R0=0.0, a=0.0)`; torch field/scalar loading, interpolation, weight, and return methods |
| `Poincare` | `Poincare(...)`; condition, solver, LCFS, output, and `run` methods |
| `FluxCalculator` | `FluxCalculator(io_handler, field, input_params)`; `run` |
| `FluxInterpolator` | `FluxInterpolator(io_handler, field, input_params)`; `run` |
| `FluxGradientor` | `FluxGradientor(io_handler, field, input_params)`; `run` |
| `SOLTracer` | `SOLTracer(io_handler, magnetic_field, input_params)`; `build_initial_conditions`, `log_inputs`, `trace`, `plot`, `run` |
| `SOLRegularizer` | `SOLRegularizer(io_handler, input_params, crossing_source=None)`; `run` |
| `SOLDensity` | `SOLDensity(io_handler, input_params)`; `run` |
| `SOLPotential` | `SOLPotential(io_handler, input_params)`; `run` |
| `Boris` | `Boris(io_handler, anlys_name="Boris", tag=None)`; condition, solver, output, diagnostic, and `run` methods |
| `Collisions` | Collision-model resolution and ion-neutral and ion-ion numerical operators |

`illiad.sol` also exports `build_torch_magnetic_field`,
`load_lcfs_boundary`, `load_poincare_settings`,
`minimum_boundary_distance`, `resolve_device`, `CrossingChunk`,
`PlaneCrossingSource`, `NpyPlaneCrossingSource`, and
`open_plane_crossing_source`.

## Output and Data Compatibility

The directory convention `output/<run>/logs`, `output/<run>/data`, and
`output/<run>/plots` is supported. Stage outputs are contracts between stages
of the same ILLIAD release, not yet stable cross-version interchange formats.

## Explicitly Non-Public Interfaces

- Modules under `illiad.cli`, root launchers, and their globals.
- `misc_scripts`, including the current density and potential prototypes.
- `fastplotlib_tests`, notebooks, scratch scripts, and unpublished helpers.
- Generated arrays not explicitly documented above and local output trees.

An internal Python interface becomes public only when it is listed here and
exported through an `illiad` namespace.
