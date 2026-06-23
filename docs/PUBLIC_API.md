# ILLIAD Public API

This document defines the public interface of ILLIAD version 0.1.0. It is the
compatibility contract used for versioning and releases. Interfaces not listed
here are implementation details, even when they are importable from a source
checkout.

ILLIAD is research software for a staged field-line, flux-surface, and
impurity-ion transport workflow. Its primary public interface is therefore the
command-line workflow and its JSON configuration files. The Python interfaces
are provided for programmatic use of the same components.

## Versioning Policy

ILLIAD follows [Semantic Versioning 2.0.0](https://semver.org/). The installed
package version is available as:

```python
from illiad import __version__
```

Until 1.0.0, ILLIAD is in an initial public-release phase:

- Patch releases (`0.1.x`) will not intentionally break the supported command
  interface, documented JSON keys, or stable Python interfaces below.
- A minor pre-1.0 release may make a breaking change. It will be called out in
  the release notes and this document before or with the release.
- At 1.0.0 and later, removal or incompatible change to a public interface
  requires a major-version increment, except where a documented deprecation
  period says otherwise.

Numerical results depend on model inputs, data files, platform, dependency
versions, stochastic sampling, and hardware. Semantic versioning applies to
the software interface, not to bitwise-identical scientific output.

## Supported Command Interface

The following commands are installed by the `illiad-fieldlines` distribution:

| Command | Purpose | Configuration file |
| --- | --- | --- |
| `illiad-fieldsolver` | Generate Cartesian magnetic-field arrays from coil geometry. | `fieldsolver_inputs.json` |
| `illiad-poincare` | Trace field lines and reconstruct Poincare surfaces. | `poincare_inputs.json` |
| `illiad-flux-calc` | Integrate toroidal flux across reconstructed surfaces. | `flux_calc_inputs.json` |
| `illiad-flux-grad` | Interpolate the flux profile and generate a Cartesian electric field. | `flux_grad_inputs.json` |
| `illiad-boris` | Run full-orbit lithium-ion transport with the Boris solver. | `boris_inputs.json` |

Every command accepts:

```text
--inputs-json PATH
```

`PATH` names a UTF-8 JSON file whose top-level value is an object. Paths in the
JSON file are resolved relative to the process working directory. The field
solver, Poincare, flux-calculation, and flux-gradient commands merge supplied
values over their built-in defaults. `illiad-boris` loads its complete input
object from the supplied file and defaults to `boris_inputs.json` when the
option is omitted.

Use the installed commands rather than invoking `run*.py` files directly. The
runner module names and their `main()` functions are not public Python API.

The commands require their upstream data products. In particular, the usual
workflow uses prepared magnetic-field arrays under `input_files/`, Poincare
outputs for the flux stages, and generated density/electric-field arrays for
the Boris stage. ILLIAD does not bundle large scientific arrays in package
artifacts. See [Release Contents](RELEASE_CONTENTS.md) for the data policy.

## JSON Configuration Interface

The repository-root JSON files are complete, supported examples. Keys listed
below are the public configuration schema for their command. Values must use
JSON types; array-like values are JSON arrays.

### `fieldsolver_inputs.json`

`illiad-fieldsolver` accepts:

| Key | Meaning |
| --- | --- |
| `output_name` | Basename for the generated magnetic-field array. |
| `mesh_size` | Three integer mesh dimensions. |
| `I_toro`, `I_heli`, `I_vert` | Toroidal, helical, and vertical coil currents in amperes. |
| `coilfile` | Coil-geometry input file. |
| `RMAJOR`, `RMINOR` | Major and minor radii in meters. |
| `mesh_periodicity` | Three-element mesh periodicity descriptor. |

### `poincare_inputs.json`

`illiad-poincare` accepts:

| Group | Keys |
| --- | --- |
| Magnetic configuration | `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD` |
| Initial field lines | `IC_PHI_DEG`, `IC_THETA_DEG`, `START_RADIUS`, `END_RADIUS`, `NLINES` |
| Trace and solver controls | `SPINS`, `NPLANES`, `SOLVER`, `RTOL`, `ATOL`, `NTHREADS`, `DOUBLE_LINE` |
| Output | `OUTPUT_DIR` |

Currents are expressed in kiloamperes; radii are expressed in meters; angles
are expressed in degrees. `OUTPUT_DIR` identifies the run directory below
`output/`.

### `flux_calc_inputs.json`

`illiad-flux-calc` accepts:

| Group | Keys |
| --- | --- |
| Input/output location | `ANLYS_DIR`, `ANLYS_SUBDIR`, `FIELD_FILE_TOR`, `FIELD_FILE_HEL` |
| Magnetic configuration | `CURRENT_TOR`, `CURRENT_HEL`, `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD` |
| Surface sampling | `LCFS_INDEX`, `NPHI`, `NTHETA`, `PHI_GENs` |
| Flux integration | `MAX_SUBSETS`, `SMOOTH_FCTR`, `INTEGRATE_EPSABS`, `INTEGRATE_EPSREL`, `ISLAND_ALGORITHM`, `HIST_BINS` |
| Diagnostics | `PLOT_ALL`, `BIG_MESH` |

`PHI_GENs` is optional. When it is absent or `null`, ILLIAD derives it from
`NPHI` as `numpy.linspace(360 // NPHI, 360, NPHI)` in degrees. Use an `NPHI`
that divides 360, or provide `PHI_GENs` explicitly for a different sampling
scheme.

### `flux_grad_inputs.json`

`illiad-flux-grad` accepts the shared location, magnetic-configuration, and
surface-sampling keys from `flux_calc_inputs.json`, plus:

| Group | Keys |
| --- | --- |
| Prior flux selection | `SMALLEST_ISLAND_INDEX` |
| Interpolation and electric field | `ALPHA`, `DEBUG`, `INV_SURF_INDICES`, `GUESS_PHI_INDEX`, `OUTPUT_FILE_NAME` |

`INV_SURF_INDICES` is a JSON array of surface indices. `OUTPUT_FILE_NAME` is
the electric-field output basename, without a required `.npy` suffix.

### `boris_inputs.json`

`illiad-boris` accepts:

| Group | Keys |
| --- | --- |
| Magnetic configuration | `CONFIG_TOR`, `CONFIG_HEL`, `ENABLE_ERRFIELD`, `TOROIDAL_CURRENT`, `HELICAL_CURRENT` |
| Upstream fields | `FIELD_FILE_DENSITY`, `FIELD_FILE_ELECTRIC` |
| Collision controls | `ION_NEUTRAL_COLLISIONS`, `ION_ION_COLLISIONS`, `NEUTRAL_GAS_DENSITY`, `PLASMA_DENSITY` |
| Ion properties | `ION_MASS`, `ION_TEMP`, `CHARGE_NUM` |
| Electric-field scale | `FIELD_SCALE_ELECTRIC` |
| Particle initialization | `LCFS_INDEX`, `DELTRS`, `NPHI`, `NTHETA`, `NPARTICLES_PER_EMITTER` |
| Time integration | `DT`, `TMAX` |
| Trace selection | `TRACK_NPHI`, `TRACK_NTHETA`, `TRACK_NPARTICLES_PER_EMITTER`, `STRIDE` |
| Output | `OUTPUT_DIRECTORY_NAME`, `TAG` |

`STRIDE` is a positive integer controlling trace sampling. `TRACE_STRIDE` is
accepted by the runner as a legacy equivalent when `STRIDE` is not present,
but new configurations should use `STRIDE`.

Unknown configuration keys are not part of the API. Current runners do not
provide schema validation for every key, so a misspelled key can be ignored or
fail later in a workflow. Start from the supplied example files for each
release.

## Python API

The preferred import root is `illiad`. New code must not import public
functionality through `classes.*`, `utility.*`, or `plot_funcs.*`; those paths
remain internal compatibility paths for the repository's current scripts.

### Stable Python Interfaces

The following interfaces are the stable programmatic surface for the 0.1
series:

```python
from illiad import __version__
from illiad.flux import calculate_flux, interpolate_flux, build_electric_field
from illiad.run_config import load_inputs_json, merge_input_params, normalize_phi_gens
```

| Import | Contract |
| --- | --- |
| `__version__` | Installed ILLIAD version string. |
| `calculate_flux(input_params=None)` | Run flux-surface integration using a mapping compatible with `flux_calc_inputs.json`; returns the selected island index. |
| `interpolate_flux(input_params=None)` | Generate the flux/density interpolation from a mapping compatible with `flux_grad_inputs.json`. |
| `build_electric_field(input_params=None)` | Generate the electric field from a mapping compatible with `flux_grad_inputs.json`. |
| `load_inputs_json(path, label="Inputs")` | Load and return a top-level JSON object, exiting with a readable error for unreadable or invalid input. |
| `merge_input_params(defaults, overrides=None)` | Return a shallow copy of `defaults`, updated by `overrides` when supplied. |
| `normalize_phi_gens(input_params)` | Mutate and return the mapping, deriving or normalizing `PHI_GENs` from `NPHI`. |

The historical `illiad.flux` names `fluxCalculator`, `fluxInterpolator`, and
`fluxGradientor` remain supported compatibility aliases for the three
snake-case functions above. New code should use the snake-case names.

### Provisional Research Interfaces

The following imports are public in the sense that they are documented and
supported for exploratory programmatic use. Their object models and method
signatures are still being consolidated from research code. They may change in
a minor pre-1.0 release; such changes will be documented in release notes.

```python
from illiad.io import IOHandler
from illiad.mesh import Mesh, TorchMesh
from illiad.particle import Particle, FieldLine, Ion
from illiad.poincare import Poincare
from illiad.boris import Boris
from illiad.collisions import Collisions
from illiad.coordtrans import RTP_to_XYZ, XYZ_to_RTP
from illiad.point_generators import generateSeedShells, generate_MB_velocities, ionInitializer
from illiad.plotting import plotFuncs
```

The current documented members of these objects are:

| Object | Constructor and methods |
| --- | --- |
| `IOHandler` | `IOHandler(run_name)`; `startLog`, `createSubDir`, `setActiveSubDir`, `saveNumpyData`, `loadNumpyData`, `saveFig`, `loadPorts_fromCSV`, `loadCSV`, `saveCSV`, `inputsBoilerplate`, `borisBoilerplate` |
| `Mesh` | `Mesh(R0=0.72, a=0.19)`; `loadCartesianField`, `addFieldPerturbation`, `setErrorField`, `set_nonPer_errField`, `rot_vecXYZ_byPHI`, `interpField`, `loadScalarField`, `interpScalarField` |
| `TorchMesh` | `TorchMesh(R0=0.0, a=0.0)`; `loadCartesianField`, `addFieldPerturbation`, `loadScalarField`, `setErrorField`, `rot_vecXYZ_byPHI`, `get_weights`, `return_vecs`, `return_scalars`, `interpField`, `interpScalarField` |
| `Particle` | `Particle(type)` |
| `FieldLine` | `FieldLine(init_XYZ, maxlength, direction=1.0)`; `pushXYZ`, `storePath` |
| `Ion` | `Ion(init_XYZ, mass_amu, charge_z, maxlife=0.0)`; `initVelocity`, `initOutput`, `setPosition`, `setVelocity` |
| `Poincare` | `Poincare(io_handler, solvr="LSODA", r_tol=1e-6, a_tol=1e-16, workers=-1, double_line=False, anlys_name="Poincare")`; `set_conditions`, `parallel_solver`, `single_solver`, `post_solver`, `identifyLCFS`, `save_output`, `run` |
| `Boris` | `Boris(io_handler, anlys_name="Boris", tag=None)`; `setConditions`, `parallel_solver`, `post_solver`, `save_output`, `run` |
| `Collisions` | `viscous_drag_hstep`, `langevin_in_hstep`, `linearFP_ii_hstep`, `chandrasekhar_psi`, `chandrasekhar_psi_prime`, `coulomb_fp_rates_li_he`, `fokker_planck_ii_hstep` |

`illiad.collisions` also exports the physical constants `kg_per_amu`,
`kboltz`, `eps0`, `sqrt_pi`, `Li_mass`, and `He_mass`.

The coordinate module exports `RTP_to_XYZ`, `XYZ_to_RTP`, `XYZ_to_RTP2`,
`RTP_to_XYZ_many`, `XYZ_to_RTP_many`, `rot_vecXYZ_byPHI`, `RTP_XYZ_JAC`,
`RTP_XYZ_JAC2`, `axisShift`, and `align_z_to_vector`.

`illiad.point_generators` exports `generateSeedShells`,
`generate_MB_velocities`, and `ionInitializer`. `illiad.plotting` exports the
existing `plotFuncs` plotting module. Their detailed return shapes and plot
formats are provisional.

## Output and Data Compatibility

The directory convention `output/<run>/logs`, `output/<run>/data`, and
`output/<run>/plots` is supported. Stage commands produce files consumed by
later stages in the same ILLIAD release.

Raw NumPy array layouts, output basenames beyond the workflow examples, log
text, plot appearance, and files below `fastplotlib_tests/` are not stable
public data formats yet. Pin the ILLIAD version and retain the source JSON
configuration with a scientific result. A documented interchange format or
data-schema version will be required before these files are treated as
cross-version public API.

## Explicitly Non-Public Interfaces

The following are outside the public compatibility contract:

- Direct imports from `classes.*`, `utility.*`, and `plot_funcs.*`.
- Source runner modules (`runFieldsolver.py`, `runPoincare.py`,
  `runFluxCalc.py`, `runFluxGrad.py`, and `runBoris_new.py`) and their globals.
- `fastplotlib_tests/`, `misc_runFiles/`, notebooks, scratch scripts, and
  unpublished analysis helpers.
- Generated input/output arrays, local output directories, and files not
  documented in this file.

When an internal interface is promoted, it will first be added here and exposed
through the `illiad` namespace.
