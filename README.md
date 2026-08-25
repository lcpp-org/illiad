```
    ----------------------------------------------------
    |  ██╗██╗    ██╗    ██╗ █████╗ ██████╗             |
    |  ██║██║    ██║    ██║██╔══██╗██╔══██╗            |
    |  ██║██║    ██║    ██║███████║██║  ██║            |
    |  ██║██║    ██║    ██║██╔══██║██║  ██║            |
    |  ██║██████╗██████╗██║██║  ██║██████╔╝            |
    |  ╚═╝╚═════╝╚═════╝╚═╝╚═╝  ╚═╝╚═════╝             |
    | Illinois Lagrangian Ion Advection and Deposition |
    ----------------------------------------------------
```

# ILLIAD

**ILLIAD** (*Illinois Lagrangian Ion Advection and Deposition*) is a Python
modeling framework for reconstructing three-dimensional magnetic and
electrostatic fields and simulating trace impurity-ion transport through the
scrape-off layer (SOL).

The code was developed for lithium evaporation experiments in the HIDRA
stellarator at the University of Illinois Urbana-Champaign. ILLIAD connects
reconstructed HIDRA fields to kinetic particle tracing so wall-deposition
patterns can be compared with magnetic topology, background-plasma models,
and lithium ion dynamics.

## Release Status

The current source tree identifies as version 1.0.0; this does not by itself
indicate that a Git tag or package release has been published. Version 1.0.0
establishes the supported active command names and JSON configuration interface
documented in [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md). SOL density and
potential construction is implemented by the package-native `SOLDensity` and
`SOLPotential` classes and their installed commands. See
[CHANGELOG.md](CHANGELOG.md) for the unreleased summary.

## Features

- Reconstruct HIDRA vacuum magnetic fields from coil geometry with
  Biot-Savart integration and fitted field corrections.
- Trace field lines, generate Poincare sections, identify the last closed flux
  surface (LCFS), and diagnose low-order island chains from measured
  rotational transform.
- Integrate toroidal flux and interpolate normalized interior profiles with
  independent 2-D or periodically wrapped local 3-D RBF fits.
- Trace open SOL field lines on CPU or CUDA with the PyTorch-backed
  `SOLTracer`, retaining compact plane-sorted crossing data.
- Generate a Cartesian electric field from either a newly interpolated flux
  profile or an existing regular scalar field.
- Advance lithium ions with a Boris-Buneman full-orbit pusher, optional
  ion-neutral and ion-ion collisions, wall termination, trace capture, and
  deposition diagnostics.

## Repository Layout

- `illiad/cli/`: adapters behind the installed commands.
- `illiad/sol/`: the official `SOLTracer` analysis and shared LCFS helpers.
- `illiad/flux/`: flux calculation, interpolation, and gradient analyses.
- `illiad/mesh/`: NumPy-backed `Mesh` and PyTorch-backed `TorchMesh` classes.
- `illiad/`: Poincare, Boris, collision, particle, IO, and plotting code.
- `run*.py`: thin source-checkout launchers for installed command modules.
- `misc_scripts/`: source-only research and plotting scripts. They are not
  installed and are not public API.

- `input_files/*.example.json`: tracked configuration templates.
- `input_files/`: tracked reference inputs plus ignored local JSON overrides
  and generated scientific data.
- `output/`: generated logs, arrays, and plots.
- `fastplotlib_tests/`: standalone interactive trace viewers.

Release artifact contents are summarized in
[`docs/RELEASE_CONTENTS.md`](docs/RELEASE_CONTENTS.md).

## Requirements and Installation

ILLIAD requires Python 3.11. Current development uses NumPy 2.3.1, SciPy
1.16.0, Matplotlib 3.10.3, pandas 2.3.1, tqdm 4.67.1, PyTorch 2.7.1, Pillow,
and torchrbf. Production-size SOL and ion tracing are intended for a
CUDA-capable GPU, although supported analyses can fall back to CPU.

From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

Replace `cu126` with the appropriate PyTorch build for the local system.
Optional fitting, viewer, and export dependencies are available as extras:

```bash
pip install -e ".[fitting,viewer,export]"
```

## Commands

Eight active workflow commands use JSON inputs:

```bash
illiad-fieldsolver --inputs input_files/fieldsolver_inputs.example.json
illiad-poincare --inputs input_files/poincare_inputs.example.json
illiad-flux-calc --inputs input_files/flux_calc_inputs.example.json
illiad-flux-grad --inputs input_files/flux_grad_inputs.example.json
illiad-sol-trace --inputs input_files/sol_trace_inputs.example.json
illiad-sol-density --inputs input_files/sol_density_inputs.example.json
illiad-sol-potential --inputs input_files/sol_potential_inputs.example.json
illiad-boris --inputs input_files/boris_inputs.example.json
```

The tracked templates may be passed directly to their commands, although
analyses still require their documented upstream field or run data. Before
customizing a template, copy it to the same directory without `.example`:

```bash
cp input_files/poincare_inputs.example.json input_files/poincare_inputs.json
illiad-poincare --inputs input_files/poincare_inputs.json
```

Working `*.json` files are ignored by Git. This keeps run-specific settings
out of commits without hiding the release templates.

Each active command also accepts the JSON path positionally:

```bash
illiad-poincare input_files/poincare_inputs.json
```

Supply either the positional path or `--inputs`, but not both.

All relative inputs and the `output/` tree are resolved from the process
working directory. The preferred Python imports include:

```python
from illiad.io import IOHandler
from illiad.mesh import Mesh, TorchMesh
from illiad.flux import FluxCalculator, FluxInterpolator, FluxGradientor
from illiad.sol import SOLDensity, SOLPotential, SOLTracer
```

See [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md) for command and configuration
contracts and [`docs/PUBLIC_NAMESPACE.md`](docs/PUBLIC_NAMESPACE.md) for
public imports.

## Analysis Workflow

### 1. Trace Poincare surfaces

Start from `input_files/poincare_inputs.example.json` and copy it before
changing run-specific values:

```bash
illiad-poincare --inputs input_files/poincare_inputs.example.json
```

The command saves Poincare planes, wall intersections, plots, and a log under
`output/<OUTPUT_DIR>/`. Downstream analyses read the selected LCFS.

### 2. Calculate toroidal flux

Configure `ANLYS_DIR`, `ANLYS_SUBDIR`, `LCFS_INDEX`, and sampling in
`input_files/flux_calc_inputs.example.json` or an ignored working copy:

```bash
illiad-flux-calc --inputs input_files/flux_calc_inputs.example.json
```

Island detection measures rotational transform from ordered Poincare
crossings, compares it with low-order rational values whose denominator is no
larger than `MAX_SUBSETS`, and splits matched chains with strided subsets.
`ISLAND_ALGORITHM` and `HIST_BINS` remain accepted legacy inputs but no longer
select the active detector.

### 3. Interpolate the interior scalar field and calculate its gradient

```bash
illiad-flux-grad --inputs input_files/flux_grad_inputs.example.json
```

With `RUN_INTERPOLATOR` set to `true`, `FLUX_INTERPOLATION_MODE` selects
independent `2d` fits or a periodically wrapped local `3d` fit. The
interpolator writes a float64 `nField_<OUTPUT_FILE_NAME>.npy`, and the gradient
stage writes `Efield_<OUTPUT_FILE_NAME>.npy` using periodic centered angular
derivatives in radians.

To generate an electric field from an existing regular scalar field, set
`RUN_INTERPOLATOR` to `false` and set `INPUT_FIELD_NAME` to a path relative to
`output/<ANLYS_DIR>/data/`.

### 4. Trace the SOL

Configure the magnetic field, LCFS, seed grid, integration, device, and plots
in `input_files/sol_trace_inputs.example.json` or an ignored working copy:

```bash
illiad-sol-trace --inputs input_files/sol_trace_inputs.example.json
```

`SOLTracer` samples exterior seed points, traces both field directions until
the wall or configured length limit, and captures crossings at regular
toroidal planes. Compact outputs include `raw_points_rtp.npy`,
`raw_fieldline_id.npy`, `raw_source_direction.npy`, `plane_offsets.npy`,
`plane_phi_deg.npy`, and `fieldline_connection_length_m.npy`, plus seed,
directional, and wall-hit metadata.

### 5. SOL density and potential

Build the density and normalized electrostatic-potential fields from the
linear interior profile, the regular SOL connection-length field, and saved
Poincare surfaces:

```bash
illiad-sol-density --inputs input_files/sol_density_inputs.example.json
illiad-sol-potential --inputs input_files/sol_potential_inputs.example.json
```

`SOLDensity` and `SOLPotential` preserve the float64
`(phi, theta, rho)` scalar-field and coordinate-file contract used by
`FluxGradientor`. Both models anchor at the LCFS, bridge the unsampled
LCFS-to-SOL interval, integrate the connection-length-dependent attenuation
to the wall, and can produce contour and midplane diagnostics.

### 6. Run lithium ion transport

Point `input_files/boris_inputs.example.json` or an ignored working copy at the
prepared density and electric fields:

```bash
illiad-boris --inputs input_files/boris_inputs.example.json
```

Ion-neutral collisions accept `viscous_drag`, `langevin`, or `null`. Ion-ion
collisions accept `linear_fp`, `fokker_planck`, or `null`. Enabled operators
are applied as half-steps around the Boris push.

Particles are stored emitter-major. Initial speeds follow the configured
Maxwellian model, and launch directions are cosine-weighted over a hemisphere
about the local electric-field direction, with the outward geometric LCFS
normal as fallback. Every run writes `E0_Dist.png` and `V0_Dist.png` before
transport.

## Outputs and Reproducibility

Analyses use a common layout:

- `output/<run>/logs/`: merged inputs and run logs.
- `output/<run>/data/`: Poincare, flux, field, particle, and impact arrays.
- `output/<run>/plots/`: topology, field, initialization, trajectory, and
  deposition figures.

Runs that reuse directory and subdirectory names may overwrite earlier
products. Retain the input JSON, package version, dependency environment, and
provenance for separately distributed scientific inputs with each result.

## License

ILLIAD is licensed under the GNU General Public License, version 3.0 only
(`GPL-3.0-only`). See [LICENSE](LICENSE) for the full terms.
