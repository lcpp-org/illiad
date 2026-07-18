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

**ILLIAD** (*Illinois Lagrangian Ion Advection and Deposition*) is a
Python modeling framework for reconstructing three-dimensional HIDRA magnetic
and electrostatic fields and simulating trace impurity-ion transport through the
scrape-off layer (SOL).

The code was developed for lithium evaporation experiments in the HIDRA
stellarator at the University of Illinois Urbana-Champaign. Post-operational
wall images from these experiments show narrow lithium deposition streaks on
the vacuum vessel. ILLIAD connects reconstructed HIDRA fields to kinetic
particle tracing so those wall-impact patterns can be compared with magnetic
topology, plasma-background assumptions, and lithium ion dynamics.

The workflow combines Biot-Savart magnetic-field reconstruction, field-line and
Poincare analysis, flux-surface-based background plasma models, electrostatic
field construction, and GPU-accelerated full-orbit particle tracing. It
produces deposition diagnostics including wall-impact locations, normalized
deposition maps, incidence angles, toroidal impact directions, impact energies,
trajectory traces, residence-time estimates, and collisionality comparisons.

## Features

- **HIDRA magnetic-field reconstruction:** Compute 3D vacuum magnetic fields
  from toroidal, helical, and vertical coil geometries using the Biot-Savart
  law. Fields are stored on structured toroidal-coordinate grids in Cartesian
  components for interpolation and particle pushing.

- **Field corrections and validation:** Apply coil-field attenuation factors
  and a non-periodic perturbative error field to reproduce measured HIDRA field
  strengths, toroidal asymmetries, and island topology.

- **Field-line tracing and Poincare analysis:** Trace field lines through many
  toroidal transits, generate Poincare sections, identify closed flux surfaces,
  island structures, stochastic/open-field regions, and estimate the last
  closed flux surface (LCFS).

- **Flux-surface background models:** Construct a normalized flux-surface
  coordinate from the reconstructed magnetic topology and use it to prescribe
  surrogate helium plasma density and plasma-potential profiles.

- **Electrostatic-field construction:** Derive SOL electric fields from the
  flux-surface-dependent plasma-potential model while retaining the 3D structure
  of the fitted HIDRA magnetic field.

- **GPU-accelerated lithium ion tracing:** Advance large particle ensembles
  with a Boris-Buneman full-orbit pusher using interpolated 3D electric and
  magnetic fields. The runner supports Maxwellian initial energies, LCFS-based
  launch distributions, wall-intersection termination, trace capture, and
  optional ion-neutral and ion-ion collision models.

- **Deposition diagnostics:** Record particle impact location, incidence angle,
  toroidal impact direction, and deposition energy on the HIDRA vessel wall.
  Generate 2D wall maps, deposition-fluence plots, trajectory visualizations,
  and statistical impact distributions.

- **Analysis and plotting tools:** Provide scripts for magnetic-field plots,
  flux-surface diagnostics, deposition summaries, survival functions, and
  standalone fastplotlib trace viewers.

## Repository Layout
The runner scripts are JSON-configured, with built-in defaults that also allow each script to be run directly.

- `runFieldsolver.py`: optional magnetic-field generation from coil geometry.
- `runPoincare.py`: field-line tracing and LCFS identification.
- `runFluxCalc.py`: flux-surface integration from Poincare output.
- `runFluxGrad.py`: flux interpolation plus electric-field generation.
- `runBoris.py`: lithium ion transport runner.
- `*_inputs.json`: example JSON configuration files for the corresponding runner scripts.
- `illiad/utilities/`: coordinate transforms, point generation, workflow
  configuration, event functions, and physical constants.
- `classes/`: mesh, field-line, particle, collision, Boris, IO helpers, 
  flux calculation, interpolation, and gradient tools.
- `plot_funcs/`: plotting scripts and reusable plotting helpers.
- `input_files/`: coil geometry, pre-generated fields, fitted profiles, and
  supporting CSV/NumPy inputs.
- `output/`: generated analysis products, logs, figures, and simulation data.
- `fastplotlib_tests/`: standalone interactive trace-viewer prototypes.

Release artifact contents are summarized in
[`docs/RELEASE_CONTENTS.md`](docs/RELEASE_CONTENTS.md). Large generated field
arrays and run outputs are intentionally kept out of package artifacts.

## Dependencies

The versions below are the current development versions used on the Illinois
Campus Cluster. Other versions may work, but have not been validated.

- Python 3.11.11
- NumPy 2.3.1
- SciPy 1.16.0
- Matplotlib 3.10.3
- pandas 2.3.1
- tqdm 4.67.1
- PyTorch 2.7.1
- Pillow
- torchrbf

Optional tools used by development, plotting, or standalone viewer scripts:

- scikit-learn, used by a few fitting/validation utilities in `misc_runFiles/`
  and `plot_funcs/`.
- fastplotlib, pygfx, wgpu, and rendercanvas, used by the interactive viewers in
  `fastplotlib_tests/`.
- imageio or an ffmpeg-capable matplotlib backend, useful for animation export.

PyTorch is used both for GPU field construction and for the Boris particle
solver. The code will select CUDA when available and otherwise fall back to CPU,
but production-size particle tracing is intended for a CUDA-capable GPU.

## Getting Started

1. Clone the repository and create a Python 3.11 environment.

   ```bash
   git clone https://github.com/lcpp-org/illiad.git
   cd illiad
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy scipy matplotlib pandas tqdm pillow torch torchrbf
   ```

   Install the optional packages above only if you need the fitting utilities,
   interactive trace viewers, or animation export.

   For editable development installs, the repository also includes packaging
   metadata:

   ```bash
   pip install -e .
   ```

   Optional dependency extras are available for adjacent tooling:

   ```bash
   pip install -e ".[fitting,viewer,export]"
   ```

   Editable installs expose the canonical workflow commands:

   ```bash
   illiad-fieldsolver --inputs-json fieldsolver_inputs.json
   illiad-poincare --inputs-json poincare_inputs.json
   illiad-flux-calc --inputs-json flux_calc_inputs.json
   illiad-flux-grad --inputs-json flux_grad_inputs.json
   illiad-boris --inputs-json boris_inputs.json
   ```

   Public Python imports are available through the `illiad` namespace:

   ```python
   from illiad.mesh import Mesh, TorchMesh
   from illiad.io import IOHandler
   from illiad.flux import calculate_flux, interpolate_flux, build_electric_field
   ```

   See [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md) for the versioned public API
   contract and [`docs/PUBLIC_NAMESPACE.md`](docs/PUBLIC_NAMESPACE.md) for a
   concise import reference.

2. Confirm that the input data are available.

   The current analysis scripts expect pre-generated field and profile files in
   `input_files/` and previously generated stage outputs under `output/`.
   `illiad-fieldsolver` can regenerate magnetic field arrays from
   `input_files/coils.wega_with_VFCoils`, but the standard analysis path starts
   from prepared `.npy` field files such as:

   - `input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy`
   - `input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy`
   - generated density and electric-field files referenced by
     `boris_inputs.json`

3. Run field-line tracing and identify the LCFS.

   Edit the currents, initial field-line locations, solver tolerances, and
   `OUTPUT_DIR` in `poincare_inputs.json`, then run:

   ```bash
   illiad-poincare --inputs-json poincare_inputs.json
   ```

   This writes Poincare surfaces, wall-intersection data, plots, and logs under
   `output/<OUTPUT_DIR>/`.

4. Calculate normalized flux surfaces.

   Set `ANLYS_DIR`, `ANLYS_SUBDIR`, `LCFS_INDEX`, and `NPHI` in
   `flux_calc_inputs.json` so they match the Poincare output, then run:

   ```bash
   illiad-flux-calc --inputs-json flux_calc_inputs.json
   ```

   This integrates toroidal flux for each reconstructed surface and saves files
   such as `CalculatedFLuxes.npy`, `CalculatedFLuxes-normalized.npy`,
   `ValidSurfaces.npy`, and surface point meshes.

5. Interpolate the flux profile and build the electric field.

   Update `flux_grad_inputs.json` for the same analysis directory and LCFS
   selection, then run:

   ```bash
   illiad-flux-grad --inputs-json flux_grad_inputs.json
   ```

   This calls the flux interpolator to generate `density_field.npy`, then takes
   the gradient of that field to produce a Cartesian electric-field array such
   as `Efield_LCFS15.npy`.

6. Run lithium ion transport.

   Edit `boris_inputs.json` to point at the magnetic configuration, generated
   density field, generated electric field, collision-model choices, ion
   properties, particle counts, timestep, and output tag. Then run:

   ```bash
   illiad-boris --inputs-json boris_inputs.json
   ```

   The runner initializes lithium ions near the LCFS, advances them with the
   Boris solver, records wall hits and selected traces, and writes plots/data to
   `output/<OUTPUT_DIRECTORY_NAME>/`.

7. Inspect outputs and make figures.

   Stage logs are written under `output/<run>/logs/`, NumPy arrays under
   `output/<run>/data/`, and figures under `output/<run>/plots/`. The
   `plot_funcs/` scripts provide additional plotting workflows for magnetic
   fields, flux profiles, deposition maps, impact-energy distributions,
   incidence-angle distributions, survival functions, and article figures.
   The scripts in `fastplotlib_tests/` are standalone interactive viewers for
   saved Boris trace arrays.

## Analysis Workflow

The full pipeline is:

1. Generate or load magnetic fields.
2. Trace magnetic field lines and generate Poincare surfaces.
3. Identify the LCFS and open-field regions.
4. Integrate toroidal flux across reconstructed surfaces.
5. Interpolate a normalized flux-surface parameter onto the simulation grid.
6. Convert the plasma-potential profile into a 3D electric field.
7. Initialize lithium ions near selected LCFS offsets.
8. Run the Boris full-orbit particle solver.
9. Post-process wall impacts, trajectories, survival functions, and deposition
   statistics.

For the current public workflow, stages 2 through 9 are the primary analysis
path. Stage 1 is available in the repository, but most downstream runs are
configured to use prepared magnetic-field files rather than regenerating those
fields every time.

## Outputs

ILLIAD uses `classes/iohandler.py` to organize analysis products. A typical run
creates:

- `output/<run>/logs/`: input summaries and stage logs.
- `output/<run>/data/`: NumPy arrays for Poincare surfaces, flux fields,
  electric fields, initial conditions, wall hits, energies, and traces.
- `output/<run>/plots/`: Poincare plots, flux diagnostics, field slices,
  deposition maps, trajectory plots, and statistical summaries.

Existing output directories may be overwritten by scripts that reuse the same
`OUTPUT_DIR`, `ANLYS_DIR`, `ANLYS_SUBDIR`, or Boris `TAG`, so choose unique run
names when preserving earlier results.

## Scientific Context

ILLIAD supports kinetic modeling of lithium impurity transport in HIDRA during
controlled lithium evaporation. The central question is how three-dimensional
magnetic topology, SOL electric fields, and weakly collisional impurity dynamics
shape the narrow lithium deposition streaks observed on the HIDRA vessel wall.

The model reconstructs HIDRA's toroidal and helical coil fields, applies
experimentally motivated correction factors, traces magnetic surfaces and open
field lines, constructs flux-surface-aligned plasma background fields, and
pushes lithium ions until they intersect the wall. The resulting deposition
maps are intended for comparison with post-operational wall images and for
estimating quantities relevant to plasma-facing component studies, including
particle fluence, incidence direction, and deposition energy.
