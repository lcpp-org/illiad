# Release Contents

This file defines what belongs in public source releases and Python package
artifacts. It is intentionally narrower than the full working tree used during
research runs.

## Include

- Source code in `illiad/`, `classes/`, `utility/`, `plot_funcs/`, and
  `fastplotlib_tests/`.
- Public runner scripts: `runFieldsolver.py`, `runPoincare.py`,
  `runFluxCalc.py`, `runFluxGrad.py`, and `runBoris_new.py`.
- Example configuration files: `boris_inputs.json` and
  `animation_inputs.json`.
- Project metadata: `pyproject.toml`, `README.md`, `LICENSE`, and
  `MANIFEST.in`.
- Small reference input files under `input_files/`, such as CSV geometry files,
  profile tables, and `coils.wega_with_VFCoils`.

## Exclude

- Generated analysis products under `output/`.
- Python caches, test caches, virtual environments, editor state, local
  installers, and cluster job files.
- Large generated scientific arrays such as `.npy`, `.npz`, `.h5`, and `.hdf5`
  field/profile files under `input_files/`.
- Manuscript drafts and large diagnostic measurement folders under
  `input_files/`.
- Development scratch scripts under `misc_runFiles/`.

## Data Policy

The source release should be installable and inspectable without bundled large
simulation data. Public examples may refer to generated field files, but large
arrays should be distributed separately through a DOI-backed archive, Git LFS,
or a documented data-download step.

Small reference files that are required to understand geometry, ports, coil
definitions, or default profiles may stay in git and in source distributions.
