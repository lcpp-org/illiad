# Release Contents

This file describes the intended public source distribution and installed
Python package. Release artifacts are narrower than the research working tree.

## Installed Distribution

The wheel contains:

- The `illiad` package, including `illiad.cli`, `illiad.flux`, `illiad.mesh`,
  `illiad.sol`, and the Poincare, Boris, collision, IO, particle, utility, and
  plotting modules.
- `fastplotlib_tests` standalone viewer modules.
- Distribution metadata, dependencies, the GPL-3.0-only license, and nine
  command entry points:
  `illiad-fieldsolver`, `illiad-poincare`, `illiad-flux-calc`,
  `illiad-flux-grad`, `illiad-sol-trace`, `illiad-sol-regularize`,
  `illiad-sol-density`, `illiad-sol-potential`, and `illiad-boris`.

`illiad-sol-density` and `illiad-sol-potential` use the package-native
`SOLDensity` and `SOLPotential` implementations. Their shared geometry and
attenuation helpers live in `illiad.sol.stitching`; the installed commands do
not depend on `misc_scripts`.

Root launchers, example JSON files, documentation sources, and reference input
files are source-distribution resources; callers should not assume they are
runtime files beside an installed wheel.

## Source Distribution

The source archive includes:

- `README.md`, `CHANGELOG.md`, `LICENSE`, `pyproject.toml`, `MANIFEST.in`, and
  Markdown files under `docs/`.
- Root launchers:
  `runFieldsolver.py`, `runPoincare.py`, `runFluxCalc.py`, `runFluxGrad.py`,
  `runSOLTrace.py`, `runSOLRegularize.py`, `runSOLDensity.py`,
  `runSOLPotential.py`, and `runBoris.py`.
- Complete tracked JSON templates under `input_files/`:
  `fieldsolver_inputs.example.json`, `poincare_inputs.example.json`,
  `flux_calc_inputs.example.json`, `flux_grad_inputs.example.json`,
  `sol_trace_inputs.example.json`, `sol_regularize_inputs.example.json`,
  `sol_density_inputs.example.json`, `sol_potential_inputs.example.json`,
  `boris_inputs.example.json`, and `animation_inputs.example.json`.
- Python sources under `illiad/` and `fastplotlib_tests/`.
- Small reference CSV files and `input_files/coils.wega_with_VFCoils`.

SOL tracing, regularization, and profile construction are implemented by
`SOLTracer`, `SOLRegularizer`, `SOLDensity`, and `SOLPotential` under
`illiad.sol`.

## Excluded Content

- The complete `misc_scripts/` research-script directory.
- Local working JSON files under `input_files/`; only `*.example.json`
  templates are included.
- Generated analysis products under `output/`.
- Python/test caches, virtual environments, editor state, installers, object
  files, and cluster job/output files.
- Large generated scientific arrays such as `.npy`, `.npz`, `.h5`, and
  `.hdf5` inputs.
- Figures, videos, logs, spreadsheets, and PDF model/manuscript files.
- Large diagnostic or measurement directories such as
  `input_files/RLP_Results/`.

## Data Policy

The package is installable and inspectable without production-size fields or
run results. Examples may refer to locally generated arrays, but those arrays
should be distributed separately through an archival data record, Git LFS, or
a documented download step.

A scientific result should retain the ILLIAD version, configuration inputs,
dependency environment, and provenance for separately distributed data.
