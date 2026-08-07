# Changelog

Notable changes to ILLIAD are recorded here. The project follows Semantic
Versioning for the stable interfaces defined in `docs/PUBLIC_API.md`.

## [1.0.0] - Unreleased

### Added

- The official PyTorch-backed `illiad.sol.SOLTracer` analysis and
  `illiad-sol-trace` command for open-field-line connection-length tracing.
- Reserved `illiad-sol-density` and `illiad-sol-potential` command names. These
  commands are placeholders and do not yet implement profile analyses.
- Periodically wrapped local 3-D flux interpolation and support for generating
  gradients from an existing regular scalar field.
- Rotational-transform-based island-chain identification and strided subset
  splitting.
- Tracked `input_files/*.example.json` templates with ignored local JSON
  working copies.

### Changed

- Consolidated supported imports under the `illiad` namespace and documented
  the installed commands as the canonical workflow interface.
- Standardized `--inputs PATH` as the active commands' configuration option.
- Added optional positional JSON input paths with explicit conflict checking.
- Defined the Boris collision selectors as `viscous_drag` and `langevin` for
  ion-neutral collisions and `linear_fp` and `fokker_planck` for ion-ion
  collisions.
- Standardized Boris emitter-major particle ordering, cosine-weighted launch
  directions, and initial-condition diagnostics.
- Updated release packaging to include small reference inputs and JSON
  templates while excluding research scripts and generated scientific data.

### Removed

- Deprecated CPU and prototype connection-length volume scripts superseded by
  `SOLTracer`.
- Root-level JSON configuration files and iota-specific Poincare examples;
  tracked templates now live under `input_files/`.

### Not Included

- Official SOL density and potential analysis classes. The current prototype
  scripts remain outside the installed package and public compatibility
  contract.
