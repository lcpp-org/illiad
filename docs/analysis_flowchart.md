# ILLIAD analysis flowchart

![ILLIAD analysis workflow](figures/illiad_analysis_flowchart.png)

Use the [SVG](figures/illiad_analysis_flowchart.svg) to edit labels, colors,
boxes, and arrows in a vector editor. The [PDF](figures/illiad_analysis_flowchart.pdf)
is a vector export for poster placement; the PNG is a 250 dpi preview.
The SVG retains text rather than converting it to outlines.

Edit the labels and layout in
[`generate_analysis_flowchart.py`](figures/generate_analysis_flowchart.py)
and regenerate all three formats with:

```bash
MPLCONFIGDIR=/tmp/illiad-mplconfig python3 docs/figures/generate_analysis_flowchart.py
```

Suggested caption: **ILLIAD reconstructs the magnetic field and topology,
builds prescribed background-plasma profiles, and follows impurity-ion
orbits to predict wall-deposition patterns.**

## Scope and correspondence to the code

This is a conceptual data-flow overview of the current code, including its
SOL extension. Arrows show dependencies, not a single automated execution
sequence. Saved intermediate data can be reused. Profile parameters and
particle-source settings are supplied separately through analysis inputs.
The plasma background is prescribed; no feedback from transported impurity
ions to the background is implied.

| Diagram block | Implementation / commands |
| --- | --- |
| Magnetic-field reconstruction | `illiad-fieldsolver`; `illiad.mesh` loads and combines saved fields with configured currents and corrections |
| Magnetic topology | `illiad-poincare`; flux analysis diagnoses islands and uses the selected LCFS |
| Interior flux profiles | `illiad-flux-calc`; interpolation stage of `illiad-flux-grad` |
| SOL connection lengths | `illiad-sol-trace` + `illiad-sol-regularize`, or `illiad-sol-connection-length trace_regularize` |
| Background-plasma model | `illiad-sol-density`, `illiad-sol-potential`, then `illiad-flux-grad` on the saved potential with `RUN_INTERPOLATOR=false` |
| Impurity-ion transport | `illiad-boris`; initialization, Boris integration, optional collision operators and wall termination |
| Analysis outputs | Saved trajectory/impact arrays and plotting/post-processing routines |

The potential and density builders use the interior profile, regular SOL
connection lengths, and saved LCFS geometry. Their normalized outputs are
scaled by the configured density and potential amplitudes when loaded for
transport. The diagram writes the resulting physical relation as **E = −∇V**.
The reconstructed magnetic field is used for topology, flux integration,
SOL tracing, and ion transport; only the long connection to transport is
drawn separately to keep the overview readable. Density enters enabled
collision models rather than the Lorentz-force update itself.

For a poster restricted to the original interior-profile workflow, omit the
SOL branch and change the background description to “Prescribe density n
and potential V from flux profiles.” The SOL extension is not required for
every ILLIAD run. This figure describes software structure, not validation
of any particular simulation result.
