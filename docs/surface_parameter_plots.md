# Surface-parameter contours

Enable cross-section plots in flux-calc with:

```json
"PLOT_SURFACE_PARAMETER": true,
"SURFACE_PARAMETER": "both",
"SURFACE_PARAMETER_STYLE": "both",
"SURFACE_PARAMETER_LEVELS": 16,
"SURFACE_PARAMETER_CMAP": "viridis"
```

`SURFACE_PARAMETER` selects `flux`, `iota`, or `both`. Flux means the existing
normalized surface parameter **1 - Psi_s / Psi_LCFS**, zero at the LCFS and one
at the axis. Iota is the existing per-surface winding measurement from ordered
Poincare crossings, not the island-local straight-angle return frequency.
`SURFACE_PARAMETER_STYLE` selects `lines`, `filled`, or `both`.
`SURFACE_PARAMETER_LEVELS` specifies the number of evenly spaced level values,
including endpoints (filled plots have one fewer color band). Flux levels span
0 to 1; iota levels span the finite measured range across the requested planes,
so all planes share a color scale. Defaults are disabled, flux, lines, and 16.
The vacuum vessel is drawn on every plot.

These plots interpolate the scalar linearly on triangles connecting adjacent,
valid, nested geometric contours of the same island multiplicity. Lobe centers
are matched between neighboring surfaces. No straight-angle acceptance is
required, and SL error/point-cap settings do not affect these plots. A constant
scalar contour need not coincide with an original sampled surface. Nonmonotonic
iota can produce multiple contours at the same value.

Failed fits, missing surfaces, curves protruding outside the LCFS, nonnested curves,
changes of island multiplicity,
and the region inside the innermost resolved contour remain blank. In particular,
this does not invent an interpolation across separatrices or an axis value of
iota. Evenly spaced levels do not imply evenly spaced distances. The plots inherit
the existing flux integration and normalization; they do not independently
validate the calculated flux or supply a new island flux convention.

Outputs are PNG and PDF files under
`plots/<ANLYS_SUBDIR>/surface_parameter/{flux,iota}_{lines,filled}_NNN`.
New flux-calc runs also save `data/<ANLYS_SUBDIR>/Iota.npy`, indexed by surface
and requested phi plane, alongside the existing flux arrays.

To plot an existing run without loading the magnetic field or reintegrating flux:

```bash
MPLBACKEND=Agg python -m misc_scripts.plot_surface_parameters \
  --inputs input_files/flux_calc/flux_calc_inputs.json
```

Use the original run's surface, phi-plane order, and fit settings. Flux plots
load `CalculatedFLuxes-normalized.npy`; iota-only plots need only Poincare data.
The script reconstructs geometric fits and recomputes the measured iota. Legacy
flux arrays do not record their phi ordering, so matching input settings are
required even when array shapes match.

## Normalized iota and weighted profiles

Additional quantity selections are `iota_normalized`, `blend`, and `all`.
`both` retains its original meaning (flux and raw iota); `all` outputs all four
quantities. Normalized iota and blend support the same line/filled styles and
0-to-1 evenly spaced levels as flux.

```json
"SURFACE_PARAMETER": "all",
"PLOT_PARAMETER_PROFILES": true,
"IOTA_WEIGHT": 0.5,
"IOTA_AXIS_VALUE": null
```

For each phi plane, define

```
iota_hat_raw = (iota_s - iota_LCFS) / (iota_axis - iota_LCFS)
iota_hat = clip(iota_hat_raw, 0, 1)
p_blend = (1 - IOTA_WEIGHT) * clip(flux_parameter, 0, 1)
          + IOTA_WEIGHT * iota_hat
```

`IOTA_WEIGHT` defaults to 0.5 and must be in [0, 1]. At 0 the blend is exactly
the flux parameter; at 1 it is exactly normalized iota. Missing data in the
unused component do not invalidate these pure endpoints. At intermediate weights,
both components must be finite. The convex blend remains in [0, 1] without an
additional min/max rescaling, which would change the weighting interpretation.

`IOTA_AXIS_VALUE` defaults to `null`: use the innermost finite non-axis surface
as an estimate of axis iota, independently per plane. The final surface row is
the axis trace, whose measured winding is not used. A finite numerical override
sets the same axis-iota reference for every plane. The normalized axis row is
assigned 1 as a boundary condition, and the LCFS is 0. Missing LCFS iota or equal
axis/LCFS references leave the entire normalized profile undefined (NaN), with
a warning. This procedure is an endpoint normalization, not a claim that iota
is monotonic. Overshoots are clipped for plotting, logged, and preserved in the
unclipped saved array. The innermost-surface estimate is not an independent
measurement of the exact on-axis rotational transform.

`PLOT_PARAMETER_PROFILES` defaults to false and independently enables per-plane
`Profiles_NNN.png`/`.pdf` plots versus surface index, comparing flux, normalized
iota, and the blend. The plots annotate the axis-reference surface or explicit
override. They can be enabled without cross-section plots in flux-calc.

`data/<ANLYS_SUBDIR>/surface_parameter/profiles.npz` stores `surface_index`,
`phi_deg`, `flux`, raw `iota`, `iota_normalized`, `iota_normalized_unclipped`,
`blend`, `iota_weight`, `lcfs_index`, `iota_axis_reference`, and
`iota_axis_surface_index` (-1 for explicit or unavailable reference).
These arrays include all surface rows; comparison plots start at the LCFS.
Replotting blends or comparison profiles requires the saved normalized flux file.

`SURFACE_PARAMETER_CMAP` selects a registered Matplotlib colormap for both
contour lines and filled cross-sections, including their colorbars. The default
is `viridis`; examples include `Blues`, `plasma`, and `viridis_r` (reversed).
It applies to all selected quantities, including normalized iota and blends.
Profile comparison curves retain their distinct line colors and styles.
