# Interior flux profiles joined to saved SOL solutions

`misc_scripts/stitch_sol_profiles.py` combines prescribed interior density and
temperature with a saved Model A/B solution. It interpolates a saved **linear
normalized flux-surface parameter**, applies adjustable peaks and alpha
coefficients, and plots all grid locations as markers in the computational phi
plane. It performs no tracing and no additional transport solve.

## Example and adjustable parameters

The [example input](../input_files/stitch_sol_profiles.example.json) uses a density
peak of `2e18 m^-3`, temperature peak of `100 eV`, and both alphas equal to 1.
These are editable illustrative inputs. The LCFS values come from the SOL run's
`n_lcfs_m3` and `T_lcfs_eV`; changing them here would invalidate the join, so they
are not independently adjustable.

From the repository root, plot the phi006 solution currently being inspected:

```bash
conda run --no-capture-output -n testenv python misc_scripts/stitch_sol_profiles.py \
  output/sol_model_b_extended_batch_He17_Li17_iso/phi006 \
  --config input_files/stitch_sol_profiles.example.json \
  --output output/stitched_sol_He17_Li17_iso_phi006 --individual
```

To process every saved plane, pass the **batch directory** instead:

```bash
conda run --no-capture-output -n testenv python misc_scripts/stitch_sol_profiles.py \
  output/sol_model_b_extended_batch_He17_Li17_iso \
  --config input_files/stitch_sol_profiles.example.json \
  --output output/stitched_sol_He17_Li17_iso_batch --individual
```

Use `--phi-deg 0 6` to select saved planes. CLI overrides include
`--density-peak-m3 3e18 --temperature-peak-ev 150`,
`--density-alpha 1.5 --temperature-alpha 2`, and `--temperature-field Tu` or `Te`.
The peaks must be at least the respective positive LCFS values; alpha must be
positive and finite. `--marker-size` defaults to **35 points squared**.
`--density-scale linear` replaces the default logarithmic density colors.
Plots are automatic; `--individual` adds separate density and temperature PNGs.
Paths in the input JSON are relative to the working directory.

## Interior profile convention

Let `psi` be the saved linear normalized parameter, with **1 at the magnetic axis
and 0 at the reference LCFS**. This is the ILLIAD convention, opposite to the
usual outward-increasing normalized enclosed flux. The density and temperature
profiles use independent parameters:

```
f(psi, alpha) = 1 - (1 - psi)^alpha
n_core = n_lcfs + (n_peak - n_lcfs) * f(psi, density_alpha)
T_core = T_lcfs + (T_peak - T_lcfs) * f(psi, temperature_alpha)
```

This is the existing density-profile convention in `illiad/sol/density.py`.
It is **not** `psi^alpha`. At `psi=0`, `df/dpsi=alpha`, so for the same flux map
the edge gradient scales with alpha and the peak-to-edge difference. Alpha above
1 flattens the center and steepens the edge; alpha below 1 produces a sharper
center and can have a singular derivative with respect to psi at the axis.
Use alpha at least 1 when a smooth central profile is important. The physical
spatial gradient also depends on the flux map's gradient.

Saved interpolation overshoots are clipped to `[0,1]` before applying alpha;
counts and raw samples are retained. A finite grid may not land on the magnetic
axis, so the **sampled maximum can be below the specified peak**. It is not
renormalized to whichever grid node happens to be largest. All duplicate
`rho=0` locations receive the same averaged parameter; the raw spread is logged.

## Geometry and flux input

The source `solution.npz` supplies the `(rho, theta)` grid, LCFS polygon, phi angle,
SOL seed indices and solved values. Interior classification uses the same
star-shaped polar boundary representation as Model A/B. Display coordinates are
`x=R-R0=rho*cos(theta)` and `z=rho*sin(theta)`, with no wall-facing angle shift.
Every grid location is represented. Excluded SOL seeds and unsolved wall nodes
remain **NaN**, shown as gray markers; they are never filled with invented plasma
values. Repeated origin points overlap on the plot.

`flux_file` must be an unscaled **linear** normalized `nField` array, shaped
`(phi, theta, rho)`, using the native coordinates:

- phi: `360/Nphi, ..., 360` degrees (360 is equivalent to zero);
- theta: `2*pi/Ntheta, ..., 2*pi` radians;
- rho: `0, ..., flux_vessel_radius_m`.

The script reads only the two needed phi planes, interpolates periodically across
the phi/theta seams, and linearly interpolates on the theta/rho grid. Array axes
are explicitly transposed by coordinate evaluation into the SOL `(rho, theta)`
layout. The flux vessel radius must match the SOL grid. Do not supply a potential,
a density field, or an already alpha-scaled profile as `flux_file`. The format
has no embedded coordinate metadata, so these are user-supplied contracts.

**Current example mismatch:** the SOL runs use surface 21, while the example flux
map was generated using surface 36 in the same IOTA3 analysis. Surface indices in
this analysis run inward; surface 36 is inside surface 21. There is no matching
surface-21 flux map among the inspected inputs. The existing map is sampled as
provided; it is not silently relabeled or recomputed. Its outer interior region
may be flat after clipping, and some interpolated boundary values can remain
positive. The script warns about the differing indices, saves the raw flux values
on the SOL boundary, and uses the explicit physical boundary anchors below.
This is an exploratory composite, not validation that the two magnetic boundaries
are equivalent. Prefer a linear flux map built for the same surface/configuration
when available. Both paths and LCFS indices are saved for provenance.

## What continuity is enforced

The interior formula is evaluated at interior grid points. The saved SOL samples
are copied **exactly**, preserving the solved exterior profiles. At each grid
poloidal angle an additional boundary anchor is inserted at the saved SOL LCFS:

```
n(boundary) = n_lcfs
T(boundary) = T_lcfs
```

`radial_profile(fields, 'density' or 'temperature', theta_index, rho_m)` provides
a piecewise-linear reconstruction through these grid samples and the anchor.
The final interior interval ends at the anchor; the first exterior interval
starts there. Both one-sided limits equal the anchor, giving **C0 continuity**
where the adjacent SOL cell exists. If a grid node is exactly on the LCFS, the
boundary value takes precedence. When the raw flux map does not vanish at the
SOL boundary, this final interval explicitly replaces its unanchored continuation.
Between samples it interpolates density/temperature themselves, not the nonlinear
alpha formula. The phi-plane figures show the actual grid samples and outline,
without adding fictitious solver cells at these anchors.

Missing exterior nodes are kept in the interpolation as NaNs, so the reconstruction
does not bridge excluded cells. At such rays no exterior continuity claim is made.

**This does not enforce C1 smoothness or particle/heat-flux continuity.**
`boundary_diagnostics.csv` reports the one-sided radial slopes using the last core
and immediately adjacent solved SOL node. These are radial slopes, not general
LCFS normal derivatives; a missing adjacent SOL node yields NaN. With a nonzero
raw boundary parameter the anchor can produce a steep last interval, and gradient
estimates depend on grid spacing. Adjusting peak/alpha changes core slopes, but
one global alpha generally cannot match a poloidally varying SOL gradient.
A physical transport match would require interior diffusivities, normal fluxes,
and a coupled boundary/transport solve. No smoothing changes the saved SOL
solution, and its original conservation budget is not extended to this composite.

`temperature_field="Tu"` is the default because Model A/B impose their LCFS
thermal boundary condition on Tu. For isothermal SOL data, Tu equals Te. Selecting
`Te` for Spitzer-Harm data joins the reconstructed local temperature to the chosen
LCFS anchor for display, **not** to a separately solved local-Te boundary
condition; the script warns and records that distinction. The interior temperature
is prescribed in both cases.

## Outputs and checks

Each plane produces:

- `stitched_overview.png`: density and temperature, with shared interior/SOL color
  scales for each quantity; optional `stitched_density.png` and
  `stitched_temperature.png`.
- `stitched_profiles.npz`: full `(rho,theta)` grids, raw/clipped psi, coordinates,
  boundary anchors, radial slopes, original SOL IDs and indices, and region codes
  `0=unavailable`, `1=prescribed core`, `2=solved SOL`.
- `profiles.csv`: every grid index/location with region, psi, density and temperature.
  Psi is NaN in the SOL, where it is not used to calculate the solved fields.
- `boundary_diagnostics.csv`: anchors and both one-sided slopes, in `m^-4` for
  density and `eV/m` for temperature.
- `run_settings.json`: selected source, flux provenance/interpolation, peak/alpha
  controls, clipping/coverage counts and continuity limitations.

The output is separate from the original solution. It is a visualization and
profile-prototyping artifact, not yet a production Boris background-field file.

```bash
python3 -m py_compile misc_scripts/stitch_sol_profiles.py tests/test_stitch_sol_profiles.py
env MPLCONFIGDIR=/tmp/illiad-mplconfig conda run --no-capture-output -n testenv \
  python -m unittest discover -s tests -p 'test_stitch_sol_profiles.py' -v
```

Tests cover profile endpoints/alpha slopes, periodic coordinate interpolation,
exact preservation of SOL values, boundary limits with deliberately mismatched
flux values, missing-cell gaps, batch selection, parameter overrides and exports.
