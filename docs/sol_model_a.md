# Model A SOL prototype

`misc_scripts/sol_model_a.py` implements the prescribed-upstream-temperature model
in [ILLIAD_SOL_Model_A_Prescribed_Tu.pdf](ILLIAD_SOL_Model_A_Prescribed_Tu.pdf).
It reads `summary.npz` and `settings.json` from
`trace_lcfs_distance_cpu_warp.py` (or CPU v2), solves selected toroidal planes on the CPU,
and writes separate prototype outputs. Saved paths, Warp, magnetic field files,
and the original Poincare run are not needed. Dependencies are NumPy and SciPy.
It does not modify the production density/potential pipeline.

For a coupled density and upstream-temperature power-balance solve using the
same trace/mesh conventions, see [Model B](sol_model_b.md).

## Run

For an existing multi-plane summary:

```bash
conda run -n testenv python misc_scripts/sol_model_a.py \
  output/lcfs_grid_distance_warp_some3 --phi-deg 12 \
  --config input_files/sol_model_a.example.json \
  --incomplete no_flux --output output/sol_model_a_phi12
```

For a single-plane summary, omit `--phi-deg`. Point to the directory that actually
contains `summary.npz`, including a `phiDDD` subdirectory when tracing in batches.
The default output directory is `<trace_dir>/model_a_phi<angle>`.

### Batch planes from one summary

Use `--all-phi` to solve every saved plane, or `--phi-deg 0 2 5` to select multiple
planes in that order. Each plane is solved independently and sequentially with
the same model settings; there is no coupling between planes.

`output/lcfs_grid_distance_warp_some` contains angles **0, 1, 2, 3, 4, 5 degrees**:

```bash
conda run --no-capture-output -n testenv python misc_scripts/sol_model_a.py \
  output/lcfs_grid_distance_warp_some --all-phi \
  --config input_files/sol_model_a_phi000.example.json \
  --incomplete long_leg --output output/sol_model_a_some_batch --plot-individual
```

The input filename refers to the original test case; its model parameters apply
to every selected plane. Batch results go in `<output>/phi000/`, `phi001/`, etc.,
each with its own solution, CSV, settings, and optional `plots/` directory.
Use `--plot` for only the six-panel overview per plane, or omit plotting flags
to save only numerical results. Without `--output`, the batch root is
`<trace_dir>/model_a_batch`. A single `--phi-deg` keeps the original single-plane
output layout; `--all-phi` always uses per-plane subdirectories.

Missing or duplicate angle selections are rejected before any solve. A solve or
plot failure stops the batch; already completed plane outputs remain available.
The source must be a directory containing one combined `summary.npz` and
`settings.json`, rather than a parent directory of separate tracer batches.

### Single-plane test

For `phi000` in `output/lcfs_grid_distance_warp_some`, use the supplied test input:

```bash
conda run --no-capture-output -n testenv python misc_scripts/sol_model_a.py \
  output/lcfs_grid_distance_warp_some --phi-deg 0 \
  --config input_files/sol_model_a_phi000.example.json \
  --incomplete long_leg --output output/sol_model_a_phi000
```

This input retains the example plasma parameters and uses a one-degree toroidal
wedge (`pi/180` radians). Of its 16,125 seeds, 15,875 have both wall hits,
160 have one wall hit, and 90 have neither. This test retains the 160 one-hit
cells using the 100 m missing-leg approximation and excludes the 90 neither-hit cells.

The example inputs are exploratory values, not calibrated HIDRA parameters. Edit
[sol_model_a.example.json](../input_files/sol_model_a.example.json) or pass another
JSON containing any subset of `SOLConfig` fields. Unknown keys are rejected.
Temperatures are eV; lengths meters; density m^-3; diffusion m^2/s; particle source
m^-3/s. `wedge_angle_rad` sets the toroidal thickness for integrated particle rates;
the density solution is independent of this common factor.

**Incomplete traces:** the default `--incomplete error` rejects a plane containing
any capped/invalid trace. Prefer longer valid traces. The explicitly selected
`--incomplete no_flux` mode excludes those cells, writes their original seed IDs,
and removes every diffusion face touching them. These artificial internal no-flux
boundaries can isolate cells from the LCFS, giving zero density when the source is
zero. Conservation then applies only to the retained domain. This option does not
repair missing lengths or establish the accuracy of that domain approximation.

`--incomplete long_leg` retains cells with exactly one wall hit and finite positive
traced lengths and a valid minimum LCFS distance. It substitutes only the missing leg:

```
missing_leg_effective_m = max(missing_leg_length_m, missing_leg_traced_m)
Lc_effective_m = wall_hit_leg_m + missing_leg_effective_m
```

Set `missing_leg_length_m` in the input JSON (default **100 m**); it must be finite
and positive. The traced distance is a lower bound, so it is never shortened.
Neither-hit cells and invalid trace records are still excluded with no-flux
boundaries. Two-hit cells retain their measured lengths. Perpendicular diffusion
continues normally through the retained one-hit cells.

Both the `2*cs/Lc` density sink and the Spitzer–Härm reconstruction use the effective
lengths. The sampled minimum LCFS distance and its signed upstream displacement
are retained. The latter must remain inside the original traced interval, even
when the effective endpoint is farther away. An untraced segment might approach
the LCFS more closely, so both the upstream prescription and local temperature
are approximate on these cells. Increasing the configured length weakens their
density sink; it is a sensitivity parameter, not a measured wall position. The
default `--incomplete error` and the previous `no_flux` mode remain available.

## Equations and conventions

The default replaceable upstream function is

```
Tu = T_lcfs * exp(-max(d_min, 0) / lambda_T_u)
```

For a custom profile, import `solve_plane` and pass
`profile=my_function`, with signature `my_function(d_min_m, T_lcfs_eV, cfg)`.
It must return finite nonnegative temperatures broadcastable to the cell array.
The prototype uses constant LCFS density and reference temperature.

`parallel_temperature_model` selects `isothermal` (`Te = Tu`) or `spitzer_harm`
(the default). Tracer `length_m` columns are forward (+B), backward (-B).
`lcfs_min_signed_s_m` points from the seed toward closest approach. Thus:

```
Lc = L_forward + L_backward
s_grid = L_backward
s_upstream = s_grid + signed_s_to_min
leg_minus = s_upstream
leg_plus = Lc - s_upstream
xi = abs(signed_s_to_min) / leg_on_grid_side
Te = Tu * ((1-xi) + target_T_fraction**3.5 * xi)**(2/7)
```

At closest approach `xi = 0`; at either wall `xi = 1`. The nearest one-sided
`wall_connection_length_m` is deliberately not used for `Lc` or the branch fraction.
Signed distances and +/- direction flags are checked. Isothermal mode does not
read those fields, so older summaries can work if their other required fields exist.
Both modes apply the selected incomplete-trace policy. In `long_leg` mode, the
equations above use substituted endpoints for the flagged one-hit cells.

The density equation is

```
-div(D_perp grad(n)) + (2*cs_upstream/Lc)*n = source
cs_upstream = sqrt(e * (Tu + gamma_i*Ti) / (ion_mass_u * atomic_mass))
```

A shared face contributes `D_perp * area / distance * (n_i - n_j)` to cell i
and its negative to cell j. The LCFS has prescribed `n_lcfs`; the vessel has
zero in-plane diffusive flux; theta is periodic. SciPy's sparse direct solver
solves the linear system. Because sound speed uses **Tu**, switching the local
Te reconstruction leaves density unchanged.

## Geometry and prototype limits

Only zero-offset traces with the same exclusion and distance-reference surface
are accepted. The saved regular polar grid must have contiguous radial SOL seeds
in every theta column, and the saved LCFS section must be star-shaped about the
minor-radius origin. Unsupported geometry fails explicitly.

Each seed represents a finite volume bounded by radial/angular midpoints. The
first radial cell extends inward to the saved LCFS, and the last extends outward
to the circular vessel, since the tracer omits seeds on the vessel. This merges
boundary slivers without introducing unknowns that have no trace data. LCFS
vertices define a periodic, piecewise-linear **rho(theta)** approximation; this
is not exact Cartesian-polygon clipping. Boundary curvature is limited by the
saved LCFS resolution.

Volumes integrate `(R0 + rho*cos(theta))*rho d(rho)d(theta)d(phi)`; face areas
include the same physical major-radius factor and wedge angle. Cut volumes,
open radial faces, and LCFS faces use Gauss quadrature split at boundary vertices
and radial crossings. `--quadrature 8` is the default; increase it to check
integration convergence. Poloidal faces use overlapping open radial intervals,
including the periodic seam. Uncut volumes agree with the PDF's analytic wedge
formula. Tests also check a shaped LCFS and boundary quadrature refinement.

Radial neighbor distances use delta-rho; poloidal distances use mean-rho times
delta-theta. The LCFS Dirichlet conductance uses the shortest seed-to-saved-polygon
distance. This is a two-point finite-volume approximation: strongly slanted or
merged boundary cells need a mesh-convergence study and ultimately nonorthogonal
flux corrections. All in-plane diffusion is treated as perpendicular, as in the
specification; there is no magnetic projection tensor or toroidal-plane coupling.

This is a particle-balance solve with a prescribed thermal background, not a
self-consistent energy solution. Potential is not recalculated. Atomic rates,
recycling, and conductive heat-flux/power diagnostics are deferred. The zero-target
SH default is a conducting-wall limit, not a sheath-entrance closure.

## Outputs and checks

- `solution.npz`: retained `seed_id`, original `rho_index`, `theta_index`, plane
  index/angle and grid axes; `n_e_m3`, `Tu_eV`, `Te_eV`, `Lc_m`, `d_min_m`, `xi`,
  upstream-to-wall legs, sound speed, cell volumes, cellwise particle residuals,
  and `rejected_seed_id`.
- `Lc_m` is the effective total length used by the solver. `Lc_traced_m` is the
  sum of the original traced distances, which is a lower bound on incomplete
  traces. `traced_length_m` and `effective_length_m` preserve the separate
  `(forward, backward)` lengths; `hit_wall` preserves the original wall flags.
  The boolean `connection_approximated` flags retained one-hit cells even when
  their effective and traced lengths are numerically equal.
- The archive also includes `n_e_m3_grid`, `Tu_eV_grid`, and `Te_eV_grid`, with
  shape **(rho, theta)** matching the tracer's plane mask. Unsolved cells are NaN.
  These are not production `(phi, theta, rho)` background fields. Isothermal
  `xi` and leg values are NaN because their geometry is not read.
- `profiles.csv`: one row per retained seed, with indices and principal fields,
  including `Lc_traced_m` and `connection_approximated` (0 or 1).
- `run_settings.json`: model settings, source directory, selected plane, geometry
  and incomplete-trace policy, residual, positivity and integrated balance checks.
  Rates are particles/s for the configured wedge. Boundary influx plus volume
  source must equal parallel loss over retained cells.
  Diagnostics count complete, approximated, rejected, and zero-density cells.

## Plot saved output

```bash
conda run --no-capture-output -n testenv python misc_scripts/plot_sol_model_a.py \
  output/sol_model_a_phi000
```

This writes `plots/sol_model_a_overview.png` inside the solution directory.
Plot output is PNG only. The six panels show density, upstream and local electron temperatures,
effective connection length, sampled minimum LCFS distance, and trace treatment
(both wall hits / approximated one-hit cells / excluded cells).

The plot uses actual cell-center samples in the tracer's `(R-R0, Z)` plane;
it does not interpolate through excluded cells. Black curves show the LCFS and
vessel. Both temperatures share one linear color scale. Density and connection
length use logarithmic colors; zero density is marked with a black cross.
The default marker size is **35 points squared** for all sample markers, including
trace status and zero-density markers. Use `--marker-size 20` to override it,
`--density-scale linear` for linear density colors, and `--output DIR` to choose
another plot directory.

Add `--individual` to also save all six panels as separate PNGs:

```bash
conda run --no-capture-output -n testenv python misc_scripts/plot_sol_model_a.py \
  output/sol_model_a_phi000 --individual
```

The additional filenames are `sol_model_a_density.png`, `sol_model_a_Tu.png`,
`sol_model_a_Te.png`, `sol_model_a_connection_length.png`,
`sol_model_a_lcfs_distance.png`, and `sol_model_a_trace_status.png`. Individual
panels use the same data, color limits, and marker size as the overview.

To plot automatically after solving, pass `--plot` to the solver. Use
`--plot-individual` to produce the overview and all individual panels in one run
(it implies `--plot`):

```bash
conda run --no-capture-output -n testenv python misc_scripts/sol_model_a.py \
  output/lcfs_grid_distance_warp_some --phi-deg 0 \
  --config input_files/sol_model_a_phi000.example.json \
  --incomplete long_leg --output output/sol_model_a_phi000 --plot-individual
```

The solver accepts the same `--marker-size` and `--density-scale` options.
Plots go in `<output>/plots`. Without either plotting flag, the solver only saves
the numerical output and does not import Matplotlib.

New solver outputs include `boundary_xz_m`, `rejected_rho_index` and
`rejected_theta_index` so plots can be generated without the trace archive.
For older outputs, the plotter reads those positions from the trace directory
recorded in `run_settings.json`; if unavailable, it warns and omits the missing
geometry. No solve or retracing is performed.

Run the focused checks with:

```bash
python3 -m py_compile misc_scripts/sol_model_a.py tests/test_sol_model_a.py
conda run -n testenv python -m unittest discover -s tests -p test_sol_model_a.py -v
git diff --check
```

Tests cover exponential and custom profiles, both parallel branches and walls,
invalid directions, the temperature switch and density invariance, incomplete
traces and one-hit substitution (including lower bounds and original-data retention),
source indices, periodic/physical mesh geometry, integrated conservation,
and second-order convergence to the finite no-flux slab's analytic cosh solution
with decay length `sqrt(D_perp*Lc/(2*cs))`. These are numerical checks of the
prototype, not experimental validation of the closures.
