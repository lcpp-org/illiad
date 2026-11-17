# Boris poloidal crossing plots

`misc_scripts/plot_boris_plane.py` reads a full `Ion_traces.npy` position history,
intersects every consecutive valid saved segment with a chosen toroidal
**half-plane**, and makes a two-panel Matplotlib figure:

- Raw crossing points colored by estimated kinetic energy.
- A rectangular `(R, Z)` histogram, optionally normalized to area, crossing
  flux, or a time-averaged density estimate.

Both panels can have the matching saved magnetic Poincaré points overlaid.
Matplotlib is sufficient here because only extracted crossings are rendered.
The 3-D histories are memory-mapped and scanned in chunks; there is no particle
or time subsampling. `--chunk-steps` changes working memory and throughput, not
numerical resolution. PDF/SVG exports rasterize the scatter/mesh layers while
keeping text and axes as vectors. Fastplotlib is not required.

## Run

From the repository root, in the normal ILLIAD environment:

```bash
python misc_scripts/plot_boris_plane.py \
  0mm_LCFS78_2eV_25V_Z1_HeEvap2_NoIslands2 \
  --phi 0 --poincare --histogram area --log-hist
```

The positional argument is the **subanalysis directory name**, defaulting to
this example. The script loads
`<analysis-dir>/data/<subanalysis>/Ion_traces.npy` and its matching log.
The analysis root defaults to this repository's
`output/IOTA3_1000sp_atol1e-9`; select another with
`--analysis-dir output/OTHER_ANALYSIS`.
For a density estimate using the default subanalysis:

```bash
python misc_scripts/plot_boris_plane.py --phi 36 --poincare \
  --histogram density --bins 100 100 --log-hist --formats png pdf
```

`--show` opens the figure. `--output DIRECTORY` changes where it is saved.
Without it, files go in
`<analysis-dir>/plots/<subanalysis>/plane_phi<angle>/`:

- `plane.png` (and other requested formats).
- `crossings.npz`: `R_m`, `Z_m`, `time_s`, `energy_eV`, signed
  `v_normal_m_s`, `speed_m_s`, `trace_column`, `segment_index`.
- `histogram.npz`: `values` in `(R_bin, Z_bin)` order, `R_edges_m`,
  `Z_edges_m`, and the quantity label.
- `settings.json`: input path, resolved timing/mass, estimator, weighting,
  exclusions, and CLI settings.

`trace_column` identifies the column in the saved trace, not the original
simulation particle ID (only a subset of ions may have been traced).
Running again into the same output directory replaces these products.

## Timing and energy

The matching `logs/<run>/boris.log` provides **ION MASS**, **TIME STEP**, and
**TRACE OUTPUT STRIDE**. Do not use the background gas mass for the ion mass.
The regular saved interval is `TIME STEP * TRACE OUTPUT STRIDE`, not `DT` alone.
For the supplied example this is `1e-8 * 15 = 1.5e-7 s`, with ion mass
`6.941 amu`. The completed Torch run has 200,000 nominal steps; the actual last
advance is step 199,999. Its appended final interval is only `4e-8 s`.

When the completed Torch log (or explicit early-stop message) and saved row
count agree, the script reconstructs that last interval. Warp early-stop timing
is not currently recorded sufficiently in the log, so it is not inferred from
nominal duration alone. If the final interval cannot be established, the script
warns and excludes it. Supply `--final-interval SECONDS` to include it, or
`--times saved_times.npy` with one strictly increasing timestamp per saved row.
`--sample-dt SECONDS` and `--mass-amu AMU` override log values. An explicit
sample interval disables log-based inference of the final interval.

`Ion_traces.npy` stores **positions only**. The plotted energy is

```text
v_segment = (x[k+1] - x[k]) / (t[k+1] - t[k])
E_estimated = mass * |v_segment|² / (2 * joules_per_eV).
```

This is a segment estimate of crossing energy, not saved instantaneous kinetic
energy. A large stride can average over gyromotion, underestimate speed/energy,
and hide repeated crossings within one saved interval. Collisions and the final
segment before a particle freezes at its wall position also limit this estimate.
Reducing plotting chunk size cannot recover missing temporal resolution. For
accurate instantaneous energies, future Boris output must save velocities (or
energies) and timestamps at adequate cadence; this script does not change the
solver or its output format.

## What the histogram measures

Let `A = ΔR ΔZ`, `T` be the observation duration, `w` the particle weight,
and `v_n` the velocity normal to the selected plane. Both travel directions
are included by default and repeat crossings are retained.

| Option | Bin value | Interpretation |
| --- | --- | --- |
| `count` | `sum(w)` | Crossing events, not unique particles |
| `area` | `sum(w) / A` | Crossing fluence per poloidal area |
| `flux` | `sum(w) / (A T)` | Mean unsigned normal crossing flux |
| `density` | `sum(w / abs(v_n)) / (A T)` | Estimated time-averaged local density of the traced ensemble |

Density needs **inverse normal speed**, not multiplication by total speed.
In a thin slab of thickness `dl`, a crossing contributes residence time
`dl / abs(v_n)`. Dividing the accumulated residence by volume `A dl` and time
`T` cancels the slab thickness. Thus a separate arbitrary toroidal volume factor
should not be applied to plane crossing counts. This is the thin-slab limit of
a residence-time estimator; it is not a finite-angle wedge occupancy calculation.
The relation between density, particle speed and track-based flux estimators is
described in the [OpenMC tally methods](https://docs.openmc.org/en/stable/methods/tallies.html).

Default `w=1` describes only the saved ensemble. `--particle-weight` is an
explicit common physical-particle weight; no scaling from the traced subset to
the full run or experimental plasma density is guessed. Nonuniform particle
weights/source sampling would require additional provenance and are not inferred.
`T` defaults to the included saved time span and can be set with `--duration`.
For a pulse launch this is a time average of that decaying ensemble, not a
steady-state source density or an instantaneous density snapshot.

Near-tangent crossings have large inverse-speed weights. Use
`--min-normal-speed SPEED` to exclude them if desired; the exclusion count and
cutoff are recorded. This introduces a bias, so compare results as the cutoff,
trace stride and bin size change. No cutoff is imposed silently.
`--direction positive` or `negative` selects one contribution instead of both.
A trajectory confined to the plane contributes no transverse crossings; this
estimator cannot describe that singular population.

## Geometry and overlays

Phi is ILLIAD's **clockwise computational angle**, `phi = -atan2(y, x)`;
`R` is cylindrical major radius and `Z` is height in metres. Positive normal
velocity means increasing computational phi. The opposite half-plane at
`phi + 180°` is rejected. Initial points on the plane are not crossings;
exact sampled hits require a continuation across it. Tangential touches and
plane-dwell endpoints are excluded. Nonfinite samples and all-zero padding are
never joined across gaps. Frozen terminal tails produce no repeated events.

`--poincare` finds the matching `data/Poincare/Poincare_<phi>.npy`;
`--poincare PATH` selects another native directory or a file with a matching
angle in its filename. There is no silent nearest-angle substitution. Native
arrays store `(theta, rho)` per surface; they are transformed as
`R = R0 + rho*cos(theta)`, `Z = rho*sin(theta)`. The default `R0=0.72 m` and
circular vessel radius `a=0.19 m` match HIDRA; both are editable.
`--poincare-size`, `--poincare-alpha`, and `--poincare-color` control visibility.

Automatic bounds include all crossings, the vessel outline, and the overlay.
`--r-range MIN MAX` / `--z-range MIN MAX` select a view; excluded crossings are
reported but retained in `crossings.npz`. Histogram colors are clipped exactly
at the circular vessel outline; empty bins inside retain the lowest color.
Saved bin values and area normalization are unchanged: boundary cells still
use their full rectangular area, so interpret partially filled cells accordingly.

## Magnetic mesh markers and average energy

Use `--plot-type nodes` for the node view, or `--plot-type both` to save both
`plane.png` and `plane_nodes.png`. `--node-size` defaults to **35 points squared**.
For example:

```bash
python misc_scripts/plot_boris_plane.py \
  0mm_LCFS78_2eV_25V_Z1_HeEvap2_NoIslands2 \
  --phi 0 --poincare --plot-type both --histogram energy --log-hist
```

`--histogram energy` computes the arithmetic mean of crossing energies in each
cell (not a residence-weighted mean). This works for both views. Empty energy
cells are saved as NaN because their mean is undefined; they are displayed at
the lowest color. Empty count/density cells retain numerical zero and the same
colored floor. The energy remains an estimate from saved position differences.

The default magnetic field is the same native toroidal field file used by the
Boris CLI. Override it with `--magnetic-field PATH`. Only its shape is read:
the native `(3, nrho, ntheta, nphi)` format implies uniform rho in `[0,a]` and
periodic theta nodes at `dtheta, ..., 2π`, matching the field loader. This option
is for that native mesh convention, not arbitrary nonuniform field arrays.
`--major-radius` and `--minor-radius` must match the field geometry.

Each crossing is assigned to a node-centered polar cell, bounded by radial and
periodic angular midpoints. Cell area is `0.5*(rho_hi²-rho_lo²)*delta_theta`.
The coincident axis nodes are merged into one disk/marker and the outer cells
end at the vessel radius. This is a local histogram estimate at mesh nodes,
not a magnetic-field interpolation or smoothing of the rectangular histogram.
All geometrically distinct nodes are displayed, including empty ones; markers
are clipped at the vessel wall. `--bins` only controls the rectangular view.

`node_histogram.npz` saves node coordinates, cell areas, crossing counts, values,
labels, and exclusion counts. Crossings outside the vessel are excluded from
node tallies and reported. Existing particle-weight, duration, direction, and
density speed-cutoff options apply to both views. Choose separate `--output`
directories when keeping multiple quantities, since filenames are reused.
