# Magnetic-axis Boris source

Run from the repository root in an ILLIAD environment:

```bash
python misc_scripts/boris_axis_source.py input_files/boris_axis_source.example.json
```

The script uses the shared Boris runner, its collision models, and its usual
wall histogram, deposition-energy/angle plots, running fraction and selected
trajectory plots. `BORIS_METHOD` accepts `torch` (default) or `warp`. Warp retains
its normal physical timestep, chunking and active-particle compaction behavior.

## Source and axis input

`NPHI` emitters span the full torus with spacing `360/NPHI` degrees, starting at
`AXIS_PHI_START_DEG`. Each emitter launches `NPARTICLES_PER_EMITTER` particles.
The Cartesian velocity components are independent zero-mean Gaussians with
standard deviation `sqrt(e * ION_TEMP / mass_kg)`. Speeds are Maxwellian,
directions are uniform in solid angle, and mean initial energy is `3/2 ION_TEMP`.
The reference direction used in the initial-velocity plot is local +phi; it
introduces no launch bias. `SOURCE_SEED` controls initial velocities only, not
stochastic collisions.

`AXIS_FILE` explicitly selects saved axis centers for the magnetic configuration.
It accepts a numeric `.npy` with shape `(nphi, 2)` or native
`(nphi, nslots, 2)` in **(theta radians, rho metres)** order, relative to the
geometric center. Slot zero is the axis; additional slots must be zero/NaN
padding. Files containing multiple nonzero island centers are rejected.
Use the innermost-surface center file used by the flux interpolator, not an
arbitrary surface's island centers.

`AXIS_PHI_DEG: null` means native planes `360/nphi, ..., 360`. Supply an explicit
list for a different plane ordering. Interpolation is periodic and linear in
`(R-R0, Z)`, avoiding angular wrap artifacts. The supplied IOTA3 example uses
`LCFS21/fSurf_104_center.npy`; its six planes are 60, 120, ..., 360 degrees,
confirmed against the matching flux log. This is an approximation from saved
centers, not a new magnetic-axis solve. More emitters do not improve the source
axis resolution. Use a denser axis file if its toroidal variation matters.

All field files, currents, axis provenance, temperature, counts and tracking
settings are editable in the JSON. `LCFS_INDEX=0` and `0mm` in the shared output
subdirectory name are naming placeholders for this source, not an LCFS launch.

## Optional paths

| Setting | Meaning |
| --- | --- |
| `TRACK_NPHI` | Number of evenly sampled emitter indices to track |
| `TRACK_NPARTICLES_PER_EMITTER` | Number of sampled particles at each tracked emitter |
| `STRIDE` | Save positions every this many integration steps, plus initial/final samples |

Set either tracking count to zero to disable saved trajectories. Set `STRIDE=1`
to retain every integration step of the selected particles. The example tracks
12 particles (6 emitters times 2 particles). Counts cannot exceed the available
emitters or particles; the script rejects invalid settings.

`Ion_traces.npy` retains the normal float32 `(sample, tracked_particle, xyz)`
format; `Trace_particle_IDs.npy` identifies its columns using zero-based indices
in the initial-condition array. IDs are emitter-major:
`emitter_index * NPARTICLES_PER_EMITTER + particle_within_emitter`.
Only the selected columns are allocated during tracing. Paths still scale with
tracked particles times saved timesteps; setting both tracking counts to zero
leaves a zero-column trace array with no position payload.

## Optional streaming plane histogram

Set `PLANE_HISTOGRAM_PHI_DEG` to a computational toroidal angle in degrees, or
`null` to disable it. Computational phi is `-atan2(y,x)`; it is the same
coordinate used by the field mesh, not the shifted wall-plot display angle.

All particles contribute to **one combined histogram**, including repeated
crossings by the same particle. Separate increasing/decreasing phi counts are
also retained. The default plot shows raw event counts, not unique-particle counts. New runs
also accumulate sums of crossing energy and inverse normal speed, allowing
energy and normalized plots without saving events or per-particle histograms.

Every physical Boris position update is tested, including steps that terminate
at the wall. A crossing is the linear Cartesian segment's intersection with
the selected positive-R half-plane, retained only within the vessel. The
opposite toroidal half-plane is excluded. Segments use `(start, end]`: arrival
at the plane counts once, departure from it does not, and particles initialized
on the plane do not generate an initial event. Distances within `1e-14 m` of
the plane are treated as zero. Timestep resolution still limits the physical
trajectory; crossings within unsampled gyromotion cannot be recovered.

Cell **edges are the magnetic-field mesh nodes**, not node-centered bins:

- `rho_edges_m = 0, dr, ..., a`, giving `mesh.nr - 1` radial cells.
- `theta_edges_rad = 0, dtheta, ..., 2*pi`, giving `mesh.ntheta` angular cells.
  The native final node at `2*pi` is also the periodic node at zero.
- Cells are lower-inclusive and upper-exclusive, with the outer vessel edge
  included in the final radial cell. Rho is a distance in metres, not normalized
  flux or distance from the magnetic axis.

The tally is independent of `STRIDE` and trajectory selection. Torch uses device
integer scatter additions; Warp uses int64 atomic additions inside each physical
substep, including within a multi-step kernel launch. The four persistent directional tallies occupy
`2 * (nr-1) * ntheta * 8 * 4` bytes, independent of particle count and simulation
length. For a 191 by 180 mesh this is 2,188,800 bytes (about 2.09 MiB).
An additional mass buffer has one float64 value per particle; no velocity or
crossing history is retained.

The shared run's `data/<subanalysis>/` contains:

- `Plane_crossings.npz`: `counts`, `counts_increasing_phi`,
  `counts_decreasing_phi` (all `(nr-1, ntheta)` int64 arrays), both edge arrays,
  exact polar-cell areas, phi, vessel dimensions, timestep and particle count.
  Schema version 2 adds `energy_sum_eV`, `inverse_normal_speed_sum_s_m`,
  `density_excluded_counts`, and their `_increasing_phi`/`_decreasing_phi` variants,
  plus `observation_duration_s`, `min_normal_speed_m_s`, and `energy_method`.
- `AxisIonSeedPts.npy`, `AxisIonSeedPhi_deg.npy`, `initVelPos.npy`,
  `axis_source_settings.json`, selected trajectories/IDs and normal wall outputs.

`plots/<subanalysis>/Plane_crossings.png` shows counts in R-Z mesh cells,
including zero-count cells inside the vessel. The exterior has no cells.
Normal wall plots are produced when at least one particle hits the wall.

## Validation

`tests/test_plane_crossings.py` checks analytic crossing energies, inverse normal
speed weighting, normalizations, early termination, empty plots, legacy files,
CLI exports and Torch/Warp CPU statistics agreement across chunk/compaction
settings. `tests/test_boris_methods.py` covers existing backend behavior.
CPU agreement is not GPU validation or timestep convergence evidence.

## Replot saved crossings

No Boris run, field loading, or trajectory files are needed:

```bash
python misc_scripts/plot_plane_crossings.py SUBANALYSIS --histogram count --log-hist
python misc_scripts/plot_plane_crossings.py SUBANALYSIS --histogram energy --cmap magma --vmin 0 --vmax 20
python misc_scripts/plot_plane_crossings.py SUBANALYSIS --histogram density --log-hist --particle-weight 1
python misc_scripts/plot_plane_crossings.py path/to/Plane_crossings.npz --histogram area --formats png pdf
```

`SUBANALYSIS` resolves under `--analysis-dir/data/` (default analysis root:
`output/IOTA3_1000sp_atol1e-9`). Plots and derived numerical maps go under the
matching `plots/<subanalysis>/` directory. `--output` overrides that directory.
The original archive is never rewritten. Each output uses
`Plane_crossings_<histogram>_<direction>`, so different modes coexist;
rerunning the same mode/direction replaces that replot only.

For cell area A, observation duration T, physical particle weight w,
number of crossings C, energy sum S_E, and inverse normal speed sum S_v:

| `--histogram` | Value |
| --- | --- |
| `count` | w C (raw crossing events when w=1) |
| `area` | w C / A (crossings per square metre) |
| `flux` | w C / (A T), unsigned time-averaged crossing flux |
| `density` | w S_v / (A T), thin-slab time-averaged particle density estimate |
| `energy` | S_E / C, arithmetic mean energy of crossing events in eV |

Energy uses the actual velocity that advances the Boris position during that
physical step, **before the second collision half-step**. The approximation is
the solver's constant drift velocity over that step, not an estimate from
strided output paths. Repeated crossings contribute repeatedly. It is neither
a residence-time-weighted energy nor a temperature. Empty energy cells remain
NaN in derived `.npz` maps and display at the color floor.

Density sums `1/abs(v_phi)` only when the normal speed exceeds
`PLANE_HISTOGRAM_MIN_NORMAL_SPEED` (m/s; default 0). This cutoff affects density
only, not counts or energies. Choose a positive cutoff before running if grazing
crossings dominate the estimate. The cutoff and excluded event counts are
saved. Replotting cannot change the cutoff because individual speeds are not
stored; `--min-normal-speed` only asserts the saved value and rejects a mismatch.

`--duration` overrides the saved observation span for flux/density; it does not
select a time window. The saved default is the actual number of position updates
times DT, including early termination when all particles have hit the wall.
Use the intended longer observation window explicitly if that is the desired
ensemble averaging convention. `--particle-weight` is a constant physical
particle multiplier; it cancels out of mean energy. Absolute physical density
requires an appropriate weight and observation interval.

Other controls: `--direction both|positive|negative` (computational phi),
`--cmap`, `--log-hist`, `--vmin`, `--vmax`, `--r-range`, `--z-range`,
`--figsize`, `--dpi`, `--formats png pdf svg`, and `--show`.
`--poincare` overlays the exact saved phi plane from the analysis's Poincare
folder; an explicit file/folder and size/alpha/color controls are also accepted.
Only display limits change: the stored mesh edges and crossing membership remain
fixed. A different phi plane or finer/different bins requires a new run.

Old count-only archives support count/area replotting. Flux additionally needs
an explicit `--duration`, since old archives have no observation span. Energy
and density fail with a clear missing-statistic error and require a new run.

### Signed net flux

Use `--histogram net-flux` with the default `--direction both` to plot
`particle_weight * (counts_increasing_phi - counts_decreasing_phi) / (area * duration)`.
Positive means transport toward increasing computational phi; negative means
transport toward decreasing phi. A toroidal plane has no intrinsic inside/outside:
this is the chosen positive-normal convention. `flux` remains unsigned (the sum
of both directions). One-way fluxes use `--histogram flux --direction positive`
or `--direction negative`.

Net flux uses a diverging color map centered at zero with symmetric automatic
limits. Custom limits must satisfy `vmin < 0 < vmax`; logarithmic scaling is
rejected because it cannot represent signed values. Empty and exactly balanced
cells are zero. Existing directional-count archives suffice; supply `--duration`
if the archive predates saved observation spans. No simulation rerun is needed.
Physical absolute units require an appropriate `--particle-weight`; the default
weight of one describes the simulated particle ensemble.
