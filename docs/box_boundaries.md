# Oriented-box boundary prototype

Boris (Torch and Warp), Cartesian Poincare tracing, and Torch SOL tracing accept
`BOUNDARIES`, a list of stationary boxes. `[]` retains the existing tracing path.
The shared Boris runner also supports the axis and planar source scripts.
The specialized standalone LCFS-distance scripts are not wired to this prototype.

Copy the keys in `input_files/box_boundaries.example.json` into your existing
workflow JSON. That file is a configuration fragment, not a complete source/field
configuration. Choose a new `TAG` / output subdirectory for a new experiment:
the event writer requires a fresh `Boundary_events` directory and refuses to
silently overwrite an earlier event archive.

```json
"BOUNDARIES": [
  {
    "name": "collector",
    "shape": "box",
    "width_m": 0.02,
    "depth_m": 0.04,
    "height_m": 0.06,
    "center_rzphi": [0.82, 0.0, 0.0],
    "rotation_local_xyz_deg": [0.0, 0.0, 25.0],
    "mode": "absorbing"
  }
]
```

The example placement is illustrative; adjust it for the object you intend to
model. Positive finite dimensions and unique nonempty names are required.
Missing rotation means zero rotation; missing mode means `absorbing`.

## Coordinates and rotations

Specify `center_rzphi: [R_m, Z_m, phi_deg]`: R and Z are Cartesian coordinates
within the selected poloidal plane. R is the cylindrical distance from the
machine axis (not minor radius or offset from R0); Z is vertical height. Phi is
the computational toroidal angle in degrees, increasing clockwise viewed from
+Z. The conversion is `X=R*cos(phi)`, `Y=-R*sin(phi)`, `Z=Z`, using the same
convention as ILLIAD's existing RTP transforms. Wall-facing display offsets do
not apply. Angles wrap periodically; R must be nonnegative.

For example, `[0.72, -0.12, 90.0]` places the center at XYZ `[0, -0.72, -0.12]` m.
The example JSON retains your box's physical position as `[0.72, -0.12, 0.0]`.
New manifests (version 2) save centers in this format. Legacy `center_m` continues
to mean XYZ metres for existing configurations and version-1 archives; supplying
both center keys is an error.

Width is local x, depth local y, height local z. Zero rotation still aligns those
axes with simulation XYZ. Selecting a different center phi moves the box; it does
not implicitly rotate it. Set `rotation_local_xyz_deg` to orient its faces.
Objects do not repeat automatically across magnetic periods.

Rotations are right handed, in degrees, applied about the **moving local x,
then moving local y, then moving local z** axes. The local-to-world matrix is
`R=Rx@Ry@Rz`, with column-vector convention `world=center+R@local`.
The inverse is `local=R.T@(world-center)`. Rotation order matters.

Faces are indexed `0..5 = -x,+x,-y,+y,-z,+z`. On an axis-i face,
`u=local[(i+1)%3]`, `v=sign*local[(i+2)%3]`. Thus `(u,v,normal)` is right handed
on both sides. Edge/corner events have a bit mask of all touched faces and are
counted once, assigned to the lowest face index. Their reported normal and UV
use that primary face; there is no unique physical corner normal.

## Integration semantics and numerical limits

Every physical Boris step is tested, regardless of trace-save stride or Warp
chunk size. A slab intersection detects both entry and exit even when both
endpoints are outside the box. `absorbing` terminates on first contact and freezes
selected trajectories there; `transparent` records all entry/exit crossings and
leaves the position advance unchanged. Neither mode modifies B/E/density fields.
Initial positions inside or on an absorbing box fail validation. Transparent
seeds may start inside: their first event is an exit. The initial point itself
is not an event.

Contacts along a face or at a tangent edge/corner are classified separately.
A segment following a face records contact at its finite-box entry and exit.
Numerical event ownership is `(start,end]` with `BOUNDARY_TOLERANCE_M` (default
`1e-10 m`) suppressing duplicate shared endpoints. Features/crossings closer
than that tolerance cannot be distinguished. Boxes use their six finite faces,
not infinite planes.

The first absorbing surface wins; equal-time box ties use input order, and the
vessel wins an exact box/vessel tie. Transparent events later in the step are
excluded. Existing plane histograms and Poincare maps stop at a box hit too.
Boris vessel output retains its legacy first-outside sample convention; only
vessel hits enter `Wallpt_OUTPUT.npy`. New object hits live in the event archive.
The Boris running-fraction plot includes both kinds of loss.

Boris and fixed-step SOL crossings are exact for the numerical straight segment,
not for the unresolved curved physical orbit. Particle event velocity is the
velocity used to advance that segment, before its second collision half-step;
energy is calculated from that velocity and particle mass. No fractional-time
collision model, reflection, sheath, sputtering, or field response is introduced.
Reduce `DT` / `STEP_SIZE_M` to check convergence, particularly for thin objects
and gyrating trajectories. Vessel-ordering refinement assumes the usual short
inside-to-outside step; it is not a general multi-intersection torus solver.

Poincare uses the selected SciPy solver, its existing tolerances, a maximum
accepted step `BOUNDARY_MAX_STEP_M` (default `0.005 m`), and dense-output
subdivision controlled by `BOUNDARY_CURVE_TOLERANCE_M` (default `1e-7 m`).
The chord tolerance is also capped at 1/20 of the smallest box dimension.
Face roots are refined on dense output. Exact mathematical tangencies are
numerically delicate: refine both step size and chord tolerance when they matter.
Transparent objects do not alter the RHS, but enabling the explicit maximum
step can change numerical sampling relative to an older unconstrained run.
Compare with matching solver settings. Absorbing Poincare runs skip automatic
LCFS identification because survival is then geometry dependent.

Torch SOL (including trace-and-regularize) uses eager one-step chunks when any
boxes are present. This costs throughput but retains the configured physical
step size and checks every step before advancing again. Added `hit_boundary.npy`
and `boundary_intersection_xyz.npy` distinguish object hits from `hit_wall.npy`.
Connection lengths now end at either absorbing objects or the vessel; they are
geometrical surface-to-surface lengths, not necessarily vessel-to-vessel lengths.
Existing length-limit behavior is retained. Interpret downstream SOL closures
accordingly; no new object-specific plasma closure is supplied.

## Event files and memory

Boris writes `data/<subanalysis>/Boundary_events/`; SOL uses its analysis
subdirectory, and Poincare uses `data/Boundary_events/`. Each contains
`manifest.json` and compressed NPZ shards. Poincare workers write separate
`line_*` subdirectories and the final root manifest lists their shards.

```python
from illiad.boundaries import iter_events
for events in iter_events("output/RUN/data/SUBANALYSIS/Boundary_events"):
    entry = events[events['kind'] == -1]
```

The structured array in each shard has:

| Field | Meaning |
|---|---|
| `trajectory_id` | Zero-based original particle row / field-line seed index |
| `source_id` | Original `Particle.particleID`; SOL uses seed index |
| `direction` | 0 for particles; +1/-1 for the field-line tracing direction |
| `boundary_id` | Index into manifest `boxes` |
| `face_id`, `face_mask` | Primary face and all touched faces |
| `kind` | -1 entry, +1 exit, 0 contact |
| `step`, `fraction` | One-based accepted step and fraction within that step |
| `coordinate` | Particle time in seconds or positive traced arc length in metres; see manifest |
| `xyz_m`, `uv_m` | Global and primary-face positions |
| `normal_xyz` | Primary-face outward unit normal |
| `vector_xyz` | Particle advance velocity or field-line tracing tangent |
| `energy_eV` | Particle kinetic energy; NaN for field lines |

Transparent crossings are visits, not deposited particles. Forward/backward
field-line branches share the seed index but retain separate direction values.
Use `(trajectory_id,direction)` for branch identity. Ordering across shards is
not chronological: sort by identity and coordinate when reconstructing histories.

`BOUNDARY_BUFFER_ROWS` (default 65536) controls event buffering, not numerical
resolution. Warp downloads and clears its bounded buffer after each launch.
If one launch exceeds capacity, tracing fails explicitly and the archive remains
marked incomplete. Increase this setting or reduce `WARP_STEP_CHUNK_SIZE` for
many simultaneous crossings. CPU buffering holds at most roughly twice this
many buffered rows before flushing, plus the current hit batch; terminated particle times additionally require O(N)
small bookkeeping. Large dense all-trajectory recording can be costly.

A manifest has `complete=false` until its writer finishes. Only listed shards
are valid. Failed Poincare workers leave per-line manifests for inspection;
without a completed root manifest there is no complete combined archive.
Python tests can use an in-memory `BoundaryRecorder`; production workflows always
write shards. Geometry metadata, conventions, and tolerance travel with the data.

## Replotting without tracing

```bash
python misc_scripts/plot_boundary_events.py \
  output/RUN/data/SUBANALYSIS/Boundary_events \
  --output output/RUN/plots/SUBANALYSIS/boundaries \
  --bins 40 --metric count --kind all
```

This produces six face heatmaps per box (PNG/PDF), numerical histogram NPZ files,
and a 3D preview with local axes. The preview adds a faint, cropped torus wall
and highlights the poloidal wall curve at each box's phi. A small R-Z locator
shows the full vessel section and the first box's orthogonal projection onto
that plane (not a box/plane intersection). A bottom-right top-down locator shows
the torus footprint and each box's horizontal projection. The North-side split
(computational phi=162 degrees) points up; increasing computational phi runs
clockwise, so phi=252 degrees points right. Main-view limits follow the boxes
and include nearby wall only, keeping the boxes prominent. For a very small box
far from the wall, the locator supplies vessel context without zooming out.
Each box uses a common color scale across faces.
Face heatmaps use a connected, dimensionally scaled unfolded-box net:

```text
     -z
     +y
 -x  +z  +x
     -y
```

The +z face retains local x/y orientation. Neighboring maps are rotated so
identical physical points coincide along the shared edges; -z attaches to +y's
outer edge. Dashed lines mark folds. This display applies to all metrics and
event filters, including `*count_all.png`; saved histogram arrays retain their
original face order and face-local UV coordinates.

Metrics are `count`, `area` (crossings/m²), `unique` (distinct trajectory IDs per
bin, combining field-line directions), `energy`, and `angle` (degrees from the
primary-face normal, using absolute normal projection). Select `--kind entry`,
`exit`, or `contact` to filter events. `unique` retains a set of visited
trajectory/bin combinations and can require substantial memory. It is not
additive across bins. Empty mean bins are NaN.

All events contribute to the face histograms. The 3D preview shows only the first
`--preview-limit` selected events (default 10000); it is a placement check, not an
unbiased statistical sample. No particle weight or absolute fluence is inferred.

Preview placement directly from your input JSON, before tracing:

```bash
python misc_scripts/plot_boundary_events.py \
  --geometry input_files/box_boundaries.example.json \
  --output /tmp/box_preview
```

`--preview-padding` controls the view margin (default 1.25; minimum 1.05).
Smaller values zoom closer. Trace event manifests now retain the actual circular
torus `major_radius_m` and `minor_radius_m`. Geometry-only inputs use
`MAJOR_RADIUS_M` and `VESSEL_RADIUS_M` when present. Older archives or inputs
without these values use an explicitly labeled reference of R0=0.72 m, a=0.19 m.
Override either with `--major-radius` / `--vessel-radius` (metres). These plotting
options never change tracing geometry or saved event positions.
