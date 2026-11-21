# Straight-field-line coordinate plots

The additional diagnostic plots show periodic Fourier cross-sections and
`SL_NTHETA` lines of constant straight-field-line poloidal angle. Enable them
in the normal flux calculator JSON:

```json
{
  "PLOT_STRAIGHT_ANGLE": true,
  "SL_NTHETA": 16
}
```

The active `input_files/flux_calc/flux_calc_inputs.json` enables 16 lines.
`NTHETA` still specifies the original flux-calculator output mesh. The new
plot is independent of `PLOT_ALL` and does not alter flux integration or its
saved meshes. The vacuum vessel is drawn at the field's minor radius, with
fixed axis limits across planes. Light-gray curves reuse the production
geometric contour reconstruction, including the original island-center shifts,
without integrating flux. They remain visible even when the angle map fails.
Dark-gray curves are the new parametric Fourier fits in straight angle. Colored curves
join equal straight-angle samples on adjacent accepted, nested sections.

To regenerate only this diagnostic from saved Poincare crossings:

```bash
MPLCONFIGDIR=/tmp/illiad-mplconfig OPENBLAS_NUM_THREADS=1 \
  conda run --no-capture-output -n testenv \
  python -m misc_scripts.plot_straight_field_coordinates \
  --inputs input_files/flux_calc/flux_calc_inputs.json --n-theta 16
```

Optionally add `--planes 36 180` for a smaller comparison. This command writes
only the `straight_angle` products; it performs no magnetic-field loading or
flux integration. BLAS threading is limited in the example to avoid overhead
from many small least-squares fits; this does not change resolution.

## Source and implementation

Source: I. Predebon et al., *Plasma Phys. Control. Fusion* **60** (2018) 045003,
[local PDF](Predebon_2018_FluxSurfaceMethod.pdf), sections 2.1–2.4,
equations (3)–(6). The paper constructs a straight angle from **unwrapped
toroidal progression**, not by scaling geometric poloidal angle. It requires
separate domains of nested conserved surfaces and excludes chaotic layers.

The implementation follows the return-map construction with these explicit
numerical adaptations:

1. Preserve chronological crossing indices. For a q-lobe chain, take every
   qth crossing to obtain successive returns to one lobe. The existing island
   detector supplies q. This q-return extension goes beyond the paper's
   illustrated m=1 island cases; each lobe has its own coordinate chart.
2. Calculate squared distance from a sampled outboard extremum. Interpolate
   distance versus return index with a cubic spline, locate near-reference
   minima, and average reciprocal return intervals as in equation (4).
   Minima farther than 15% of the maximum squared distance are excluded,
   avoiding inboard minima where they are distinguishable. Multiple spline
   extrema separated by less than one sample are merged because they cannot
   be resolved as separate returns.
3. Refine the frequency by minimizing positional residuals of a periodic
   Fourier fit. Geometric winding, equation (3), identifies the initial
   frequency basin; it is not used as the final angle. This refinement is
   an addition to the paper: a small frequency error accumulates over the
   hundreds of saved returns and scrambles equation (6)'s ordering.
4. Assign `theta_sl = 2*pi*f_return*(j/q + delta_phi/(2*pi*q)) - phase_origin`,
   modulo `2*pi`, where j is the original chronological crossing index and
   delta_phi is the directed displacement from the logged launch plane.
   A q-lobe return thus spans q toroidal turns. The reported local slope is
   `iota_sl = SL_TOROIDAL_DIRECTION*f_return/q`. It is not the global rational
   rotational transform used by the island detector.
5. Fit **both coordinates** `(R-R0, Z)` as periodic Fourier functions of this
   angle. Parametric curves allow concave island shapes that a single-valued
   radius about the global axis cannot represent. This substitutes a
   regularized Fourier fit for section 2.3's parametric spline.

One accepted lobe sets the frequency for a given surface and q; every subsequent
lobe and toroidal plane must fit that same frequency. The phase origin is set
at a fitted maximum-R point on the first accepted reference plane for each
surface/lobe, then kept fixed on all subsequent planes. A surface-dependent
constant offset is allowed without changing `dtheta_sl/dphi`; choosing a new
offset independently on every plane would not be allowed. The reference plane
and phase origin are recorded. Changing the requested plane order can change
this gauge, so use the same plane order when comparing runs.

## Controls and validation

| Input | Default | Meaning |
| --- | --- | --- |
| `PLOT_STRAIGHT_ANGLE` | `false` | Enable the new diagnostic during flux calculation. |
| `SL_NTHETA` | `16` | Number of equally spaced angle levels, without a duplicate 360-degree line. |
| `SL_MODES` | `8` | Highest Fourier harmonic in each Cartesian cross-section coordinate. |
| `SL_REGULARIZATION` | `1e-8` | m^4 curvature penalty, with sample-count normalization. |
| `SL_MAX_REL_ERROR` | `0.03` | Maximum held-out RMS position error divided by the bounding-box diagonal. |
| `SL_MAX_POINTS` | `null` | Uncapped by default; a positive integer keeps only the first N consecutive crossings per field line and phi plane for straight-angle fitting. Applied before island-lobe splitting. |
| `SL_TOROIDAL_DIRECTION` | `1` | Sign of computational toroidal progression in the saved ordered trace. |

The cap affects only straight-angle reconstruction, including frequency estimation and
held-out validation. Geometric contours and flux integration continue to use all
points. Each island lobe receives approximately N/q crossings; insufficient
samples still cause rejection. This truncates history, rather than thinning it.
For example, set `"SL_MAX_POINTS": 512` to use the first 512 crossings, or
`null` to use all crossings. `"SL_MAX_REL_ERROR": 0.03` means 3%.

The default direction is for forward increasing-phi data. Reverse progression
requires `-1`; it is an explicit assumption, not inferred from sorted points.
The source must log `DOUBLE_LINE=False`: legacy concatenated forward/backward
arrays have no per-crossing direction/transit metadata. Missing interior
crossings also cause rejection rather than silently renumbering the data.

Once a frequency is established, later planes do not repeat the distance-minimum
frequency-estimation test. That test depends on section shape and can fail on a
regular surface whose geometric contour is easy to fit. Later planes still
have to predict their held-out positions and provide adequate angle coverage.
Counts in the plot title distinguish geometric contours from accepted/rejected
angle maps; island lobes count separately. Counts can vary when island detection,
coverage, or positional errors vary with phi, and rejection does not hide the
separately reconstructed geometric fit.

This implementation supports the **unaliased, zero-integer-turn branch** of the
local return map, with frequency less than half a poloidal turn per return.
It does not implement the paper's integer-turn recovery in equation (5).
Single-plane samples cannot exclude hidden complete turns; the user must
establish the branch from sufficiently resolved tracing. This restriction is
particularly important for configurations with high rotational transform.

At least `max(80, 6*(2*SL_MODES+1))` crossings per lobe and four local turns
are required. The return-minimum estimate is now diagnostic only: missing
minima or disagreement with geometric winding does not reject the mapping.
Frequency refinement starts from the winding estimate and must still pass
the positional, stability, and coverage checks below. This bypass is an
explicit departure from relying on equation (4) for the final frequency. A fit
learned from the first 70% of crossings must predict the last 30% within the
selected error threshold. Full-history frequency refinement may accumulate
at most 0.15 rad relative to the training estimate, and the phase coverage
may not have a gap larger than 90 degrees. These are numerical acceptance
checks, not a proof of integrability or an independent field-model validation.

Rejected sections remain gaps. Connections also stop at changes in q,
chronological lobe identity, missing surface indices, or loss of nesting
between sampled Fourier contours. Straight segments between accepted surfaces
are only a plotting interpolation: no volume coordinates, Jacobian, or metric
tensor are constructed. Regions near separatrices often lack the sampling
needed for this reconstruction, as the paper discusses. A rejection does not
by itself establish that a surface is chaotic; insufficient return-time
resolution can reject regular surfaces too.

## Outputs

Under `output/<ANLYS_DIR>/`:

* `plots/<ANLYS_SUBDIR>/straight_angle/Coordinates_NNN.png` and `.pdf`:
  Fourier contours and colored constant-angle lines at each requested plane.
* `data/<ANLYS_SUBDIR>/straight_angle/Coordinates_NNN.npz`: angle levels,
  surface/q/lobe identifiers, Fourier coefficients, and sampled coordinate
  points. Coordinates are `(R-R0, Z)` in metres. Coefficient order is constant,
  cosine harmonics 1..M, sine harmonics 1..M, with one column per coordinate.
* `data/<ANLYS_SUBDIR>/straight_angle/diagnostics.json`: accepted/rejected
  sections, reasons, coarse and refined frequencies, held-out residuals,
  coverage, phase origins, reference planes, and configuration.

Running again replaces these diagnostic products for the requested planes.
The JSON describes only the current invocation; old images of unrequested
planes can remain in the directory.

Tests in `tests/test_straight_angle.py` cover a known nonlinear geometric-angle
mapping, a concave contour, sparse/unstable input rejection, merged-direction
rejection, and three-lobe coordinate output. Use the same dependency-complete
environment as the flux calculator.
