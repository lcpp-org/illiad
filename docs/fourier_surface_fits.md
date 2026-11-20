# Regular-surface Fourier reconstruction experiment

This tests periodic fitting of geometric contours before attempting a full flux
coordinate reconstruction. Predebon et al., *Plasma Physics and Controlled Fusion*
**60** (2018) 045003, [doi:10.1088/1361-6587/aaaa49](https://doi.org/10.1088/1361-6587/aaaa49),
describe straight-field-line coordinates reconstructed from Poincare maps. The
accessible publisher preview establishes that scope; this implementation does
not claim to reproduce the paper's complete algorithm. In particular, it does
not construct a straight-field-line poloidal angle, rotational-transform-based
angle mapping, metric tensor, or Jacobian.

## Representation and controls

`illiad.flux.fourier.fit_fourier_contour` represents a regular cross-section as

```text
r(theta) = a0 + sum[m=1..M] (a_m cos(m theta) + b_m sin(m theta))
objective = sum_i w_i (r(theta_i) - r_i)^2
            + lambda sum_m m^4 (a_m^2 + b_m^2)
```

Here theta is the **geometric angle about the existing fitted magnetic axis**,
exactly as for the flux calculator's spline input. Both fits use the same axis
and saved crossings. This is not necessarily the vessel-centered geometric
angle and is not a straight-field-line angle. Points are shifted back using
the existing coordinate transforms when generating the calculator's mesh.

Angular-cell weights (half the gaps on each side of a sample, normalized to sum
to one) reduce bias from uneven sampling. Exact duplicate angles are averaged.
The constant coefficient is unpenalized. An augmented least-squares solve avoids
normal equations. Values and first/second derivatives are analytically periodic.

The normal flux JSON accepts these additional inputs:

| Input | Default | Meaning |
| --- | --- | --- |
| `SURFACE_FIT` | `"spline"` | Use `"fourier"` for unsplit surfaces; island subsets still use splines. |
| `FOURIER_MODES` | `8` | Highest harmonic; larger values allow finer contour structure. |
| `FOURIER_REGULARIZATION` | `1e-6` | Dimensionless lambda; larger values suppress high harmonics more strongly. |
| `FOURIER_MAX_GAP_DEG` | `90` | Reject a fit with a larger gap between sampled angles. |

These are starting values for an experiment, not calibrated settings. Lambda
is not interchangeable with `SMOOTH_FCTR`, which retains the existing FITPACK
residual-budget meaning. `NTHETA` still controls the output sampling, while M
controls the fitted shape's resolution. The magnetic interpolation and flux
integration tolerances remain those selected by the existing workflow.

Use a new `ANLYS_SUBDIR` for each production experiment to preserve the spline
baseline. The normal flux output filenames/shapes remain unchanged. Failed fits
have NaN geometry/flux and false validity; an invalid LCFS aborts a Fourier run
because it cannot provide the normalization boundary. An island subset failure
cannot be hidden by a later successful subset.

The fitter requires at least `2*M+1` distinct angles, adequate angular coverage,
a full-rank weighted design matrix, positive input radii, and positive output
radii on a dense validation grid. These checks do **not** prove the contour is
star-shaped or a regular flux surface. A single-valued radius cannot represent
folds, island chains around the global axis, or a stochastic band. The existing
island detector decides which calculator surfaces are split; an unsplit result
is not itself a topology certificate. Inspect residuals, gaps, traces, and
neighboring surfaces before using a fit for physics.

## Reproducible comparison

From the repository root, in the dependency-complete environment:

```bash
MPLCONFIGDIR=/tmp/illiad-mplconfig conda run --no-capture-output -n testenv \
  python -m misc_scripts.compare_flux_surface_fits \
  --analysis IOTA3_1000sp_atol1e-9 \
  --subanalysis fourier_regular_test \
  --surfaces 36 48 80 --planes 60 180 300 --spins 32
```

The subanalysis must not already exist. Source arrays and prior results are
preserved. `--no-trace` produces a fit-only comparison and reports independent
metrics as unavailable. `--smoothing`, `--modes`, `--regularization`, and
`--max-gap-deg` make the fitting choices explicit. Sparse surfaces or detected
island chains are recorded as rejected rather than silently assigned a fit.

Fresh traces start from the first finite **measured** crossing, never from a
fitted curve, and run in both directions. These are independent integrations
of the same field model, not independent experimental measurements or a second
magnetic-field model. The source log supplies currents, field configurations,
error-field enablement, solver, and tolerances. Field loading, Cartesian field
interpolation, normalized FieldLine RHS, vessel event, and dense-output crossing
extraction use the existing CPU Poincare implementation. The latter retains its
existing root-solve/linear-fallback behavior. A maximum step of 0.05 m is imposed;
`--max-step`, `--method`, `--rtol`, and `--atol` permit explicit convergence checks.
`--spins` specifies arc length divided by `2*pi*R0` **per direction**, not an
exact number of toroidal returns.

The source log cannot establish historical identity of magnetic-field files.
The report records current field paths, sizes, and modification times, and
hashes each source Poincare array. Historical source-field hashes are unavailable.
The comparison currently uses the standard toroidal/helical field files, R0=0.72 m,
vessel radius 0.19 m, and the current default error-field magnitude/direction.
Do not interpret it as a matched comparison for a run with custom mesh files,
vessel dimensions, or error-field parameters.

Outputs under `output/<analysis>/`:

* `data/<subanalysis>/metrics.json`: sample counts, angular coverage, fit
  conditioning, training and independent residuals, separate +/-B results,
  seam jumps, trace termination/length, solver settings, and a fixed M/lambda
  sensitivity grid. Independent samples never enter any coefficient fit.
* `data/<subanalysis>/phi*_surface*.npz`: editable coefficients, spline knots,
  evaluation grids, training data, validation crossings/directions, and axis.
* `plots/<subanalysis>/phi*_surface*.png` and `.pdf`: overlaid contours, r(theta),
  and signed radial residuals against fresh traces.

Radial residuals compare at the same angle. The separate contour-distance
metrics use minimum Euclidean distances to 4096 sampled curve segments, which
can be smaller for steep or oscillatory curves. Neither should be interpreted
as an error in toroidal flux. The sensitivity grid is diagnostic, not an
automatic parameter-selection procedure. Short trace lengths, limited planes,
or small residuals do not establish global nestedness or field-line invariance.

The existing `FluxCalculator.integrate_flux` coordinate transformation has a
separate, verified ordering problem: it calls `axisShift(r, theta, ...)`, although
that function accepts `(theta, radius, ...)` and returns `(theta, radius)`.
For example, local `(r,theta)=(0.1,pi/2)` with an axis at `(R-R0,Z)=(0.03,0)`
should give vessel-centered `(r,theta)=(0.104403,1.279340)`; the current integration
path produces `(0.098129,1.600649)`. It is retained here to isolate the fitting
change. The contour comparisons do not call that integration path. Full
nonuniform-field flux outputs need a separate correction and validation before
being used to assess flux-coordinate accuracy.

## Measured IOTA3 results (2026-11-20)

Fresh +/-B integrations were completed for surfaces 36, 48, and 80 at 180 deg,
and surfaces 48 and 80 at 60 and 300 deg. All completed traces reached their
requested length without a vessel hit. Surface 21 was rejected at both
additional planes because it had only nine finite crossings. Seven automated
tests passed, including the analytic uniform-field area integral.

The following comparison uses M=12, lambda=1e-6 and the legacy spline's
`s=6e-6`. Entries are radial RMS errors against fresh traces, in millimetres:

| Plane | Surface | Fresh crossings | Fourier | Spline |
| --- | --- | --- | --- | --- |
| 60 deg | 48 | 16 | 0.143 | 0.106 |
| 60 deg | 80 | 16 | 0.059 | 0.069 |
| 180 deg | 48 | 66 | 0.093 | 0.108 |
| 180 deg | 80 | 66 | 0.056 | 0.075 |
| 300 deg | 48 | 16 | 0.129 | 0.119 |
| 300 deg | 80 | 16 | 0.075 | 0.063 |

The 60/300 deg runs used eight length-equivalent spins per direction and source
LSODA tolerances (rtol=2.49e-12, atol=1e-9), with max step 0.05 m. The table's
180 deg results used 32 spins per direction, rtol=2.49e-13, atol=1e-10, and max
step 0.025 m. Comparisons within a row share identical fresh crossings; the
different lengths and tolerances limit comparisons across rows.

A separate 180 deg pilot used 16 spins per direction at source tolerances.
Matching the first crossings in each direction between pilot and tighter runs
gave position differences of 0.00132 mm RMS (surface 48) and 0.00233 mm RMS
(surface 80), with maxima 0.00250 and 0.00442 mm. These are smaller than the
fitting residuals. This is a local tolerance check, not long-time convergence.

At surface 36, the original spline developed large spikes between saved
crossings; Fourier removed those spikes. Nevertheless, the initial M=8 fit's
fresh-trace radial RMS was 0.912 mm versus the spline's 0.532 mm (33 crossings).
The source points have a largest angular gap of 31.1 deg. This surface should
not be treated as a demonstrated regular contour merely because it is unsplit.

The result supports Fourier as an explicit, smooth experimental representation,
but does not support replacing every spline with one universal M/lambda pair.
M=12 improved the 180 deg inner-surface results relative to M=8; that parameter
sensitivity is reported rather than hidden by changing the default after testing.

Reports, coefficients, and figures live under the corresponding `data/` and
`plots/` subdirectories of `output/IOTA3_1000sp_atol1e-9/`:

* `fourier_regular_pilot_v2`: 180 deg pilot, M=8, with M/lambda sensitivity grid.
* `fourier_regular_convergence`: 180 deg, M=12 and tighter tracing settings.
* `fourier_regular_planes_short`: 60/300 deg, M=12, plus sparse-data rejection.

A preliminary longer extra-plane process terminated with exit 143 before
producing a comparison. It is not included in these results.

## Automated checks

```bash
python3 -m py_compile illiad/flux/fourier.py illiad/flux/calculator.py \
  illiad/cli/flux_calc.py misc_scripts/compare_flux_surface_fits.py \
  tests/test_fourier_contours.py
MPLCONFIGDIR=/tmp/illiad-mplconfig conda run -n testenv \
  python -m unittest discover -s tests -p test_fourier_contours.py
```

Tests cover analytic harmonic/derivative recovery, seam continuity, regularized
prediction at unseen angles, duplicate/sample-count invariance, sparse-input
rejection, optional calculator routing, a uniform-field analytic area integral,
and bidirectional tracing in an analytic toroidal field. These CPU checks are not GPU validation or validation of a
complete straight-field-line coordinate system.
