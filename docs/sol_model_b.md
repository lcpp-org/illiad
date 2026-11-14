# Model B: solved upstream temperature and particle/power balance

`misc_scripts/sol_model_b.py` implements the baseline equations in
[ILLIAD_SOL_Model_B_Power_Balance_Tu.pdf](ILLIAD_SOL_Model_B_Power_Balance_Tu.pdf).
It solves density `n_e` and upstream representative temperature `Tu` together on
one or more saved trace planes. Local `Te` is then reconstructed as isothermal or
Spitzer–Härm. Atomic processes are optional (Model B extended, PDF Section 8.1);
with an empty neutral mixture the baseline equations are unchanged. Potential
is not recalculated.

The prototype shares Model A's trace selection, original seed indices, clipped
polar geometry, periodic faces, temperature reconstruction, batch selection, and
output/plot conventions. Geometry retains the limitations described in
[Model A](sol_model_a.md#geometry-and-prototype-limits), including the star-shaped
LCFS assumption, boundary-cell extensions, and two-point approximation on
nonorthogonal boundary faces. This is a reduced quasi-2D closure, not a full 3D
field-aligned energy solution.

## Run an example

```bash
conda run --no-capture-output -n testenv python misc_scripts/sol_model_b.py \
  output/lcfs_grid_distance_warp_some --phi-deg 0 \
  --config input_files/sol_model_b.example.json \
  --incomplete long_leg --output output/sol_model_b_phi000 --plot-individual
```

Use `--all-phi` instead of `--phi-deg 0` to solve all saved angles; or select a
subset with `--phi-deg 0 2 5`. Batch output has `phi000/`, `phi001/`, etc. below
`--output`. Every plane is solved independently with the same configuration.
`--plot` saves only the six-panel overview; `--plot-individual` saves it and all
six individual PNGs. Marker area defaults to 35 points squared; use
`--marker-size` to change it. `--density-scale linear` changes density colors.

The [example input](../input_files/sol_model_b.example.json) is an **illustrative
numerical case**, not calibrated operating conditions: `T_lcfs_eV=50`,
`n_lcfs_m3=1e18`, `D_perp_m2_s=10`, `chi_perp_m2_s=10`, `gamma_e=5.5`, and
`source_power_total_W=2500` (6.9444 W per one-degree wedge). Edit these values for the
physical case. Without a JSON input, the class defaults use `D=chi=1`, a
one-radian wedge, no neutrals, and **zero additional volumetric power**; boundary heating is
still present. The inherited `lambda_T_u_m` sets only the initial temperature
guess. It does not prescribe the converged temperature.

## Model B extended: prescribed homogeneous neutrals

The [extended example input](../input_files/sol_model_b_extended.example.json)
uses the same plasma/power settings as the current baseline example, with
`neutral_densities_m3 = {"He": 1e16, "Li": 1e14}` and `plt_bounds = "clip"`.
These are illustrative densities, not measurements. Run one plane with plots:

```bash
conda run --no-capture-output -n testenv python misc_scripts/sol_model_b.py \
  output/lcfs_grid_distance_warp_some --phi-deg 0 \
  --config input_files/sol_model_b_extended.example.json \
  --incomplete long_leg --output output/sol_model_b_extended_phi000 --plot-individual
```

Use `--all-phi` and a different output directory for a batch. To override the
**entire** mixture, repeat `--neutral He=1e16 --neutral Li=1e14`; to disable all
atomic processes use `--no-neutrals` or an empty JSON dictionary `{}`. Species
symbols are case insensitive; duplicates, negative/nonfinite densities, and
positive-density species without the required tables are rejected. He and Li
are currently supported by the available excitation tables. Each neutral
density is homogeneous throughout every plane and fixed during the solve.

For species `s`, only neutral → singly ionized transitions are enabled:

```
Sn_ion = sum_s n_e * n0_s * k_ion_s(Te)                    [m^-3 s^-1]
Q_ion  = sum_s n_e * n0_s * k_ion_s(Te) * E_ion_s * e     [W/m^3]
Q_exc  = sum_s n_e * n0_s * PLT_s(Te, n_e)               [W/m^3]
Q_atomic = Q_ion + Q_exc
```

Each event supplies one electron to the particle equation. The energy equation
includes `Q_atomic` as a sink. Rates use the **reconstructed local Te** at each
retained grid sample, recomputed within every nonlinear iteration; they are
not integrated along traced field lines. Sound speed and sheath power still use
Tu. Therefore, with neutrals enabled, changing isothermal/Spitzer–Härm can change
both solved density and Tu. This is an explicit reduced closure choice.

`misc_scripts/atomic_rates.py` shares the existing Horvath prototype functions:
Voronov Maxwellian ionization fits from `input_files/voronov_table.dat` and ADAS
neutral PLT power from `input_files/excitation_coefs/plt96_he.dat` / `plt96_li.dat`.
Horvath continues exporting its existing public atomic functions. PLT is already
a power coefficient, converted from W cm^3 to W m^3; it must **not** be multiplied
by another transition energy or electron charge. No separate excitation-radiation
term is added on top of PLT. The electrons form one Maxwellian population and
neutral targets are stationary; Horvath's prescribed fast-electron population is
not part of Model B.

`plt_bounds="clip"` explicitly holds coefficients at the nearest table edge
when local temperature or density lies outside the table. It does not floor the
solved fields. `plt_bounds="raise"` (also `--plt-bounds raise`) rejects an
out-of-range evaluation, including an intermediate iterate. Voronov fits retain
the existing extrapolation and `E_ion/Te > 80` zero-rate convention. Per-species
final-state coverage counts are saved and printed, so boundary holding and
out-of-fit temperatures are visible. These checks describe coverage, not accuracy
outside the tabulated ranges.

**Recycling is zero.** Neutral depletion, recombination, charge-state evolution,
ion-species transport, and electron–ion exchange remain omitted. Electron density
is the single solved particle field and `ion_mass_u` remains the prescribed
single effective ion mass used by the baseline closure. In a He/Li mixture this
is a helium-background/impurity approximation unless the user chooses another
effective mass; a large impurity fraction requires a more complete ion model.

## Equations and power convention

With temperatures in eV and `e = 1.602176634e-19 J/eV`:

```
cs(Tu) = sqrt(e*(Tu + gamma_i*Ti) / (ion_mass_u*atomic_mass))
nu(Tu) = 2*cs(Tu)/Lc
-div(D_perp*grad(n)) + nu*n = source_m3_s + Sn_ion
-div(n*e*chi_perp*grad(Tu)) + gamma_e*n*e*Tu*nu + Q_atomic = source_W_m3
```

`Sn_ion` and `Q_atomic` are zero for the baseline.

`chi_perp_m2_s` is defined by `q = -n*e*chi*grad(Tu)`; no additional `3/2`
heat-capacity factor is inserted. The sheath coefficient `gamma_e=5.5` is
provisional and editable. Both sound speed and sheath loss use **Tu**, regardless
of the local-temperature reconstruction switch.

Shared heat-face conductances are `e*chi*((n_i+n_j)/2)*A/distance`. An LCFS
heat face uses `(n_i+n_lcfs)/2` and the prescribed `T_lcfs_eV`. Shared fluxes enter
adjacent equations with opposite signs. Both particle and heat flux through the
outer in-plane vessel boundary are zero; parallel losses already represent wall
losses. LCFS density and temperature are Dirichlet conditions.

The additional heating kernel defaults to

```
w = exp(-max(d_min, 0)/source_lambda_m)
source_W_m3 = source_power_wedge_W * w / sum(cell_volume_m3*w)
```

The denominator includes the **retained solution domain** after trace and
connectivity exclusions. The volume integral equals the requested wedge power to
floating-point precision. Negative, nonfinite, or entirely zero weights are
rejected; `source_power_wedge_W=0` disables additional heating. For a different
kernel, call `solve_plane(..., weight_func=my_function)` or `solve_coupled(...)`;
`my_function(d_min_m, cfg)` returns nonnegative weights. Multiplying all weights
by a common factor does not change the source. The exponential is a heuristic
source shape, not a microscopic heat transport operator.

Alternatively, set `source_power_wedge_W` to `null` and specify
`source_power_total_W`. The solver assumes toroidally uniform total power and
uses `P_wedge = P_total*wedge_angle_rad/(2*pi)`. Exactly one power definition must
be active. A batch of six one-degree planes samples six degrees; it does not
redistribute the whole-torus power across those six planes.

**Source power is additional volumetric heating, not total SOL input power.**
The fixed LCFS temperature supplies a separately computed heat flux. The budget is

```
lcfs_heat_influx_W + source_power_wedge_W = sheath_power_W + atomic_power_W
lcfs_influx_s_inv + prescribed_particle_source_s_inv + ionization_source_s_inv
    = parallel_loss_s_inv
```

LCFS influx is signed. Strong volumetric heating can send heat back toward the
core, giving a negative value; it is not clipped. Avoid interpreting both this
source and the LCFS heat influx as the same prescribed physical input.

## Nonlinear solution and failure behavior

The solver starts from Model A's exponential as an initial guess, then alternates:

1. Solve density with the current temperature-dependent particle-loss coefficient
   and ionization source evaluated at the previous density/local temperature.
   With positive neutral densities, relax the density update as well.
2. Build heat conductances from the new density and solve temperature with a
   frozen sound speed in the sheath coefficient and a positive atomic sink
   coefficient `Q_atomic(Tu_old, n_new)/Tu_old` multiplying the trial Tu.
3. Relax temperature with `relaxation` (default 0.5).
4. Recompute both **original nonlinear** residuals, including ionization and
   excitation at the updated density/local temperature. At convergence the
   linearized atomic sink equals the original loss; no extra loss is introduced.

Convergence requires relative updates in both fields below `state_tolerance`
(default `1e-7`), cellwise normalized particle/energy residuals below
`residual_tolerance` (default `1e-8`), and integrated budgets within that tolerance.
Cellwise normalization uses the sum of absolute face, boundary, source and sink
terms. Integrated normalization uses the incoming/outgoing powers or particle
rates, including signed LCFS exchange. Every iteration is saved.

Nonpositive or nonfinite densities/temperature trials are rejected; no artificial
floor is applied. Failure to converge within `max_iterations` (default 200) raises
an error and does not publish a new solution for that plane. A previous output
at the same path is not removed, so use distinct output directories for distinct
parameter trials. Batch execution stops at the first failure; completed planes
remain available.

A sparse region with finite imposed heating can produce very large temperature
trials and fail the positivity checks in later iterations. In particular, a trial
using `D=chi=1` and a 1 W source in the original one-radian `phi000` mesh failed;
this is not evidence of a converged physical solution or proof that no steady
state exists. Adjusting relaxation, using continuation from a weaker source,
and checking the closure's applicability may be necessary for harder parameter
sets. No automatic continuation or Newton fallback is implemented in this
prototype. Very cold converged temperatures are reported without flooring and
should also be assessed against the physical assumptions.

## Incomplete traces and disconnected components

The default `--incomplete error` follows the PDF: every retained line must reach
both walls. `--incomplete no_flux` excludes incomplete/invalid records and removes
faces touching them. `--incomplete long_leg` is the explicitly requested extension
from Model A; **it departs from the PDF's strict tracing requirement**. It retains
one-hit cells, sets the missing length to
`max(missing_leg_length_m, actually_traced_length)`, and still excludes neither-hit
or invalid records. The default missing-leg length is 100 m. Original lengths
and hit flags are preserved and each approximation is flagged. Both parallel
particle and power sinks then use that effective length. Sampled closest approach
and reconstructed temperatures are also approximate on incomplete lines.

A connected component with no LCFS particle boundary and zero prescribed
volumetric particle source admits `n=0`. Its temperature equation is degenerate, so Model B
explicitly excludes it **before normalizing source power**, warns, and saves its
seed IDs separately. This is the principal additional domain check learned from
Model A's isolated zero-density cells. It changes the retained domain, not the
physics by imposing a density floor. Components with a positive prescribed
particle source remain eligible. Electron-impact ionization requires existing
electrons and does not count as an independent particle supply for this exclusion
rule. Self-sustained isolated discharge branches are not solved. All budget claims
apply to the retained domain.

## Local temperature and plots

`parallel_temperature_model` is `isothermal` by default. `spitzer_harm` applies
the same signed branch reconstruction as Model A, using the converged Tu.
`target_T_fraction` sets the target temperature relative to Tu. This switch must
leave both solved `n_e` and `Tu` unchanged **when atomic processes are disabled**,
and is tested that way. With atomic processes enabled, local Te feeds back into
the balances through the rates.

The reconstruction does **not** replace the sheath-limited energy sink with a
conductive sink. Its implied parallel conductive flux need not agree with that
sink. It also does not enforce identical Tu across different sampled cells on
the same 3D field line. The zero-target choice remains a conducting-wall limit,
not a physical sheath-entrance boundary condition.

The shared plotter recognizes Model B metadata and labels the field **Solved
upstream temperature**. Replot saved data without solving:

```bash
conda run --no-capture-output -n testenv python misc_scripts/plot_sol_model_a.py \
  output/sol_model_b_phi000 --individual
```

The existing plotter filename is retained for compatibility. Model B produces
`sol_model_b_overview.png` and `sol_model_b_<panel>.png`, with the same six panels
and shared temperature scale as Model A. Power densities and budgets are in the
numerical outputs below.

## Output and validation

- `solution.npz`: Model A-compatible seed/grid indices and trace geometry, density,
  solved `Tu_eV`, reconstructed `Te_eV`, and `(rho, theta)` grids with NaN outside
  solved cells. Additional arrays are `source_W_m3`, `sheath_loss_W_m3`,
  `energy_residual_W`, `iteration_history`, and `disconnected_seed_id`. Atomic fields
  are `atomic_temperature_eV`, `ionization_source_m3_s`, `ionization_loss_W_m3`,
  `excitation_loss_W_m3`, and `atomic_loss_W_m3`, plus per-species fields such as
  `ionization_source_He_m3_s` and `excitation_loss_Li_W_m3`.
- `profiles.csv`: retained-cell fields and original/effective connection lengths.
- `power_balance.csv`: seed ID, heating, sheath and atomic losses in W/m^3, and
  cell-integrated nonlinear energy residual in W.
- `atomic_processes.csv`: seed ID, local atomic temperature, total and per-species
  ionization particle sources and ionization/excitation power densities.
- `iteration_history.csv`: relative density/temperature updates, normalized
  nonlinear residuals, and integrated particle/energy imbalances at every iterate.
- `run_settings.json`: model label `B`, input configuration, selected plane,
  source/geometry/trace conventions, convergence metrics, retained/excluded counts,
  and signed particle and power budgets. `variant` is `baseline` or `extended`.
  `volume_source_s_inv` includes both prescribed and ionization particle sources;
  separate totals and per-species powers/table-coverage counts are also saved.

Run the numerical checks:

```bash
python3 -m py_compile misc_scripts/sol_model_a.py misc_scripts/sol_model_b.py \
  misc_scripts/plot_sol_model_a.py tests/test_sol_model_b.py
conda run -n testenv python -m unittest discover -s tests -p 'test_sol_model*.py' -v
```

They cover source normalization/custom kernels, whole-torus power scaling,
particle and energy conservation, negative LCFS heat influx against an independent
one-cell root solve, coupled density/temperature grid convergence against an
independent continuum BVP, initial-guess independence, heating sensitivity,
reconstruction-switch invariance, explicit failure paths, and trace/connectivity
exclusions. Model A's existing geometry, slab, trace, and batch checks also run.
These validate the numerical implementation; they are not experimental validation
of the source or transport closures. Atomic tests also cover mixture additivity,
threshold/PLT units, table bounds, an independent one-cell nonlinear root, exact
zero-neutral regression, local-temperature feedback, and saved atomic fields.
Recycling, recombination, electron-ion energy exchange, and a consistent
conduction/sheath closure remain deferred.
