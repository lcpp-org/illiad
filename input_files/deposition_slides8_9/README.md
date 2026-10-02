# Deposition cases for slides 8–9

Five new Boris inputs extend the collisional evaporation cases in
`input_files/collisionality_slides3_5/`. Reuse those existing IOTA3 inputs/results
as the fitted-field baseline; no duplicate IOTA3 production run is needed.

| Slide | New case | Error field | LCFS index | Helical current |
| --- | --- | --- | --- | --- |
| 8 | IOTA3 ideal, flattened islands | Off | 29 | 0.900 |
| 9 | IOTA4, full islands | On | 30 | 0.790 |
| 9 | IOTA4, flattened islands | On | 30 | 0.790 |
| 9 | IOTA5, full islands | On | 11 | 0.710 |
| 9 | IOTA5, flattened islands | On | 11 | 0.710 |

All use toroidal current 0.486. Each input points to density and electric fields
generated for its own magnetic configuration and uses that configuration's
Poincare directory for launch positions.

## Shared background and sampling

All new inputs preserve the previous evaporation settings: peak plasma-density
multiplier 2e18 m^-3, neutral He density 3e17 m^-3, explicit peak plasma potential
25 V, lithium launch/background-He ion temperature 5 eV, neutral temperature
300 K, lithium mass 6.941 amu, and charge +1. `ELECTRON_TEMP_EV` is omitted.
Ion-neutral `langevin` and ion-ion `fokker_planck` collisions are enabled.

Every run uses `NPHI=180`, `NTHETA=90`, and `NPARTICLES_PER_EMITTER=500`:
**8,100,000 launched particles**. Tracking uses `TRACK_NPHI=45`,
`TRACK_NTHETA=45`, and `TRACK_NPARTICLES_PER_EMITTER=1`:
**2,025 tracked particles**, at `STRIDE=15`. The timestep remains 1e-8 s,
maximum time 1 ms, and radial launch offset zero.

## Comparisons

For slide 8 compare `boris_iota3_ideal_flat_evap_collisions_on.json` against
`../collisionality_slides3_5/boris_iota3_flat_evap_collisions_on.json`.
The complete available ideal fields are the LCFS29 no-islands/flattened model.
The old ideal full-island potential directory does not contain a stitched
potential or electric field, so no full-island ideal Boris input is supplied.
The ideal and fitted cases use their own background fields and LCFS surfaces
(indices 29 and 40 respectively); this is a self-consistent configuration
comparison, not a magnetic-field toggle with identical background geometry.

For slide 9 compare IOTA3, IOTA4, and IOTA5 within the same island treatment.
Use the existing full/flat IOTA3 evaporation collisional inputs as the two
baselines. The new IOTA4/IOTA5 cases retain the fitted error field, matching
those baselines. Full/flat comparisons remain a background-model sensitivity
check. Additional collisionless or typical-operation cases are not needed for
the primary evaporation robustness comparisons in the outline.

Use identical deposition bins, coordinate convention, and color limits across
the comparison plots. Normalize counts by 8,100,000 launched ions in every case.
Use the saved `Wallpt_OUTPUT.npy` for final shared-scale plots; the automatic
per-run plots do not enforce common limits. Each run samples independent initial
velocities/noise, as the current CLI provides no random-seed input.

## Commands

Run from `/home/sgula/code/illiad` in the simulation environment:

```bash
conda activate testenv
illiad-boris --inputs input_files/deposition_slides8_9/boris_iota3_ideal_flat_evap_collisions_on.json
```

Run the four additional slide 9 cases sequentially:

```bash
for config in iota4 iota5; do
  for profile in full flat; do
    illiad-boris --inputs "input_files/deposition_slides8_9/boris_${config}_${profile}_evap_collisions_on.json" || break 2
  done
done
```

Outputs use distinct tags under each configuration's `output/<analysis>/data/`,
`plots/`, and `logs/` trees. Example data directories:

- Ideal IOTA3: `output/IOTA3_ideal_1000sp_atol1e-9/data/0mm_LCFS29_5eV_25V_Z1_Slide8_IOTA3_ideal_flat_evap_collisions_on/`
- Full IOTA4: `output/IOTA4_1000sp_atol1e-9/data/0mm_LCFS30_5eV_25V_Z1_Slide9_IOTA4_full_evap_collisions_on/`
- Full IOTA5: `output/IOTA5_1000sp_atol1e-9/data/0mm_LCFS11_5eV_25V_Z1_Slide9_IOTA5_full_evap_collisions_on/`

For the flat IOTA4/IOTA5 directories, replace `full` with `flat` in the tag.
Input preparation does not run these simulations.
