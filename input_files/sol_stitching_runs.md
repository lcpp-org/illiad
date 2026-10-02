Stitching configurations cover iota3, iota4, and iota5, each with full and flat
islands, for both potential and density (12 runs). All use ALPHA=1.0 and
SOL_BETA=0.5, and the existing SOLtrace_500c_RegularGrid connection lengths.

The SOL/Core ratio means the SOL drop divided by the core drop:
- Potential: DELTA_PHI_SOL / (DELTA_PHI_0W - DELTA_PHI_SOL) = 0.33.
- Density: (N_LCFS - N_WALL) / (N_AXIS - N_LCFS) = 0.33.

The total potential drop and axis density remain normalized to 1.0; density
at the wall remains 0.0001. Linear refers to the interior profile exponent;
the SOL retains the connection-length attenuation model.

Prerequisite: supply the linear interior profiles at these paths beneath
output/IOTA<iota>_1000sp_atol1e-9/data/:

| iota | Full island input | Flat island input |
| --- | --- | --- |
| 3 | LCFS40_full/nField_LCFS40_linear.npy | LCFS40_flat/nField_LCFS40_linear.npy |
| 4 | LCFS30_full/nField_LCFS30_linear.npy | LCFS30_flat/nField_LCFS30_linear.npy |
| 5 | LCFS11_full/nField_LCFS11_linear.npy | LCFS11_flat/nField_LCFS11_linear.npy |

These interior files were absent when the JSONs were created. The paths are
chosen names, not verified existing artifacts; adjust NFIELD_SUBDIR and
NFIELD_FILENAME if the profiles are generated elsewhere. Full/flat behavior
comes from those input profiles, not a stitcher switch. They must match the
SOL grid and selected LCFS and use the linear 1 - Psi_bar convention.

From the repository root, after preparing the interior profiles:

```bash
for iota in 3 4 5; do
  for island in full flat; do
    python runSOLPotential.py --inputs "input_files/sol_potential_inputs_iota${iota}_${island}.json"
    python runSOLDensity.py --inputs "input_files/sol_density_inputs_iota${iota}_${island}.json"
  done
done
```

Each JSON selects a distinct output subdirectory. Plots are enabled.
The stitching runs have not been executed.
