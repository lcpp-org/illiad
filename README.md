```
    -----------------------------------------------------------
    |  ██╗██╗     ██╗     ██╗ █████╗ ██████╗                  |
    |  ██║██║     ██║     ██║██╔══██╗██╔══██╗                 |
    |  ██║██║     ██║     ██║███████║██║  ██║                 |
    |  ██║██║     ██║     ██║██╔══██║██║  ██║                 |
    |  ██║███████╗███████╗██║██║  ██║██████╔╝                 |
    |  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═════╝                  |
    |                                                         |
    |  Illinois Lagrangian Impurity Advection and Deposition  |
    -----------------------------------------------------------
```

# ILLIAD

**ILLIAD** (*Illinois Lagrangian Impurity Advection and Deposition*) is a Python-based modeling framework for reconstructing three-dimensional magnetic and electrostatic fields in **HIDRA** and simulating trace impurity-ion transport through the scrape-off layer (SOL). The code was developed to study lithium impurity motion during controlled lithium evaporation experiments, where post-operational wall images show narrow, field-aligned deposition streaks on the HIDRA vacuum vessel.

ILLIAD combines Biot–Savart magnetic-field reconstruction, field-line and flux-surface analysis, flux-surface-based background plasma models, and GPU-accelerated full-orbit particle tracing. The workflow is designed to connect experimentally constrained HIDRA fields to predictive deposition diagnostics, including wall-impact locations, normalized deposition fluence, incidence angles, impact energies, residence times, and collisionality estimates.

## Features

- **HIDRA magnetic-field reconstruction:** Compute the 3D vacuum magnetic field from toroidal, helical, and vertical coil geometries using the Biot–Savart law, with fields stored on structured toroidal-coordinate grids in Cartesian components for particle pushing and interpolation.

- **Non-ideal field corrections and validation:** Include scalar coil-field attenuation factors and a uniform non-periodic perturbative error field to reproduce experimentally measured HIDRA field strengths, toroidal asymmetries, and island topology.

- **Field-line tracing and Poincaré analysis:** Trace magnetic field lines over many toroidal transits, generate Poincaré sections, identify nested flux surfaces, island surfaces, stochastic/open-field regions, and estimate the last closed flux surface (LCFS).

- **Flux-surface and background plasma modeling:** Construct a normalized flux-surface parameter from the reconstructed magnetic topology and use it to prescribe surrogate helium plasma density and plasma-potential profiles.

- **Electrostatic-field construction:** Derive the SOL electric field from the flux-surface-dependent plasma-potential model using `E ≈ -Vp ∇ψ̂`, while retaining the full non-axisymmetric 3D structure introduced by the fitted magnetic field.

- **GPU-accelerated kinetic impurity tracing:** Advance large ensembles of impurity ions with the Boris–Buneman full-orbit particle pusher using interpolated 3D electric and magnetic fields. Simulations support Maxwellian initial energies, hemisphere-directed launch distributions from the LCFS, and wall-intersection termination.

- **Deposition and wall-impact diagnostics:** Record particle impact location, incidence angle, toroidal impact direction, and deposition energy on the HIDRA vessel wall. Generate 2D `(θ, φ)` deposition maps, normalized deposition-fluence plots, trajectory visualizations, and statistical impact distributions.

- **Residence-time and collisionality analysis:** Compute survival functions and mean SOL residence times from particle-loss histories, then compare against estimated ion-neutral and ion-ion collision times to assess whether HIDRA operating regimes are collisionless, weakly collisional, or collisional.

- **Experiment-facing analysis workflow:** Compare simulated deposition structures with observed lithium streak patterns and evaluate how magnetic topology, electrostatic acceleration, ion temperature, and operating regime shape impurity deposition.

- **Batch processing and data analysis:** Automate large parameter scans and post-processing workflows using `pandas`, `tqdm`, and GPU-enabled PyTorch simulations.

## Dependencies

*Note that these are the current versions used on the Illinois Campus Cluster (ICC).  
Use with other versions is not guaranteed.*
- python==3.11.11
- numpy==2.3.1
- scipy==1.16.0
- matplotlib==3.10.3
- pandas==2.3.1
- tqdm==4.67.1
- torch==2.7.1 (PyTorch)

## Getting Started

1. Install the required dependencies.
2. Explore the `run....py` scripts in the base directory for demo scripts covering field modeling, Poincaré map generation, flux calculations, and kinetic ion tracing.
3. Refer to the documentation for API details and advanced usage.

