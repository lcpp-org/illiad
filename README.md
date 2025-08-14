
```
    ----------------------------------------------------
    |               ███  ███ ████████ ███  ███ ███████ |
    |               ███  ███   ████   ███  ███ ███     |
    | FIELDLINES    ███  ███   ████   ███  ███ ███     |
    |               ████████ ████████ ████████ ███████ |
    ----------------------------------------------------
```
## Summary

`fieldlines-uiuc` is a Python package for modeling, visualizing, and analyzing impurity ion transport in a toroidal device, with a primary focus on lithium deposition in **HIDRA**, the Hybrid Illinois Device for Reasearch and Applications. It is designed as a 'ones stop shop,' capable of simulating the magentic field from the Biot-Savart law, addition of perturbative error fields, fieldline tracing and Poincare map generation and analysis, and kinetic ion tracing. It is designed for both educational and research use, providing a flexible framework for exploring field line dynamics and plasma physics phenomena.

## Features

- **Magnetic Field Modeling:** Simulate and analyze complex magnetic field configurations, including custom field definitions and sources.
- **Field Line Tracing:** Accurately trace magnetic field lines in 2D and 3D domains.
- **Poincaré Maps:** Generate Poincaré sections to study field line topology and magnetic surfaces.
- **Flux and Electric Field Calculation:** Compute magnetic flux surfaces and evaluate electric fields derived from the modeled systems.
- **Kinetic Ion Tracing:** Run high-performance kinetic ion trajectory simulations on the GPU using PyTorch, enabling large-scale particle tracing and analysis.
- **Interactive Visualization:** Visualize field lines, Poincaré maps, and particle trajectories interactively with Matplotlib.
- **Batch Processing and Data Analysis:** Automate simulations and analyze results efficiently using Pandas and TQDM for progress tracking.

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

