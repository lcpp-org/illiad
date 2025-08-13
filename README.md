# fieldlines-uiuc

`fieldlines-uiuc` is a Python package for modeling, visualizing, and analyzing electromagnetic fields, with a primary focus on magnetic field structures. It is designed for both educational and research use, providing a flexible framework for exploring field line dynamics and plasma physics phenomena.

## Features

- **Magnetic Field Modeling:** Simulate and analyze complex magnetic field configurations, including custom field definitions and sources.
- **Field Line Tracing:** Accurately trace magnetic field lines in 2D and 3D domains.
- **Poincaré Maps:** Generate Poincaré sections to study field line topology and magnetic surfaces.
- **Flux and Electric Field Calculation:** Compute magnetic flux surfaces and evaluate electric fields derived from the modeled systems.
- **Kinetic Ion Tracing:** Run high-performance kinetic ion trajectory simulations on the GPU using PyTorch, enabling large-scale particle tracing and analysis.
- **Interactive Visualization:** Visualize field lines, Poincaré maps, and particle trajectories interactively with Matplotlib.
- **Batch Processing and Data Analysis:** Automate simulations and analyze results efficiently using Pandas and TQDM for progress tracking.

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- tqdm
- torch (PyTorch)

## Getting Started

1. Install the required dependencies.
2. Explore the `examples/` directory for demonstration scripts covering field modeling, field line tracing, Poincaré map generation, flux and electric field calculations, and kinetic ion tracing.
3. Refer to the documentation for API details and advanced usage.
