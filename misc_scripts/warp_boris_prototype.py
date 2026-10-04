"""Compatibility imports for the former standalone Warp prototype.

The maintained implementation now lives in illiad.boris_warp. Use
Boris(..., method="warp") in package workflows.
"""
from illiad.boris_warp import (
    VectorGrid, DensityGrid, CollisionConfig, minor_radius, rotate_z,
    sample_vector, sample_density, initialize_velocity, push_and_record_wall,
    make_grid, make_density_grid, make_collision_config, run_wall_only,
)
