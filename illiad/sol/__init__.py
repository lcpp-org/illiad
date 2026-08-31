"""Scrape-off-layer tracing and geometry helpers."""

from . import stitching
from .crossings import (
    CrossingChunk,
    NpyPlaneCrossingSource,
    PlaneShardWriter,
    PlaneCrossingSource,
    ShardedPlaneCrossingSource,
    open_plane_crossing_source,
)
from .density import SOLDensity
from .potential import SOLPotential
from .regularizer import SOLRegularizer
from .workflow import SOLTraceRegularizer
from .tracer import (
    SOLTracer,
    build_torch_magnetic_field,
    load_lcfs_boundary,
    load_poincare_settings,
    minimum_boundary_distance,
    resolve_device,
)

__all__ = [
    "SOLDensity",
    "SOLPotential",
    "SOLRegularizer",
    "SOLTracer",
    "SOLTraceRegularizer",
    "CrossingChunk",
    "NpyPlaneCrossingSource",
    "PlaneShardWriter",
    "PlaneCrossingSource",
    "ShardedPlaneCrossingSource",
    "stitching",
    "build_torch_magnetic_field",
    "load_lcfs_boundary",
    "load_poincare_settings",
    "minimum_boundary_distance",
    "open_plane_crossing_source",
    "resolve_device",
]
