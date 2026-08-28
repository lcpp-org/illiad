"""Regularize raw SOL connection-length crossings onto a field mesh.

The saved field follows the ILLIAD scalar-field convention
``(phi, theta, rho)``. Samples are accumulated into their nearest regular
grid nodes, then missing cells outside the LCFS are filled in the seam-free
poloidal ``(x, z)`` plane.
"""

from contextlib import nullcontext
import gc
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from illiad.utilities.coordtrans import XYZ_to_RTP_many
from .crossings import open_plane_crossing_source
from .tracer import load_lcfs_boundary, load_poincare_settings


RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"
INDEX_SPOOL_DTYPE = np.dtype(
    [("cell_index", "<i4"), ("fieldline_id", "<i4")],
    align=False,
)


def validate_regularizer_settings(params):
    """Validate the public regularization settings."""
    for key in (
        "N_RHO",
        "N_THETA",
        "IDW_NEIGHBORS",
        "RAW_CHUNK_SIZE",
        "N_LEVELS",
        "DPI",
        "TREE_WORKERS",
    ):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer.")
    if params["N_RHO"] < 2:
        raise ValueError("N_RHO must be at least 2.")
    if params["N_THETA"] < 3:
        raise ValueError("N_THETA must be at least 3.")
    if not 0.0 <= params["RHO_MIN"] < params["RHO_MAX"]:
        raise ValueError("Require 0 <= RHO_MIN < RHO_MAX.")
    if params["RAW_CHUNK_SIZE"] <= 0:
        raise ValueError("RAW_CHUNK_SIZE must be positive.")
    if params["INTERPOLATION_SPACE"] not in {"linear", "log"}:
        raise ValueError(
            'INTERPOLATION_SPACE must be "linear" or "log".'
        )
    if params["FILL_METHOD"] not in {"idw", "nearest", "none"}:
        raise ValueError(
            'FILL_METHOD must be "idw", "nearest", or "none".'
        )
    if params["IDW_NEIGHBORS"] < 1:
        raise ValueError("IDW_NEIGHBORS must be positive.")
    if params["TREE_WORKERS"] == 0 or params["TREE_WORKERS"] < -1:
        raise ValueError("TREE_WORKERS must be -1 or a positive integer.")
    if (
        not np.isfinite(params["IDW_POWER"])
        or params["IDW_POWER"] <= 0.0
    ):
        raise ValueError("IDW_POWER must be positive and finite.")
    if params["N_LEVELS"] < 2:
        raise ValueError("N_LEVELS must be at least 2.")
    if params["DPI"] <= 0:
        raise ValueError("DPI must be positive.")
    if (
        not np.isfinite(params["VESSEL_RADIUS_M"])
        or params["VESSEL_RADIUS_M"] <= 0.0
    ):
        raise ValueError("VESSEL_RADIUS_M must be positive and finite.")
    if params["COLOR_SCALE"] not in {"linear", "log"}:
        raise ValueError('COLOR_SCALE must be "linear" or "log".')
    if params["CONTOUR_EXTEND"] not in {
        "auto",
        "neither",
        "both",
        "min",
        "max",
    }:
        raise ValueError("Invalid CONTOUR_EXTEND setting.")
    for key in ("GENERATE_PLOTS", "SHOW_PROGRESS"):
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} must be a boolean.")


def make_regular_grid(n_rho, n_theta, rho_min, rho_max):
    """Return flux-compatible radial and poloidal node arrays."""
    rho = np.linspace(rho_min, rho_max, n_rho, dtype=np.float64)
    theta = np.linspace(
        2.0 * np.pi / n_theta,
        2.0 * np.pi,
        n_theta,
        dtype=np.float64,
    )
    grid_theta, grid_rho = np.meshgrid(theta, rho, indexing="ij")
    grid_x = grid_rho * np.cos(grid_theta)
    grid_z = grid_rho * np.sin(grid_theta)
    return rho, theta, grid_rho, grid_theta, grid_x, grid_z


def positive_data_range(source, chunk_size):
    """Return the finite positive range of the source value population."""
    data_min = np.inf
    data_max = -np.inf
    for chunk in source.iter_value_chunks(chunk_size):
        finite = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if finite.size:
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths are available.")
    return data_min, data_max


def make_color_scale(source, params):
    """Resolve plot levels and normalization from source connection lengths."""
    data_min, data_max = positive_data_range(
        source,
        params["RAW_CHUNK_SIZE"],
    )
    value_min = data_min if params["VMIN"] is None else params["VMIN"]
    value_max = data_max if params["VMAX"] is None else params["VMAX"]
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
    if not value_min < value_max:
        raise ValueError("Resolved color limits require VMIN < VMAX.")

    if params["COLOR_SCALE"] == "log":
        if value_min <= 0.0:
            raise ValueError("Logarithmic plots require VMIN > 0.")
        levels = np.geomspace(value_min, value_max, params["N_LEVELS"])
        norm = LogNorm(vmin=value_min, vmax=value_max)
    else:
        levels = np.linspace(value_min, value_max, params["N_LEVELS"])
        norm = Normalize(vmin=value_min, vmax=value_max)

    if params["CONTOUR_EXTEND"] == "auto":
        below = data_min < value_min
        above = data_max > value_max
        if below and above:
            extend = "both"
        elif below:
            extend = "min"
        elif above:
            extend = "max"
        else:
            extend = "neither"
    else:
        extend = params["CONTOUR_EXTEND"]
    return levels, norm, extend, value_min, value_max


def grid_indices(points_rtp, n_rho, n_theta, rho_min, rho_max):
    """Map RTP points to their nearest periodic regular-grid nodes."""
    radial_spacing = (rho_max - rho_min) / (n_rho - 1)
    rho_index = np.rint(
        (points_rtp[:, 0] - rho_min) / radial_spacing
    ).astype(np.int64)
    theta_spacing = 2.0 * np.pi / n_theta
    theta_index = (
        np.rint(
            np.remainder(points_rtp[:, 1], 2.0 * np.pi) / theta_spacing
        ).astype(np.int64)
        - 1
    ) % n_theta
    return rho_index, theta_index


class RegularGridAccumulator:
    """Fixed-size sufficient statistics for direct trace regularization."""

    def __init__(self, plane_count, params):
        self.plane_count = int(plane_count)
        self.n_theta = params["N_THETA"]
        self.n_rho = params["N_RHO"]
        self.interpolation_space = params["INTERPOLATION_SPACE"]
        shape = (self.plane_count, self.n_theta, self.n_rho)
        self.value_sum = np.zeros(shape, dtype=np.float64)
        self.sample_count = np.zeros(shape, dtype=np.int64)
        self.used_samples = np.zeros(self.plane_count, dtype=np.int64)

    @property
    def cell_count(self):
        return int(self.value_sum.size)

    def add_indexed(self, cell_index, connection_length_m):
        """Add resolved positive samples addressed by global cell index."""
        cell_index = np.asarray(cell_index)
        values = np.asarray(connection_length_m)
        valid = (
            (cell_index >= 0)
            & (cell_index < self.cell_count)
            & np.isfinite(values)
            & (values > 0.0)
        )
        if not np.any(valid):
            return 0
        cell_index = cell_index[valid].astype(np.int64, copy=False)
        values = values[valid]
        if self.interpolation_space == "log":
            values = np.log(values)
        flat_sum = self.value_sum.ravel()
        flat_count = self.sample_count.ravel()
        np.add.at(flat_sum, cell_index, values)
        np.add.at(flat_count, cell_index, 1)
        plane_index = cell_index // (self.n_theta * self.n_rho)
        self.used_samples += np.bincount(
            plane_index,
            minlength=self.plane_count,
        )
        return int(cell_index.size)

    def plane_mean(self, plane_index):
        """Return one plane's direct cell means and occupancy counts."""
        counts = self.sample_count[plane_index]
        occupied = counts > 0
        gridded = np.full(counts.shape, np.nan, dtype=np.float64)
        gridded[occupied] = (
            self.value_sum[plane_index][occupied] / counts[occupied]
        )
        if self.interpolation_space == "log":
            gridded[occupied] = np.exp(gridded[occupied])
        return gridded, counts


class GridIndexSpool:
    """Temporarily spool one paired trace batch as cell/field-line IDs."""

    def __init__(self, scratch_path, plane_count, major_radius, params):
        self.path = Path(scratch_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._handle = self.path.open("wb", buffering=0)
        self.n_planes = int(plane_count)
        self.major_radius = float(major_radius)
        self.params = params
        self.counts = np.zeros(self.n_planes, dtype=np.int64)
        self.record_count = 0

    def _append_points(self, plane_index, points_rtp, fieldline_id):
        points_rtp = np.asarray(points_rtp)
        fieldline_id = np.asarray(fieldline_id)
        finite = (
            np.all(np.isfinite(points_rtp[:, :2]), axis=1)
            & (points_rtp[:, 0] >= self.params["RHO_MIN"])
            & (points_rtp[:, 0] <= self.params["RHO_MAX"])
        )
        if not np.any(finite):
            return
        points_rtp = points_rtp[finite]
        fieldline_id = fieldline_id[finite]
        rho_index, theta_index = grid_indices(
            points_rtp,
            self.params["N_RHO"],
            self.params["N_THETA"],
            self.params["RHO_MIN"],
            self.params["RHO_MAX"],
        )
        valid = (rho_index >= 0) & (rho_index < self.params["N_RHO"])
        if not np.any(valid):
            return
        local_index = (
            theta_index[valid] * self.params["N_RHO"] + rho_index[valid]
        )
        global_index = (
            int(plane_index) * self.params["N_THETA"] * self.params["N_RHO"]
            + local_index
        )
        if global_index.max(initial=0) > np.iinfo(np.int32).max:
            raise ValueError("Regular field exceeds the int32 cell-index range.")
        records = np.empty(global_index.size, dtype=INDEX_SPOOL_DTYPE)
        records["cell_index"] = global_index
        records["fieldline_id"] = fieldline_id[valid]
        records.tofile(self._handle)
        count = int(records.size)
        self.counts[int(plane_index)] += count
        self.record_count += count

    def append_xyz(
        self,
        plane_index,
        xyz,
        fieldline_id,
        source_direction,
    ):
        del source_direction
        points_rtp = XYZ_to_RTP_many(np.asarray(xyz), self.major_radius)
        self._append_points(plane_index, points_rtp, fieldline_id)

    def append_rtp(
        self,
        plane_index,
        points_rtp,
        fieldline_id,
        source_direction,
    ):
        del source_direction
        self._append_points(plane_index, points_rtp, fieldline_id)

    def consume(self, accumulator, fieldline_connection_length_m, chunk_size):
        """Accumulate the completed batch and remove its temporary spool."""
        self._handle.flush()
        self._handle.close()
        self._handle = None
        try:
            if self.record_count:
                records = np.memmap(
                    self.path,
                    mode="r",
                    dtype=INDEX_SPOOL_DTYPE,
                    shape=(self.record_count,),
                )
                try:
                    for start in range(0, self.record_count, chunk_size):
                        stop = min(start + chunk_size, self.record_count)
                        chunk = records[start:stop]
                        fieldline_id = np.asarray(chunk["fieldline_id"])
                        accumulator.add_indexed(
                            np.asarray(chunk["cell_index"]),
                            np.asarray(fieldline_connection_length_m)[fieldline_id],
                        )
                finally:
                    del records
        finally:
            self.path.unlink(missing_ok=True)

    def abort(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.path.unlink(missing_ok=True)


def accumulate_plane(source, plane_index, params):
    """Aggregate every finite sample from one plane into regular cells."""
    n_rho = params["N_RHO"]
    n_theta = params["N_THETA"]
    cell_count = n_rho * n_theta
    value_sum = np.zeros(cell_count, dtype=np.float64)
    sample_count = np.zeros(cell_count, dtype=np.int64)
    used_samples = 0

    for chunk in source.iter_plane_chunks(
        plane_index,
        params["RAW_CHUNK_SIZE"],
    ):
        point_chunk = chunk.points_rtp
        value_chunk = chunk.connection_length_m
        finite = (
            np.all(np.isfinite(point_chunk[:, :2]), axis=1)
            & np.isfinite(value_chunk)
            & (value_chunk > 0.0)
            & (point_chunk[:, 0] >= params["RHO_MIN"])
            & (point_chunk[:, 0] <= params["RHO_MAX"])
        )
        if not np.any(finite):
            continue

        point_chunk = point_chunk[finite]
        value_chunk = value_chunk[finite]
        if params["INTERPOLATION_SPACE"] == "log":
            value_chunk = np.log(value_chunk)
        rho_index, theta_index = grid_indices(
            point_chunk,
            n_rho,
            n_theta,
            params["RHO_MIN"],
            params["RHO_MAX"],
        )

        valid_index = (rho_index >= 0) & (rho_index < n_rho)
        flat_index = (
            theta_index[valid_index] * n_rho + rho_index[valid_index]
        )
        value_sum += np.bincount(
            flat_index,
            weights=value_chunk[valid_index],
            minlength=cell_count,
        )
        sample_count += np.bincount(flat_index, minlength=cell_count)
        used_samples += int(np.count_nonzero(valid_index))

    occupied = sample_count > 0
    gridded = np.full(cell_count, np.nan, dtype=np.float64)
    gridded[occupied] = value_sum[occupied] / sample_count[occupied]
    if params["INTERPOLATION_SPACE"] == "log":
        gridded[occupied] = np.exp(gridded[occupied])
    return (
        gridded.reshape(n_theta, n_rho),
        sample_count.reshape(n_theta, n_rho),
        used_samples,
    )


def exterior_mask(boundary, grid_x, grid_z):
    """Return the regular cells lying outside a closed LCFS boundary."""
    closed_boundary = np.vstack((boundary, boundary[0]))
    inside = MplPath(closed_boundary).contains_points(
        np.column_stack((grid_x.ravel(), grid_z.ravel()))
    )
    return ~inside.reshape(grid_x.shape)


def fill_missing_cells(field, exterior, grid_x, grid_z, params):
    """Fill unsampled exterior cells from occupied nodes in x-z space."""
    field = field.copy()
    field[~exterior] = np.nan
    occupied = exterior & np.isfinite(field) & (field > 0.0)
    missing = exterior & ~occupied
    if not np.any(missing) or params["FILL_METHOD"] == "none":
        return field, int(np.count_nonzero(occupied)), 0
    if not np.any(occupied):
        raise ValueError("No occupied regular-grid cells remain outside the LCFS.")

    source_points = np.column_stack((grid_x[occupied], grid_z[occupied]))
    target_points = np.column_stack((grid_x[missing], grid_z[missing]))
    source_values = field[occupied]
    if params["INTERPOLATION_SPACE"] == "log":
        source_values = np.log(source_values)

    tree = cKDTree(source_points)
    neighbor_count = (
        1
        if params["FILL_METHOD"] == "nearest"
        else min(params["IDW_NEIGHBORS"], source_points.shape[0])
    )
    distances, indices = tree.query(
        target_points,
        k=neighbor_count,
        workers=params["TREE_WORKERS"],
    )
    if neighbor_count == 1:
        filled_values = source_values[indices]
    else:
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        weights = distances ** (-params["IDW_POWER"])
        filled_values = np.sum(
            weights * source_values[indices],
            axis=1,
        ) / np.sum(weights, axis=1)
    if params["INTERPOLATION_SPACE"] == "log":
        filled_values = np.exp(filled_values)
    field[missing] = filled_values
    return (
        field,
        int(np.count_nonzero(occupied)),
        int(np.count_nonzero(missing)),
    )


def regularize_field(
    analysis_dir,
    source,
    lcfs_index,
    rho,
    theta,
    grid_x,
    grid_z,
    output_path,
    sim_io,
    params,
):
    """Build and atomically save the regular connection-length field."""
    temporary_path = output_path.with_name(
        f".{output_path.stem}.building.npy"
    )
    temporary_path.unlink(missing_ok=True)
    field = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float64,
        shape=(source.plane_count, theta.size, rho.size),
    )

    start_time = perf_counter()
    progress = tqdm(
        range(source.plane_count),
        desc="Regularizing connection length",
        unit="plane",
        dynamic_ncols=True,
        disable=not params["SHOW_PROGRESS"],
    )
    log_context = (
        logging_redirect_tqdm(loggers=[sim_io.log])
        if params["SHOW_PROGRESS"]
        else nullcontext()
    )
    try:
        with log_context:
            for plane_index in progress:
                phi_deg = float(source.plane_phi_deg[plane_index])
                boundary, _ = load_lcfs_boundary(
                    analysis_dir,
                    phi_deg,
                    lcfs_index,
                )
                exterior = exterior_mask(boundary, grid_x, grid_z)
                binned, counts, used_samples = accumulate_plane(
                    source,
                    plane_index,
                    params,
                )
                regular, occupied_count, filled_count = fill_missing_cells(
                    binned,
                    exterior,
                    grid_x,
                    grid_z,
                    params,
                )
                field[plane_index] = regular
                plane_samples = source.plane_sample_count(plane_index)
                sim_io.log.info(
                    "Regularized phi=%03.0f deg: %d/%d raw samples used, "
                    "%d directly occupied exterior cells, %d filled cells, "
                    "%d samples in the busiest cell.",
                    phi_deg,
                    used_samples,
                    plane_samples,
                    occupied_count,
                    filled_count,
                    int(counts.max(initial=0)),
                )
                if (plane_index + 1) % 10 == 0:
                    field.flush()
                    gc.collect()
        field.flush()
        del field
        field = None
        os.replace(temporary_path, output_path)
    except Exception:
        if field is not None:
            del field
        temporary_path.unlink(missing_ok=True)
        raise

    sim_io.log.info(
        "REGULAR CONNECTION-LENGTH FIELD FINISHED IN %.3f seconds.",
        perf_counter() - start_time,
    )
    return np.load(output_path, mmap_mode="r")


def regularize_accumulator(
    analysis_dir,
    accumulator,
    plane_phi_deg,
    lcfs_index,
    rho,
    theta,
    grid_x,
    grid_z,
    output_path,
    sim_io,
    params,
):
    """Fill LCFS-exterior gaps and atomically save direct statistics."""
    temporary_path = output_path.with_name(f".{output_path.stem}.building.npy")
    temporary_path.unlink(missing_ok=True)
    field = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float64,
        shape=(accumulator.plane_count, theta.size, rho.size),
    )
    try:
        for plane_index, phi_deg in enumerate(plane_phi_deg):
            boundary, _ = load_lcfs_boundary(
                analysis_dir,
                float(phi_deg),
                lcfs_index,
            )
            exterior = exterior_mask(boundary, grid_x, grid_z)
            binned, counts = accumulator.plane_mean(plane_index)
            regular, occupied_count, filled_count = fill_missing_cells(
                binned,
                exterior,
                grid_x,
                grid_z,
                params,
            )
            field[plane_index] = regular
            sim_io.log.info(
                "Regularized direct phi=%03.0f deg: %d samples, %d "
                "directly occupied exterior cells, %d filled cells, %d "
                "samples in the busiest cell.",
                phi_deg,
                accumulator.used_samples[plane_index],
                occupied_count,
                filled_count,
                int(counts.max(initial=0)),
            )
        field.flush()
        del field
        field = None
        os.replace(temporary_path, output_path)
    except Exception:
        if field is not None:
            del field
        temporary_path.unlink(missing_ok=True)
        raise
    return np.load(output_path, mmap_mode="r")


def plot_plane(
    plane,
    rho,
    theta,
    phi_deg,
    boundary,
    levels,
    norm,
    extend,
    sim_io,
    params,
):
    """Plot one regular field plane in the Cartesian cross-section."""
    plot_theta = np.concatenate(([0.0], theta))
    plot_data = np.vstack((plane[-1], plane))
    plot_theta_grid, plot_rho_grid = np.meshgrid(
        plot_theta,
        rho,
        indexing="ij",
    )
    plot_x = plot_rho_grid * np.cos(plot_theta_grid)
    plot_z = plot_rho_grid * np.sin(plot_theta_grid)

    fig, ax = plt.subplots(figsize=(7, 6))
    color_artist = ax.contourf(
        plot_x,
        plot_z,
        plot_data,
        levels=levels,
        norm=norm,
        cmap=params["COLORMAP"],
        extend=extend,
    )
    closed_boundary = np.vstack((boundary, boundary[0]))
    ax.plot(
        closed_boundary[:, 0],
        closed_boundary[:, 1],
        color="black",
        linewidth=1.0,
        label="LCFS",
    )
    vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    vessel_radius = params["VESSEL_RADIUS_M"]
    ax.plot(
        vessel_radius * np.cos(vessel_angle),
        vessel_radius * np.sin(vessel_angle),
        color="0.35",
        linewidth=1.0,
        label="Vessel wall",
    )
    physical_phi = (
        phi_deg + params["PHYSICAL_PHI_OFFSET_DEG"]
    ) % 360.0
    ax.set_title(
        "Regular-grid connection length\n"
        f"$\\phi_{{phy}}={physical_phi:03.0f}^\\circ$ CW from North split, "
        f"$\\phi_c={phi_deg:03.0f}^\\circ$"
    )
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_xlim(-vessel_radius, vessel_radius)
    ax.set_ylim(-vessel_radius, vessel_radius)
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4, color="0.75")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(color_artist, ax=ax, pad=0.03)
    colorbar.set_label("Connection length [m]")

    plot_name = f"connection_length_field_{phi_deg:03.0f}.png"
    sim_io.saveFig(
        plot_name,
        dpi=params["DPI"],
        subdir=params["ANLYS_SUBDIR"],
    )
    sim_io.log.info(
        "Saved regular-grid contour: %s/%s",
        params["ANLYS_SUBDIR"],
        plot_name,
    )
    plt.close(fig)
    gc.collect()


def plot_field(
    analysis_dir,
    field,
    rho,
    theta,
    source,
    lcfs_index,
    sim_io,
    params,
):
    """Plot every plane of a completed regular field."""
    levels, norm, extend, value_min, value_max = make_color_scale(
        source,
        params,
    )
    sim_io.log.info(
        "Regular-grid plot color range: %g to %g m (%s).",
        value_min,
        value_max,
        params["COLOR_SCALE"],
    )
    progress = tqdm(
        range(source.plane_count),
        desc="Plotting regular field",
        unit="plane",
        dynamic_ncols=True,
        disable=not params["SHOW_PROGRESS"],
    )
    log_context = (
        logging_redirect_tqdm(loggers=[sim_io.log])
        if params["SHOW_PROGRESS"]
        else nullcontext()
    )
    with log_context:
        for plane_index in progress:
            phi_deg = float(source.plane_phi_deg[plane_index])
            boundary, _ = load_lcfs_boundary(
                analysis_dir,
                phi_deg,
                lcfs_index,
            )
            plot_plane(
                field[plane_index],
                rho,
                theta,
                phi_deg,
                boundary,
                levels,
                norm,
                extend,
                sim_io,
                params,
            )


class SOLRegularizer:
    """Build a regular SOL connection-length field from crossing chunks."""

    def __init__(self, io_handler, input_params, crossing_source=None):
        self.simIO = io_handler
        self.input_params = dict(input_params)
        self.source = crossing_source
        self.lcfs_index = None
        self.field = None
        self.output_path = None
        self.data_dir = None

    def _resolve_inputs(self):
        params = self.input_params
        validate_regularizer_settings(params)
        if self.source is None:
            raw_data_dir = Path(self.simIO.data_dir) / params["TRACE_SUBDIR"]
            self.source = open_plane_crossing_source(raw_data_dir)
        else:
            raw_data_dir = getattr(self.source, "data_dir", None)

        self.lcfs_index = params["LCFS_INDEX"]
        if self.lcfs_index is None:
            settings = load_poincare_settings(params["ANLYS_DIR"])
            self.lcfs_index = settings.get("LCFS_INDEX")
        if self.lcfs_index is None:
            raise ValueError(
                "No LCFS index was found; provide LCFS_INDEX explicitly."
            )
        if (
            isinstance(self.lcfs_index, bool)
            or not isinstance(self.lcfs_index, (int, np.integer))
            or self.lcfs_index < 0
        ):
            raise ValueError("LCFS_INDEX must be a nonnegative integer.")
        return raw_data_dir

    def run(self):
        """Regularize, save, and optionally plot the connection length."""
        params = self.input_params
        raw_data_dir = self._resolve_inputs()
        rho, theta, _, _, grid_x, grid_z = make_regular_grid(
            params["N_RHO"],
            params["N_THETA"],
            params["RHO_MIN"],
            params["RHO_MAX"],
        )
        self.data_dir = Path(self.simIO.data_dir) / params["ANLYS_SUBDIR"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.data_dir / params["OUTPUT_FIELD_FILENAME"]

        run_settings = {
            **params,
            "RAW_DATA_DIR": None if raw_data_dir is None else str(raw_data_dir),
            "INPUT_FORMAT": self.source.input_format,
            "RAW_SAMPLE_COUNT": self.source.sample_count,
            "LCFS_INDEX_RESOLVED": int(self.lcfs_index),
            "N_PHI": self.source.plane_count,
            "FIELD_SHAPE": (
                self.source.plane_count,
                theta.size,
                rho.size,
            ),
        }
        self.simIO.inputsBoilerplate(
            "CONNECTION-LENGTH REGULAR-FIELD INPUTS",
            run_settings,
        )
        self.simIO.saveNumpyData(
            rho,
            RHO_FILENAME.removesuffix(".npy"),
            subdir=params["ANLYS_SUBDIR"],
        )
        self.simIO.saveNumpyData(
            theta,
            THETA_FILENAME.removesuffix(".npy"),
            subdir=params["ANLYS_SUBDIR"],
        )
        self.simIO.saveNumpyData(
            np.asarray(self.source.plane_phi_deg),
            PHI_FILENAME.removesuffix(".npy"),
            subdir=params["ANLYS_SUBDIR"],
        )

        self.field = regularize_field(
            params["ANLYS_DIR"],
            self.source,
            int(self.lcfs_index),
            rho,
            theta,
            grid_x,
            grid_z,
            self.output_path,
            self.simIO,
            params,
        )
        self.simIO.log.info(
            "Saved regular connection-length field: %s",
            self.output_path,
        )

        if params["GENERATE_PLOTS"]:
            plot_field(
                params["ANLYS_DIR"],
                self.field,
                rho,
                theta,
                self.source,
                int(self.lcfs_index),
                self.simIO,
                params,
            )
            self.simIO.log.info(
                "Saved %d regular-grid contour plots: %s",
                self.source.plane_count,
                Path(self.simIO.plot_dir) / params["ANLYS_SUBDIR"],
            )
        self.simIO.log.info(
            "## CONNECTION-LENGTH REGULAR-FIELD ANALYSIS FINISHED ##"
        )
        return self.field
