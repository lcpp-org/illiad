"""Chunk-oriented access to saved SOL plane crossings."""

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from illiad.utilities.coordtrans import XYZ_to_RTP_many


SHARD_DIRECTORY = "raw_crossings"
SHARD_MANIFEST = "manifest.json"
SHARD_FORMAT = "illiad-plane-crossings-v1"
TRACE_LENGTH_LIMIT_FILENAME = "trace_length_limit_m.npy"
TRACE_SPINS_FILENAME = "trace_spins.npy"
TRACE_LCFS_INDEX_FILENAME = "trace_lcfs_index.npy"
TRACE_VESSEL_RADIUS_FILENAME = "trace_vessel_radius_m.npy"
SHARD_DTYPE = np.dtype(
    [
        ("rtp", "<f8", (3,)),
        ("fieldline_id", "<i4"),
        ("source_direction", "i1"),
    ],
    align=False,
)


@dataclass(frozen=True)
class CrossingChunk:
    """One bounded chunk of points and their connection lengths."""

    points_rtp: np.ndarray
    connection_length_m: np.ndarray


class PlaneCrossingSource(Protocol):
    """Interface consumed by regular-grid connection-length analyses."""

    @property
    def plane_phi_deg(self) -> np.ndarray:
        """Toroidal plane coordinates in saved order."""

    @property
    def plane_count(self) -> int:
        """Number of toroidal planes."""

    @property
    def sample_count(self) -> int:
        """Total number of raw samples."""

    @property
    def input_format(self) -> str:
        """Short description of the underlying value representation."""

    def plane_sample_count(self, plane_index: int) -> int:
        """Return the number of samples assigned to one plane."""

    def iter_plane_chunks(
        self,
        plane_index: int,
        chunk_size: int,
    ) -> Iterator[CrossingChunk]:
        """Yield bounded point/value chunks for one toroidal plane."""

    def iter_value_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield the smallest saved population defining the value range."""


class NpyPlaneCrossingSource:
    """Read the compact or legacy expanded plane-sorted NumPy output."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        required_paths = {
            "points": self.data_dir / "raw_points_rtp.npy",
            "offsets": self.data_dir / "plane_offsets.npy",
            "phi": self.data_dir / "plane_phi_deg.npy",
        }
        missing = [
            str(path) for path in required_paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing connection-length crossing data:\n"
                + "\n".join(missing)
            )

        self._points = np.load(required_paths["points"], mmap_mode="r")
        self._offsets = np.load(required_paths["offsets"], mmap_mode="r")
        self._plane_phi_deg = np.load(required_paths["phi"], mmap_mode="r")

        expanded_path = self.data_dir / "raw_connection_length_m.npy"
        fieldline_id_path = self.data_dir / "raw_fieldline_id.npy"
        fieldline_values_path = (
            self.data_dir / "fieldline_connection_length_m.npy"
        )
        if fieldline_id_path.is_file() and fieldline_values_path.is_file():
            self._expanded_values = None
            self._fieldline_id = np.load(fieldline_id_path, mmap_mode="r")
            self._fieldline_values = np.load(
                fieldline_values_path,
                mmap_mode="r",
            )
            self._input_format = "compact_fieldline_indexed"
        elif expanded_path.is_file():
            self._expanded_values = np.load(expanded_path, mmap_mode="r")
            self._fieldline_id = None
            self._fieldline_values = None
            self._input_format = "expanded"
        else:
            raise FileNotFoundError(
                "Missing connection-length values. Expected either "
                "raw_connection_length_m.npy or both raw_fieldline_id.npy "
                "and fieldline_connection_length_m.npy."
            )

        self._validate()

    @property
    def plane_phi_deg(self):
        return self._plane_phi_deg

    @property
    def plane_count(self):
        return int(self._plane_phi_deg.size)

    @property
    def sample_count(self):
        return int(self._points.shape[0])

    @property
    def input_format(self):
        return self._input_format

    def _validate(self):
        if self._points.ndim != 2 or self._points.shape[1] != 3:
            raise ValueError(
                "raw_points_rtp.npy must have shape (sample, 3); "
                f"found {self._points.shape}."
            )
        if self._plane_phi_deg.ndim != 1:
            raise ValueError("plane_phi_deg.npy must be one-dimensional.")
        if (
            self._offsets.ndim != 1
            or self._offsets.size != self._plane_phi_deg.size + 1
        ):
            raise ValueError(
                "plane_offsets.npy must have one more entry than "
                "plane_phi_deg.npy."
            )
        if (
            self._offsets[0] != 0
            or self._offsets[-1] != self._points.shape[0]
        ):
            raise ValueError(
                "plane_offsets.npy does not span the complete raw sample "
                "array."
            )
        if np.any(np.diff(self._offsets) < 0):
            raise ValueError(
                "plane_offsets.npy must be monotonically increasing."
            )

        if self._expanded_values is not None:
            if (
                self._expanded_values.ndim != 1
                or self._expanded_values.shape[0] != self._points.shape[0]
            ):
                raise ValueError(
                    "raw_connection_length_m.npy must contain one value per "
                    "raw point."
                )
        else:
            if (
                self._fieldline_id.ndim != 1
                or self._fieldline_id.shape[0] != self._points.shape[0]
            ):
                raise ValueError(
                    "raw_fieldline_id.npy must contain one ID per raw point."
                )
            if self._fieldline_values.ndim != 1:
                raise ValueError(
                    "fieldline_connection_length_m.npy must be "
                    "one-dimensional."
                )

    def _values_for_slice(self, start, stop):
        if self._expanded_values is not None:
            return np.asarray(self._expanded_values[start:stop])
        fieldline_id = np.asarray(self._fieldline_id[start:stop])
        return np.asarray(self._fieldline_values[fieldline_id])

    def plane_sample_count(self, plane_index):
        self._validate_plane_index(plane_index)
        return int(
            self._offsets[plane_index + 1] - self._offsets[plane_index]
        )

    def iter_plane_chunks(self, plane_index, chunk_size):
        self._validate_plane_index(plane_index)
        _require_positive_chunk_size(chunk_size)
        plane_start = int(self._offsets[plane_index])
        plane_stop = int(self._offsets[plane_index + 1])
        for start in range(plane_start, plane_stop, chunk_size):
            stop = min(start + chunk_size, plane_stop)
            yield CrossingChunk(
                points_rtp=np.asarray(self._points[start:stop]),
                connection_length_m=self._values_for_slice(start, stop),
            )

    def iter_value_chunks(self, chunk_size):
        _require_positive_chunk_size(chunk_size)
        values = (
            self._fieldline_values
            if self._fieldline_values is not None
            else self._expanded_values
        )
        for start in range(0, values.size, chunk_size):
            yield np.asarray(values[start : start + chunk_size])

    def _validate_plane_index(self, plane_index):
        if not 0 <= plane_index < self.plane_count:
            raise IndexError(
                f"Plane index {plane_index} is outside [0, "
                f"{self.plane_count})."
            )


class PlaneShardWriter:
    """Append crossings directly to one packed binary file per plane."""

    def __init__(self, data_dir, plane_phi_deg, major_radius):
        self.data_dir = Path(data_dir)
        self.shard_dir = self.data_dir / SHARD_DIRECTORY
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.plane_phi_deg = np.asarray(plane_phi_deg, dtype=np.float64)
        if self.plane_phi_deg.ndim != 1 or not self.plane_phi_deg.size:
            raise ValueError("plane_phi_deg must be a nonempty vector.")
        self.n_planes = int(self.plane_phi_deg.size)
        self.major_radius = float(major_radius)
        self.counts = np.zeros(self.n_planes, dtype=np.int64)
        self._handles = OrderedDict()
        self._max_open_files = 64
        manifest_path = self.shard_dir / SHARD_MANIFEST
        manifest_path.unlink(missing_ok=True)
        manifest_path.with_suffix(".building.json").unlink(missing_ok=True)
        for stale_shard in self.shard_dir.glob("plane_*.bin"):
            stale_shard.unlink()
        for plane_index in range(self.n_planes):
            path = self._plane_path(plane_index)
            path.open("wb").close()

    def _plane_path(self, plane_index):
        return self.shard_dir / f"plane_{plane_index:04d}.bin"

    def _handle_for_plane(self, plane_index):
        handle = self._handles.pop(plane_index, None)
        if handle is None:
            if len(self._handles) == self._max_open_files:
                _, oldest_handle = self._handles.popitem(last=False)
                oldest_handle.close()
            handle = self._plane_path(plane_index).open("ab", buffering=0)
        self._handles[plane_index] = handle
        return handle

    def append_xyz(
        self,
        plane_index,
        xyz,
        fieldline_id,
        source_direction,
    ):
        """Convert one bounded XYZ block and append it to its plane shard."""
        plane_index = int(plane_index)
        if not 0 <= plane_index < self.n_planes:
            raise IndexError(f"Invalid plane index {plane_index}.")
        xyz = np.asarray(xyz)
        fieldline_id = np.asarray(fieldline_id)
        source_direction = np.asarray(source_direction)
        count = int(xyz.shape[0])
        if xyz.shape != (count, 3):
            raise ValueError("xyz must have shape (sample, 3).")
        if fieldline_id.shape != (count,) or source_direction.shape != (count,):
            raise ValueError("Crossing metadata must have one entry per point.")
        if count == 0:
            return
        if np.any(fieldline_id < 0) or np.any(
            fieldline_id > np.iinfo(np.int32).max
        ):
            raise ValueError("fieldline_id exceeds the int32 raw-ID range.")

        records = np.empty(count, dtype=SHARD_DTYPE)
        records["rtp"] = XYZ_to_RTP_many(xyz, self.major_radius)
        records["rtp"][:, 2] = np.deg2rad(self.plane_phi_deg[plane_index])
        records["fieldline_id"] = fieldline_id
        records["source_direction"] = source_direction
        records.tofile(self._handle_for_plane(plane_index))
        self.counts[plane_index] += count

    def append_rtp(
        self,
        plane_index,
        points_rtp,
        fieldline_id,
        source_direction,
    ):
        """Append points that are already represented in RTP coordinates."""
        plane_index = int(plane_index)
        if not 0 <= plane_index < self.n_planes:
            raise IndexError(f"Invalid plane index {plane_index}.")
        points_rtp = np.asarray(points_rtp)
        fieldline_id = np.asarray(fieldline_id)
        source_direction = np.asarray(source_direction)
        count = int(points_rtp.shape[0])
        if points_rtp.shape != (count, 3):
            raise ValueError("points_rtp must have shape (sample, 3).")
        if fieldline_id.shape != (count,) or source_direction.shape != (count,):
            raise ValueError("Crossing metadata must have one entry per point.")
        if count == 0:
            return
        if np.any(fieldline_id < 0) or np.any(
            fieldline_id > np.iinfo(np.int32).max
        ):
            raise ValueError("fieldline_id exceeds the int32 raw-ID range.")
        records = np.empty(count, dtype=SHARD_DTYPE)
        records["rtp"] = points_rtp
        records["rtp"][:, 2] = np.deg2rad(self.plane_phi_deg[plane_index])
        records["fieldline_id"] = fieldline_id
        records["source_direction"] = source_direction
        records.tofile(self._handle_for_plane(plane_index))
        self.counts[plane_index] += count

    def finish(self, fieldline_connection_length_m):
        """Close all shards and publish the manifest last."""
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._handles.clear()
        fieldline_values = np.asarray(fieldline_connection_length_m)
        manifest = {
            "format": SHARD_FORMAT,
            "record_itemsize": SHARD_DTYPE.itemsize,
            "plane_phi_deg": self.plane_phi_deg.tolist(),
            "plane_counts": self.counts.tolist(),
            "sample_count": int(self.counts.sum()),
            "fieldline_count": int(fieldline_values.size),
        }
        manifest_path = self.shard_dir / SHARD_MANIFEST
        temporary_path = manifest_path.with_suffix(".building.json")
        temporary_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
        return manifest_path

    def abort(self):
        """Close handles and remove incomplete current-run shards."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        (self.shard_dir / SHARD_MANIFEST).unlink(missing_ok=True)
        for partial_shard in self.shard_dir.glob("plane_*.bin"):
            partial_shard.unlink()


class ShardedPlaneCrossingSource:
    """Read the append-only plane-sharded raw crossing representation."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.shard_dir = self.data_dir / SHARD_DIRECTORY
        manifest_path = self.shard_dir / SHARD_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing crossing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != SHARD_FORMAT:
            raise ValueError(
                f"Unsupported crossing shard format: {manifest.get('format')!r}."
            )
        if manifest.get("record_itemsize") != SHARD_DTYPE.itemsize:
            raise ValueError("Crossing shard record size does not match this reader.")
        self._plane_phi_deg = np.asarray(
            manifest["plane_phi_deg"], dtype=np.float64
        )
        self._plane_counts = np.asarray(
            manifest["plane_counts"], dtype=np.int64
        )
        if (
            self._plane_phi_deg.ndim != 1
            or self._plane_counts.shape != self._plane_phi_deg.shape
            or np.any(self._plane_counts < 0)
        ):
            raise ValueError("Invalid plane arrays in crossing manifest.")
        self._fieldline_values = np.load(
            self.data_dir / "fieldline_connection_length_m.npy",
            mmap_mode="r",
        )
        if self._fieldline_values.ndim != 1:
            raise ValueError(
                "fieldline_connection_length_m.npy must be one-dimensional."
            )
        if int(manifest["fieldline_count"]) != self._fieldline_values.size:
            raise ValueError(
                "Crossing manifest field-line count is inconsistent with "
                "fieldline_connection_length_m.npy."
            )
        if int(manifest["sample_count"]) != int(self._plane_counts.sum()):
            raise ValueError("Crossing manifest sample counts are inconsistent.")
        for plane_index, count in enumerate(self._plane_counts):
            expected_size = int(count) * SHARD_DTYPE.itemsize
            path = self._plane_path(plane_index)
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ValueError(
                    f"Crossing shard {path} does not contain {int(count)} records."
                )

    def _plane_path(self, plane_index):
        return self.shard_dir / f"plane_{plane_index:04d}.bin"

    @property
    def plane_phi_deg(self):
        return self._plane_phi_deg

    @property
    def plane_count(self):
        return int(self._plane_phi_deg.size)

    @property
    def sample_count(self):
        return int(self._plane_counts.sum())

    @property
    def input_format(self):
        return "plane_sharded_fieldline_indexed"

    def _validate_plane_index(self, plane_index):
        if not 0 <= plane_index < self.plane_count:
            raise IndexError(
                f"Plane index {plane_index} is outside [0, {self.plane_count})."
            )

    def plane_sample_count(self, plane_index):
        self._validate_plane_index(plane_index)
        return int(self._plane_counts[plane_index])

    def iter_plane_chunks(self, plane_index, chunk_size):
        self._validate_plane_index(plane_index)
        _require_positive_chunk_size(chunk_size)
        count = self.plane_sample_count(plane_index)
        if count == 0:
            return
        records = np.memmap(
            self._plane_path(plane_index),
            mode="r",
            dtype=SHARD_DTYPE,
            shape=(count,),
        )
        try:
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                chunk = records[start:stop]
                fieldline_id = np.asarray(chunk["fieldline_id"])
                if np.any(fieldline_id < 0) or np.any(
                    fieldline_id >= self._fieldline_values.size
                ):
                    raise ValueError(
                        f"Crossing shard for plane {plane_index} contains an "
                        "invalid field-line ID."
                    )
                yield CrossingChunk(
                    points_rtp=np.asarray(chunk["rtp"]),
                    connection_length_m=np.asarray(
                        self._fieldline_values[fieldline_id]
                    ),
                )
        finally:
            del records

    def iter_value_chunks(self, chunk_size):
        _require_positive_chunk_size(chunk_size)
        for start in range(0, self._fieldline_values.size, chunk_size):
            yield np.asarray(
                self._fieldline_values[start : start + chunk_size]
            )


def _require_positive_chunk_size(chunk_size):
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")


def open_plane_crossing_source(data_dir):
    """Open the currently supported saved crossing representation."""
    manifest_path = Path(data_dir) / SHARD_DIRECTORY / SHARD_MANIFEST
    if manifest_path.is_file():
        return ShardedPlaneCrossingSource(data_dir)
    return NpyPlaneCrossingSource(data_dir)
