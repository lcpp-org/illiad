"""Chunk-oriented access to saved SOL plane crossings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


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


def _require_positive_chunk_size(chunk_size):
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")


def open_plane_crossing_source(data_dir):
    """Open the currently supported saved crossing representation."""
    return NpyPlaneCrossingSource(data_dir)
