from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class THZLoadError(ValueError):
    """Raised when a THz sweep file cannot be loaded or validated."""


def _infer_delimiter(file_path: Path, delimiter: str | None) -> str | None:
    if delimiter is not None:
        return delimiter
    return "," if file_path.suffix.lower() == ".csv" else None


def _load_numeric_table(file_path: Path, delimiter: str | None, skip_header: int) -> np.ndarray:
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            comments="#",
            skip_header=skip_header,
            dtype=float,
        )
    except OSError as exc:
        raise THZLoadError(f"Could not read file '{file_path}': {exc}") from exc

    if data.size == 0:
        raise THZLoadError(f"File '{file_path}' did not contain any numeric data.")

    if data.ndim == 1:
        data = data[:, np.newaxis]

    return np.asarray(data, dtype=float)


def _validate_uniform_time_axis(time_axis: np.ndarray) -> float:
    if time_axis.ndim != 1:
        raise THZLoadError("The time axis must be one-dimensional.")
    if time_axis.size < 2:
        raise THZLoadError("The time axis must contain at least two points.")

    steps = np.diff(time_axis)
    if not np.all(np.isfinite(steps)):
        raise THZLoadError("The time axis contains non-finite values.")

    reference_step = steps[0]
    if np.isclose(reference_step, 0.0):
        raise THZLoadError("The time axis step cannot be zero.")

    if not np.allclose(steps, reference_step, rtol=1e-5, atol=1e-12):
        raise THZLoadError("The time axis must be uniformly spaced.")

    return float(reference_step)


def load_thz_sweep(
    file_path: str | Path,
    *,
    delimiter: str | None = None,
    skip_header: int = 0,
    time_column: int = 0,
    waveform_columns: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a THz sweep as ``(n_timepoints, n_angles)`` plus a time axis.

    The file is expected to contain numeric data in CSV or plain-text format.
    By default the first column is interpreted as the time axis and the
    remaining columns are treated as waveforms.
    """

    path = Path(file_path)
    if not path.exists():
        raise THZLoadError(f"File '{path}' does not exist.")

    resolved_delimiter = _infer_delimiter(path, delimiter)
    table = _load_numeric_table(path, resolved_delimiter, skip_header)

    if table.ndim != 2:
        raise THZLoadError(f"Expected a 2D table in '{path}', got shape {table.shape}.")
    if table.shape[1] < 2:
        raise THZLoadError(
            f"Loaded data from '{path}' must have at least 2 columns, got {table.shape[1]}."
        )

    if waveform_columns is None:
        if time_column < 0 or time_column >= table.shape[1]:
            raise THZLoadError(
                f"time_column={time_column} is out of bounds for data with {table.shape[1]} columns."
            )
        waveform_columns = tuple(index for index in range(table.shape[1]) if index != time_column)

    time_axis = np.asarray(table[:, time_column], dtype=float)
    waveform_matrix = np.asarray(table[:, waveform_columns], dtype=float)

    if waveform_matrix.ndim == 1:
        waveform_matrix = waveform_matrix[:, np.newaxis]

    _validate_uniform_time_axis(time_axis)

    n_timepoints, n_angles = waveform_matrix.shape
    time_start = float(time_axis[0])
    time_end = float(time_axis[-1])
    print(
        f"Loaded THz sweep from {path}: shape=({n_timepoints}, {n_angles}), "
        f"time_range=[{time_start:.6g}, {time_end:.6g}] ps, waveforms={n_angles}"
    )

    return waveform_matrix, time_axis
