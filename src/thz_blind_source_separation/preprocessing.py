from __future__ import annotations

from typing import Tuple

import numpy as np


def preprocess_waveforms(
    waveforms: np.ndarray,
    time_axis: np.ndarray,
    baseline_frac: float = 0.1,
    lp_cutoff_thz: float = 4.5,
    filter_order: int = 4,
) -> np.ndarray:
    """Apply baseline correction, normalization, then low-pass filtering.

    Parameters
    - waveforms: array shape (n_timepoints, n_angles)
    - time_axis: 1D array of time in picoseconds
    - baseline_frac: fraction of the early window to use for baseline
    - lp_cutoff_thz: low-pass cutoff in THz
    - filter_order: Butterworth filter order

    Returns a cleaned waveform matrix of the same shape.
    """

    if waveforms.ndim != 2:
        raise ValueError("waveforms must be a 2D array (n_timepoints, n_angles)")
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be a 1D array")
    if waveforms.shape[0] != time_axis.shape[0]:
        raise ValueError("time_axis length must match number of waveform rows")

    n_time = waveforms.shape[0]
    n_baseline = max(1, int(np.ceil(baseline_frac * n_time)))

    cleaned = waveforms.astype(float).copy()

    # Baseline correction
    baseline = np.mean(cleaned[:n_baseline, :], axis=0)
    cleaned = cleaned - baseline[np.newaxis, :]

    # Global normalization — preserves relative amplitudes across angles
    global_peak = np.max(np.abs(cleaned))
    if global_peak == 0.0:
        global_peak = 1.0
    cleaned = cleaned / global_peak

    # Low-pass filtering using scipy.signal
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception as exc:  # pragma: no cover - environment issues
        raise RuntimeError(
            "scipy is required for filtering; install scipy (pip install scipy)"
        ) from exc

    dt_ps = float(time_axis[1] - time_axis[0])
    fs_hz = 1.0 / (dt_ps * 1e-12)
    cutoff_hz = float(lp_cutoff_thz) * 1e12
    nyq = fs_hz / 2.0

    if cutoff_hz >= nyq:
        return cleaned

    normal_cutoff = cutoff_hz / nyq
    sos = butter(filter_order, normal_cutoff, btype="low", analog=False, output="sos")
    filtered = sosfiltfilt(sos, cleaned, axis=0)

    return filtered
