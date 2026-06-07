"""Synthetic THz data generator for blind-source-separation experiments.

Physics background
------------------
THz time-domain pulses are well approximated by the *first derivative of a
Gaussian* (sometimes called a "monocycle"):

    s(t) = -A · (t - t0) / σ² · exp(-(t - t0)² / (2σ²))

The mixing model implemented here mimics the polarimetric dependence seen in
anisotropic THz measurements.  For an analyser rotated by angle α the
detected intensity components follow Malus-like trigonometric weights:

    x(α) = w1·sin(2α)  +  w2·cos(4α)  +  w3·sin(4α)  +  w4  +  noise

Each source k is mixed with its own set of four weights derived from the
above basis functions, so that the overall observation matrix is

    X(t, α) = S(t) @ A(α)  +  noise

where S is (n_timepoints × K) and A is (K × n_angles).
"""

from __future__ import annotations

import numpy as np


def _gaussian_derivative_pulse(
    time_axis: np.ndarray,
    center: float,
    width: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return a single Gaussian-derivative (monocycle) pulse.

    Parameters
    ----------
    time_axis : 1-D array, shape (T,)
        Time points in picoseconds.
    center : float
        Peak location in picoseconds.
    width : float
        Gaussian standard deviation σ in picoseconds.
    amplitude : float
        Signed peak amplitude.

    Returns
    -------
    pulse : 1-D array, shape (T,)
    """
    tau = time_axis - center
    pulse = -amplitude * (tau / width**2) * np.exp(-(tau**2) / (2.0 * width**2))
    return pulse


def generate_synthetic_thz(
    K: int,
    n_angles: int,
    n_timepoints: int,
    noise_level: float,
    *,
    t_start_ps: float = 0.0,
    t_end_ps: float = 10.0,
    pulse_width_ps: float = 0.3,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic THz polarimetric sweep dataset.

    Each of the *K* hidden sources is a Gaussian-derivative (monocycle) pulse
    with a distinct centre time and amplitude.  The *K* sources are linearly
    mixed across *n_angles* measurement angles using the four polarimetric
    basis functions

        sin(2α),  cos(4α),  sin(4α),  1  (constant)

    so that the mixing coefficient for source *k* at angle α is

        A[k, α] = c1[k]·sin(2α) + c2[k]·cos(4α) + c3[k]·sin(4α) + c4[k]

    The mixing coefficients ``c1…c4`` are drawn randomly (uniform [-1, 1])
    and then normalised so that each row of the mixing matrix has unit ℓ²-norm.

    Independent Gaussian noise (zero mean, standard deviation = *noise_level*
    × peak signal amplitude) is added to the final observation matrix.

    Parameters
    ----------
    K : int
        Number of independent THz source waveforms to generate (≥ 1).
    n_angles : int
        Number of analyser/waveplate angles in the sweep (≥ 1).
    n_timepoints : int
        Number of time samples per waveform (≥ 2).
    noise_level : float
        Noise standard deviation expressed as a fraction of the peak-to-peak
        signal amplitude (e.g. 0.05 → 5 % noise).
    t_start_ps : float, optional
        Start of the time axis in picoseconds.  Default 0.0 ps.
    t_end_ps : float, optional
        End of the time axis in picoseconds.  Default 10.0 ps.
    pulse_width_ps : float, optional
        Gaussian width σ for every source pulse in picoseconds.  Default 0.3 ps.
    rng : numpy Generator, int seed, or None
        Random-number source.  Pass an integer for reproducible results.

    Returns
    -------
    X : ndarray, shape (n_timepoints, n_angles)
        Observed (noisy) mixed waveform matrix, one column per angle.
    S : ndarray, shape (n_timepoints, K)
        Ground-truth source waveforms *before* mixing or noise addition.

    Examples
    --------
    >>> X, S = generate_synthetic_thz(K=3, n_angles=36, n_timepoints=256,
    ...                               noise_level=0.05, rng=42)
    >>> X.shape
    (256, 36)
    >>> S.shape
    (256, 3)
    """
    # ------------------------------------------------------------------ setup
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")
    if n_angles < 1:
        raise ValueError(f"n_angles must be at least 1, got {n_angles}")
    if n_timepoints < 2:
        raise ValueError(f"n_timepoints must be at least 2, got {n_timepoints}")
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    rng = np.random.default_rng(rng)

    # ----------------------------------------------------- time & angle grids
    time_axis: np.ndarray = np.linspace(t_start_ps, t_end_ps, n_timepoints)
    angles_rad: np.ndarray = np.linspace(0.0, np.pi, n_angles, endpoint=False)

    # -------------------------------------------------- ground-truth sources S
    # Spread pulse centres uniformly over the middle 60 % of the time window
    # so no pulse is clipped at the boundary.
    t_range = t_end_ps - t_start_ps
    # Divide by max(K-1, 1) so the K centres are evenly spread from
    # 20 % to 80 % of the time window (not just compressed into the lower half).
    centers = t_start_ps + t_range * (0.2 + 0.6 * np.arange(K) / max(K - 1, 1))
    amplitudes = rng.uniform(0.5, 1.5, size=K) * rng.choice([-1, 1], size=K)

    S = np.column_stack(
        [
            _gaussian_derivative_pulse(
                time_axis,
                center=centers[k],
                width=pulse_width_ps,
                amplitude=amplitudes[k],
            )
            for k in range(K)
        ]
    )  # shape (n_timepoints, K)

    # --------------------------------------------------- mixing matrix A(α)
    # Four polarimetric basis functions evaluated at each angle
    B = np.column_stack(
        [
            np.sin(2.0 * angles_rad),   # sin(2α)
            np.cos(4.0 * angles_rad),   # cos(4α)
            np.sin(4.0 * angles_rad),   # sin(4α)
            np.ones(n_angles),           # constant term
        ]
    )  # shape (n_angles, 4)

    # Random mixing coefficients for each source, then row-normalise
    C = rng.uniform(-1.0, 1.0, size=(K, 4))  # (K, 4)
    row_norms = np.linalg.norm(C, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    C = C / row_norms  # unit-norm rows ensure sources contribute equally

    # A[k, α] = C[k, :] @ B[α, :]ᵀ  →  A shape (K, n_angles)
    A = C @ B.T  # (K, n_angles)

    # ------------------------------------------- clean observations + noise
    X_clean = S @ A  # (n_timepoints, n_angles)

    peak_amplitude = np.max(np.abs(X_clean)) if X_clean.size > 0 else 1.0
    if peak_amplitude == 0.0:
        peak_amplitude = 1.0

    noise = rng.standard_normal(size=X_clean.shape) * noise_level * peak_amplitude
    X = X_clean + noise

    return X, S
