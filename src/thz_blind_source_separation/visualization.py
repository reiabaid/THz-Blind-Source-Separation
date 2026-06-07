from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_waveforms_overlay(
    waveforms: np.ndarray,
    time_axis: np.ndarray,
    angles: np.ndarray | None = None,
    save_path: str | Path = "waveforms_overlay.png",
) -> None:
    """Overlay all waveforms on one axes, colored blue-to-red by angle index.

    Parameters
    ----------
    waveforms : (n_timepoints, n_angles)
    time_axis : (n_timepoints,) in picoseconds
    angles    : (n_angles,) polarization angles in degrees; if None, uses
                0-indexed integers
    save_path : destination file for the PNG
    """
    if waveforms.ndim != 2:
        raise ValueError("waveforms must be a 2D array (n_timepoints, n_angles)")

    n_angles = waveforms.shape[1]
    if angles is None:
        angles = np.arange(n_angles, dtype=float)

    cmap = plt.colormaps["coolwarm"].resampled(n_angles)
    norm = mcolors.Normalize(vmin=float(angles[0]), vmax=float(angles[-1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_angles):
        ax.plot(time_axis, waveforms[:, i], color=cmap(norm(angles[i])), lw=0.8, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Polarization angle (deg)")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (norm.)")
    ax.set_title("THz waveforms — all angles overlaid")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlay plot -> {save_path}")


def plot_heatmap(
    waveforms: np.ndarray,
    time_axis: np.ndarray,
    angles: np.ndarray | None = None,
    save_path: str | Path = "heatmap_time_angle.png",
) -> None:
    """2-D heatmap: time on x-axis, polarization angle on y-axis, amplitude as colour.

    A diverging colormap is used so that positive and negative field values
    are both clearly visible.

    Parameters
    ----------
    waveforms : (n_timepoints, n_angles)
    time_axis : (n_timepoints,) in picoseconds
    angles    : (n_angles,) polarization angles in degrees; if None, uses
                0-indexed integers
    save_path : destination file for the PNG
    """
    if waveforms.ndim != 2:
        raise ValueError("waveforms must be a 2D array (n_timepoints, n_angles)")

    n_angles = waveforms.shape[1]
    if angles is None:
        angles = np.arange(n_angles, dtype=float)

    # Symmetric limits so zero maps to the colormap centre
    abs_max = float(np.max(np.abs(waveforms)))
    if abs_max == 0.0:
        abs_max = 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        time_axis,
        angles,
        waveforms.T,          # transpose: rows=angles, cols=time
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        shading="auto",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Amplitude (norm.)")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Polarization angle (deg)")
    ax.set_title("THz waveform heatmap — time vs. angle")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap -> {save_path}")


def plot_fft(
    waveform: np.ndarray,
    time_axis: np.ndarray,
    save_path: str | Path = "fft_analysis.png",
    *,
    noise_floor_frac: float = 0.25,
    db_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT of a single waveform and save an annotated amplitude plot.

    The function computes a one-sided (positive-frequency) amplitude spectrum,
    converts the frequency axis to THz, estimates a noise floor from the
    high-frequency tail of the spectrum, and marks it with a dashed horizontal
    line.

    Parameters
    ----------
    waveform : 1-D array, shape (n_timepoints,)
        Time-domain THz waveform.
    time_axis : 1-D array, shape (n_timepoints,)
        Uniformly sampled time axis in **picoseconds**.
    save_path : str or Path, optional
        Destination file for the PNG.  Default ``"fft_analysis.png"``.
    noise_floor_frac : float, optional
        Fraction of the high-frequency tail used to estimate the noise floor.
        E.g. 0.25 means the top-25 % of frequency bins are averaged.
        Default 0.25.
    db_scale : bool, optional
        If *True* (default) the y-axis is plotted in dB
        (``20·log10(amplitude)``); otherwise linear amplitude.

    Returns
    -------
    freqs_thz : 1-D array
        Positive frequency axis in THz.
    amplitude : 1-D array
        One-sided amplitude spectrum (linear, before any dB conversion).

    Raises
    ------
    ValueError
        If *waveform* or *time_axis* are not 1-D, have mismatched length, or
        *time_axis* has fewer than 2 points.
    """
    waveform = np.asarray(waveform, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    if waveform.ndim != 1:
        raise ValueError(
            f"waveform must be 1-D, got shape {waveform.shape}"
        )
    if time_axis.ndim != 1:
        raise ValueError(
            f"time_axis must be 1-D, got shape {time_axis.shape}"
        )
    if waveform.size != time_axis.size:
        raise ValueError(
            f"waveform length ({waveform.size}) must match time_axis length ({time_axis.size})"
        )
    if time_axis.size < 2:
        raise ValueError("time_axis must contain at least 2 points")

    n = waveform.size
    dt_ps = float(time_axis[1] - time_axis[0])   # sampling interval in ps
    dt_s  = dt_ps * 1e-12                         # convert to seconds

    # --- FFT ------------------------------------------------------------------
    fft_vals   = np.fft.rfft(waveform)            # one-sided complex spectrum
    freqs_hz   = np.fft.rfftfreq(n, d=dt_s)       # frequency axis in Hz
    freqs_thz  = freqs_hz * 1e-12                 # convert to THz

    # Two-sided → one-sided amplitude normalisation
    amplitude  = np.abs(fft_vals) / n
    amplitude[1:-1] *= 2                          # double non-DC, non-Nyquist bins

    # --- Noise floor ----------------------------------------------------------
    # Use the median of the top `noise_floor_frac` of frequency bins.
    # The median is more robust to spectral lines than the mean.
    n_noise    = max(1, int(np.ceil(noise_floor_frac * amplitude.size)))
    noise_floor = float(np.median(amplitude[-n_noise:]))

    # --- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    if db_scale:
        # Avoid log(0): clip to a small positive floor before taking log
        eps  = amplitude[amplitude > 0].min() * 1e-3 if np.any(amplitude > 0) else 1e-12
        plot_amp   = 20.0 * np.log10(np.maximum(amplitude, eps))
        noise_line = 20.0 * np.log10(max(noise_floor, eps))
        ylabel     = "Amplitude (dB)"
    else:
        plot_amp   = amplitude
        noise_line = noise_floor
        ylabel     = "Amplitude (a.u.)"

    ax.plot(freqs_thz, plot_amp, color="#4C9BE8", lw=1.5, label="Spectrum")

    ax.axhline(
        noise_line,
        color="#E8724C",
        lw=1.2,
        linestyle="--",
        label=f"Noise floor ≈ {noise_floor:.3g} a.u.",
    )

    # Shade the noise floor region
    ax.fill_between(
        freqs_thz,
        plot_amp.min(),
        noise_line,
        alpha=0.08,
        color="#E8724C",
        label="_nolegend_",
    )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(ylabel)
    ax.set_title("THz Pulse — FFT Amplitude Spectrum")
    ax.legend(framealpha=0.85)
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved FFT plot -> {save_path}")

    return freqs_thz, amplitude


def plot_angle_dependence(
    data_matrix: np.ndarray,
    time_axis: np.ndarray,
    angles: np.ndarray | None = None,
    save_path: str | Path = "angle_dependence.png",
    *,
    time_index: int | None = None,
) -> tuple[int, float]:
    """Plot E-field amplitude vs polarization angle at the pulse peak.

    At the time point of maximum RMS signal (or a user-specified index), the
    measured E-field across all angles is plotted together with a scaled
    ``sin(2alpha)`` reference curve.  The reference amplitude is chosen by
    least-squares projection so the overlay is meaningful regardless of the
    true mixing weights.

    Parameters
    ----------
    data_matrix : ndarray, shape (n_timepoints, n_angles)
        Waveform matrix — one column per polarization angle.
    time_axis : 1-D array, shape (n_timepoints,)
        Time axis in picoseconds.
    angles : 1-D array, shape (n_angles,), optional
        Polarization angles in **degrees**.  If *None*, evenly spaced angles
        on ``[0, 180)`` are assumed.
    save_path : str or Path, optional
        Destination PNG file.  Default ``"angle_dependence.png"``.
    time_index : int, optional
        Row index to use as the fixed time point.  If *None* (default), the
        index of the maximum RMS across angles is used (i.e. the pulse peak).

    Returns
    -------
    peak_index : int
        Row index of the selected time point.
    peak_time_ps : float
        Corresponding time in picoseconds.

    Raises
    ------
    ValueError
        If ``data_matrix`` is not 2-D, shapes are inconsistent, or
        ``time_index`` is out of range.
    """
    data_matrix = np.asarray(data_matrix, dtype=float)
    time_axis   = np.asarray(time_axis,   dtype=float)

    if data_matrix.ndim != 2:
        raise ValueError(
            f"data_matrix must be 2-D (n_timepoints, n_angles), got shape {data_matrix.shape}"
        )
    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1-D, got shape {time_axis.shape}")
    if data_matrix.shape[0] != time_axis.size:
        raise ValueError(
            f"data_matrix rows ({data_matrix.shape[0]}) must match "
            f"time_axis length ({time_axis.size})"
        )

    n_timepoints, n_angles = data_matrix.shape

    # ---------------------------------------------------------------- angles
    if angles is None:
        angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    else:
        angles = np.asarray(angles, dtype=float)
        if angles.size != n_angles:
            raise ValueError(
                f"angles length ({angles.size}) must match data_matrix columns ({n_angles})"
            )

    angles_rad = np.deg2rad(angles)

    # ------------------------------------------ locate the peak time point
    if time_index is None:
        # RMS across angles at each time step — robust when sources cancel
        rms_vs_time = np.sqrt(np.mean(data_matrix ** 2, axis=1))
        peak_index  = int(np.argmax(rms_vs_time))
    else:
        if not (0 <= time_index < n_timepoints):
            raise ValueError(
                f"time_index={time_index} is out of range for "
                f"n_timepoints={n_timepoints}"
            )
        peak_index = int(time_index)

    peak_time_ps = float(time_axis[peak_index])
    e_field      = data_matrix[peak_index, :]          # shape (n_angles,)

    # ---------------------- least-squares amplitude of sin(2alpha) reference
    # Fit: e_field ≈ A * sin(2*alpha) + B
    # Using a two-column design matrix [sin(2a), 1] so a DC offset is absorbed.
    design = np.column_stack([np.sin(2.0 * angles_rad), np.ones(n_angles)])
    coeffs, _, _, _ = np.linalg.lstsq(design, e_field, rcond=None)
    A_fit, B_fit    = float(coeffs[0]), float(coeffs[1])

    # Dense reference curve for smooth overlay
    alpha_dense  = np.linspace(0.0, 180.0, 500)
    ref_curve    = A_fit * np.sin(2.0 * np.deg2rad(alpha_dense)) + B_fit

    # --------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left panel: waveform with peak marker ----
    ax_wave = axes[0]
    # Show the RMS waveform so the peak is unambiguous
    rms_wave = np.sqrt(np.mean(data_matrix ** 2, axis=1))
    ax_wave.plot(time_axis, rms_wave, color="#4C9BE8", lw=1.5, label="RMS across angles")
    ax_wave.axvline(
        peak_time_ps,
        color="#E8724C",
        lw=1.4,
        linestyle="--",
        label=f"Peak @ {peak_time_ps:.2f} ps",
    )
    ax_wave.set_xlabel("Time (ps)")
    ax_wave.set_ylabel("RMS E-field (a.u.)")
    ax_wave.set_title("Pulse waveform — selected time point")
    ax_wave.legend(framealpha=0.85)

    # ---- Right panel: angle dependence ----
    ax_ang = axes[1]

    # Measured data as scatter + connecting line
    ax_ang.plot(
        angles, e_field,
        color="#4C9BE8",
        lw=1.5,
        marker="o",
        markersize=5,
        label="Measured E-field",
        zorder=3,
    )

    # sin(2alpha) reference
    ax_ang.plot(
        alpha_dense,
        ref_curve,
        color="#E8724C",
        lw=2.0,
        linestyle="--",
        label=r"$A\,\sin(2\alpha) + B$  (LSQ fit)",
        zorder=2,
    )

    # Zero line for reference
    ax_ang.axhline(0.0, color="white" if plt.rcParams["axes.facecolor"] == "black"
                   else "grey", lw=0.8, linestyle=":", alpha=0.6)

    ax_ang.set_xlabel("Polarization angle, alpha (deg)")
    ax_ang.set_ylabel("E-field amplitude (a.u.)")
    ax_ang.set_title(
        f"Angle dependence at t = {peak_time_ps:.2f} ps\n"
        f"sin(2alpha) fit: A = {A_fit:.3f}, B = {B_fit:.3f}"
    )
    ax_ang.set_xlim(angles[0], angles[-1])
    ax_ang.legend(framealpha=0.85)

    fig.suptitle("THz Polarimetric Angle Dependence", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved angle-dependence plot -> {save_path}")

    return peak_index, peak_time_ps

