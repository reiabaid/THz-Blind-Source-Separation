from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    cmap = cm.get_cmap("coolwarm", n_angles)
    norm = mcolors.Normalize(vmin=float(angles[0]), vmax=float(angles[-1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_angles):
        ax.plot(time_axis, waveforms[:, i], color=cmap(norm(angles[i])), lw=0.8, alpha=0.85)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Polarization angle (deg)")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (norm.)")
    ax.set_title("THz waveforms — all angles overlaid")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlay plot → {save_path}")


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
    print(f"Saved heatmap → {save_path}")
