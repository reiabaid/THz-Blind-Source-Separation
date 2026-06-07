"""Week 1 integration script — synthetic data, preprocess, and produce all plots."""

import numpy as np

from thz_blind_source_separation import (
    generate_synthetic_thz,
    plot_angle_dependence,
    plot_fft,
    plot_heatmap,
    plot_waveforms_overlay,
    preprocess_waveforms,
)

N_TIMEPOINTS = 512
N_ANGLES = 36
T_START_PS = 0.0
T_END_PS = 10.0
NOISE_LEVEL = 0.05
K = 4

time_axis = np.linspace(T_START_PS, T_END_PS, N_TIMEPOINTS)
angles = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

X, S = generate_synthetic_thz(
    K=K,
    n_angles=N_ANGLES,
    n_timepoints=N_TIMEPOINTS,
    noise_level=NOISE_LEVEL,
    t_start_ps=T_START_PS,
    t_end_ps=T_END_PS,
    rng=42,
)
print(f"Synthetic data: X={X.shape}, S={S.shape}")

preprocessed = preprocess_waveforms(X, time_axis)
print(f"Preprocessed:   shape={preprocessed.shape}")

plot_waveforms_overlay(preprocessed, time_axis, angles=angles)
plot_heatmap(preprocessed, time_axis, angles=angles)
plot_fft(preprocessed[:, 0], time_axis)
plot_angle_dependence(preprocessed, time_axis, angles=angles)

print("\nAll outputs saved: waveforms_overlay.png, heatmap_time_angle.png, fft_analysis.png, angle_dependence.png")
