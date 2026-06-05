"""Week 1 integration script — load, preprocess, and plot a THz sweep."""

import sys
import numpy as np

from thz_blind_source_separation import (
    load_thz_sweep,
    plot_heatmap,
    plot_waveforms_overlay,
    preprocess_waveforms,
)

file_path = sys.argv[1] if len(sys.argv) > 1 else "data/thz_sweep.csv"

waveforms, time_axis = load_thz_sweep(file_path)

preprocessed = preprocess_waveforms(waveforms, time_axis)

n_angles = preprocessed.shape[1]
angles = np.linspace(0, 360, n_angles, endpoint=False)

plot_waveforms_overlay(preprocessed, time_axis, angles=angles)
plot_heatmap(preprocessed, time_axis, angles=angles)

print("All outputs saved: waveforms_overlay.png, heatmap_time_angle.png")
