# THz Blind Source Separation — Implementation Plan
### Based on Advisor Feedback (June 2026)

---

## Background & Problem Statement

Real THz time-domain spectroscopy (TDS) waveforms vary systematically with external physical parameters:
- **Angle** (analyser / waveplate rotation)
- **Magnetic field** (magneto-optical Faraday / Kerr effect)
- **Temperature** (thermal phonon/carrier effects)
- …and other sample-specific parameters

Acquiring a full multi-dimensional experimental dataset (varying all of these simultaneously) is currently **not feasible** in the lab. The advisor's direction is therefore to:

> *Generate those waveforms **mathematically** by changing amplitudes and phases systematically. Train the BSS program on these synthetic waveforms to detect amplitude and phase changes — without giving it the amplitude/phase labels directly. Then compare what the program recovers against the known ground truth and check if a functional dependence on the physical parameters can be established.*

The current codebase (`synthetic.py`) already generates synthetic waveforms mixed over angle using a polarimetric model. This plan extends and restructures that work across all three stages below.

---

## High-Level Roadmap

```
Stage 1  →  Stage 2  →  Stage 3
Richer         BSS /         Functional
synthetic      ML model       dependence
data gen.      training       recovery
```

---

## Stage 1 — Richer Synthetic Waveform Generation

### What needs to be done

The existing `generate_synthetic_thz()` only varies angle. We need waveforms that vary **amplitude and phase** as explicit, controlled functions of every physical parameter.

### Mathematical model to implement

A THz monocycle at parameter set `p = (angle α, field B, temperature T, ...)` is:

```
s(t; p) = A(p) · s₀( t − Δτ(p) )
```

where:
- `s₀(t)` is the **reference pulse** (Gaussian derivative, already implemented)
- `A(p)` = parameter-dependent **amplitude scaling**
- `Δτ(p)` = parameter-dependent **time-delay / phase shift**

### Functional dependences to encode

| Parameter | Amplitude law | Phase / delay law |
|-----------|--------------|-------------------|
| Angle α (deg) | `A(α) = A₀ · cos²(α)` (Malus's law) | `Δτ(α) = τ_max · sin(2α)` |
| Magnetic field B (T) | `A(B) = A₀ · (1 + g_A · B)` | `Δτ(B) = φ_B · B` (Faraday rotation proxy) |
| Temperature T (K) | `A(T) = A₀ · exp(−γ · T)` | `Δτ(T) = τ_T · (T / T_ref)` |

These are physically motivated approximations — exact functional forms can be refined later when real data arrives.

### Steps

1. **`src/thz_blind_source_separation/synthetic_parametric.py`** — new file
   - Define a `ParametricTHzDataset` class (or factory function) that accepts a grid of parameter values and returns a 3-D array: `X[time, param_1, param_2, ...]`
   - Each waveform is generated from the reference pulse with the amplitude and phase shift dictated by the parameter combination
   - Store ground-truth `A_true` and `Δτ_true` arrays alongside the dataset so they can be used for validation later

2. **Extend `__init__.py`** to export the new generator

3. **Write a generation script** `scripts/generate_parametric_dataset.py`
   - Sweeps over a grid: e.g., 36 angles × 10 field values × 10 temperature values = 3600 waveforms
   - Saves the dataset to a `.npz` file (`data/synthetic_parametric.npz`) containing:
     - `X` — waveform array
     - `time_axis`
     - `params` — structured array with the value of every physical parameter for each waveform
     - `A_true`, `tau_true` — ground-truth amplitude and phase arrays

4. **Visualization check** — regenerate existing plots for the new parametric dataset to confirm the amplitude/phase trends are visually sensible

---

## Stage 2 — BSS / ML Model Training

### Objective

Feed the model **raw waveforms only** (no amplitude or phase labels) and train it to extract the underlying amplitude and phase for each waveform.

### Approach options (to be decided)

| Option | Method | Rationale |
|--------|--------|-----------|
| A | **ICA (FastICA / JADE)** | Classic BSS; recovers statistically independent sources |
| B | **PCA / SVD** | Linear decomposition baseline; interpretable |
| C | **Autoencoder (VAE or standard)** | Non-linear; latent vector encodes A & Δτ implicitly |
| D | **Supervised regression (CNN/MLP)** | Treat A and Δτ as regression targets; uses ground truth only for evaluation |

> **Recommended starting point:** Option B (SVD/PCA) as a baseline, then Option C (Autoencoder) for the main model, because the advisor wants the model to find the structure without being given the labels.

### Steps

1. **`src/thz_blind_source_separation/bss.py`** — new file
   - Implement `run_svd_bss(X)` → returns component matrix + explained variance
   - Implement `run_ica_bss(X)` → wrapper around `sklearn.decomposition.FastICA`

2. **`src/thz_blind_source_separation/autoencoder.py`** — new file
   - Define a simple 1-D convolutional autoencoder in PyTorch (or alternatively in NumPy/SciPy if keeping dependency-light)
   - **Encoder**: maps a single waveform (length T) → latent vector of dimension 2 (one for amplitude, one for phase, by design pressure or unsupervised)
   - **Decoder**: maps latent vector back to waveform
   - Loss: mean-squared reconstruction error

3. **`scripts/train_model.py`** — training script
   - Load `data/synthetic_parametric.npz`
   - Apply preprocessing (existing `preprocess_waveforms`)
   - Split into train / validation sets (80/20)
   - Train autoencoder; log loss per epoch
   - Save model checkpoint to `models/autoencoder.pt`

4. **`scripts/run_bss.py`** — BSS baseline script
   - Run SVD and ICA on the same dataset
   - Save component waveforms and mixing coefficients

---

## Stage 3 — Comparison & Functional Dependence Recovery

### Objective

Compare the model's output (recovered amplitudes / phases) against `A_true` and `tau_true` from Stage 1. Establish whether the recovered latent variables are smooth, monotonic functions of the physical parameters.

### Steps

1. **`scripts/evaluate_model.py`** — evaluation script
   - Load saved model and the parametric dataset
   - Pass each waveform through the encoder → get latent vector `(z₁, z₂)`
   - Collect `(z₁, z₂)` alongside the true `(A_true, tau_true)` for every waveform

2. **Scatter plots & correlation analysis** — for each physical parameter:
   - Plot `z₁` vs `A_true` → should be linear if the model found amplitude
   - Plot `z₂` vs `tau_true` → should be linear if the model found phase
   - Compute Pearson/Spearman correlation coefficient

3. **Functional curve fitting** — if correlation is high (> 0.9):
   - Fit `z₁ = f(α, B, T)` using `scipy.optimize.curve_fit` with the known functional form
   - Report fitted parameters vs. ground-truth injected parameters

4. **Visualizations to produce**:
   - `latent_vs_angle.png` — z₁, z₂ as a function of angle
   - `latent_vs_field.png` — z₁, z₂ as a function of B
   - `latent_vs_temperature.png` — z₁, z₂ as a function of T
   - `recovery_error_heatmap.png` — |z₁ − A_true| and |z₂ − tau_true| over the parameter grid

5. **`src/thz_blind_source_separation/evaluation.py`** — new file
   - `compute_recovery_metrics(z_latent, A_true, tau_true)` → returns dict of correlation, RMSE, max error
   - `fit_functional_form(z, param_values, form)` → returns fitted parameters

---

## Proposed File / Directory Structure (after all three stages)

```
THz-Blind-Source-Separation/
│
├── IMPLEMENTATION_PLAN.md          ← this document
│
├── data/
│   └── synthetic_parametric.npz    ← generated in Stage 1
│
├── models/
│   └── autoencoder.pt              ← saved in Stage 2
│
├── scripts/
│   ├── generate_parametric_dataset.py   ← Stage 1
│   ├── train_model.py                   ← Stage 2
│   ├── run_bss.py                       ← Stage 2
│   └── evaluate_model.py               ← Stage 3
│
├── src/thz_blind_source_separation/
│   ├── __init__.py                 ← extend exports
│   ├── loader.py                   ← (existing)
│   ├── preprocessing.py            ← (existing)
│   ├── synthetic.py                ← (existing, angle-only)
│   ├── synthetic_parametric.py     ← NEW — Stage 1
│   ├── bss.py                      ← NEW — Stage 2
│   ├── autoencoder.py              ← NEW — Stage 2
│   └── evaluation.py               ← NEW — Stage 3
│
├── week1_main.py                   ← (existing)
└── pyproject.toml                  ← add new deps (torch / sklearn)
```

---

## New Dependencies to Add

| Package | Purpose | Stage |
|---------|---------|-------|
| `scikit-learn` | FastICA, PCA, metrics | 2 |
| `torch` | Autoencoder training | 2 |
| `pandas` | Parameter grid management | 1 |
| `seaborn` | Heatmap visualizations | 3 |

---

## Execution Order

```
Step 1  Implement synthetic_parametric.py  (Stage 1)
Step 2  Run generate_parametric_dataset.py → data/synthetic_parametric.npz
Step 3  Visual sanity check on generated waveforms
Step 4  Implement bss.py + run_bss.py       (Stage 2 — baseline)
Step 5  Implement autoencoder.py + train_model.py (Stage 2 — main)
Step 6  Train and save model
Step 7  Implement evaluation.py + evaluate_model.py (Stage 3)
Step 8  Produce correlation plots & curve fits
Step 9  Write up results — does recovered latent space match injected A(p), τ(p)?
```

---

## Success Criteria

- [ ] Recovered amplitude `z₁` correlates with `A_true` with Pearson r > 0.95 across the parameter grid
- [ ] Recovered phase `z₂` correlates with `tau_true` with Pearson r > 0.95
- [ ] Fitted functional form parameters (e.g., `g_A`, `φ_B`, `γ`) are within 10% of ground-truth injected values
- [ ] All results reproducible with a fixed random seed

---

## Open Questions / Decisions Needed

1. **Latent dimension**: Should the autoencoder bottleneck be forced to dim=2 (explicit amplitude + phase), or should we let it be larger (e.g., 8) and do post-hoc analysis? A dim=2 bottleneck is cleaner for interpretation but may hurt reconstruction if the signal has more complexity.

2. **Model framework**: Use PyTorch (more flexible, heavier dependency) or stick to NumPy/SciPy only (lighter, less expressive)? This affects what can be done with limited lab compute.

3. **Noise level**: Should noise be kept at 5% (current default) or increased to stress-test the BSS recovery?

4. **Parameter grid resolution**: How fine should the grid be? Finer = better training data but larger files and longer compute. A 36×10×10 grid (~3600 waveforms) is a reasonable starting point.

5. **Physical functional forms**: The amplitude/phase laws in the table above are approximate. If you have references for more precise forms (e.g., from the Faraday angle formula), those should be plugged in.
