---
title: "Diffusion Imputation for Ocean Buoy Data"
description: "CSDI-based conditional score diffusion for reconstructing ~70% missing wave height observations using multi-feature conditioning from NOAA buoy stations."
tags: ["diffusion", "PyTorch", "time-series", "imputation", "CSDI"]
status: "completed"
date: "2024 – 2025"
featured: true
order: 1
---

## TL;DR

NOAA buoy stations record wave height (WVHT) at ~10-minute intervals, but real-world sensor failures and ocean erosion cause roughly **70% of wave height values to be missing**, while auxiliary variables like wind speed, pressure, and temperature remain far more complete. We use a Conditional Score-based Diffusion Model (CSDI) with multi-feature conditioning to reconstruct missing wave data by leveraging these more robust environmental signals.

## Problem

The National Data Buoy Center (NDBC) maintains ocean buoys that record 9 meteorological and oceanographic variables — but wave height, the most operationally important one, is also the most unreliable. Approximately 70% of WVHT records are missing due to sensor malfunctions, ocean erosion, and insufficient maintenance. Classical interpolation methods (linear, spline) ignore the complex temporal dynamics of ocean waves and the information contained in correlated auxiliary variables (WSPD, WDIR, DPD, APD, MWD, PRES, ATMP, DEWP).

## Data

- **Source**: NDBC station 42001 (mid-Gulf of Mexico), standard meteorological data
- **Frequency**: ~10-minute intervals
- **Target variable**: Significant wave height (WVHT, meters)
- **Conditioning variables**: Wind speed, wind direction, dominant period, average period, mean wave direction, atmospheric pressure, air temperature, dew point
- **Missing rate**: ~70% for WVHT; auxiliary variables substantially more complete
- **Temporal split** (strict chronological, no leakage):
  - Train: through December 31, 2022
  - Validation: January 1 – December 31, 2023
  - Test: January 1, 2024 – present
- **Preprocessing**: z-score normalization, pickle serialization via `preprocess_ndbc_data.py`

## Method

The model follows the CSDI architecture (Tashiro et al., NeurIPS 2021):

1. **Input representation**: The buoy data is structured as a `(B, 2, K, L)` tensor — batch, (observed + mask channels), K features, L timesteps
2. **Side information**: Time embeddings and feature embeddings are injected as conditioning
3. **Multi-feature conditioning**: The 8 auxiliary variables (WSPD, PRES, ATMP, etc.) serve as observed conditioning — the model only needs to generate missing WVHT values
4. **Score network**: Stacked residual blocks with `Conv1d`, multi-head self-attention (8 heads), and diffusion time embeddings. Architecture: 4 layers, 64 channels.
5. **Diffusion**: 50-step schedule; the model learns the score function ∇log p(x) and reverses the noise process conditioned on observed entries

```
Input (B, 2, K, L) → Conv1d → ResidualBlocks → Output (B, K, L)
                         ↑
                    Side Info (time + feature embeddings)
```

The key insight is that multi-feature conditioning lets the diffusion model use highly-complete auxiliary signals to guide reconstruction of the sparse target variable — rather than trying to impute WVHT from its own incomplete history alone.

## Results

Multi-feature conditioning significantly improved reconstruction quality:

- **Short-period RMSE**: ~0.06 (with multi-feature conditioning)
- Enhanced reconstruction accuracy compared to single-feature baseline
- More stable uncertainty sampling — the 90% prediction interval shows better calibration

The model produces probabilistic imputations (not just point estimates) by drawing multiple samples from the reverse diffusion process, providing calibrated uncertainty for downstream applications.

<!-- TODO: Add reconstruction visualization -->
<div class="placeholder-img">📊 Figure: WVHT reconstruction — observed (red), imputed (green median), 90% interval (shaded) (coming soon)</div>

## Debugging Notes

- **Double normalization**: Data was being z-scored twice — once in `preprocess_ndbc_data.py` and again inside the model forward pass. This produced suspiciously low RMSE. Caught by inspecting raw predictions before denormalization.
- **Mask channel alignment**: The `(B, 2, K, L)` input packs observed values and a binary mask together. An early bug had the mask inverted for auxiliary features, causing the model to treat complete variables as missing.
- **Diffusion steps at inference**: 50 steps (matching training config) works well; increasing to 200 didn't improve CRPS meaningfully.

## Next Steps

- Extend to multi-station joint imputation
- Benchmark against GP-VAE and BRITS baselines
- Investigate learned noise schedules (cosine vs. linear)

## Links

- [WaveCast2 codebase](https://github.com/LASTCASTGSY/wave_cast2)
- [CSDI paper (Tashiro et al., NeurIPS 2021)](https://arxiv.org/abs/2107.03502)
- [NDBC data](https://www.ndbc.noaa.gov/)
