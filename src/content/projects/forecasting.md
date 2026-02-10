---
title: "Multi-Horizon Wave Forecasting"
description: "Conditional diffusion model for probabilistic wave height forecasting at 6h/24h/48h/72h horizons using NDBC buoy data at 10-minute resolution."
tags: ["forecasting", "time-series", "CSDI", "PyTorch"]
status: "completed"
date: "2024 – 2025"
featured: true
order: 2
---

## TL;DR

Extended the CSDI imputation framework to multi-horizon forecasting: given a 72-hour context window of buoy observations, the model generates probabilistic forecasts at 6h, 24h, 48h, and 72h horizons (36 to 432 time steps at 10-minute cadence). Forecasting is framed as "imputation of future values" — the future window is treated as missing data conditioned on the observed past.

## Problem

Wave height forecasting is critical for maritime navigation, offshore safety, and recreational fishing. Physics-based models (WaveWatch III, SWAN) are computationally expensive and require full atmospheric forcing. Statistical models (ARIMA, Kalman filters) struggle with the nonlinear, oscillatory, and non-stationary dynamics of ocean waves. We want a data-driven model that produces sharp, calibrated probabilistic forecasts directly from buoy observations — including honest uncertainty estimates for operational decision-making.

## Data

- **Source**: NDBC station 42001 (mid-Gulf of Mexico)
- **Frequency**: ~10-minute intervals (NDBC1 dataset type)
- **Features**: 9 variables — WVHT, WSPD, WDIR, DPD, APD, MWD, PRES, ATMP, DEWP
- **Context window**: 72 hours = 432 time steps
- **Forecast horizons** (at 10-minute cadence):
  - 6h → 36 steps
  - 24h → 144 steps
  - 48h → 288 steps
  - 72h → 432 steps
- **Temporal split** (deterministic, strict chronological):
  - Train: through December 31, 2022
  - Validation: January 1 – December 31, 2023
  - Test: January 1, 2024 – present

## Method

The forecasting mode re-uses the CSDI score network but changes the masking strategy. Instead of randomly masking individual timesteps, the mask has a deterministic block structure:

1. **Context** (observed): The past 432 steps are always unmasked
2. **Target** (to generate): The next H steps are fully masked
3. **Multi-feature conditioning**: All 9 buoy variables in the context window serve as conditioning; the model generates the full feature vector for future steps
4. **Sampling**: Draw N samples from the reverse diffusion process to form a predictive distribution

```python
# Forecasting mask — past is observed, future is missing
# At forecast origin t:
#   Context: buoy data [t - 432 : t]       (observed)
#   Target:  future    [t+1 : t+H]         (to generate)
python exe_wave.py --mode forecasting --forecast_horizon 24 --station 42001
```

### Multi-horizon evaluation

The pipeline supports running all four horizons sequentially via `--mode forecasting_multi`, or a full "golden run" that produces model checkpoints, config snapshots, and high-resolution forecast visualizations:

```bash
python exe_wave.py --mode golden_run --station 42001
```

### Architecture details

- 4 residual layers, 64 channels, 8 attention heads
- 50 diffusion steps
- Training: 200 epochs, batch size 16, learning rate 0.001

## Results

| Metric | 6h | 24h | 48h | 72h |
|--------|-----|------|------|------|
| RMSE | — | — | — | — |
| MAE | — | — | — | — |
| CRPS | — | — | — | — |
| 90% Coverage | — | — | — | — |

> *Results table to be filled from golden run outputs. Each cell reports metrics on un-normalized WVHT.*

### Visualization

Forecast plots from `vis_wave.py` include:
- **Red points**: Observed context data
- **Blue crosses**: Ground truth targets
- **Green line**: Median prediction (50th percentile)
- **Shaded green**: 90% prediction interval (5th–95th percentile)
- **Gray dashed**: Context/horizon boundary

<!-- TODO: Add forecast fan chart from golden run -->
<div class="placeholder-img">📈 Figure: 24h forecast — context (red), truth (blue), median prediction (green), 90% interval (shaded) (coming soon)</div>

## Debugging Notes

- **Temporal split leakage (V1 → V2)**: V1 used ratio-based random splitting, which leaked future data into training. V2 introduced strict chronological boundaries — this is the single most important change between versions.
- **t₀ boundary discontinuity**: The model initially produced a sharp drop at the context-forecast boundary because training used random scattered masks that never exhibited a clean block boundary. Fixed by ensuring the forecasting masking strategy is used during training (not just inference).
- **Normalization scope**: Statistics must be computed on the training set only, then applied to validation and test. Per-window normalization was leaking information.

## Next Steps

- Compare with transformer baselines (Informer, PatchTST)
- Investigate performance on stations with different wave climates
- Test sensitivity to context window length

## Links

- [WaveCast2 codebase](https://github.com/LASTCASTGSY/wave_cast2)
- [NDBC station 42001](https://www.ndbc.noaa.gov/station_page.php?station=42001)
