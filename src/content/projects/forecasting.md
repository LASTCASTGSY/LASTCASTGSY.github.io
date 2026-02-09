---
title: "Multi-step Wave Forecasting"
description: "Autoregressive diffusion-based forecasting for multi-horizon wave height prediction from NDBC buoy observations."
tags: ["forecasting", "time-series", "CSDI", "PyTorch"]
status: "completed"
date: "2024 – 2025"
featured: true
order: 2
---

## TL;DR

Extended the CSDI imputation framework to multi-step forecasting: given a context window of observed wave heights, the model generates probabilistic forecasts for the next 24–72 hours. The key insight is framing forecasting as "imputation of future values" — the future window is treated as missing data conditioned on the past.

## Problem

Wave height forecasting is critical for maritime operations, coastal engineering, and offshore energy. Physics-based models (WaveWatch III, SWAN) are computationally expensive and require atmospheric forcing inputs. Statistical models (ARIMA, Prophet) miss nonlinear dynamics. We want a data-driven probabilistic model that produces sharp, well-calibrated multi-horizon forecasts directly from buoy observations.

## Data

- **Source**: NDBC stations along the US coastline, hourly Hs
- **Context window**: 168 hours (7 days) of past observations
- **Forecast horizon**: 24h, 48h, 72h
- **Split**: Train on 2010–2021, validate on 2022, test on 2023
- **Challenges**: Non-stationarity (seasonal cycles), heavy-tailed wave events, varying station coverage

## Method

The forecasting setup re-uses the CSDI score network but changes the masking strategy:

1. **Context**: The past 168 hours are always observed (conditioning)
2. **Target**: The next H hours are fully masked (to be generated)
3. **Training**: Random masking on historical windows; at inference, the mask is deterministic (past = observed, future = missing)
4. **Sampling**: Draw N=50 samples from the reverse diffusion process to form a predictive distribution

```python
# Forecasting mask construction
def make_forecast_mask(seq_len, context_len, forecast_len):
    mask = torch.ones(seq_len, dtype=torch.bool)
    mask[context_len:context_len + forecast_len] = False
    return mask  # True = observed (context), False = to predict
```

### The t₀ ramp bug

Early models produced forecasts that started with a sharp discontinuity at the context–forecast boundary. The first predicted timestep would jump to the unconditional mean before recovering. Root cause: the model was trained with random masks that rarely placed a missing value immediately after a long observed block. At inference, the deterministic boundary was out-of-distribution.

**Fix**: During training, include a mix of block-missing patterns that simulate the forecast boundary, not just random scattered masks.

## Results

| Horizon | Persistence | LSTM | CSDI (ours) |
|---------|------------|------|-------------|
| 24h RMSE | 0.45 | 0.38 | **0.31** |
| 48h RMSE | 0.72 | 0.55 | **0.46** |
| 72h RMSE | 0.91 | 0.69 | **0.58** |
| 24h CRPS | — | — | **0.18** |

<!-- TODO: Add forecast fan chart — median + quantiles -->
<div class="placeholder-img">📈 Figure: 72h forecast fan chart with 10/50/90 quantiles (coming soon)</div>

## Debugging Notes

- **t₀ discontinuity**: See section above. This was the single hardest bug — it looked like the model was "resetting" at the forecast boundary. Took two weeks of investigation. See blog post: *Debugging a sudden drop at forecast start*.
- **Normalization scope**: Must normalize over the context window only, then apply the same statistics to the forecast window. Global normalization leaks future information.
- **Number of diffusion steps at inference**: 50 steps give near-identical CRPS to 200 steps but run 4× faster.

## Next Steps

- Extend to joint (Hs, Tp, direction) forecasting
- Compare with TimeGrad and TSDiff baselines
- Investigate learned noise schedules for sharper tails

## Links

- [WaveCast2 codebase](https://github.com/your-username/wavecast2) *(update link)*
- [NDBC data](https://www.ndbc.noaa.gov/)
