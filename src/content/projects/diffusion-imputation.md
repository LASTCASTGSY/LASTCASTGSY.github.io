---
title: "Diffusion Imputation"
description: "CSDI-style score-based diffusion for imputing ocean buoy time series with extreme missingness (~80%)."
tags: ["diffusion", "PyTorch", "time-series", "imputation"]
status: "completed"
date: "2024 – 2025"
featured: true
order: 1
---

## TL;DR

Trained a Conditional Score-based Diffusion model (CSDI) to impute missing wave observations from NDBC buoys. The model handles ~80% missing rates — common in real ocean monitoring — and produces calibrated probabilistic imputations, not just point estimates.

## Problem

NDBC buoy stations record wave height, period, and direction, but sensor failures, transmission gaps, and maintenance windows result in severe data loss. Classical interpolation (linear, spline) ignores the temporal structure of ocean waves. We need a generative approach that respects the data distribution and provides uncertainty estimates for downstream forecasting.

## Data

- **Source**: NDBC historical archives, hourly observations
- **Features**: significant wave height (Hs), dominant period (Tp), mean wave direction
- **Missing pattern**: block-missing and scattered-missing, with overall rates between 60–85%
- **Preprocessing**: z-score normalization per feature, sliding windows of 168 hours (7 days)

## Method

The model follows the CSDI architecture (Tashiro et al., 2021):

1. **Forward process**: Gradually add Gaussian noise to observed values over T diffusion steps
2. **Reverse process**: A transformer-based denoiser learns to recover clean signals conditioned on the observed (non-missing) entries
3. **Conditioning**: Observed values are injected at every reverse step — the model only generates values at missing positions
4. **Score network**: Two stacked transformer layers with time-embedding and feature-embedding, operating on the (time × feature) grid

```python
# Simplified reverse diffusion step
def p_sample(model, x_t, t, cond_mask, cond_obs):
    noise_pred = model(x_t, t, cond_mask, cond_obs)
    x_t_minus_1 = (1 / alpha[t].sqrt()) * (
        x_t - (beta[t] / (1 - alpha_bar[t]).sqrt()) * noise_pred
    )
    if t > 0:
        x_t_minus_1 += sigma[t] * torch.randn_like(x_t)
    # Re-inject observed values
    x_t_minus_1[cond_mask] = cond_obs[cond_mask]
    return x_t_minus_1
```

## Results

| Metric | Linear Interp | GP | CSDI (ours) |
|--------|--------------|-----|-------------|
| RMSE   | 0.82         | 0.61| **0.43**    |
| MAE    | 0.64         | 0.47| **0.33**    |
| CRPS   | —            | 0.38| **0.24**    |

> *Values are on held-out test windows with 80% artificial masking. CRPS is computed over 50 posterior samples.*

<!-- TODO: Add imputation visualization — observed vs. imputed with confidence bands -->
<div class="placeholder-img">📊 Figure: Imputation results with 95% confidence intervals (coming soon)</div>

## Debugging Notes

- **Double normalization bug**: Early runs showed suspiciously low RMSE because data was being z-scored twice — once in the data loader and once in the model forward pass. Caught this by inspecting raw predictions before denormalization.
- **Mask leakage**: Initially, the conditioning mask was being applied after noise addition instead of at every step, causing the model to "see through" the noise on observed values. Fixed by re-injecting observations at each reverse step.
- **Diffusion steps**: 50 steps works nearly as well as 200 during inference. Training always uses 200.

## Next Steps

- Integrate learned noise schedules (cosine vs. linear)
- Benchmark against GP-VAE and BRITS baselines
- Test on multi-station joint imputation

## Links

- [CSDI paper (Tashiro et al., 2021)](https://arxiv.org/abs/2107.03502)
- [WaveCast2 codebase](https://github.com/your-username/wavecast2) *(update link)*
