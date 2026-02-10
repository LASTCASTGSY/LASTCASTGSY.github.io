---
title: "From imputation to forecasting: reframing the mask"
description: "How the same CSDI architecture handles both missing data reconstruction and multi-horizon wave forecasting by changing the masking strategy."
date: "2025-01-28"
tags: ["forecasting", "CSDI", "architecture"]
draft: false
---

One of the most elegant things about the CSDI framework is that imputation and forecasting are the same problem with different masks. This wasn't obvious to me when I started — I thought forecasting would require a fundamentally different architecture. It doesn't.

## Imputation mask: random scatter

For imputation, the mask is stochastic. Each timestep-feature pair has some probability of being masked (missing). In our NDBC data, the real-world missingness pattern is a mix of scattered and block-missing:

```
Imputation: ○ ● ○ ○ ● ● ○ ● ○ ○ ● ○ ● ○ ○ ● ● ○
            (○ = observed, ● = missing)
```

The model learns to fill in gaps of varying sizes surrounded by observed anchors.

## Forecasting mask: block boundary

For forecasting, the mask is deterministic. Everything in the context window is observed; everything in the forecast window is missing:

```
Forecasting: ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ● ● ● ●
             |---- 72h context ----|-- H forecast --|
             432 steps @ 10min      36-432 steps
```

The model generates the future conditioned on the past. At 10-minute resolution on NDBC1 data, our four horizons translate to:

| Horizon | Steps |
|---------|-------|
| 6h | 36 |
| 24h | 144 |
| 48h | 288 |
| 72h | 432 |

## The V1 → V2 data leakage lesson

In V1 of WaveCast2, I used ratio-based random splitting — grabbing random windows from the full time range for train/val/test. This is fine for imputation (where the task is filling internal gaps), but catastrophic for forecasting: the model had already seen "future" data during training.

V2 introduced strict chronological splitting:
- Train: through December 31, 2022
- Validation: January 1 – December 31, 2023
- Test: January 1, 2024 – present

This is the single most important change between V1 and V2. RMSE numbers that looked great under random splitting degraded significantly with proper temporal splits — which is exactly what you'd expect when you remove data leakage.

```python
# V1 (bad): random windows from any time period
# V2 (good): strict temporal boundaries
python exe_wave.py --mode forecasting_multi --station 42001
```

## The t₀ boundary problem

Even after fixing the split, the model produced a discontinuity at the context-forecast boundary. The first predicted timestep would snap toward the unconditional mean before recovering over 3–5 steps.

The cause: during imputation training, the model sees random scattered masks. It never encounters a clean block boundary where 432 observed steps are immediately followed by 432 missing steps. At inference, this exact pattern is out-of-distribution.

The fix was to include block-missing patterns during training that simulate the forecast boundary — not just random scatter. After this change, the t₀ discontinuity disappeared.

## Golden runs

WaveCast2 has a `--mode golden_run` option that runs the full pipeline and saves everything needed to reproduce results: the exact `config.yaml`, the best model checkpoint, and high-resolution forecast visualizations showing context (red), truth (blue), median prediction (green), and 90% prediction intervals (shaded).

```bash
python exe_wave.py --mode golden_run --station 42001
```

<!-- TODO: Add golden run visualization -->
<div class="placeholder-img">📈 Figure: Golden run output — 24h forecast with context, truth, and prediction interval (coming soon)</div>

## Takeaway

If you have a working CSDI imputation model, you already have a forecasting model — you just need to change the mask. But be careful about (1) temporal splitting (no leakage!) and (2) training mask distribution (include block masks that match the forecasting pattern). These two changes took WaveCast2 from a toy imputation demo to a real forecasting pipeline.
