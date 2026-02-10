---
title: "CRPS, coverage, and honest evaluation for wave forecasts"
description: "Why RMSE alone is insufficient for diffusion-based forecasting, and how we evaluate the full predictive distribution in WaveCast2."
date: "2025-02-05"
tags: ["evaluation", "CRPS", "metrics", "probabilistic"]
draft: false
---

When I first reported results for WaveCast2, I computed RMSE on the sample mean and called it a day. A diffusion model generates 50 forecast samples per window — collapsing those into a single mean and measuring RMSE throws away the most valuable output of the model: its uncertainty.

## The evaluation protocol in WaveCast2

We report four metrics for every forecast horizon (6h, 24h, 48h, 72h), all computed on **un-normalized** WVHT values so the numbers are directly interpretable in meters:

1. **RMSE** (on sample mean) — for comparison with deterministic baselines
2. **MAE** (on sample median) — more robust to heavy-tailed wave events
3. **CRPS** — Continuous Ranked Probability Score, evaluating the full predictive distribution
4. **90% Coverage** — does the 5th–95th percentile interval actually contain 90% of observations?

## What CRPS captures that RMSE doesn't

CRPS is a strictly proper scoring rule that evaluates the predicted CDF against the observed value. For an ensemble of N forecast samples, the empirical CRPS decomposes into two terms:

```
CRPS = E|X - y| - ½ E|X - X'|
```

The first term penalizes bias (how far are the samples from the truth?). The second term rewards sharpness (how tightly concentrated are the samples?). A model that hedges by spreading samples wide gets penalized — you can't game CRPS by just increasing variance.

In WaveCast2, the CRPS calculation uses quantile loss across the sample distribution:

```python
# From utils.py — quantile-based CRPS
# For each quantile q in Q:
#   ρ_q(y - ŷ) = q·max(y-ŷ, 0) + (1-q)·max(ŷ-y, 0)
# CRPS ≈ (1/|Q|) Σ ρ_q(y_i - ŷ_i^(q)) / Σ|y_i|
```

This is normalized by the absolute values of the targets, making it scale-independent and comparable across stations with different wave climates.

## 90% coverage as a calibration check

Coverage answers a simple question: when the model says "I'm 90% confident WVHT will be between 1.2m and 2.8m," is the truth actually inside that interval 90% of the time?

```python
# Empirical coverage at the 90% level
lower = samples.quantile(0.05, dim=0)  # 5th percentile
upper = samples.quantile(0.95, dim=0)  # 95th percentile
coverage = ((truth >= lower) & (truth <= upper)).float().mean()
# Ideal: coverage ≈ 0.90
```

In practice, I've found that:
- **Underdispersed models** (coverage < 90%) are overconfident — they produce tight intervals that miss the truth. This often happens when the model hasn't learned to represent multimodal futures (e.g., "could be calm OR a storm").
- **Overdispersed models** (coverage > 90%) are hedging — they produce wide intervals that always contain the truth but aren't useful for decision-making.

## Why un-normalized metrics matter

Early in the project, I was reporting RMSE on z-scored data, which made numbers look small (~0.06) but impossible to interpret. Is 0.06 good? It depends entirely on the normalization statistics, which change per station and per training run.

WaveCast2 V2 reports all metrics on un-normalized WVHT in meters. If the RMSE at 24h is 0.35m, that means the model is typically off by about 35 centimeters on wave height — a number that's directly meaningful for maritime operations.

## The golden run captures everything

The `--mode golden_run` pipeline automatically computes all four metrics at all four horizons and saves them alongside the model checkpoint and config. This ensures reproducibility — you can always trace back exactly which parameters produced which numbers.

```bash
# Outputs:
#   config.yaml       — exact parameters
#   model.pth         — best checkpoint
#   visualizations/   — forecast plots with context, truth, median, 90% interval
#   metrics.json      — RMSE, MAE, CRPS, coverage per horizon
python exe_wave.py --mode golden_run --station 42001
```

<!-- TODO: Add reliability diagram -->
<div class="placeholder-img">📊 Figure: Reliability diagram — observed coverage vs. nominal coverage across quantile levels (coming soon)</div>

## Summary

If you're building probabilistic forecasts with diffusion models, CRPS and coverage should be your primary metrics. RMSE is fine for comparing against deterministic baselines, but it's blind to the distributional quality that makes diffusion models worth the computational cost. And always report in physical units — your users care about meters of wave height, not standard deviations.
