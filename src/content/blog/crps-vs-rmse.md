---
title: "CRPS vs RMSE: how I evaluate probabilistic forecasts"
description: "Why point metrics aren't enough for diffusion models, and how CRPS captures the quality of a full predictive distribution."
date: "2025-02-05"
tags: ["evaluation", "CRPS", "metrics"]
draft: false
---

When I started working with diffusion-based forecasting, I reported RMSE on the sample mean and called it a day. It took me a while to realize this was leaving most of the model's value on the table. A diffusion model doesn't produce a single forecast — it produces a distribution. Evaluating that distribution with a point metric is like judging a weather forecast only by whether the predicted temperature was correct, ignoring whether it said there was a 90% chance of rain.

## The problem with RMSE for probabilistic models

RMSE measures the error of a single point prediction against the true value. For a diffusion model that generates N=50 samples, you have to somehow collapse those samples into a single number first. The standard approach is to take the sample mean:

```python
forecasts = diffusion_model.sample(context, n_samples=50)  # (50, T, D)
point_forecast = forecasts.mean(dim=0)                       # (T, D)
rmse = ((point_forecast - target) ** 2).mean().sqrt()
```

This works, but it throws away all the distributional information. A model that predicts "the wave height will be between 1.5m and 3.0m with a mode at 2.2m" gets the same RMSE as a model that predicts "2.2m with certainty" — even though the first model is far more useful when the actual height is 2.8m.

## Enter CRPS

The Continuous Ranked Probability Score (CRPS) is a strictly proper scoring rule that evaluates the full predictive CDF against the observed value. Intuitively, it measures how close the predicted distribution is to a point mass at the true value.

For an ensemble of forecasts, the empirical CRPS can be computed as:

```python
def crps_ensemble(forecasts, observation):
    """
    forecasts: (N,) array of ensemble members
    observation: scalar true value
    """
    N = len(forecasts)
    # Term 1: expected absolute error
    mae = np.abs(forecasts - observation).mean()
    # Term 2: expected pairwise spread
    spread = 0
    for i in range(N):
        for j in range(N):
            spread += np.abs(forecasts[i] - forecasts[j])
    spread /= (2 * N * N)
    return mae - spread
```

The first term (MAE between ensemble and observation) penalizes bias. The second term (pairwise spread) rewards sharpness — a model that concentrates its samples tightly around the truth gets a better score than one that spreads them out "just in case."

## CRPS in practice

In my wave forecasting experiments, CRPS revealed things that RMSE couldn't:

**Observation 1**: Two models with nearly identical RMSE (0.46 vs. 0.47) had very different CRPS (0.23 vs. 0.31). The second model was hedging — its sample variance was too large, producing unnecessarily wide prediction intervals. CRPS penalized this; RMSE didn't notice.

**Observation 2**: During storm events, a well-calibrated model with high RMSE can have *better* CRPS than a model with low RMSE but poorly calibrated uncertainty. If the truth is Hs = 4.0m and model A predicts 3.2m ± 0.3m while model B predicts 3.5m ± 1.5m, model A has lower RMSE but model B has better CRPS because its distribution actually covers the true value.

<!-- TODO: Add a visual showing two predictive distributions with different CRPS -->
<div class="placeholder-img">📊 Figure: Two forecasts with similar RMSE but different CRPS — sharp vs. diffuse predictive distributions (coming soon)</div>

## My evaluation protocol

For every model I train, I report four numbers:

1. **RMSE** (on sample mean) — for comparison with deterministic baselines
2. **MAE** (on sample median) — more robust to outliers
3. **CRPS** (on 50-sample ensemble) — distributional quality
4. **Coverage at 90%** — does the 90% prediction interval actually contain 90% of observations?

```python
def evaluate(model, test_loader, n_samples=50):
    metrics = {'rmse': [], 'mae': [], 'crps': [], 'cov90': []}
    for context, target in test_loader:
        samples = model.sample(context, n_samples)  # (N, T, D)
        mean_pred = samples.mean(0)
        median_pred = samples.median(0).values
        lower = samples.quantile(0.05, dim=0)
        upper = samples.quantile(0.95, dim=0)

        metrics['rmse'].append(rmse(mean_pred, target))
        metrics['mae'].append(mae(median_pred, target))
        metrics['crps'].append(crps_ensemble(samples, target))
        metrics['cov90'].append(((target >= lower) & (target <= upper)).float().mean())
    return {k: np.mean(v) for k, v in metrics.items()}
```

## Practical note: CRPS is cheap

One concern I had was computational cost — CRPS involves pairwise comparisons between ensemble members, which is O(N²). In practice, with N=50 samples, this is negligible compared to the cost of generating those samples in the first place. The `properscoring` Python package has an optimized implementation that sorts the ensemble and computes CRPS in O(N log N).

```bash
pip install properscoring
```

```python
from properscoring import crps_ensemble
score = crps_ensemble(observation, ensemble_members)
```

## Summary

If you're building probabilistic forecasts with diffusion models, CRPS should be your primary metric. RMSE is fine for comparing against deterministic baselines, but it's blind to the thing that makes your model special: the quality of its uncertainty.
