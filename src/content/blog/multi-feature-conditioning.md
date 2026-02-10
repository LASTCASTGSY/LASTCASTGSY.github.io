---
title: "Why multi-feature conditioning matters for diffusion imputation"
description: "How leveraging complete auxiliary variables (wind, pressure, temperature) dramatically improves reconstruction of sparse wave height data."
date: "2025-01-15"
tags: ["diffusion", "conditioning", "imputation"]
draft: false
---

When I first set up CSDI for wave height imputation, I treated it as a single-variable problem — just reconstruct WVHT from its own incomplete history. The results were mediocre. What changed everything was realizing that the *other* 8 variables on the buoy are almost fully observed.

## The asymmetry in NDBC data

NDBC buoys record 9 variables, but the missingness is wildly asymmetric:

- **WVHT** (significant wave height): ~70% missing — sensor failures, ocean erosion, maintenance gaps
- **WSPD, WDIR, PRES, ATMP, DEWP, DPD, APD, MWD**: substantially more complete

This asymmetry is the key. Wave height doesn't fluctuate independently — it's physically driven by wind speed, wind direction, and atmospheric pressure. If you know it was blowing 25 m/s from the southeast for the last 6 hours, you have strong prior information about what WVHT should be, even if the wave sensor was offline.

## Single-feature vs. multi-feature conditioning

In the CSDI framework, conditioning works through the observed mask. During the reverse diffusion process, observed values are re-injected at every denoising step — the model only generates values at missing positions.

With single-feature conditioning, the model only sees the sparse WVHT observations as anchors. With multi-feature conditioning, it also sees the nearly-complete wind, pressure, and temperature records. These serve as dense conditioning signals that constrain the diffusion to physically plausible wave heights.

```python
# In the CSDI input tensor (B, 2, K, L):
#   Channel 0: observed values (dense for aux vars, sparse for WVHT)
#   Channel 1: binary mask (1 = observed, 0 = missing)
#
# Multi-feature conditioning means K=9 features where
# 8 of them provide near-complete conditioning signal
```

## What I observed

The improvement was substantial — short-period RMSE dropped to approximately 0.06 with multi-feature conditioning, and the uncertainty sampling became much more stable. The 90% prediction intervals actually covered ~90% of observations, instead of being overconfident during calm conditions and underconfident during storms.

The model effectively learned that "high wind speed + low pressure → elevated wave height" without us encoding any physics explicitly. The diffusion process converges faster and produces sharper samples when the auxiliary variables are dense, because the reverse process has more information to condition on at each step.

<!-- TODO: Add comparison — single-feature vs. multi-feature imputation -->
<div class="placeholder-img">📊 Figure: Imputation quality with 1 vs. 9 conditioning features (coming soon)</div>

## Practical takeaway

If you're applying diffusion models to time-series imputation, look for this kind of asymmetry in your data. The target variable might be heavily missing, but correlated auxiliary variables might be nearly complete. Feeding all of them into the conditioning mask gives the model far more to work with during the reverse process. In our case, the auxiliary buoy variables are free — they come from the same data download. The only cost is a slightly larger input tensor.
