---
title: "Transformer vs diffusion for time-series forecasting: what I observed"
description: "Comparing deterministic transformer models with score-based diffusion for wave height forecasting — where each approach wins."
date: "2025-01-28"
tags: ["diffusion", "transformers", "comparison"]
draft: false
---

This isn't a rigorous benchmark paper — it's a collection of observations from training both transformer-based and diffusion-based models on the same wave forecasting dataset. The goal is to document what I've seen, not to make definitive claims.

## Setup

Both models are trained on the same NDBC buoy dataset: 168h context → 72h forecast of significant wave height (Hs). The transformer is a vanilla encoder-decoder with learned positional embeddings, trained with MSE loss. The diffusion model is the CSDI architecture with 200 training steps and 50 inference steps.

## Observation 1: Point accuracy is close

On RMSE alone, the two models are surprisingly close — the transformer is sometimes within 5% of the diffusion model for 24h forecasts. The gap widens at longer horizons (48h, 72h), where the diffusion model's ability to capture multimodal futures seems to help.

```
24h RMSE: Transformer 0.34  |  CSDI 0.31
48h RMSE: Transformer 0.52  |  CSDI 0.46
72h RMSE: Transformer 0.68  |  CSDI 0.58
```

## Observation 2: Calibration is where diffusion wins

The diffusion model produces prediction intervals by drawing multiple samples from the reverse process. These intervals are well-calibrated — the 90% interval actually contains about 90% of observations. The transformer, being deterministic, produces a single point forecast with no natural uncertainty estimate.

You *can* get uncertainty from a transformer via MC dropout or deep ensembles, but in my experiments these intervals were poorly calibrated — too narrow for calm conditions, too wide for storms.

<!-- TODO: Reliability diagram comparing CSDI intervals vs. MC dropout intervals -->
<div class="placeholder-img">📊 Figure: Reliability diagram — observed coverage vs. nominal coverage for CSDI and MC dropout (coming soon)</div>

## Observation 3: The transformer is 10× faster

Training the transformer takes about 2 hours on a single A10G. The diffusion model takes 18–24 hours for comparable convergence. At inference, the transformer produces a forecast in a single forward pass (~5ms). The diffusion model needs 50 sequential denoising steps × 50 samples = 2,500 forward passes for a full predictive distribution.

For real-time operational forecasting, this matters a lot. There are ways to speed up diffusion inference (DDIM, distillation), but the transformer is inherently faster.

## Observation 4: Diffusion handles non-stationarity better

During storm events — where wave heights jump to 3–5× their typical values — the transformer tends to underpredict the peak. Its MSE loss encourages conservative, mean-regressing predictions. The diffusion model, sampling from a learned distribution, can produce rare but plausible extreme trajectories.

```python
# Pseudo-analysis: fraction of storm peaks captured within prediction intervals
storm_events = test_data[test_data['Hs'] > 3.0]
csdi_coverage = (csdi_upper[storm_events] > storm_events['Hs']).mean()
# csdi_coverage ≈ 0.85
```

## Observation 5: Transformers are easier to debug

The transformer training loop is straightforward: forward pass → MSE loss → backward pass. You can inspect gradients, attention maps, and intermediate representations directly. Debugging the diffusion model involves reasoning about noise schedules, reverse process dynamics, and the interaction between conditioning and generation. The t₀ bug I wrote about [in another post](/blog/debugging-t0-drop/) would have been trivial to diagnose in a transformer.

## My current take

I use the diffusion model when I need uncertainty estimates or when the forecast distribution is multimodal (storm vs. calm scenarios). I'd use a transformer for fast, deterministic single-point forecasts where calibration isn't critical. In practice, I think the two approaches are complementary rather than competitive.

<!-- TODO: Side-by-side forecast plots — transformer (single line) vs. diffusion (fan chart) for a storm event -->
<div class="placeholder-img">📈 Figure: Storm event comparison — transformer point forecast vs. CSDI probabilistic forecast (coming soon)</div>
