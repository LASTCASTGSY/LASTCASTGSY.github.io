---
title: "Debugging a sudden drop at forecast start (t₀)"
description: "How a mismatch between training masks and inference masks caused a systematic discontinuity at the forecast boundary."
date: "2025-01-15"
tags: ["debugging", "diffusion", "forecasting"]
draft: false
---

This is a debugging story. It took me about two weeks to figure out, and the fix was two lines of code.

## The symptom

After training CSDI for multi-step wave forecasting, the model produced forecasts that looked great — except for the very first timestep. At the boundary between the context window (observed past) and the forecast window (generated future), there was a sharp, sudden drop. The predicted value at t₀ would snap toward the unconditional mean, then gradually recover over the next 3–5 steps.

The effect was consistent across test samples. It wasn't random noise — it was a systematic bias at the forecast boundary.

<!-- TODO: Add a plot showing the t₀ discontinuity -->
<div class="placeholder-img">📈 Figure: Forecast with visible t₀ discontinuity — the first predicted step drops sharply before recovering (coming soon)</div>

## What I tried first

My initial hypothesis was that the diffusion scheduler was misconfigured — maybe the noise at the boundary was too high, or the model was struggling with the transition from "clean" observed data to "noisy" generated data. I tried:

- Adjusting the number of diffusion steps at inference (50 → 200 → 500)
- Changing the noise schedule (linear → cosine)
- Adding a linear warmup to the forecast mask

None of these helped. The discontinuity was unchanged.

## The actual cause

The root cause was a **training–inference mismatch in the masking pattern**.

During training, I was using random scattered masks — each timestep had an independent probability of being masked. This meant the model almost never encountered a pattern where a *long contiguous block of observed values was immediately followed by a long contiguous block of missing values*. But that's exactly what happens at inference time:

```
Training mask:  ○ ● ○ ○ ● ● ○ ● ○ ○ ● ○  (scattered)
Inference mask: ○ ○ ○ ○ ○ ○ ● ● ● ● ● ●  (block boundary)
                              ^ t₀
```

The model had never seen the specific boundary pattern it was being asked to generate from.

## The fix

The fix was to augment the training masking strategy with block-missing patterns that simulate the forecast boundary:

```python
def sample_training_mask(seq_len, context_ratio=0.5):
    if random.random() < 0.3:  # 30% of training samples
        # Block mask: simulate forecast boundary
        ctx_len = int(seq_len * random.uniform(0.3, 0.7))
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[ctx_len:] = False
        return mask
    else:
        # Random scattered mask (original behavior)
        return torch.rand(seq_len) > random.uniform(0.2, 0.8)
```

After retraining with this augmented masking, the t₀ discontinuity disappeared completely.

<!-- TODO: Add a before/after comparison plot -->
<div class="placeholder-img">📊 Figure: Before vs. after — the same test case with scattered-only masks vs. augmented masks (coming soon)</div>

## Takeaway

The model is only as good as the distribution of masks it trains on. If the inference-time masking pattern is out-of-distribution relative to training, the model will fail at exactly the points where the pattern diverges. This is a general principle for conditional diffusion models — and it's easy to miss because the loss curves look fine during training.
