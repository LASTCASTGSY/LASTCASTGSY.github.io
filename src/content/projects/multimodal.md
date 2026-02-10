---
title: "Multimodal Wave Forecasting"
description: "Cross-attention fusion of buoy time series and ERA5 atmospheric reanalysis for conditioned probabilistic wave forecasts."
tags: ["multimodal", "ViT", "cross-attention", "ERA5", "PyTorch"]
status: "ongoing"
date: "2025 – present"
featured: true
order: 3
---

## TL;DR

Extending WaveCast2 to condition wave forecasts on gridded atmospheric data (ERA5 reanalysis). A lightweight Vision Transformer encodes ERA5 fields into context tokens, which are injected into the diffusion backbone via cross-attention. Early results suggest that wind and pressure fields significantly improve 48h+ forecasts compared to buoy-only baselines.

## Problem

Buoy-only forecasting is limited by the local observation horizon — a single station cannot capture the upstream atmospheric dynamics that generate swells. Physics-based models solve this by ingesting global wind fields, but at high computational cost. We want to give our diffusion model access to the same upstream information via learned representations of ERA5 grids, without hand-engineering physical relationships.

## Data

- **Time series**: NDBC buoy Hs, Tp (same as forecasting project)
- **ERA5 fields**: 10m wind (u, v), mean sea level pressure, on a regional grid (e.g., 10°×10° around the buoy station)
- **Resolution**: ERA5 at 0.25° spatial, hourly temporal — downsampled to match buoy frequency
- **Fusion challenge**: Aligning a 1D time series with a 3D spatiotemporal tensor (time × lat × lon × channels)

## Method

### Architecture

```
ERA5 grid → [Patch Embed → ViT Encoder → context tokens]
                                              ↓ (cross-attention)
Buoy TS → [CSDI Diffusion Backbone ← cross-attn ← ERA5 context] → Forecast
```

1. **ERA5 encoder**: Lightweight ViT (4 layers, 128-dim, 4 heads) processes each ERA5 snapshot as a set of spatial patches. Temporal snapshots are concatenated along the sequence dimension.
2. **Cross-attention fusion**: Each ResidualBlock in the CSDI backbone includes a cross-attention layer where time-series features (queries) attend to ERA5 context tokens (keys/values).
3. **Classifier-Free Guidance (CFG)**: During training, ERA5 conditioning is randomly dropped (p=0.1) so the model learns both conditional and unconditional generation. At inference, guidance weight w scales the ERA5 influence:

```python
# CFG inference
noise_uncond = model(x_t, t, era5_context=None)
noise_cond = model(x_t, t, era5_context=era5_tokens)
noise_pred = noise_uncond + w * (noise_cond - noise_uncond)
```

### Key design decisions

- **ViT context computed once**: The ERA5 encoder runs once per forward pass. Context tokens are cached and reused across all diffusion steps to avoid O(T × ViT) cost.
- **No pressure features**: After ablation, we found that removing MSLP and keeping only wind (u, v) improved results. Pressure was redundant with wind information and added noise.
- **Cross-attention placement**: Added to every other ResidualBlock (not every block) — a tradeoff between expressiveness and memory.

## Results (Preliminary)

| Model | 48h RMSE | 72h RMSE | 48h CRPS |
|-------|----------|----------|----------|
| CSDI (buoy only) | 0.46 | 0.58 | 0.27 |
| CSDI + ERA5 concat | 0.44 | 0.56 | 0.26 |
| CSDI + ERA5 cross-attn | **0.40** | **0.51** | **0.23** |
| CSDI + ERA5 cross-attn + CFG (w=1.5) | **0.39** | **0.49** | **0.22** |

> *Preliminary numbers on a single test station. Full multi-station evaluation in progress.*

<!-- TODO: Add ablation plot — RMSE vs. guidance weight w -->
<div class="placeholder-img">📊 Figure: RMSE as a function of CFG guidance weight w (coming soon)</div>

<!-- TODO: Add attention map visualization — which ERA5 patches does the model attend to? -->
<div class="placeholder-img">🗺️ Figure: Cross-attention heatmap over ERA5 grid (coming soon)</div>

## Debugging Notes

- **Model ignoring ERA5**: The first integration attempt used simple concatenation of ERA5 features with the time-series embedding. The model learned to ignore ERA5 entirely (ablation showed no difference). Cross-attention fixed this by forcing explicit queries over ERA5 context.
- **Double normalization (again)**: ERA5 data was being normalized in both the data pipeline and inside the ViT encoder. Manifested as flat, uninformative context tokens.
- **Memory scaling**: With cross-attention at every block, training OOM'd on 24GB GPUs for sequences > 336 steps. Reduced to every-other-block and added gradient checkpointing.

## Next Steps

- Full multi-station evaluation (10+ NDBC stations, diverse wave climates)
- Ablation study on ViT depth, patch size, and number of ERA5 variables
- Compare with MicroClimaX-style architectures
- Investigate temporal attention over ERA5 (current approach flattens time into sequence)

## Links

- [ERA5 dataset (Copernicus CDS)](https://cds.climate.copernicus.eu/)
- [MCD-TSF paper (Multimodal Conditional Diffusion)](https://arxiv.org/abs/) *(link TBD)*
- [WaveCast2 codebase](https://github.com/lastcastgsy/wavecast2) *(update link)*
