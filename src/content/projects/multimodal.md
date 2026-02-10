---
title: "Multimodal Forecasting with ERA5 Spatial Covariates"
description: "Cross-attention fusion of NDBC buoy time series and ERA5 atmospheric reanalysis grids (u10, v10, msl) for conditioned probabilistic wave forecasts."
tags: ["multimodal", "ERA5", "cross-attention", "CNN", "PyTorch"]
status: "ongoing"
date: "2025 – present"
featured: true
order: 3
---

## TL;DR

Extending WaveCast2 to condition wave forecasts on **gridded ERA5 atmospheric reanalysis data** — 10m wind components (u10, v10) and mean sea level pressure (msl) on a 0.25° spatial grid centered on the buoy. A 2D CNN spatial encoder compresses each ERA5 snapshot into a context vector, which is fused into the diffusion backbone via **cross-attention**. Early results show that atmospheric spatial context improves forecasts at 48h+ horizons compared to buoy-only baselines.

## Problem

Buoy-only forecasting is limited by the local observation horizon — a single station cannot see the upstream atmospheric dynamics that generate swells. Physics-based models solve this by ingesting global wind fields, but at enormous computational cost. We want the diffusion model to learn from the same atmospheric context through cross-attention, without hand-engineering physical relationships.

## Data

**Buoy time series** (same as forecasting project):
- NDBC station 42001, 9 variables at 10-minute cadence
- 72h context window (432 steps)

**ERA5 reanalysis grids**:
- **Variables**: 10m u-wind (u10), 10m v-wind (v10), mean sea level pressure (msl)
- **Spatial resolution**: 0.25° × 0.25°
- **Grid window**: 3° × 3° or 5° × 5° centered on buoy location
- **Temporal resolution**: Hourly (aligned to buoy timestamps)
- **Temporal constraint (no-cheating rule)**: ERA5 data only from `[t - T_era : t]` — strictly causal, never includes future atmospheric data

```
For forecast origin t:
  Context x:   buoy data  [t - T_ctx : t]        (observed)
  Exogenous:   ERA5 grids [t - T_era : t]        (causal only!)
  Target y:    future buoy [t+1 : t+H]           (to generate)
```

## Method

### Architecture

```
ERA5 Grid (B, T, C, H, W) → 2D CNN SpatialEncoder → (B, T, D)
                                                        ↓ (cross-attention)
Buoy Data → CSDI Diffusion Backbone + CrossAttention → Forecast
```

1. **ERA5 spatial encoder**: A 2D CNN processes each ERA5 snapshot (3 channels × H × W) into a D-dimensional vector. Temporal snapshots are stacked along the sequence dimension, producing `(B, T, D)` context tokens.
2. **Cross-attention fusion**: ResidualBlocks in the CSDI backbone include cross-attention layers where buoy time-series features (queries) attend to ERA5 spatial context (keys/values).
3. **Strict temporal alignment**: ERA5 context is limited to the causal window `[t - T_era : t]`. This is enforced in `Wave_Dataset_ERA5_Forecasting` class.

### Configuration

```yaml
era5:
  enabled: true
  context_hours: 24
  spatial_window_deg: 3.0
  # Variables: u10, v10, msl at 0.25° resolution
```

```bash
python exe_wave.py --mode forecasting --use_era5 --era5_path ./data/era5 --forecast_horizon 24
```

## Results (Preliminary)

| Model | 24h RMSE | 48h RMSE | 72h RMSE |
|-------|----------|----------|----------|
| CSDI (buoy only) | — | — | — |
| CSDI + ERA5 cross-attn | — | — | — |

> *Results to be filled from ongoing evaluation. ERA5 conditioning expected to show largest gains at longer horizons where upstream atmospheric dynamics matter most.*

<!-- TODO: Add comparison plot — buoy-only vs ERA5-conditioned forecasts -->
<div class="placeholder-img">📊 Figure: 48h forecast comparison — buoy-only vs. ERA5-conditioned (coming soon)</div>

<!-- TODO: Add cross-attention heatmap over ERA5 grid -->
<div class="placeholder-img">🗺️ Figure: Cross-attention weights over ERA5 spatial grid — which atmospheric regions does the model attend to? (coming soon)</div>

## Debugging Notes

- **Model ignoring ERA5**: The first attempt used simple concatenation of ERA5 features with the time-series embedding. The model learned to ignore ERA5 entirely (ablation showed zero difference). Cross-attention with a dedicated spatial encoder fixed this by creating an explicit query-key-value pathway.
- **Double normalization (again)**: ERA5 data was being normalized in both the data pipeline and inside the CNN encoder. This produced flat, uninformative context tokens.
- **Memory scaling**: With cross-attention at every residual block, training OOM'd on 24GB GPUs for longer sequences. Reduced to every-other-block placement and added gradient checkpointing.
- **Pressure variable**: After ablation, removing MSLP and keeping only wind (u10, v10) improved results. Wind encodes most of the useful atmospheric signal; pressure was redundant and added noise.

## Future Directions

From the poster presentation: the next step is to **integrate buoy camera imagery using a Vision Transformer (ViT)** as an additional conditioning stream, enabling the model to combine visual and sensor data. We also plan to introduce transformer-based temporal encoders to better capture long-range dependencies and support real-time forecasting.

## Links

- [WaveCast2 codebase](https://github.com/LASTCASTGSY/wave_cast2)
- [ERA5 dataset (Copernicus CDS)](https://cds.climate.copernicus.eu/)
- [CSDI paper (Tashiro et al., NeurIPS 2021)](https://arxiv.org/abs/2107.03502)
