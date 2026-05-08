# Ankara 2026 Rolling +24h Backtest

Run completed on 2026-05-08.

## Scope

- Requested location: Ankara, Turkey (`39.9334N`, `32.8597E`).
- Model grid point used: `40.0N`, `33.0E`.
- Initializations: `2026-01-01T00` through `2026-04-29T00`.
- Verification times: `2026-01-02T00` through `2026-04-30T00`.
- Lead: +24h (`4` autoregressive 6-hour model steps).
- Rows: `119 days * 4 variables = 476`.
- Variables are pressure-level model fields, not surface weather:
  `Z500`, `T850`, `U850`, `Q850`.

## Accuracy Summary

| Variable | Unit | Days | RMSE | MAE | Bias | Persistence RMSE | Skill vs Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z500 | m | 119 | 7.8293 | 6.0773 | -1.5433 | 69.7721 | 0.9874 |
| T850 | K | 119 | 1.1942 | 0.9141 | -0.4493 | 3.4133 | 0.8776 |
| U850 | m/s | 119 | 1.9281 | 1.5626 | 0.0470 | 4.5229 | 0.8183 |
| Q850 | kg/kg | 119 | 0.000460 | 0.000372 | -0.000216 | 0.001033 | 0.8017 |

Skill is `1 - model_mse / persistence_mse`; positive values mean the model beat
the no-change persistence baseline.

## Runtime

The run used JAX GPU backend on the RTX 5090:

```text
jax_default_backend: gpu
device: CudaDevice(id=0), NVIDIA GeForce RTX 5090
```

For the 118 timed resume days after the first smoke day:

| Stage | Mean seconds/day | Median seconds/day | Total seconds |
|---|---:|---:|---:|
| Load global ERA5 initial condition | 562.44 | 479.25 | 66,367.36 |
| Forecast on RTX 5090 | 2.85 | 2.84 | 336.47 |
| Load Ankara ERA5 truth point | 362.51 | 318.45 | 42,776.74 |

The model forecast itself was fast. The wall-clock cost came from repeated
Google ARCO ERA5 Zarr reads, especially loading each global initial condition.

## Artifacts

- `point_daily_metrics.csv`: one row per date and variable.
- `point_summary.csv`: aggregate accuracy summary.
- `timing_device.json`: device snapshot and per-day timing.
- `run.log`: execution log.

