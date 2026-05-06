# Accuracy Benchmark Attempt

Date run: 2026-05-06

## What completed

- WSL2/JAX saw the RTX 5090 as `CudaDevice(id=0)` with default backend `gpu`.
- A one-step smoke benchmark for `2012-01-01T00` completed and wrote valid
  metrics for Z500, T850, U850, and Q850.
- The smoke forecast/JIT segment took 11.9 seconds on the RTX 5090.

## What did not complete

The full pilot command was started:

```bash
uv run scripts/04_cuda_accuracy_benchmark.py --quiet-runner
```

with a 7200 second Linux `timeout`. It reached the first init time and began
loading 40 ERA5 truth verification times, but did not complete the first init
before the 2 hour limit. No full 640-row accuracy CSV was produced.

The observed bottleneck is ARCO/ERA5 Zarr data access, not model inference:

- Smoke initial-condition load: 595.6 seconds
- Smoke one-step forecast/JIT: 11.9 seconds
- Smoke one-time truth load: 534.2 seconds

The benchmark scripts remain ready to run, but the full pilot needs either a
nearer/faster ERA5 mirror, pre-staged NetCDF/Zarr truth data, or a much longer
unattended run window.

