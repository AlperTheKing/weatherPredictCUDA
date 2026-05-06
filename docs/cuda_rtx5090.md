# RTX 5090 CUDA Benchmark Workflow

This fork keeps the original Keisler 2022 model intact and adds repeatable
measurement scripts for an RTX 5090 CUDA/JAX setup.

## Why WSL2

JAX GPU wheels are Linux-focused. On this Windows machine the RTX 5090 is
visible from WSL2 Ubuntu, so run CUDA benchmarks there rather than in native
Windows Python.

From PowerShell:

```powershell
wsl -e bash -lc "cd /mnt/e/Projects/weatherPredictCUDA && /usr/lib/wsl/lib/nvidia-smi"
```

## Install

From PowerShell, install `uv` in WSL if it is missing:

```powershell
wsl -e bash -lc "curl -LsSf https://astral.sh/uv/install.sh | sh"
```

Then install the project with CUDA 13 and plotting dependencies. CUDA 13 JAX
wheels require Python 3.11 or newer; the WSL Python on this machine is 3.12.

```powershell
wsl -e bash -lc "export PATH=\"$HOME/.local/bin:$PATH\" && cd /mnt/e/Projects/weatherPredictCUDA && uv sync --extra cuda13 --extra scripts"
```

Verify that JAX sees the GPU:

```powershell
wsl -e bash -lc "export PATH=\"$HOME/.local/bin:$PATH\" && cd /mnt/e/Projects/weatherPredictCUDA && uv run python -c \"import jax; print(jax.default_backend(), jax.devices())\""
```

The backend should be GPU/CUDA and the device list should include the RTX 5090.

## Accuracy Benchmark

The pilot benchmark runs four 2012 initializations for a 10-day horizon and
writes compact CSV/JSON outputs under `results/pilot_2012_cuda5090/`.

```powershell
wsl -e bash -lc "export PATH=\"$HOME/.local/bin:$PATH\" && cd /mnt/e/Projects/weatherPredictCUDA && uv run scripts/04_cuda_accuracy_benchmark.py"
```

Outputs:

- `accuracy_metrics.csv`: one row per init time, lead time, and variable.
- `accuracy_summary.csv`: compact per-init/per-variable summary.
- `timing_device.json`: device snapshot and load/forecast/truth/metric timings.

The metric rows cover `4 dates * 40 leads * 4 variables = 640` rows.

On the first RTX 5090 run, model inference was fast but ARCO/ERA5 data access
was the bottleneck. A one-step smoke run produced `accuracy_smoke_metrics.csv`
and `accuracy_smoke_timing_device.json`; the full 640-row pilot exceeded a
7200 second run limit while loading ERA5 truth for the first initialization.
See `results/pilot_2012_cuda5090/accuracy_attempt_report.md`.

## Sensitivity Benchmark

The default sensitivity target is Istanbul T850 at +72h.

```powershell
wsl -e bash -lc "export PATH=\"$HOME/.local/bin:$PATH\" && cd /mnt/e/Projects/weatherPredictCUDA && uv run scripts/05_cuda_sensitivity_benchmark.py --jax-cache .jax_cache"
```

Outputs:

- `sensitivity_top_locations.csv`: strongest point/channel sensitivities.
- `sensitivity_channel_norms.csv`: weighted channel norms for all 78 inputs.
- `sensitivity_finite_difference.csv`: finite-difference sanity check.
- `sensitivity_normalized.png`: normalized gradient map.
- `sensitivity_physical_display.png`: physical-unit gradient map.
- `sensitivity_timing_device.json`: device snapshot and timing.

## Tests

```powershell
wsl -e bash -lc "export PATH=\"$HOME/.local/bin:$PATH\" && cd /mnt/e/Projects/weatherPredictCUDA && uv run pytest"
```

Some upstream tests are marked `network` or `slow` and may download ERA5 or
WeatherBench data.
