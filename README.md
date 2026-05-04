# keisler-2022
This repo contains code to run the machine learning weather model described in [Forecasting Global Weather with Graph Neural Networks (Keisler 2022)](https://arxiv.org/abs/2202.07575).

The model uses a three-stage graph neural network: an **Encoder** maps ERA5 lat/lon fields onto an H3 hexagonal mesh, a **Processor** runs 9 rounds of message passing on the mesh, and a **Decoder** maps back to lat/lon to produce a 6-hour forecast update. Autoregressive rollout produces multi-day forecasts.

## Installation

Prerequisites: Python 3.10+ and `uv` installed.

```bash
git clone git@github.com:rkeisler/keisler-2022.git
cd keisler-2022

# CPU-only (default)
uv sync

# GPU with CUDA 12
uv sync --extra cuda12

# Optional: run tests
uv run pytest
```

## Running Forecasts

A 10-day forecast should take about one minute total (load initial conditions, run forecast, save output) on a GPU machine. It will take a bit longer on a CPU machine, e.g. 2 minutes on a 8-vCPU machine.

The model can be initialized from two data sources:

**ERA5 reanalysis** (via [Google ARCO](https://github.com/google-research/arco-era5)) — historical dates, good for evaluation:

```bash
uv run forecast.py --init 2020-01-01T00 --steps 40
```

**ECMWF IFS analysis** (via [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) on AWS) — recent dates, good for near real-time forecasting:

```bash
uv run --extra opendata forecast.py --init 2026-02-15T00 --steps 20 --input opendata
```

Use `--help` to see all available options:

```bash
uv run forecast.py --help
```

## GPU Setup

```bash
uv sync --extra cuda12
```

Verify GPU access:

```bash
uv run python -c "import jax; print('Devices:', jax.devices())"
```

You should see `[CudaDevice(id=0)]` instead of `[CpuDevice(id=0)]`.

**Troubleshooting:** If JAX falls back to CPU, make sure `LD_LIBRARY_PATH` is **not set**.
A pre-existing `LD_LIBRARY_PATH` can cause JAX to find incompatible system CUDA libraries instead of its own pip-bundled versions. To fix, run this command or put it in your `.bashrc`:

```bash
unset LD_LIBRARY_PATH
```

See the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier) for more details.


## Scripts

The `scripts/` directory contains example analysis and visualization scripts.

### `01_evaluation.py` — ERA5 Evaluation

Runs a forecast initialized from ERA5 reanalysis, computes area-weighted RMSE at each 6-hour lead time, and produces a figure of specific humidity at 850 hPa comparing ERA5 truth vs. model forecast.

```bash
uv run --extra scripts scripts/01_evaluation.py --init 2020-01-01T00 --steps 12
```

<img width="500" alt="era5_eval_q850" src="https://github.com/user-attachments/assets/cad2e38f-62e0-4b36-af91-e57cd4669866" />

### `02_sensitivity.py` — Forecast Sensitivity Maps

Computes `d(forecast)/d(initial_conditions)` using JAX autodiff for a chosen target location, field, and lead time, then visualizes the sensitivity maps.

```bash
uv run --extra scripts scripts/02_sensitivity.py --init 2026-01-03T00 --steps 12
```

<img width="500" alt="sensitivity_3day" src="https://github.com/user-attachments/assets/b0451725-e84d-40cc-949b-279a8f54e48f" />

### `03_hurricane.py` — Hurricane Sandy Tracking

Runs an 8-day forecast initialized from ERA5 at 2012-10-23T00 (2012 was a test-set year), tracks Hurricane Sandy's center via the Z1000 minimum, and compares the predicted track against the actual track.

```bash
uv run --extra scripts scripts/03_hurricane.py
```

<img width="500" alt="hurricane" src="https://github.com/user-attachments/assets/c19b13b9-3844-4709-8d96-f32fadd6c76f" />


