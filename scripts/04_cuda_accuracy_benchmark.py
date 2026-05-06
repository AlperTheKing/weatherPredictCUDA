"""Run a 10-day ERA5 accuracy benchmark for the Keisler 2022 model.

The default pilot evaluates four 2012 initialization dates, one per season,
against ERA5 truth. It writes compact CSV/JSON outputs and does not save the
large forecast fields.

Usage:
    uv run scripts/04_cuda_accuracy_benchmark.py
    uv run scripts/04_cuda_accuracy_benchmark.py --steps 40 --allow-cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

from keisler_2022.benchmarks import (
    EVAL_VARIABLES,
    area_weighted_metrics,
    require_finite_metrics,
    skill_from_mse,
    to_eval_units,
)
from keisler_2022.config import Config
from keisler_2022.io import load_arco_era5
from keisler_2022.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cuda_accuracy_benchmark")

DEFAULT_INIT_TIMES = (
    "2012-01-01T00",
    "2012-04-01T00",
    "2012-07-01T00",
    "2012-10-01T00",
)
DEFAULT_OUT_DIR = Path("results/pilot_2012_cuda5090")
TRUTH_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 10-day Keisler 2022 forecasts against ERA5 truth."
    )
    parser.add_argument(
        "--init",
        action="append",
        dest="init_times",
        help="Initialization time. Repeat to override the four-date pilot.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of 6-hour forecast steps (default: 40 = 10 days).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run even if JAX only sees CPU devices.",
    )
    parser.add_argument(
        "--quiet-runner",
        action="store_true",
        help="Disable per-step runner logs.",
    )
    return parser.parse_args()


def device_snapshot() -> dict[str, Any]:
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "repr": str(device),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "id": getattr(device, "id", None),
            }
        )
    return {
        "jax_default_backend": jax.default_backend(),
        "jax_devices": devices,
    }


def cpu_only() -> bool:
    return all(getattr(device, "device_kind", "cpu").lower() == "cpu" for device in jax.devices())


def load_era5_truth(
    times: list[pd.Timestamp],
) -> xr.Dataset:
    """Load benchmark variables from ERA5 at verification times."""
    needed_vars = sorted({item.variable for item in EVAL_VARIABLES})
    needed_levels = sorted({item.level for item in EVAL_VARIABLES})

    lat_1deg = np.arange(90, -90.1, -1.0)
    lon_1deg = np.arange(0, 360, 1.0)

    ds = xr.open_zarr(TRUTH_ZARR_URL, chunks=None, storage_options={"token": "anon"})
    ds = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop_era5t"])
    )
    ds = ds[needed_vars].sel(level=needed_levels)

    logger.info("Loading ERA5 truth for %s verification times", len(times))
    ds_truth = ds.sel(
        time=list(times),
        latitude=lat_1deg,
        longitude=lon_1deg,
        method="nearest",
    )
    ds_truth = ds_truth.assign_coords(latitude=lat_1deg, longitude=lon_1deg)
    return ds_truth.astype(np.float32).load()


def metric_rows_for_init(
    init_time: pd.Timestamp,
    ds_init: xr.Dataset,
    ds_forecast: xr.Dataset,
    ds_truth: xr.Dataset,
    steps: int,
) -> list[dict[str, object]]:
    lat = ds_forecast.latitude.values
    rows: list[dict[str, object]] = []

    for step in range(1, steps + 1):
        lead_hours = 6 * step
        valid_time = init_time + pd.Timedelta(hours=lead_hours)
        for eval_var in EVAL_VARIABLES:
            forecast = to_eval_units(
                ds_forecast[eval_var.variable]
                .sel(level=eval_var.level, time=valid_time)
                .values,
                eval_var,
            )
            truth = to_eval_units(
                ds_truth[eval_var.variable]
                .sel(level=eval_var.level, time=valid_time)
                .values,
                eval_var,
            )
            persistence = to_eval_units(
                ds_init[eval_var.variable]
                .sel(level=eval_var.level)
                .isel(time=0)
                .values,
                eval_var,
            )

            model_metrics = area_weighted_metrics(forecast, truth, lat)
            persistence_metrics = area_weighted_metrics(persistence, truth, lat)
            rows.append(
                {
                    "init_time": init_time.isoformat(),
                    "valid_time": valid_time.isoformat(),
                    "lead_hours": lead_hours,
                    "variable": eval_var.short_name,
                    "source_variable": eval_var.variable,
                    "level_hpa": eval_var.level,
                    "unit": eval_var.unit,
                    "rmse": model_metrics["rmse"],
                    "mae": model_metrics["mae"],
                    "bias": model_metrics["bias"],
                    "mse": model_metrics["mse"],
                    "persistence_rmse": persistence_metrics["rmse"],
                    "persistence_mae": persistence_metrics["mae"],
                    "persistence_mse": persistence_metrics["mse"],
                    "skill_vs_persistence_mse": skill_from_mse(
                        model_metrics["mse"], persistence_metrics["mse"]
                    ),
                }
            )

    require_finite_metrics(rows)
    return rows


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    keys = sorted({(row["init_time"], row["variable"]) for row in rows})
    for init_time, variable in keys:
        group = [
            row
            for row in rows
            if row["init_time"] == init_time and row["variable"] == variable
        ]
        group = sorted(group, key=lambda row: int(row["lead_hours"]))
        rmse = np.array([float(row["rmse"]) for row in group])
        skill = np.array([float(row["skill_vs_persistence_mse"]) for row in group])
        final = group[-1]
        summary.append(
            {
                "init_time": init_time,
                "variable": variable,
                "unit": final["unit"],
                "n_leads": len(group),
                "mean_rmse": float(np.mean(rmse)),
                "max_rmse": float(np.max(rmse)),
                "final_lead_hours": final["lead_hours"],
                "final_rmse": final["rmse"],
                "mean_skill_vs_persistence_mse": float(np.mean(skill)),
                "final_skill_vs_persistence_mse": final[
                    "skill_vs_persistence_mse"
                ],
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    out_dir: Path,
    metric_rows: list[dict[str, object]],
    timing: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "accuracy_metrics.csv", metric_rows)
    write_csv(out_dir / "accuracy_summary.csv", summarize_metrics(metric_rows))
    with (out_dir / "timing_device.json").open("w", encoding="utf-8") as fp:
        json.dump(timing, fp, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    init_times = [pd.Timestamp(item) for item in (args.init_times or DEFAULT_INIT_TIMES)]

    if cpu_only() and not args.allow_cpu:
        raise RuntimeError(
            "JAX only sees CPU devices. Run inside WSL2 with CUDA/JAX GPU support, "
            "or pass --allow-cpu for a slow debug run."
        )

    total_start = time.perf_counter()
    timing: dict[str, Any] = {
        "benchmark": "cuda_accuracy",
        "init_times": [item.isoformat() for item in init_times],
        "steps": args.steps,
        "device": device_snapshot(),
        "per_init": [],
    }

    logger.info("Device snapshot: %s", timing["device"])
    t0 = time.perf_counter()
    runner = Runner(verbose=not args.quiet_runner, config=Config())
    timing["runner_setup_seconds"] = time.perf_counter() - t0

    metric_rows: list[dict[str, object]] = []
    for init_time in init_times:
        logger.info("Starting benchmark init %s", init_time.isoformat())
        init_timing: dict[str, Any] = {"init_time": init_time.isoformat()}

        t0 = time.perf_counter()
        ds_init = load_arco_era5(init_time, cache=True).load()
        init_timing["load_initial_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ds_forecast = runner.run(ds_init, n_steps=args.steps)
        init_timing["forecast_jit_seconds"] = time.perf_counter() - t0

        verification_times = [
            init_time + pd.Timedelta(hours=6 * step)
            for step in range(1, args.steps + 1)
        ]
        t0 = time.perf_counter()
        ds_truth = load_era5_truth(verification_times)
        init_timing["truth_load_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        metric_rows.extend(
            metric_rows_for_init(
                init_time=init_time,
                ds_init=ds_init,
                ds_forecast=ds_forecast,
                ds_truth=ds_truth,
                steps=args.steps,
            )
        )
        init_timing["metrics_seconds"] = time.perf_counter() - t0
        timing["per_init"].append(init_timing)

        t0 = time.perf_counter()
        write_outputs(args.out_dir, metric_rows, timing)
        init_timing["write_seconds"] = time.perf_counter() - t0
        logger.info("Finished init %s", init_time.isoformat())

    timing["total_seconds"] = time.perf_counter() - total_start
    write_outputs(args.out_dir, metric_rows, timing)
    logger.info("Wrote benchmark outputs to %s", args.out_dir)


if __name__ == "__main__":
    main()
