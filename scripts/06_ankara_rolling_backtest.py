"""Run a rolling +24h point forecast backtest for Ankara.

Defaults:
    init dates: 2026-01-01T00 through 2026-04-29T00
    truth dates: 2026-01-02T00 through 2026-04-30T00
    location: Ankara nearest 1-degree grid point, 40N 33E

This evaluates the pressure-level variables the Keisler model actually
predicts: Z500, T850, U850, and Q850. It writes after every completed day so a
long run can be resumed.

Usage:
    uv run scripts/06_ankara_rolling_backtest.py
    uv run scripts/06_ankara_rolling_backtest.py --max-days 3 --allow-cpu
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

from keisler_2022.benchmarks import EVAL_VARIABLES, skill_from_mse, to_eval_units
from keisler_2022.config import Config
from keisler_2022.io import ARCO_ERA5_ZARR_URL, load_arco_era5_exact_1deg
from keisler_2022.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ankara_rolling_backtest")

DEFAULT_OUT_DIR = Path("results/ankara_2026_daily_backtest")
ANKARA_LAT = 39.9334
ANKARA_LON = 32.8597
LEAD_STEPS = 4
LEAD_HOURS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling Ankara +24h point forecast backtest against ERA5."
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-truth-date",
        default="2026-04-30",
        help="Last truth date to verify. Last init is one day earlier.",
    )
    parser.add_argument("--init-hour", type=int, default=0)
    parser.add_argument("--lat", type=float, default=ANKARA_LAT)
    parser.add_argument("--lon", type=float, default=ANKARA_LON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Optional cap for smoke/partial runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip init dates already present in point_daily_metrics.csv.",
    )
    parser.add_argument(
        "--no-init-cache",
        action="store_true",
        help="Disable /tmp NetCDF cache for repeated ARCO init loads.",
    )
    parser.add_argument(
        "--quiet-runner",
        action="store_true",
        help="Disable per-step runner logs.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run even if JAX only sees CPU devices.",
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
    return {"jax_default_backend": jax.default_backend(), "jax_devices": devices}


def cpu_only() -> bool:
    return all(
        getattr(device, "device_kind", "cpu").lower() == "cpu"
        for device in jax.devices()
    )


def nearest_grid_point(lat: float, lon: float) -> tuple[float, float]:
    lat_grid = np.arange(90, -90.1, -1.0)
    lon_grid = np.arange(0, 360, 1.0)
    grid_lat = float(lat_grid[int(np.argmin(np.abs(lat_grid - lat)))])
    grid_lon = float(lon_grid[int(np.argmin(np.abs(lon_grid - (lon % 360))))])
    return grid_lat, grid_lon


def init_times_for_window(
    start_date: str,
    end_truth_date: str,
    init_hour: int,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date) + pd.Timedelta(hours=init_hour)
    end_truth = pd.Timestamp(end_truth_date) + pd.Timedelta(hours=init_hour)
    end_init = end_truth - pd.Timedelta(hours=LEAD_HOURS)
    return list(pd.date_range(start=start, end=end_init, freq="1D"))


def read_completed_init_times(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as fp:
        return {row["init_time"] for row in csv.DictReader(fp)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_point_truth(
    valid_time: pd.Timestamp,
    grid_lat: float,
    grid_lon: float,
) -> xr.Dataset:
    needed_vars = sorted({item.variable for item in EVAL_VARIABLES})
    needed_levels = sorted({item.level for item in EVAL_VARIABLES})
    ds = xr.open_zarr(ARCO_ERA5_ZARR_URL, chunks=None, storage_options={"token": "anon"})
    ds = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop_era5t"])
    )
    ds_truth = ds[needed_vars].sel(level=needed_levels)
    ds_truth = ds_truth.sel(
        time=[valid_time],
        latitude=[grid_lat],
        longitude=[grid_lon],
        method="nearest",
    )
    ds_truth = ds_truth.assign_coords(latitude=[grid_lat], longitude=[grid_lon])
    return ds_truth.astype(np.float32).load()


def point_value(
    ds: xr.Dataset,
    variable: str,
    level: int,
    time_value: pd.Timestamp,
    grid_lat: float,
    grid_lon: float,
) -> float:
    value = (
        ds[variable]
        .sel(time=time_value, level=level, latitude=grid_lat, longitude=grid_lon)
        .values
    )
    return float(np.asarray(value).item())


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    variables = sorted({str(row["variable"]) for row in rows})
    for variable in variables:
        group = [row for row in rows if row["variable"] == variable]
        errors = np.array([float(row["error"]) for row in group])
        persistence_errors = np.array(
            [float(row["persistence_error"]) for row in group]
        )
        mse = float(np.mean(errors**2))
        persistence_mse = float(np.mean(persistence_errors**2))
        summary.append(
            {
                "variable": variable,
                "unit": group[0]["unit"],
                "n_days": len(group),
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
                "persistence_rmse": float(np.sqrt(persistence_mse)),
                "persistence_mae": float(np.mean(np.abs(persistence_errors))),
                "skill_vs_persistence_mse": skill_from_mse(mse, persistence_mse),
                "max_abs_error": float(np.max(np.abs(errors))),
            }
        )
    return summary


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    timing: dict[str, Any],
) -> None:
    write_csv(out_dir / "point_daily_metrics.csv", rows)
    write_csv(out_dir / "point_summary.csv", summarize(rows))
    with (out_dir / "timing_device.json").open("w", encoding="utf-8") as fp:
        json.dump(timing, fp, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if cpu_only() and not args.allow_cpu:
        raise RuntimeError(
            "JAX only sees CPU devices. Run inside WSL2 with CUDA/JAX GPU support, "
            "or pass --allow-cpu for a slow debug run."
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "point_daily_metrics.csv"

    init_times = init_times_for_window(
        args.start_date,
        args.end_truth_date,
        args.init_hour,
    )
    if args.max_days is not None:
        init_times = init_times[: args.max_days]
    completed = read_completed_init_times(metrics_path) if args.resume else set()
    grid_lat, grid_lon = nearest_grid_point(args.lat, args.lon)

    timing: dict[str, Any] = {
        "benchmark": "ankara_rolling_backtest",
        "start_date": args.start_date,
        "end_truth_date": args.end_truth_date,
        "lead_hours": LEAD_HOURS,
        "requested_location": {"latitude": args.lat, "longitude": args.lon},
        "grid_location": {"latitude": grid_lat, "longitude": grid_lon},
        "n_requested_init_times": len(init_times),
        "device": device_snapshot(),
        "per_init": [],
    }
    total_start = time.perf_counter()
    logger.info("Device snapshot: %s", timing["device"])
    logger.info("Using grid point lat=%s lon=%s", grid_lat, grid_lon)

    t0 = time.perf_counter()
    runner = Runner(verbose=not args.quiet_runner, config=Config())
    timing["runner_setup_seconds"] = time.perf_counter() - t0

    rows = load_existing_rows(metrics_path) if args.resume else []
    for init_time in init_times:
        init_key = init_time.isoformat()
        if init_key in completed:
            logger.info("Skipping completed init %s", init_key)
            continue

        valid_time = init_time + pd.Timedelta(hours=LEAD_HOURS)
        logger.info("Forecasting %s -> %s", init_key, valid_time.isoformat())
        init_timing: dict[str, Any] = {
            "init_time": init_key,
            "valid_time": valid_time.isoformat(),
        }

        t0 = time.perf_counter()
        ds_init = load_arco_era5_exact_1deg(
            init_time,
            cache=not args.no_init_cache,
        ).load()
        init_timing["load_initial_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ds_forecast = runner.run(ds_init, n_steps=LEAD_STEPS)
        init_timing["forecast_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        truth = load_point_truth(valid_time, grid_lat, grid_lon)
        init_timing["truth_point_load_seconds"] = time.perf_counter() - t0

        for eval_var in EVAL_VARIABLES:
            forecast_raw = point_value(
                ds_forecast,
                eval_var.variable,
                eval_var.level,
                valid_time,
                grid_lat,
                grid_lon,
            )
            truth_raw = point_value(
                truth,
                eval_var.variable,
                eval_var.level,
                valid_time,
                grid_lat,
                grid_lon,
            )
            persistence_raw = point_value(
                ds_init,
                eval_var.variable,
                eval_var.level,
                init_time,
                grid_lat,
                grid_lon,
            )
            forecast_value = float(to_eval_units(np.array(forecast_raw), eval_var))
            truth_value = float(to_eval_units(np.array(truth_raw), eval_var))
            persistence_value = float(
                to_eval_units(np.array(persistence_raw), eval_var)
            )
            rows.append(
                {
                    "init_time": init_key,
                    "valid_time": valid_time.isoformat(),
                    "lead_hours": LEAD_HOURS,
                    "requested_latitude": args.lat,
                    "requested_longitude": args.lon,
                    "grid_latitude": grid_lat,
                    "grid_longitude": grid_lon,
                    "variable": eval_var.short_name,
                    "source_variable": eval_var.variable,
                    "level_hpa": eval_var.level,
                    "unit": eval_var.unit,
                    "forecast": forecast_value,
                    "truth": truth_value,
                    "error": forecast_value - truth_value,
                    "abs_error": abs(forecast_value - truth_value),
                    "persistence": persistence_value,
                    "persistence_error": persistence_value - truth_value,
                    "persistence_abs_error": abs(persistence_value - truth_value),
                }
            )

        timing["per_init"].append(init_timing)
        timing["completed_init_times"] = sorted(
            {str(row["init_time"]) for row in rows}
        )
        timing["total_seconds_so_far"] = time.perf_counter() - total_start
        write_outputs(out_dir, rows, timing)

    timing["total_seconds"] = time.perf_counter() - total_start
    write_outputs(out_dir, rows, timing)
    logger.info("Wrote Ankara backtest outputs to %s", out_dir)


if __name__ == "__main__":
    main()
