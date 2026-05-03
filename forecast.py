"""Run a weather forecast with the Keisler 2022 GNN model.

Usage:
    uv run forecast.py --init 2020-01-01T00 --steps 40
    uv run forecast.py --init 2020-01-01T00 --steps 40 --timing
    uv run forecast.py --help
"""

import argparse
import logging
import time

import jax
import pandas as pd
import xarray as xr

from keisler_2022.config import Config
from keisler_2022.io import load_arco_era5, load_ecmwf_open_data
from keisler_2022.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("keisler_2022.forecast")
logger.setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a weather forecast with the Keisler 2022 GNN model."
    )
    parser.add_argument(
        "--init", required=True, help="Initialization time, e.g. 2020-01-01T00"
    )
    parser.add_argument(
        "--steps", required=True, type=int, help="Number of 6-hour steps"
    )
    parser.add_argument(
        "--out", default="/tmp/forecast.nc", help="Output NetCDF filename"
    )
    parser.add_argument(
        "--timing", action="store_true", help="Print timing information"
    )
    parser.add_argument(
        "--input",
        default="era5",
        choices=["era5", "opendata"],
        help="Input data source",
    )
    args = parser.parse_args()

    config = Config()

    total_start = time.time()

    # Load initial conditions
    t0_load = time.time()
    init_time = pd.Timestamp(args.init)

    if args.input == "opendata":
        ds = load_ecmwf_open_data(init_time)
    else:
        ds = load_arco_era5(init_time)
    ds = ds.load()
    t_load = time.time() - t0_load

    if args.timing:
        logger.info(f"Loaded initial conditions ({t_load:.1f}s)")

    runner = Runner(verbose=False, config=config)

    # Check if running on CPU only
    devices = jax.devices()
    if all(device.device_kind == "cpu" for device in devices):
        logger.warning(
            "The forecast is being run on CPU only. This will take longer than on a GPU/TPU."
        )

    if args.timing:
        # Time first step separately to get accurate JIT compilation time
        t0_first = time.time()
        ds_first = runner.run(ds, n_steps=1, timing=False)
        t_first = time.time() - t0_first

        # Time remaining steps
        ds_input_next = ds_first.isel(time=-1).expand_dims("time")
        t0_remaining = time.time()
        ds_remaining = runner.run(ds_input_next, n_steps=args.steps - 1, timing=False)
        t_remaining = time.time() - t0_remaining

        # Combine results
        ds_forecast = xr.concat(
            [ds_first, ds_remaining.isel(time=slice(1, None))], dim="time"
        )
        t_forecast = t_first + t_remaining

        if args.steps > 1:
            avg_step_time = t_remaining / (args.steps - 1)
            jit_time = max(0.0, t_first - avg_step_time)
            steps_time = t_forecast - jit_time
    else:
        t0_forecast = time.time()
        ds_forecast = runner.run(ds, n_steps=args.steps, timing=False)
        t_forecast = time.time() - t0_forecast
        jit_time = None
        avg_step_time = None
        steps_time = None

    # Write output
    t0_write = time.time()
    if args.out.endswith(".nc"):
        ds_forecast.to_netcdf(args.out)
    elif args.out.endswith(".zarr"):
        ds_forecast.to_zarr(args.out)
    else:
        raise ValueError(
            f"Output file must have .nc or .zarr extension. Got: {args.out}"
        )
    t_write = time.time() - t0_write

    total_time = time.time() - total_start

    if args.timing:
        logger.info("\nForecast Timing Summary:")
        if args.steps > 1 and jit_time is not None:
            logger.info(f"  JIT: {jit_time:.1f}s")
            logger.info(
                f"  {args.steps} Steps: {steps_time:.1f}s ({avg_step_time:.1f}s per step)"
            )
            logger.info(f"  Total: {t_forecast:.1f}s")
        else:
            logger.info(f"  Total: {t_forecast:.1f}s (JIT included)")
        logger.info(f"\nWrote forecast to {args.out} ({t_write:.1f}s)")
        logger.info(f"\nTotal time: {total_time:.1f}s")
    else:
        logger.info(
            f"Forecast finished in {total_time:.1f} seconds. Output written to {args.out}"
        )


if __name__ == "__main__":
    main()
