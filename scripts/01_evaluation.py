"""Evaluate the Keisler 2022 model against ERA5.

Run a forecast initialized from ERA5, compute area-weighted RMSE at each
6-hour lead time, and produce a Q850 comparison figure.

Usage:
    uv run scripts/01_evaluation.py --init 2020-01-01T00
    uv run scripts/01_evaluation.py --init 2020-01-01T00 --steps 20
"""

import argparse
import logging
import time

import cartopy.crs as ccrs
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from keisler_2022.config import Config
from keisler_2022.io import load_arco_era5
from keisler_2022.runner import Runner

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("era5_eval")
logger.setLevel(logging.INFO)

# Variables to evaluate: (short_name, long_name, level)
EVAL_VARS = [
    ("Z500", "geopotential", 500),
    ("T850", "temperature", 850),
    ("U850", "u_component_of_wind", 850),
    ("Q850", "specific_humidity", 850),
]


def load_era5_truth(
    times: list[pd.Timestamp],
    eval_vars: list[tuple[str, str, int]] = EVAL_VARS,
) -> xr.Dataset:
    """Load only the evaluation variables from ERA5 for verification times.

    Instead of fetching all 6 vars × 13 levels, this only grabs the specific
    variable/level combos we need for scoring.  Uses sel (not interp) to
    subsample from 0.25° to 1° since the 1° points are an exact subset.
    """
    needed_vars = sorted({long_name for _, long_name, _ in eval_vars})
    needed_levels = sorted({level for _, _, level in eval_vars})

    lat_1deg = np.arange(90, -90.1, -1.0)
    lon_1deg = np.arange(0, 360, 1.0)

    zarr_url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ds = xr.open_zarr(zarr_url, chunks=None, storage_options=dict(token="anon"))
    ds = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop_era5t"])
    )

    ds = ds[needed_vars].sel(level=needed_levels)

    datasets = []
    for t in times:
        logger.info(f"  Loading ERA5 truth for {t.isoformat()}")
        ds_t = ds.sel(time=[t], latitude=lat_1deg, longitude=lon_1deg, method="nearest")
        ds_t = ds_t.assign_coords(latitude=lat_1deg, longitude=lon_1deg)
        ds_t = ds_t.astype(np.float32).load()
        datasets.append(ds_t)

    return xr.concat(datasets, dim="time")


def area_weighted_rmse(
    forecast: np.ndarray,
    truth: np.ndarray,
    lat_deg: np.ndarray,
) -> float:
    """Compute area-weighted RMSE.

    Area weight for each pixel is cos(latitude), which accounts for the
    convergence of meridians toward the poles on a regular lat/lon grid.

    Parameters
    ----------
    forecast : array, shape (lat, lon)
    truth : array, shape (lat, lon)
    lat_deg : array, shape (lat,)  — latitude in degrees

    Returns
    -------
    Scalar RMSE value.
    """
    weights = np.cos(np.deg2rad(lat_deg))[:, np.newaxis]  # (lat, 1)
    mse = np.sum(weights * (forecast - truth) ** 2) / np.sum(
        weights * np.ones_like(forecast)
    )
    return float(np.sqrt(mse))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Keisler 2022 forecasts against ERA5 truth."
    )
    parser.add_argument(
        "--init",
        required=True,
        help="Initialization time, e.g. 2020-01-01T00",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of 6-hour steps (default: 12 = 3 days)",
    )
    parser.add_argument(
        "--fig",
        default="/tmp/era5_eval_q850.png",
        help="Output path for Q850 figure (default: /tmp/era5_eval_q850.png)",
    )
    args = parser.parse_args()

    init_time = pd.Timestamp(args.init)
    n_steps = args.steps

    # --- Load initial conditions ---
    logger.info(f"Loading initial conditions for {init_time.isoformat()}")
    ds_init = load_arco_era5(init_time)
    ds_init = ds_init.load()

    # --- Run forecast ---
    config = Config()
    runner = Runner(verbose=True, config=config)

    devices = jax.devices()
    if all(d.device_kind == "cpu" for d in devices):
        logger.warning(
            "Running on CPU only — this will be significantly slower than GPU/TPU."
        )

    logger.info(f"Running {n_steps}-step forecast...")
    t0 = time.time()
    ds_forecast = runner.run(ds_init, n_steps=n_steps)
    t_forecast = time.time() - t0
    logger.info(f"Forecast completed in {t_forecast:.1f}s")

    # --- Load ERA5 truth at each verification time ---
    verification_times = [
        init_time + pd.Timedelta(hours=6 * i) for i in range(1, n_steps + 1)
    ]

    logger.info(
        f"Loading ERA5 truth for {len(verification_times)} verification times..."
    )
    t0 = time.time()
    ds_truth = load_era5_truth(verification_times)
    t_truth = time.time() - t0
    logger.info(f"Loaded truth data in {t_truth:.1f}s")

    # --- Compute area-weighted RMSE ---
    lat = ds_forecast.latitude.values  # 90 -> -90

    lead_hours = [6 * i for i in range(1, n_steps + 1)]
    results: dict[str, list[float]] = {name: [] for name, _, _ in EVAL_VARS}

    for t_verif in verification_times:
        for short_name, long_name, level in EVAL_VARS:
            fc = ds_forecast[long_name].sel(level=level, time=t_verif).values
            tr = ds_truth[long_name].sel(level=level, time=t_verif).values
            rmse = area_weighted_rmse(fc, tr, lat)
            results[short_name].append(rmse)

    # --- Print results ---
    print()
    print("=" * 62)
    print(f"  ERA5 Evaluation — Init: {init_time.isoformat()}")
    print("=" * 62)

    header = f"{'Lead (h)':>10}"
    for short_name, _, _ in EVAL_VARS:
        header += f"  {short_name:>12}"
    print(header)
    print("-" * len(header))

    for i, lead_h in enumerate(lead_hours):
        row = f"{lead_h:>10}"
        for short_name, _, _ in EVAL_VARS:
            val = results[short_name][i]
            if val < 0.01:
                row += f"  {val:>12.6f}"
            elif val < 100:
                row += f"  {val:>12.4f}"
            else:
                row += f"  {val:>12.2f}"
        print(row)

    print("-" * len(header))

    print()

    # --- Q850 comparison figure ---
    matplotlib.use("Agg")
    data_crs = ccrs.PlateCarree()
    proj = ccrs.Robinson()
    lons = np.arange(0, 360, 1.0)
    lats = np.arange(90, -90.1, -1.0)

    fig_lead_hours = [6, 24, 48, 72]
    fig_lead_hours = [h for h in fig_lead_hours if h in lead_hours]
    n_rows = len(fig_lead_hours)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, 2.5 * n_rows),
        subplot_kw={"projection": proj},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Collect all Q850 fields to get consistent vmin/vmax
    q_fields: list[np.ndarray] = []
    for h in fig_lead_hours:
        t_verif = init_time + pd.Timedelta(hours=h)
        q_fields.append(
            ds_truth["specific_humidity"].sel(level=850, time=t_verif).values
        )
        q_fields.append(
            ds_forecast["specific_humidity"].sel(level=850, time=t_verif).values
        )
    all_q = np.concatenate([f.ravel() for f in q_fields])
    vmin = float(np.percentile(all_q, 0.2))
    vmax = float(np.percentile(all_q, 99.8))

    for row, h in enumerate(fig_lead_hours):
        t_verif = init_time + pd.Timedelta(hours=h)
        tr = ds_truth["specific_humidity"].sel(level=850, time=t_verif).values
        fc = ds_forecast["specific_humidity"].sel(level=850, time=t_verif).values

        for col, (data, label) in enumerate([(tr, "ERA5"), (fc, "Forecast")]):
            ax = axes[row, col]
            im = ax.pcolormesh(
                lons,
                lats,
                data,
                transform=data_crs,
                cmap="Blues_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.coastlines(linewidth=0.5)
            ax.set_global()
            ax.set_title(f"{label} Q850 (+{h}h)", fontsize=12)

    fig.suptitle(
        f"Q850 — Init: {init_time.isoformat()}", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    cbar_ax = fig.add_axes([0.90, 0.08, 0.015, 0.84])
    fig.colorbar(im, cax=cbar_ax, label="kg/kg")

    fig.savefig(args.fig, dpi=200)
    print(f"Figure saved to {args.fig}")


if __name__ == "__main__":
    main()
