"""Hurricane Sandy forecast and tracking with the Keisler 2022 GNN model.

Run an 8-day forecast initialized from ERA5 at 2012-10-23T00, track the
hurricane center via Z1000 (geopotential height at 1000 hPa), and compare
against the IBTrACS best track.

Usage:
    uv run scripts/03_hurricane.py
    uv run scripts/03_hurricane.py --steps 4  # faster for dev (1 day)
    uv run scripts/03_hurricane.py --no-best-track  # skip IBTrACS fetch
"""

import argparse
import logging
import os
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
logger = logging.getLogger("hurricane")
logger.setLevel(logging.INFO)

GRAVITY = 9.80665

# Approximate position of Sandy at 2012-10-23T00 (from IBTrACS)
SANDY_INIT_LAT = 17.5
SANDY_INIT_LON = 282.0  # 0-360 convention (i.e. 78W)


def fetch_best_track(
    forecast_times: list[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fetch IBTrACS best track for Hurricane Sandy (2012).

    Downloads the North Atlantic basin CSV from NOAA, filters for Sandy,
    and interpolates lat/lon to the given forecast valid times.

    Returns
    -------
    (lats, lons) arrays aligned with *forecast_times* (0-360 lon convention),
    or None if the download fails.
    """
    url = (
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for"
        "-climate-stewardship-ibtracs/v04r00/access/csv/"
        "ibtracs.NA.list.v04r00.csv"
    )
    try:
        logger.info("Downloading IBTrACS North Atlantic data (~55 MB)...")
        df = pd.read_csv(url, skiprows=[1], low_memory=False)

        mask = (df["NAME"] == "SANDY") & (df["SEASON"] == 2012)
        sandy = df.loc[mask].copy()
        if sandy.empty:
            logger.warning("Sandy 2012 not found in IBTrACS data")
            return None

        sandy["time"] = pd.to_datetime(sandy["ISO_TIME"])
        sandy["lat"] = pd.to_numeric(sandy["LAT"], errors="coerce")
        sandy["lon"] = pd.to_numeric(sandy["LON"], errors="coerce") % 360
        sandy = sandy.set_index("time")[["lat", "lon"]].sort_index()

        # Interpolate to forecast valid times
        fc_idx = pd.DatetimeIndex(forecast_times)
        all_idx = sandy.index.union(fc_idx)
        sandy = sandy.reindex(all_idx).interpolate(method="time")
        sandy = sandy.reindex(fc_idx)

        lats = sandy["lat"].values.astype(float)
        lons = sandy["lon"].values.astype(float)

        n_valid = int(np.isfinite(lats).sum())
        logger.info(f"IBTrACS: {n_valid}/{len(lats)} valid positions for Sandy")
        if n_valid == 0:
            return None
        return lats, lons

    except Exception as e:
        logger.warning(f"Failed to fetch IBTrACS data: {e}")
        return None


def track_center(
    ds_forecast,
    init_lat: float = SANDY_INIT_LAT,
    init_lon: float = SANDY_INIT_LON,
    search_radius: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Track the hurricane center as the Z1000 minimum at each forecast step.

    Uses a search-box approach: at each step, find the minimum geopotential
    height at 1000 hPa within *search_radius* degrees of the previous center.

    Returns
    -------
    (lats, lons) of the tracked center at each forecast time.
    """
    lats_grid = ds_forecast.latitude.values  # 90 to -90
    lons_grid = ds_forecast.longitude.values  # 0 to 359
    times = ds_forecast.time.values

    center_lats = np.zeros(len(times))
    center_lons = np.zeros(len(times))
    prev_lat, prev_lon = init_lat, init_lon

    for i, t in enumerate(times):
        z1000 = ds_forecast["geopotential"].sel(level=1000, time=t).values / GRAVITY

        lat_mask = (lats_grid >= prev_lat - search_radius) & (
            lats_grid <= prev_lat + search_radius
        )
        lon_lo = prev_lon - search_radius
        lon_hi = prev_lon + search_radius
        if lon_lo < 0:
            lon_mask = (lons_grid >= lon_lo % 360) | (lons_grid <= lon_hi)
        elif lon_hi >= 360:
            lon_mask = (lons_grid >= lon_lo) | (lons_grid <= lon_hi % 360)
        else:
            lon_mask = (lons_grid >= lon_lo) & (lons_grid <= lon_hi)

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

        if len(lat_idx) == 0 or len(lon_idx) == 0:
            center_lats[i], center_lons[i] = prev_lat, prev_lon
            continue

        sub = z1000[np.ix_(lat_idx, lon_idx)]
        min_pos = np.unravel_index(np.argmin(sub), sub.shape)
        center_lats[i] = lats_grid[lat_idx[min_pos[0]]]
        center_lons[i] = lons_grid[lon_idx[min_pos[1]]]
        prev_lat, prev_lon = center_lats[i], center_lons[i]

    return center_lats, center_lons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hurricane Sandy forecast and tracking."
    )
    parser.add_argument(
        "--init",
        default="2012-10-23T00",
        help="Initialization time (default: 2012-10-23T00)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of 6-hour steps (default: 32 = 8 days)",
    )
    parser.add_argument(
        "--fig",
        default="/tmp/hurricane.png",
        help="Output figure path (default: /tmp/hurricane.png)",
    )
    parser.add_argument(
        "--no-best-track",
        action="store_true",
        help="Skip IBTrACS best-track download",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Directory to cache forecast netCDF (e.g. /tmp)",
    )
    args = parser.parse_args()

    init_time = pd.Timestamp(args.init)
    n_steps = args.steps

    # --- Run (or load cached) forecast ---
    cache_path = None
    if args.cache:
        cache_path = os.path.join(
            args.cache,
            f"hurricane_{init_time.strftime('%Y%m%dT%H')}_n{n_steps}.nc",
        )

    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached forecast from {cache_path}")
        ds_forecast = xr.open_dataset(cache_path)
    else:
        logger.info(f"Loading initial conditions for {init_time.isoformat()}")
        ds_init = load_arco_era5(init_time)
        ds_init = ds_init.load()

        config = Config()
        runner = Runner(verbose=True, config=config)

        devices = jax.devices()
        if all(d.device_kind == "cpu" for d in devices):
            logger.warning(
                "Running on CPU only -- this will be significantly slower than GPU/TPU."
            )

        logger.info(f"Running {n_steps}-step forecast...")
        t0 = time.time()
        ds_forecast = runner.run(ds_init, n_steps=n_steps)
        t_forecast = time.time() - t0
        logger.info(f"Forecast completed in {t_forecast:.1f}s")

        if cache_path:
            ds_forecast.to_netcdf(cache_path)
            logger.info(f"Cached forecast to {cache_path}")

    # --- Track hurricane center via Z1000 minimum ---
    logger.info("Tracking hurricane center via Z1000 minimum...")
    fc_lats, fc_lons = track_center(ds_forecast)

    forecast_times = [pd.Timestamp(t) for t in ds_forecast.time.values]
    for i, t in enumerate(forecast_times):
        lead_h = (t - init_time).total_seconds() / 3600
        logger.info(f"  +{lead_h:5.0f}h  lat={fc_lats[i]:6.1f}  lon={fc_lons[i]:6.1f}")

    # --- Fetch IBTrACS best track (optional) ---
    best_track = None
    if not args.no_best_track:
        best_track = fetch_best_track(forecast_times)

    # --- Figure: 2x3 panels ---
    # Panels 1-5: Z1000 at lead = 0, 2, 4, 6, 8 days (no tracks)
    # Panel 6: Z1000 at 8-day lead with both tracks overlaid
    matplotlib.use("Agg")
    data_crs = ccrs.PlateCarree()
    proj = ccrs.Orthographic(central_longitude=280, central_latitude=27)
    lons = np.arange(0, 360, 1.0)
    lats = np.arange(90, -90.1, -1.0)

    lead_days = [0, 2, 4, 6, 8]
    step_indices = [d * 4 for d in lead_days]
    step_indices = [s for s in step_indices if s < len(forecast_times)]

    # Geopotential height at 1000 hPa (m)
    plot_fields = []
    for s in step_indices:
        t = forecast_times[s]
        plot_fields.append(
            ds_forecast["geopotential"].sel(level=1000, time=t).values / GRAVITY
        )

    # Consistent vmin/vmax across all panels
    all_vals = np.concatenate([f.ravel() for f in plot_fields])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 85))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        subplot_kw={"projection": proj},
    )
    axes_flat = axes.ravel()

    # Panels 1-5: Z1000 fields with forecast track
    for ax, step_idx, field in zip(axes_flat[:5], step_indices, plot_fields):
        lead_h = int(step_idx * 6)
        lead_d = lead_h // 24

        ax.pcolormesh(
            lons,
            lats,
            field,
            transform=data_crs,
            cmap="Greys_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.coastlines(linewidth=0.5)
        ax.set_global()

        # Forecast track up to this lead time
        ax.plot(
            fc_lons[: step_idx + 1],
            fc_lats[: step_idx + 1],
            "o-",
            color="crimson",
            markersize=3,
            linewidth=1.5,
            transform=data_crs,
        )
        ax.plot(
            fc_lons[step_idx],
            fc_lats[step_idx],
            "o",
            color="crimson",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1,
            transform=data_crs,
        )

        valid_time = forecast_times[step_idx].strftime("%Y-%m-%dT%H")
        ax.set_title(f"{lead_d}-day lead\nValid: {valid_time}", fontsize=12)

    # Panel 6: 8-day lead with both tracks
    last_step = step_indices[-1]
    ax6 = axes_flat[5]
    ax6.pcolormesh(
        lons,
        lats,
        plot_fields[-1],
        transform=data_crs,
        cmap="Greys_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax6.coastlines(linewidth=0.5)
    ax6.set_global()

    ax6.plot(
        fc_lons[: last_step + 1],
        fc_lats[: last_step + 1],
        "o-",
        color="crimson",
        markersize=3,
        linewidth=1.5,
        transform=data_crs,
        label="K22 Forecast",
    )
    ax6.plot(
        fc_lons[last_step],
        fc_lats[last_step],
        "o",
        color="crimson",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
        transform=data_crs,
    )

    if best_track is not None:
        bt_lats, bt_lons = best_track
        valid = np.isfinite(bt_lats[: last_step + 1]) & np.isfinite(
            bt_lons[: last_step + 1]
        )
        if valid.any():
            ax6.plot(
                bt_lons[: last_step + 1][valid],
                bt_lats[: last_step + 1][valid],
                "o-",
                color="dodgerblue",
                markersize=3,
                linewidth=1.5,
                transform=data_crs,
                label="Actual (IBTrACS)",
            )
            if last_step < len(bt_lats) and np.isfinite(bt_lats[last_step]):
                ax6.plot(
                    bt_lons[last_step],
                    bt_lats[last_step],
                    "o",
                    color="dodgerblue",
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    transform=data_crs,
                )

    ax6.legend(loc="lower left", fontsize=8)
    valid_time_8d = forecast_times[last_step].strftime("%Y-%m-%dT%H")
    ax6.set_title(f"8-day lead + tracks\nValid: {valid_time_8d}", fontsize=12)

    fig.suptitle(
        f"Hurricane Sandy — 8-Day K22 Forecast Initialized at {init_time.strftime('%Y-%m-%dT%H')}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.fig, dpi=200)
    print(f"Figure saved to {args.fig}")


if __name__ == "__main__":
    main()
