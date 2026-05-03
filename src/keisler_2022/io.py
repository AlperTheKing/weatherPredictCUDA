import os
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

# Contract: Loaders return an xr.Dataset with
# - data_vars exactly {geopotential, specific_humidity, temperature, u_component_of_wind, v_component_of_wind, vertical_velocity}
# - dims: (time=1, level=13, latitude=181, longitude=360)
# - latitude: 90 -> -90 (descending), longitude: 0..359 (ascending)
REQUIRED_VARS = (
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
)
REQUIRED_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
GRAVITY = 9.80665  # m/s^2


def load_arco_era5(time_use: pd.Timestamp | str, *, cache: bool = False) -> xr.Dataset:
    """
    Load ERA5 fields from Google ARCO to the Keisler22 input contract.

    Returns an xr.Dataset with variables/dims as specified by the contract above
    for the single initialization time `time_use` (ISO8601 string or pd.Timestamp).
    """
    time_use = pd.Timestamp(time_use)
    savename = f"/tmp/era5_from_arco_{time_use.strftime('%Y%m%d_%H%M%S')}.nc"
    if cache and os.path.exists(savename):
        return xr.open_dataset(savename)

    ds: xr.Dataset = xr.open_dataset(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        engine="zarr",
        decode_timedelta=False,
    )

    ds = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop_era5t"])
    )

    ds = ds[list(REQUIRED_VARS)]
    ds = ds.sel(level=list(REQUIRED_LEVELS))

    # Normalize time selection; raises KeyError if missing
    ds = ds.sel(time=[time_use])

    # Interpolate to 1.0° grid
    new_latitude = np.arange(-90, 90.1, 1.0)
    new_longitude = np.arange(0, 360, 1.0)
    ds = ds.interp(latitude=new_latitude, longitude=new_longitude, method="linear")

    # Ensure latitude descending (90 -> -90)
    if ds.latitude.values[0] < ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Ensure exact coordinate values
    ds = ds.assign_coords(
        latitude=np.arange(90, -90.1, -1.0), longitude=np.arange(0, 360, 1.0)
    )

    ds = ds.astype(np.float32)

    if cache:
        ds.to_netcdf(savename)

    return ds


def load_ecmwf_open_data(
    init_time: pd.Timestamp | str,
    source: Literal["aws"] = "aws",
) -> xr.Dataset:
    """Load IFS analysis fields from ECMWF Open Data into the Keisler22 input contract.

    Downloads the 0h operational forecast GRIB2 file from the ECMWF Open Data
    S3 bucket, extracts pressure-level fields, and returns an xr.Dataset
    conforming to the standard contract (see module docstring).

    Requires the ``opendata`` extra: ``uv sync --extra opendata``

    Parameters
    ----------
    init_time : str or pd.Timestamp
        Initialization time, e.g. "2026-02-15T00" (must be 00z or 12z).
    source : {"aws"}
        Cloud source for the data.
    """
    try:
        import cfgrib  # noqa: F401
        import s3fs
    except ImportError as e:
        raise ImportError(
            "cfgrib and s3fs are required for ECMWF Open Data. "
            "Install with: uv sync --extra opendata"
        ) from e

    init_time = pd.Timestamp(init_time)

    s3_buckets = {"aws": "ecmwf-forecasts"}
    grib_rename = {
        "t": "temperature",
        "q": "specific_humidity",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "w": "vertical_velocity",
        "gh": "geopotential",
    }

    # Build S3 path: {bucket}/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDD}{HH}0000-0h-oper-fc.grib2
    bucket = s3_buckets[source]
    date_str = init_time.strftime("%Y%m%d")
    hour_str = init_time.strftime("%H")
    stream = "oper" if init_time.hour in (0, 12) else "scda"
    filename = f"{date_str}{hour_str}0000-0h-{stream}-fc.grib2"
    s3_path = f"{bucket}/{date_str}/{hour_str}z/ifs/0p25/{stream}/{filename}"

    # Download GRIB2 to temp file (cfgrib/eccodes requires a local file path)
    local_grib = f"/tmp/{filename}"
    if not os.path.exists(local_grib):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(s3_path, local_grib)

    # Open all GRIB message groups; find the pressure-level dataset
    import cfgrib

    all_datasets = cfgrib.open_datasets(local_grib)  # type: ignore[possibly-missing-attribute]
    ds_pl = None
    for candidate in all_datasets:
        if "isobaricInhPa" in candidate.dims:
            ds_pl = candidate
            break
    if ds_pl is None:
        raise ValueError("No pressure-level dataset found in GRIB file")

    # Select only the variables we need
    grib_vars = list(grib_rename.keys())
    missing = [v for v in grib_vars if v not in ds_pl.data_vars]
    if missing:
        raise ValueError(
            f"GRIB file missing required variables: {missing}. "
            f"Available: {list(ds_pl.data_vars)}"
        )
    ds = ds_pl[grib_vars]

    # Convert geopotential height (gpm) -> geopotential (m2/s2)
    ds["gh"] = ds["gh"] * GRAVITY

    # Rename variables to match contract
    ds = ds.rename(grib_rename)

    # Rename level coordinate
    ds = ds.rename({"isobaricInhPa": "level"})

    # Shift longitude from [-180, 180) to [0, 360)
    lon = ds.longitude.values
    ds = ds.assign_coords(longitude=np.where(lon < 0, lon + 360, lon))
    ds = ds.sortby("longitude")

    # Add time dimension (contract requires time=1)
    ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=[init_time])

    # Interpolate from 0.25° to 1.0° grid
    new_latitude = np.arange(-90, 90.1, 1.0)
    new_longitude = np.arange(0, 360, 1.0)
    ds = ds.interp(latitude=new_latitude, longitude=new_longitude, method="linear")

    # Ensure latitude descending (90 -> -90)
    if ds.latitude.values[0] < ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Ensure exact coordinate values
    ds = ds.assign_coords(
        latitude=np.arange(90, -90.1, -1.0), longitude=np.arange(0, 360, 1.0)
    )

    # Reorder dims to match contract: (time, level, latitude, longitude)
    ds = ds.transpose("time", "level", "latitude", "longitude")

    ds = ds.astype(np.float32)

    # Drop non-contract coordinates carried over from GRIB
    contract_coords = {"time", "level", "latitude", "longitude"}
    ds = ds.drop_vars([c for c in ds.coords if c not in contract_coords])

    # Strip all GRIB metadata attrs
    ds.attrs = {}
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].attrs = {}

    return ds
