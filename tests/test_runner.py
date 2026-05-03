import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from keisler_2022.io import load_arco_era5
from keisler_2022.runner import Runner


@pytest.mark.network
def test_two_steps_regression() -> None:
    """Make a 2-step forecast, check that the forecast is close to the expected values."""
    # Initialize the runner
    runner = Runner()

    # Load the input dataset
    ds_input = load_arco_era5(pd.Timestamp("2020-01-01T00:00:00"), cache=True)

    # Run the forecast
    n_steps = 2
    ds_forecast = runner.run(ds_input, n_steps=n_steps)

    # check a specific geopotential value
    expected = 186141.34375
    measured = ds_forecast.geopotential.isel(time=-1, level=0, latitude=0, longitude=0)
    assert np.isclose(
        measured,
        expected,
        atol=1e-6,
    ), f"Expected {expected}, got {measured}, difference {measured - expected}"

    # check mean of all fields
    # Note: GPU computation introduces small stochasticity, so tolerances are relaxed
    # Observed variance: geopotential ~0.003, u_component_of_wind ~0.0001, v_component_of_wind ~0.00006
    expected_dict: dict[str, float] = {
        "geopotential": 77669.4722000,
        "specific_humidity": 0.0017015,
        "temperature": 243.1940000,
        "u_component_of_wind": 8.3083637,
        "v_component_of_wind": 0.0388912,
        "vertical_velocity": 0.0035376,
    }
    for data_var in ds_forecast.data_vars:
        expected = expected_dict[str(data_var)]
        measured = ds_forecast[data_var].mean().item()
        assert np.isclose(measured, expected, atol=0.01, rtol=1e-4), (
            f"Expected {expected}, got {measured}, difference {measured - expected}"
        )


@pytest.mark.slow
@pytest.mark.network
def test_2020_forecast() -> None:
    """Make a 2020 forecast, compare with original Keisler22 forecast.

    In 2023 we ran the Keisler22 model for all of 2020 for WeatherBench evaluation.
    Does the model still produce forecasts that are close to those forecasts?

    This test:
    - runs the model for a specific initialization time
    - loads the original forecast
    - compares the forecast to the original forecast
    - makes sure the differences are significantly smaller than the errors in the forecasts
    """
    # from https://sites.research.google/weatherbench/deterministic-scores/
    keisler22_240h_errors = {
        "z500": 787.0,
        "t850": 3.6,
        "q850": 0.0019,
        "u850": 6.1,
        "v850": 6.1,
    }

    # Initialize the runner
    runner = Runner()

    # Load the input dataset
    ds_input = load_arco_era5(pd.Timestamp("2020-01-01T00:00:00"), cache=True)

    # Run the forecast
    n_hours_forecast = 240
    n_steps = n_hours_forecast // 6
    ds_forecast = runner.run(ds_input, n_steps=n_steps)

    # Load WeatherBench comparison data
    da_weatherbench = keisler22_weatherbench_2020_forecast()
    init_date = "2020-01-01T00:00:00"

    # Dictionary to map comparison keys to forecast variables and levels
    var_mapping = {
        "z500": ("geopotential", 500),
        "t850": ("temperature", 850),
        "q850": ("specific_humidity", 850),
        "u850": ("u_component_of_wind", 850),
        "v850": ("v_component_of_wind", 850),
    }

    # Compare each variable to the Keisler 2022 forecast error
    safety_margin = 16
    for key_compare, forecast_error in keisler22_240h_errors.items():
        # Get WeatherBench data
        weatherbench_data = (
            da_weatherbench.sel(init=init_date)
            .isel(fhour=slice(0, n_steps + 1))
            .sel(band=key_compare)
        )

        # Get forecast data
        var_name, level = var_mapping[key_compare]
        forecast_data = ds_forecast[var_name].sel(level=level)

        # Compute RMS of the difference at the last timestep
        diff = weatherbench_data.values[-1] - forecast_data.values[-1]
        rms = np.sqrt(np.mean(diff**2))

        print(
            f"RMS difference for {key_compare}: {rms:.6f} ({forecast_error / rms:.1f}X smaller than {n_hours_forecast}-hours forecast error of {forecast_error:.6f})"
        )
        assert rms < (forecast_error / safety_margin), (
            f"RMS difference for {key_compare} ({rms:.6f}) exceeds threshold of {forecast_error / safety_margin:.6f}"
        )


def keisler22_weatherbench_2020_forecast() -> xr.DataArray:
    """Fetch the Keisler22 WeatherBench data for 2020

    The Keisler22 model was used to generate forecasts for 2020 for WeatherBench evaluation.
    That data was saved in a raw Zarr store. This function wraps it in an xr.DataArray.
    """
    # adapted from https://github.com/rkeisler/keisler22-predict/blob/main/read_data.ipynb
    gs_path = "gs://rk-public-auto/wb2/keisler22_2020_predict"
    ztmp = zarr.open(gs_path, mode="r")
    data = dask.array.from_zarr(ztmp["data"])

    # define the forecast initialization datetimes.
    init = pd.date_range(
        start="2020-01-01 00:00:00", end="2020-12-31 12:00:00", freq="12h"
    )

    # define the forecast hour coordinates.
    # fhour=0 corresponds to the ERA5 initialization.
    # there are 40 6-hour steps after the initialization.
    fhour = np.arange(0, 40 + 1) * 6

    # define the lat/lon coordinates.
    lat = np.arange(+90, -90 - 1, -1.0)
    lon = np.arange(0, 360, +1.0)
    lon[lon > 180] -= 360

    # define the bands, i.e. channels.
    band = [
        "z500",
        "z700",
        "z850",
        "t500",
        "t700",
        "t850",
        "q500",
        "q700",
        "q850",
        "u500",
        "u700",
        "u850",
        "v500",
        "v700",
        "v850",
    ]

    # check that the shape of the data array matches the shape of these coordinate arrays.
    assert data.shape == (len(init), len(fhour), len(lat), len(lon), len(band))

    # create the xr.DataArray.
    da = xr.DataArray(
        data,
        name="data",
        dims=("init", "fhour", "latitude", "longitude", "band"),
        coords={
            "init": init,
            "fhour": fhour,
            "latitude": lat,
            "longitude": lon,
            "band": band,
        },
    )

    return da
