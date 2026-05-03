import functools
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import haiku as hk
import jax
import jraph
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from keisler_2022.config import Config, resolve_artifact
from keisler_2022.gnn import one_step_fn
from keisler_2022.graphs import GraphBuilder
from keisler_2022.solar import centered_solar

varnames = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "geopotential",
]
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
full_varnames = [
    f"{varnames[i]}_{levels[j]:04d}"
    for i in range(len(varnames))
    for j in range(len(levels))
]


@dataclass
class PreparedForecast:
    """All objects needed to run (or differentiate) the forecast loop."""

    graphs: dict[str, jraph.GraphsTuple]
    initial_data: NDArray[np.floating]  # normalized, shape (n_node_era5, n_channels)
    transformed: Any  # hk.Transformed — call .apply(params, graphs, step_idx)
    params: Any  # immutable haiku parameter dict
    forecast_times: list[pd.Timestamp]


class Runner:
    def __init__(self, verbose: bool = True, config: Config | None = None):
        self.ind_use = list(np.arange(78))
        self.config = config if config is not None else Config()
        self.n_channels = int(6 * 13)
        self.n_lat = 181
        self.n_lon = 360

        self.verbose = verbose
        self.logger = logging.getLogger("keisler_2022")
        # Set default level based on verbose flag; user apps can override
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        self.define_graphs()
        self.define_fixed()

        self.normalizer = self.get_normalizer()

    def _log(self, message: str) -> None:
        if self.verbose:
            self.logger.info(message)

    def define_graphs(self) -> None:
        builder = GraphBuilder(self.config)
        builder.build()

        self.geometry_era5 = builder.geometry_era5
        self.lat_deg = np.rad2deg(self.geometry_era5.latlonr[:, 0])
        self.lon_deg = np.rad2deg(self.geometry_era5.latlonr[:, 1])

        self.n_node_era5 = builder.n_node_era5
        self.n_node_h3 = builder.n_node_h3
        self.n_node = builder.n_node

        assert self.n_node_era5 == 181 * 360 == 65160
        assert self.n_node_h3 == 5882

        self.static_graphs = builder.static_graphs

    def define_fixed(self) -> None:
        fixed = np.load(resolve_artifact(self.config.data.orography_landsea_file))
        self.orography = fixed["orography"]
        self.landsea = fixed["landsea"]
        self.orography = np.reshape(self.orography, (self.n_node_era5, 1))
        self.landsea = np.reshape(self.landsea, (self.n_node_era5, 1))

        self.static_graphs["e"].nodes["orography"] = np.zeros((self.n_node, 1))
        self.static_graphs["e"].nodes["orography"][: self.n_node_era5] = self.orography

        self.static_graphs["e"].nodes["landsea"] = np.zeros((self.n_node, 1))
        self.static_graphs["e"].nodes["landsea"][: self.n_node_era5] = self.landsea

    def get_normalizer(self) -> dict[str, NDArray[np.floating]]:
        path = resolve_artifact(self.config.data.normalizer_file)
        tmp = np.load(path)
        normalizer = {
            "means": np.array(tmp["means"])[self.ind_use],
            "stds": np.array(tmp["stds"])[self.ind_use],
        }
        return normalizer

    def prepare(self, ds: xr.Dataset, n_steps: int) -> PreparedForecast:
        """Validate input, normalize, build graphs, and load model.

        Returns a :class:`PreparedForecast` containing everything needed to
        execute (or differentiate) the forecast loop.
        """
        # validate the dataset
        expected_vars = {
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        }
        if set(ds.data_vars) != expected_vars:
            raise ValueError(
                f"Dataset variables must be {sorted(expected_vars)}; got {sorted(list(ds.data_vars))}"
            )
        for v in ds.data_vars:
            if ds[v].dims != ("time", "level", "latitude", "longitude"):
                raise ValueError(
                    f"Variable {v} dims must be ('time','level','latitude','longitude'); got {ds[v].dims}"
                )
            if ds[v].shape != (1, 13, 181, 360):
                raise ValueError(
                    f"Variable {v} shape must be (1,13,181,360); got {ds[v].shape}"
                )

        # make sure data is on the correct 1.0 deg grid
        latitude_expected = np.arange(90, -90.1, -1.0)
        if not (
            np.isclose(ds.latitude.values, latitude_expected).all()
            or np.isclose(ds.latitude.values, latitude_expected[::-1]).all()
        ):
            raise ValueError("Latitude is not on the expected 1.0-degree grid")
        # if latitude goes the wrong way, flip it
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            ds = ds.isel(latitude=slice(None, None, -1))
            self._log("Input latitude ascending; flipped to descending order")

        if not np.isclose(ds.latitude.values, np.arange(90, -90.1, -1.0)).all():
            raise ValueError("Latitude values not exactly on expected 1.0-degree grid")
        if not np.isclose(ds.longitude.values, np.arange(0, 360, 1.0)).all():
            raise ValueError("Longitude values not exactly on expected 1.0-degree grid")

        # get the data
        data_arrays_np: list[NDArray[np.floating]] = [
            np.asarray(ds[varnames[i]].sel(level=levels[j]))
            for i in range(len(varnames))
            for j in range(len(levels))
        ]
        data: NDArray[np.floating] = np.stack(data_arrays_np, axis=-1)
        data = data[0, :, :, :]
        assert data.shape == (181, 360, 6 * 13)

        # reshape and normalize
        this_x = np.reshape(data, (self.n_node_era5, self.n_channels))
        this_x -= self.normalizer["means"]
        this_x /= self.normalizer["stds"]

        # forecast_times should be based on the first timestamp of ds,
        # and then n_steps each 6 hours
        start_time = pd.Timestamp(ds.time.values[0])
        self._log(f"Starting forecast: {n_steps} steps from {start_time.isoformat()}")
        forecast_times: list[pd.Timestamp] = []
        for i in range(n_steps + 1):
            this_time = start_time + pd.Timedelta(hours=6 * i)
            forecast_times.append(pd.Timestamp(this_time))

        # recompute solar
        solar_list: list[NDArray[np.floating]] = []
        for this_time in forecast_times[:-1]:
            solar_list.append(centered_solar(self.lat_deg, self.lon_deg, this_time))
        this_solar: NDArray[np.floating] = np.stack(solar_list, axis=0)

        # compute day of year (DOY)
        this_doy: NDArray[np.floating] = np.zeros(
            (n_steps, self.n_node_era5, 1), dtype="float32"
        )
        assert len(forecast_times[:-1]) == n_steps
        for i_datetime, this_time in enumerate(forecast_times[:-1]):
            when = pd.Timestamp(this_time)
            if when.tz is None:
                when = when.tz_localize("UTC")
            else:
                when = when.tz_convert("UTC")
            this_doy[i_datetime, :, :] = when.dayofyear
        this_doy /= 365.0

        # move time dimension to last dimension for solar and doy
        this_solar = np.moveaxis(this_solar, 0, -1)
        this_doy = np.moveaxis(this_doy, 0, -1)
        self._log("Computed solar and day-of-year features")

        # get the graphs
        graphs = {
            "e": self.static_graphs["e"].jraph(),
            "p": self.static_graphs["p"].jraph(),
            "d": self.static_graphs["d"].jraph(),
        }

        # initialize node arrays in encoder graph
        graphs["e"].nodes["data"] = self.init_set(this_x)
        graphs["e"].nodes["all_solar"] = self.init_set(this_solar)
        graphs["e"].nodes["all_doy"] = self.init_set(this_doy)
        self._log("Initialized graphs and node features")

        # build the haiku-transformed network function
        n_node = self.n_node
        n_node_era5 = self.n_node_era5
        n_node_h3 = self.n_node_h3

        net_fn_batch = functools.partial(
            one_step_fn,
            n_features=self.config.model.n_features,
            n_mlp_layers={
                "e": self.config.model.n_mlp_layers_encoder,
                "p": self.config.model.n_mlp_layers_processor,
                "d": self.config.model.n_mlp_layers_decoder,
            },
            n_processor_blocks=self.config.model.n_processor_blocks,
            n_channels_out=self.config.model.n_channels_out,
            use_lat=self.config.model.use_lat,
            use_lon=self.config.model.use_lon,
            use_doy=self.config.model.use_doy,
            n_node=n_node,
            n_node_era5=n_node_era5,
            n_node_h3=n_node_h3,
        )
        transformed: Any = hk.without_apply_rng(hk.transform(net_fn_batch))

        # Load model parameters (weights)
        params_savename = resolve_artifact(self.config.data.weights_file)
        with open(params_savename, "rb") as fp:
            params = pickle.load(fp)
        params = hk.data_structures.to_immutable_dict(params)
        self._log("Loaded model parameters")

        return PreparedForecast(
            graphs=graphs,
            initial_data=this_x,
            transformed=transformed,
            params=params,
            forecast_times=forecast_times,
        )

    def run(self, ds: xr.Dataset, n_steps: int, timing: bool = False) -> xr.Dataset:
        prep = self.prepare(ds, n_steps)
        graphs = prep.graphs
        forecast_times = prep.forecast_times
        net_apply = jax.jit(prep.transformed.apply)
        params = prep.params
        n_channels_out = self.config.model.n_channels_out

        # initialize the output dataset.
        # it will be like input but with forecast_times as time.
        ds_forecast = ds.copy()
        ds_forecast = ds_forecast.isel(
            time=[0] * (n_steps + 1)
        )  # Repeat the single timestep n_steps+1 times
        ds_forecast["time"] = (("time",), forecast_times)
        ds_forecast.attrs = {}
        self._log("Initialized output dataset")

        for step_idx in range(n_steps):
            self._log(f"Step {step_idx + 1}/{n_steps}")

            # Take a step forward (net_apply increments i_time internally)
            graphs, i_step = net_apply(params, graphs, step_idx)

            # Extract and save prediction
            this_pred = np.array(graphs["e"].nodes["data"])
            this_pred *= self.normalizer["stds"]
            this_pred += self.normalizer["means"]
            this_pred = this_pred[: self.n_node_era5]
            this_pred = np.reshape(this_pred, (self.n_lat, self.n_lon, n_channels_out))

            if not np.isfinite(this_pred).all():
                raise FloatingPointError(
                    f"Non-finite prediction encountered at step {step_idx}"
                )

            # now put into output dataset (i_step is step_idx + 1 after net_apply increments)
            for i_var, v in enumerate(varnames):
                for i_level, level in enumerate(levels):
                    ds_forecast[v].loc[
                        {"level": level, "time": forecast_times[i_step]}
                    ] = this_pred[:, :, i_var * len(levels) + i_level]
            self._log(f"Saved predictions for {forecast_times[i_step].isoformat()}")

        self._log("Forecast complete")
        return ds_forecast

    def init_set(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        shape = list(data.shape)
        shape[0] = self.n_node
        tmp = np.zeros(tuple(shape))
        tmp[: self.n_node_era5] = data
        return tmp
