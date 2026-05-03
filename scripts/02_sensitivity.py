"""Compute forecast sensitivities (gradients) for the Keisler 2022 GNN model.

For a chosen target location, field, and lead time, compute
d(forecast)/d(initial_conditions) using JAX autodiff and visualize the
sensitivity maps for selected input fields.

Usage:
    uv run scripts/02_sensitivity.py
    uv run scripts/02_sensitivity.py --init 2026-01-03T00 --steps 12
    uv run scripts/02_sensitivity.py --init 2026-01-03T00 --steps 4  # faster for dev
"""

import argparse
import logging
import time

import cartopy.crs as ccrs
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keisler_2022.config import Config
from keisler_2022.io import load_arco_era5
from keisler_2022.runner import Runner, levels, varnames

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("sensitivity")
logger.setLevel(logging.INFO)

# Sensitivity fields to visualize: (label, variable_name, level)
SENS_FIELDS = [
    ("Z500", "geopotential", 500),
    ("U500", "u_component_of_wind", 500),
]


def channel_index(var_name: str, level: int) -> int:
    """Return the flattened channel index for a given variable and pressure level."""
    return varnames.index(var_name) * len(levels) + levels.index(level)


def node_index(lat: float, lon: float, n_lon: int = 360) -> int:
    """Return the flattened ERA5 node index for the nearest grid cell.

    The grid has latitude descending from 90 to -90 (1-degree spacing)
    and longitude from 0 to 359 (1-degree spacing, 0-360 convention).
    """
    lat_grid = np.arange(90, -90.1, -1.0)
    lon_grid = np.arange(0, 360, 1.0)
    lat_idx = int(np.argmin(np.abs(lat_grid - lat)))
    lon_idx = int(np.argmin(np.abs(lon_grid - lon)))
    return lat_idx * n_lon + lon_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute forecast sensitivity maps via JAX autodiff."
    )
    parser.add_argument(
        "--init",
        default="2026-01-03T00",
        help="Initialization time (default: 2026-01-03T00)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of 6-hour steps (default: 12 = 72h). Use 4 for faster iteration.",
    )
    parser.add_argument("--lat", type=float, default=35.7, help="Target latitude")
    parser.add_argument(
        "--lon",
        type=float,
        default=254.1,
        help="Target longitude (0-360 convention)",
    )
    parser.add_argument(
        "--target-var",
        default="temperature",
        help="Target variable name (default: temperature)",
    )
    parser.add_argument(
        "--target-level",
        type=int,
        default=850,
        help="Target pressure level in hPa (default: 850)",
    )
    parser.add_argument(
        "--fig",
        default="/tmp/sensitivity.png",
        help="Output figure path (default: /tmp/sensitivity.png)",
    )
    parser.add_argument(
        "--jax-cache",
        default=None,
        help="Directory for persistent JAX compilation cache (speeds up repeat runs)",
    )
    args = parser.parse_args()

    if args.jax_cache:
        jax.config.update("jax_compilation_cache_dir", args.jax_cache)
        logger.info(f"JAX compilation cache: {args.jax_cache}")

    init_time = pd.Timestamp(args.init)
    n_steps = args.steps
    lead_hours = n_steps * 6
    target_lon = args.lon % 360  # ensure 0-360

    # --- Load initial conditions ---
    logger.info(f"Loading initial conditions for {init_time.isoformat()}")
    ds = load_arco_era5(init_time)
    ds = ds.load()

    # --- Create runner and prepare forecast ---
    config = Config()
    runner = Runner(verbose=True, config=config)

    devices = jax.devices()
    if all(d.device_kind == "cpu" for d in devices):
        logger.warning("Running on CPU only -- gradient computation will be slow.")

    logger.info(f"Preparing {n_steps}-step forecast...")
    prep = runner.prepare(ds, n_steps)

    # --- Identify target ---
    target_ch = channel_index(args.target_var, args.target_level)
    target_node = node_index(args.lat, target_lon)
    target_label = f"{args.target_var[0].upper()}{args.target_level}"
    logger.info(
        f"Target: {target_label} at lat={args.lat}, lon={target_lon} "
        f"(node={target_node}, channel={target_ch})"
    )

    # --- Define differentiable forecast function ---
    n_node = runner.n_node
    n_node_era5 = runner.n_node_era5
    n_channels = runner.n_channels

    # Extract time-varying graph data as explicit arrays so they become
    # dynamic JAX inputs rather than baked-in constants.  This keeps the
    # XLA computation graph identical across different init times, enabling
    # the persistent compilation cache to hit.
    all_solar = jnp.array(prep.graphs["e"].nodes["all_solar"])
    all_doy = jnp.array(prep.graphs["e"].nodes["all_doy"])

    @jax.checkpoint
    def one_step(g, step_idx):
        """Single forecast step, checkpointed to save memory during backprop."""
        g, _ = prep.transformed.apply(prep.params, g, step_idx)
        return g

    def forecast_scalar(
        input_data_era5: jnp.ndarray,
        all_solar: jnp.ndarray,
        all_doy: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map normalized ERA5 initial data to a scalar forecast value."""
        padded = jnp.zeros((n_node, n_channels))
        padded = padded.at[:n_node_era5].set(input_data_era5)

        # Fresh shallow copies so tracing doesn't contaminate the template
        g = {
            k: v._replace(nodes=dict(v.nodes), edges=dict(v.edges))
            for k, v in prep.graphs.items()
        }
        g["e"].nodes["data"] = padded
        g["e"].nodes["all_solar"] = all_solar
        g["e"].nodes["all_doy"] = all_doy

        for step_idx in range(n_steps):
            g = one_step(g, step_idx)

        return g["e"].nodes["data"][target_node, target_ch]

    input_data = jnp.array(prep.initial_data)

    # --- Compute gradients (w.r.t. first arg only) ---
    logger.info("Computing gradients (includes JIT compilation on first run)...")
    grad_fn = jax.jit(jax.grad(forecast_scalar, argnums=0))

    t0 = time.time()
    sensitivities = grad_fn(input_data, all_solar, all_doy)
    sensitivities.block_until_ready()
    t_grad = time.time() - t0
    logger.info(f"Gradient computation: {t_grad:.1f}s")

    # Convert to numpy and reshape to (lat, lon, channels)
    sens_np = np.array(sensitivities).reshape(runner.n_lat, runner.n_lon, n_channels)

    # --- Visualize ---
    matplotlib.use("Agg")
    data_crs = ccrs.PlateCarree()
    proj = ccrs.Orthographic(central_longitude=target_lon, central_latitude=args.lat)
    lons = np.arange(0, 360, 1.0)
    lats = np.arange(90, -90.1, -1.0)

    # Z500 at lead=0 for contour overlay
    z500_init = ds["geopotential"].sel(level=500).values.squeeze() / 9.80665

    n_panels = len(SENS_FIELDS)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4 * n_panels, 4),
        subplot_kw={"projection": proj},
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (label, var_name, level) in zip(axes, SENS_FIELDS):
        ch = channel_index(var_name, level)
        field = sens_np[:, :, ch]

        vmax = float(np.percentile(np.abs(field), 99.9))
        if vmax == 0:
            vmax = 1.0
        _ = ax.pcolormesh(
            lons,
            lats,
            field,
            transform=data_crs,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.plot(
            target_lon,
            args.lat,
            marker="*",
            color="black",
            markersize=12,
            markeredgewidth=0.5,
            transform=data_crs,
        )
        ax.contour(
            lons,
            lats,
            z500_init,
            levels=12,
            colors="gray",
            linewidths=0.4,
            alpha=0.7,
            transform=data_crs,
        )
        ax.coastlines(linewidth=0.5)
        ax.set_global()
        ax.set_title(f"d({target_label}) / d({label})", fontsize=12)

    fig.suptitle(
        f"Sensitivity — Init: {init_time.isoformat()}, Lead: +{lead_hours}h",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.fig, dpi=200)
    logger.info(f"Figure saved to {args.fig}")


if __name__ == "__main__":
    main()
