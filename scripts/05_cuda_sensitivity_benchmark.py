"""Run an Istanbul-targeted forecast sensitivity benchmark.

The default computes d(T850 at Istanbul, +72h) / d(initial conditions) using
JAX autodiff, then writes normalized gradients, physical-unit summaries, maps,
and a finite-difference sanity check.

Usage:
    uv run scripts/05_cuda_sensitivity_benchmark.py
    uv run scripts/05_cuda_sensitivity_benchmark.py --steps 12 --allow-cpu
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
import jax.numpy as jnp
import numpy as np
import pandas as pd

from keisler_2022.benchmarks import GRAVITY, area_weighted_mean
from keisler_2022.config import Config
from keisler_2022.io import load_arco_era5
from keisler_2022.runner import Runner, levels, varnames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cuda_sensitivity_benchmark")

DEFAULT_OUT_DIR = Path("results/pilot_2012_cuda5090/sensitivity_istanbul")
DEFAULT_SENS_FIELDS = (
    ("Z500", "geopotential", 500),
    ("U500", "u_component_of_wind", 500),
)

UNIT_BY_VAR = {
    "geopotential": "m",
    "specific_humidity": "kg/kg",
    "temperature": "K",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "vertical_velocity": "Pa/s",
}

SHORT_BY_VAR = {
    "geopotential": "Z",
    "specific_humidity": "Q",
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "vertical_velocity": "W",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Istanbul forecast sensitivity maps via JAX autodiff."
    )
    parser.add_argument(
        "--init",
        default="2012-01-01T00",
        help="Initialization time (default: 2012-01-01T00).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of 6-hour steps (default: 12 = +72h).",
    )
    parser.add_argument("--lat", type=float, default=41.0, help="Target latitude.")
    parser.add_argument(
        "--lon",
        type=float,
        default=29.0,
        help="Target longitude in either -180..180 or 0..360 convention.",
    )
    parser.add_argument(
        "--target-var",
        default="temperature",
        choices=varnames,
        help="Target variable (default: temperature).",
    )
    parser.add_argument(
        "--target-level",
        type=int,
        default=850,
        choices=levels,
        help="Target pressure level in hPa (default: 850).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of strongest point/channel sensitivities to write.",
    )
    parser.add_argument(
        "--fd-normalized-step",
        type=float,
        default=0.1,
        help="Central finite-difference step along the normalized gradient direction.",
    )
    parser.add_argument(
        "--jax-cache",
        type=Path,
        default=None,
        help="Optional persistent JAX compilation cache directory.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run even if JAX only sees CPU devices.",
    )
    return parser.parse_args()


def channel_index(var_name: str, level: int) -> int:
    return varnames.index(var_name) * len(levels) + levels.index(level)


def channel_name(channel: int) -> tuple[str, int, str]:
    var_index, level_index = divmod(channel, len(levels))
    var_name = varnames[var_index]
    level = levels[level_index]
    return var_name, level, f"{SHORT_BY_VAR[var_name]}{level}"


def node_index(lat: float, lon: float, n_lon: int = 360) -> int:
    lat_grid = np.arange(90, -90.1, -1.0)
    lon_grid = np.arange(0, 360, 1.0)
    lat_idx = int(np.argmin(np.abs(lat_grid - lat)))
    lon_idx = int(np.argmin(np.abs(lon_grid - (lon % 360))))
    return lat_idx * n_lon + lon_idx


def grid_indices(lat: float, lon: float) -> tuple[int, int]:
    lat_grid = np.arange(90, -90.1, -1.0)
    lon_grid = np.arange(0, 360, 1.0)
    lat_idx = int(np.argmin(np.abs(lat_grid - lat)))
    lon_idx = int(np.argmin(np.abs(lon_grid - (lon % 360))))
    return lat_idx, lon_idx


def display_scale_for_input(var_name: str) -> float:
    if var_name == "geopotential":
        return GRAVITY
    return 1.0


def display_scales_by_channel() -> np.ndarray:
    scales = []
    for channel in range(len(varnames) * len(levels)):
        var_name, _, _ = channel_name(channel)
        scales.append(display_scale_for_input(var_name))
    return np.asarray(scales, dtype=np.float32)


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_forecast_scalar(
    runner: Runner,
    prep: Any,
    n_steps: int,
    target_node: int,
    target_channel: int,
):
    n_node = runner.n_node
    n_node_era5 = runner.n_node_era5
    n_channels = runner.n_channels

    @jax.checkpoint
    def one_step(g, step_idx):
        g, _ = prep.transformed.apply(prep.params, g, step_idx)
        return g

    def forecast_scalar(
        input_data_era5: jnp.ndarray,
        all_solar: jnp.ndarray,
        all_doy: jnp.ndarray,
    ) -> jnp.ndarray:
        padded = jnp.zeros((n_node, n_channels))
        padded = padded.at[:n_node_era5].set(input_data_era5)

        g = {
            key: value._replace(nodes=dict(value.nodes), edges=dict(value.edges))
            for key, value in prep.graphs.items()
        }
        g["e"].nodes["data"] = padded
        g["e"].nodes["all_solar"] = all_solar
        g["e"].nodes["all_doy"] = all_doy

        for step_idx in range(n_steps):
            g = one_step(g, step_idx)

        return g["e"].nodes["data"][target_node, target_channel]

    return forecast_scalar


def top_sensitivity_rows(
    normalized: np.ndarray,
    physical_display: np.ndarray,
    top_n: int,
) -> list[dict[str, object]]:
    lat_grid = np.arange(90, -90.1, -1.0)
    lon_grid = np.arange(0, 360, 1.0)
    flat_abs = np.abs(normalized).ravel()
    top_n = min(top_n, flat_abs.size)
    unordered = np.argpartition(flat_abs, -top_n)[-top_n:]
    ordered = unordered[np.argsort(flat_abs[unordered])[::-1]]

    rows = []
    for rank, flat_index in enumerate(ordered, start=1):
        lat_idx, lon_idx, channel = np.unravel_index(flat_index, normalized.shape)
        var_name, level, label = channel_name(int(channel))
        input_unit = UNIT_BY_VAR[var_name]
        rows.append(
            {
                "rank": rank,
                "latitude": float(lat_grid[lat_idx]),
                "longitude": float(lon_grid[lon_idx]),
                "channel": int(channel),
                "variable": label,
                "source_variable": var_name,
                "level_hpa": level,
                "input_unit": input_unit,
                "gradient_normalized": float(normalized[lat_idx, lon_idx, channel]),
                "abs_gradient_normalized": float(flat_abs[flat_index]),
                "gradient_physical_display": float(
                    physical_display[lat_idx, lon_idx, channel]
                ),
            }
        )
    return rows


def channel_norm_rows(
    normalized: np.ndarray,
    physical_display: np.ndarray,
) -> list[dict[str, object]]:
    lat_grid = np.arange(90, -90.1, -1.0)
    rows = []
    for channel in range(normalized.shape[-1]):
        var_name, level, label = channel_name(channel)
        field_norm = normalized[:, :, channel]
        field_phys = physical_display[:, :, channel]
        rows.append(
            {
                "channel": channel,
                "variable": label,
                "source_variable": var_name,
                "level_hpa": level,
                "input_unit": UNIT_BY_VAR[var_name],
                "normalized_weighted_rms": float(
                    np.sqrt(area_weighted_mean(field_norm**2, lat_grid))
                ),
                "normalized_weighted_mean_abs": area_weighted_mean(
                    np.abs(field_norm), lat_grid
                ),
                "normalized_max_abs": float(np.max(np.abs(field_norm))),
                "physical_display_weighted_rms": float(
                    np.sqrt(area_weighted_mean(field_phys**2, lat_grid))
                ),
                "physical_display_weighted_mean_abs": area_weighted_mean(
                    np.abs(field_phys), lat_grid
                ),
                "physical_display_max_abs": float(np.max(np.abs(field_phys))),
            }
        )
    return rows


def finite_difference_row(
    value_fn: Any,
    input_data: jnp.ndarray,
    all_solar: jnp.ndarray,
    all_doy: jnp.ndarray,
    normalized: np.ndarray,
    target_std: float,
    normalized_step: float,
) -> dict[str, object]:
    gradient_direction = normalized.reshape(input_data.shape)
    gradient_norm = float(np.linalg.norm(gradient_direction))
    if gradient_norm == 0 or not np.isfinite(gradient_norm):
        raise FloatingPointError("Cannot run finite difference with zero gradient norm")

    unit_direction = jnp.array(gradient_direction / gradient_norm)
    step = float(normalized_step)
    plus = input_data + step * unit_direction
    minus = input_data - step * unit_direction

    plus_norm = value_fn(plus, all_solar, all_doy)
    minus_norm = value_fn(minus, all_solar, all_doy)
    plus_norm.block_until_ready()
    minus_norm.block_until_ready()

    finite_difference_norm_units = float((plus_norm - minus_norm) / (2.0 * step))
    autodiff_norm_units = gradient_norm
    finite_difference_model_units = finite_difference_norm_units * target_std
    autodiff_model_units = autodiff_norm_units * target_std
    rel_error = abs(finite_difference_model_units - autodiff_model_units) / max(
        abs(autodiff_model_units), 1e-12
    )

    return {
        "check_reason": "central difference along normalized gradient direction",
        "normalized_step": step,
        "gradient_l2_norm_normalized": gradient_norm,
        "autodiff_directional_derivative_normalized": autodiff_norm_units,
        "finite_difference_directional_derivative_normalized": finite_difference_norm_units,
        "autodiff_directional_derivative_model_units": autodiff_model_units,
        "finite_difference_directional_derivative_model_units": finite_difference_model_units,
        "relative_error": rel_error,
    }


def plot_selected_maps(
    gradients: np.ndarray,
    path: Path,
    title: str,
    target_lat: float,
    target_lon: float,
    selected_fields: tuple[tuple[str, str, int], ...] = DEFAULT_SENS_FIELDS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lons = np.arange(0, 360, 1.0)
    lats = np.arange(90, -90.1, -1.0)
    n_cols = len(selected_fields)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

    for ax, (label, var_name, level) in zip(axes[0], selected_fields):
        channel = channel_index(var_name, level)
        field = gradients[:, :, channel]
        vmax = float(np.percentile(np.abs(field), 99.9))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(
            field,
            extent=(float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0])),
            origin="upper",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.scatter([target_lon % 360], [target_lat], marker="*", color="black", s=80)
        ax.set_xlim(0, 359)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.jax_cache:
        args.jax_cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(args.jax_cache))

    if cpu_only() and not args.allow_cpu:
        raise RuntimeError(
            "JAX only sees CPU devices. Run inside WSL2 with CUDA/JAX GPU support, "
            "or pass --allow-cpu for a slow debug run."
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    init_time = pd.Timestamp(args.init)
    target_lon = args.lon % 360
    target_channel = channel_index(args.target_var, args.target_level)
    target_node = node_index(args.lat, target_lon)
    target_label = f"{SHORT_BY_VAR[args.target_var]}{args.target_level}"
    lead_hours = args.steps * 6

    timing: dict[str, Any] = {
        "benchmark": "cuda_sensitivity",
        "init_time": init_time.isoformat(),
        "lead_hours": lead_hours,
        "target": {
            "label": target_label,
            "variable": args.target_var,
            "level_hpa": args.target_level,
            "latitude": args.lat,
            "longitude": target_lon,
            "node": target_node,
            "channel": target_channel,
        },
        "device": device_snapshot(),
    }
    total_start = time.perf_counter()
    logger.info("Device snapshot: %s", timing["device"])

    t0 = time.perf_counter()
    ds = load_arco_era5(init_time, cache=True).load()
    timing["load_initial_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    runner = Runner(verbose=True, config=Config())
    prep = runner.prepare(ds, args.steps)
    timing["prepare_seconds"] = time.perf_counter() - t0

    all_solar = jnp.array(prep.graphs["e"].nodes["all_solar"])
    all_doy = jnp.array(prep.graphs["e"].nodes["all_doy"])
    input_data = jnp.array(prep.initial_data)
    forecast_scalar = build_forecast_scalar(
        runner=runner,
        prep=prep,
        n_steps=args.steps,
        target_node=target_node,
        target_channel=target_channel,
    )
    grad_fn = jax.jit(jax.grad(forecast_scalar, argnums=0))
    value_fn = jax.jit(forecast_scalar)

    logger.info("Computing gradients for %s +%sh", target_label, lead_hours)
    t0 = time.perf_counter()
    gradients = grad_fn(input_data, all_solar, all_doy)
    gradients.block_until_ready()
    timing["gradient_seconds"] = time.perf_counter() - t0

    normalized = np.asarray(gradients).reshape(
        runner.n_lat, runner.n_lon, runner.n_channels
    )
    input_stds = np.asarray(runner.normalizer["stds"], dtype=np.float32)
    target_std = float(input_stds[target_channel])
    physical_model_units = normalized * target_std / input_stds[np.newaxis, np.newaxis, :]
    physical_display = physical_model_units * display_scales_by_channel()[
        np.newaxis, np.newaxis, :
    ]

    t0 = time.perf_counter()
    top_rows = top_sensitivity_rows(normalized, physical_display, args.top_n)
    norm_rows = channel_norm_rows(normalized, physical_display)
    fd_row = finite_difference_row(
        value_fn=value_fn,
        input_data=input_data,
        all_solar=all_solar,
        all_doy=all_doy,
        normalized=normalized,
        target_std=target_std,
        normalized_step=args.fd_normalized_step,
    )

    write_csv(out_dir / "sensitivity_top_locations.csv", top_rows)
    write_csv(out_dir / "sensitivity_channel_norms.csv", norm_rows)
    write_csv(out_dir / "sensitivity_finite_difference.csv", [fd_row])

    plot_selected_maps(
        normalized,
        out_dir / "sensitivity_normalized.png",
        f"Normalized sensitivity d({target_label} +{lead_hours}h)/d(input)",
        args.lat,
        target_lon,
    )
    plot_selected_maps(
        physical_display,
        out_dir / "sensitivity_physical_display.png",
        f"Physical-unit sensitivity d({target_label} +{lead_hours}h)/d(input)",
        args.lat,
        target_lon,
    )

    target_lat_idx, target_lon_idx = grid_indices(args.lat, target_lon)
    target_value = float(
        ds[args.target_var]
        .sel(level=args.target_level)
        .isel(time=0)
        .values[target_lat_idx, target_lon_idx]
    )
    timing["target_initial_value_model_units"] = target_value
    timing["write_seconds"] = time.perf_counter() - t0
    timing["total_seconds"] = time.perf_counter() - total_start

    with (out_dir / "sensitivity_timing_device.json").open("w", encoding="utf-8") as fp:
        json.dump(timing, fp, indent=2, sort_keys=True)

    logger.info("Wrote sensitivity outputs to %s", out_dir)


if __name__ == "__main__":
    main()
