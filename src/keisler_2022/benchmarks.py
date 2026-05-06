from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

GRAVITY = 9.80665


@dataclass(frozen=True)
class EvalVariable:
    short_name: str
    variable: str
    level: int
    unit: str
    scale: float = 1.0


EVAL_VARIABLES: tuple[EvalVariable, ...] = (
    EvalVariable("Z500", "geopotential", 500, "m", 1.0 / GRAVITY),
    EvalVariable("T850", "temperature", 850, "K"),
    EvalVariable("U850", "u_component_of_wind", 850, "m/s"),
    EvalVariable("Q850", "specific_humidity", 850, "kg/kg"),
)


def latitude_weights(lat_deg: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return area weights for a regular latitude-longitude grid."""
    return np.cos(np.deg2rad(lat_deg))[:, np.newaxis]


def area_weighted_mean(
    values: NDArray[np.floating],
    lat_deg: NDArray[np.floating],
) -> float:
    """Area-weighted finite mean for a 2D latitude-longitude field."""
    weights = latitude_weights(lat_deg)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    numerator = np.sum(weights * np.where(finite, values, 0.0))
    denominator = np.sum(weights * finite)
    return float(numerator / denominator)


def area_weighted_metrics(
    forecast: NDArray[np.floating],
    truth: NDArray[np.floating],
    lat_deg: NDArray[np.floating],
) -> dict[str, float]:
    """Compute area-weighted RMSE, MAE, bias, and MSE."""
    diff = np.asarray(forecast) - np.asarray(truth)
    mse = area_weighted_mean(diff**2, lat_deg)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": area_weighted_mean(np.abs(diff), lat_deg),
        "bias": area_weighted_mean(diff, lat_deg),
        "mse": mse,
    }


def skill_from_mse(model_mse: float, baseline_mse: float) -> float:
    """Return MSE skill score relative to a baseline forecast."""
    if not np.isfinite(model_mse) or not np.isfinite(baseline_mse):
        return float("nan")
    if baseline_mse <= 0:
        return float("nan")
    return float(1.0 - model_mse / baseline_mse)


def to_eval_units(
    values: NDArray[np.floating],
    eval_var: EvalVariable,
) -> NDArray[np.floating]:
    """Convert a model field to benchmark units."""
    return np.asarray(values) * eval_var.scale


def require_finite_metrics(rows: Iterable[dict[str, object]]) -> None:
    """Raise if any numeric metric field is non-finite."""
    metric_fields = {
        "rmse",
        "mae",
        "bias",
        "persistence_rmse",
        "persistence_mae",
        "skill_vs_persistence_mse",
    }
    for row in rows:
        for key in metric_fields:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, float) and not np.isfinite(value):
                raise FloatingPointError(f"Non-finite metric {key}: {row}")

