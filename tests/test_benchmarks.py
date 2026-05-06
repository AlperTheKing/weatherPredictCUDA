import numpy as np
import pytest

from keisler_2022.benchmarks import (
    EVAL_VARIABLES,
    GRAVITY,
    area_weighted_mean,
    area_weighted_metrics,
    require_finite_metrics,
    skill_from_mse,
    to_eval_units,
)


def test_area_weighted_metrics_zero_error() -> None:
    lat = np.array([60.0, 0.0, -60.0])
    truth = np.ones((3, 2), dtype=np.float32)
    forecast = truth.copy()

    metrics = area_weighted_metrics(forecast, truth, lat)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["bias"] == 0.0
    assert metrics["mse"] == 0.0


def test_area_weighted_mean_ignores_non_finite_values() -> None:
    lat = np.array([0.0, 60.0])
    values = np.array([[1.0, np.nan], [3.0, np.inf]])

    measured = area_weighted_mean(values, lat)

    expected = (1.0 * 1.0 + 0.5 * 3.0) / (1.0 + 0.5)
    assert np.isclose(measured, expected)


def test_skill_from_mse() -> None:
    assert skill_from_mse(model_mse=2.0, baseline_mse=8.0) == 0.75
    assert np.isnan(skill_from_mse(model_mse=2.0, baseline_mse=0.0))


def test_z500_converts_geopotential_to_height() -> None:
    z500 = next(item for item in EVAL_VARIABLES if item.short_name == "Z500")
    geopotential = np.array([[GRAVITY * 5000.0]], dtype=np.float32)

    converted = to_eval_units(geopotential, z500)

    assert np.isclose(converted.item(), 5000.0)


def test_require_finite_metrics_rejects_nan_skill() -> None:
    with pytest.raises(FloatingPointError):
        require_finite_metrics(
            [
                {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "bias": 0.0,
                    "persistence_rmse": 1.0,
                    "persistence_mae": 1.0,
                    "skill_vs_persistence_mse": float("nan"),
                }
            ]
        )

