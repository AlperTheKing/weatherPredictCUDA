import numpy as np
import pandas as pd

from keisler_2022.solar import (
    centered_solar,
    eq_of_time,
    q0,
    solar_alt_approx,
    toa_solar_rad_approx,
)


def test_q0() -> None:
    # Test winter solstice (approximately day 355)
    assert abs(q0(355) - 1.034) < 0.001

    # Test summer solstice (approximately day 172)
    assert abs(q0(172) - 0.966) < 0.001

    # Test spring equinox (approximately day 80)
    assert abs(q0(80) - 1.0) < 0.01


def test_solar_alt_equator_noon() -> None:
    # Test solar altitude at equator during equinox at solar noon
    lat_deg = 0.0  # equator
    lon_deg = 0.0  # prime meridian
    # March 20, 2024 12:00 UTC (approximate spring equinox)
    when = pd.Timestamp("2024-03-20T12:00:00", tz="UTC")

    alt = solar_alt_approx(lat_deg, lon_deg, when)
    # Should be close to 90 degrees at solar noon on equinox
    assert abs(alt - 90.0) < 5.0


def test_solar_alt_poles() -> None:
    # Test solar altitude at north pole during summer solstice
    lat_deg = 90.0  # north pole
    lon_deg = 0.0
    # June 20, 2024 12:00 UTC (approximate summer solstice)
    when = pd.Timestamp("2024-06-20T12:00:00", tz="UTC")

    alt = solar_alt_approx(lat_deg, lon_deg, when)
    # Should be close to 23.45 degrees (earth's tilt)
    assert abs(alt - 23.45) < 2.0


def test_centered_solar() -> None:
    lat_deg = 45.0
    lon_deg = 0.0
    valid_time = pd.Timestamp("2024-03-20T12:00:00", tz="UTC")

    result = centered_solar(lat_deg, lon_deg, valid_time)

    # Check shape (should have 11 time points)
    assert result.shape[-1] == 11

    # Values should be non-negative
    assert np.all(result >= 0)

    # Middle value should be largest (closest to solar noon)
    assert np.argmax(result) == 5


def test_toa_solar_rad_approx() -> None:
    lat_deg = 0.0  # equator
    lon_deg = 0.0  # prime meridian
    # Test at solar noon during equinox
    when = pd.Timestamp("2024-03-20T12:00:00", tz="UTC")

    rad = toa_solar_rad_approx(lat_deg, lon_deg, when)

    # Should be close to 1.0 (normalized solar constant)
    assert 0.9 < rad < 1.1


def test_eq_of_time() -> None:
    # Test equation of time at specific days
    # Values should follow known pattern with max/min values

    # Maximum deviation occurs around February 11 (day 42)
    assert abs(eq_of_time(42)) > 14.0

    # Minimum deviation occurs around November 3 (day 307)
    assert abs(eq_of_time(307)) > 16.0

    # Should be close to zero near April 15 (day 105)
    assert abs(eq_of_time(105)) < 1.0
