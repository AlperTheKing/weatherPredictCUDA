import numpy as np
import pandas as pd
from numpy.typing import NDArray


def centered_solar(
    lat_deg: NDArray[np.floating] | float,
    lon_deg: NDArray[np.floating] | float,
    valid_time: pd.Timestamp,
) -> NDArray[np.floating]:
    """
    Calculate the solar radiation at a given latitude and longitude for a given time, plus a few hours around it.
    """
    when0 = pd.Timestamp(valid_time)
    if when0.tz is None:
        when0 = when0.tz_localize("UTC")
    else:
        when0 = when0.tz_convert("UTC")
    solars: list[NDArray[np.floating]] = []
    shifts_hours = [-12, -9, -6, -3, -1, 0, +1, +3, +6, +9, +12]
    for i_shift_hours in shifts_hours:
        when = when0 + pd.Timedelta(hours=i_shift_hours)
        this_solar = toa_solar_rad_approx(lat_deg, lon_deg, when)
        solars.append(this_solar)
    solars_array: NDArray[np.floating] = np.stack(solars, axis=-1)
    return solars_array


def toa_solar_rad_approx(
    lat_deg: NDArray[np.floating] | float,
    lon_deg: NDArray[np.floating] | float,
    when: pd.Timestamp,
) -> NDArray[np.floating]:
    """
    Calculate the solar radiation at a given latitude and longitude for a given time.
    """
    alt = solar_alt_approx(lat_deg, lon_deg, when)
    small = 1e-3
    alt = np.maximum(alt, small)
    day_of_year = when.dayofyear
    q = q0(day_of_year)
    rad: NDArray[np.floating] = np.asarray(q * np.sin(np.deg2rad(alt)))
    return rad


def solar_alt_approx(
    lat_deg: NDArray[np.floating] | float,
    lon_deg: NDArray[np.floating] | float,
    when: pd.Timestamp,
) -> NDArray[np.floating]:
    """
    Calculate the solar altitude at a given latitude and longitude for a given time.
    See https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
    See https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
    """
    day_of_year = when.dayofyear
    earth_tilt_deg = 23.45
    declination_deg = earth_tilt_deg * np.sin(
        (2.0 * np.pi / 365.0) * (day_of_year - 81.0)
    )
    declination_rad = np.deg2rad(declination_deg)
    lat_rad = np.deg2rad(lat_deg)
    hour_angle = get_hour_angle(when, lon_deg)
    a = np.cos(lat_rad) * np.cos(declination_rad) * np.cos(np.deg2rad(hour_angle))
    b = np.sin(lat_rad) * np.sin(declination_rad)
    result: NDArray[np.floating] = np.asarray(np.rad2deg(np.arcsin(a + b)))
    return result


def q0(day_of_year: int) -> float:
    """
    Calculate the solar constant at a given day of the year.
    See https://en.wikipedia.org/wiki/Solar_irradiance
    """
    return float(1.0 + 0.034 * np.cos(2.0 * np.pi / 365.24 * day_of_year))


def get_hour_angle(
    when: pd.Timestamp, lon_deg: float | NDArray[np.floating]
) -> float | NDArray[np.floating]:
    """
    Calculate the hour angle at a given time and longitude.
    """
    tmp = 60.0 * when.hour + when.minute + 4.0 * lon_deg + eq_of_time(when.dayofyear)
    solar_time = tmp / 60.0
    result: float | NDArray[np.floating] = np.asarray(15.0 * (solar_time - 12.0))
    return result


def eq_of_time(day_of_year: int) -> float:
    """
    Calculate the equation of time at a given day of the year.
    See https://www.pveducation.org/pvcdrom/properties-of-sunlight/the-suns-position
    """
    tmp = 2 * np.pi / 364.0 * (day_of_year - 81)
    return float(9.87 * np.sin(2.0 * tmp) - 7.53 * np.cos(tmp) - 1.5 * np.sin(tmp))
