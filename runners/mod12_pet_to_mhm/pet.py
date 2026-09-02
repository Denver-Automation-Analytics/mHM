"""PET calculation kernels — pure math, no I/O.

Authors
-------
- Matthias Kelbling
- Simon Lüdke
- Stephan Thober
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


METHODS_REQUIRING_TMAX_TMIN = {
    "hargreaves_samani",
    "hargreaves-samani",
    "HS",
    "baier_robertson",
    "baier-robertson",
}
METHODS_REQUIRING_TAVG = {
    "hargreaves_samani",
    "hargreaves-samani",
    "HS",
    "oudinmcguinness_bordne",
    "mcguinness-bordne",
    "hamon",
    "jensen_haise",
    "jensen-haise",
    "penman_monteith",
    "penman-monteith",
}


def _daylength_hours(lat_deg: np.ndarray, time: datetime) -> np.ndarray:
    """
    Approximate day length (hours) from latitude + day-of-year.

    Good enough for PET empirical methods (Hamon, Blaney-Criddle, Thornthwaite).
    """
    lat = np.deg2rad(lat_deg)
    doy = time.timetuple().tm_yday
    # solar declination (radians); common approximation
    delta = 0.409 * np.sin(2.0 * np.pi * (doy - 81) / 365.0)
    # sunset hour angle
    cos_omega = -np.tan(lat) * np.tan(delta)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)
    # day length in hours
    return 24.0 / np.pi * omega


def _sat_vapor_pressure_kpa(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (kPa) for temperature in °C (FAO-56)."""
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


def e_rad_calculator(time: datetime, lat: np.ndarray) -> np.ndarray:
    """Calculate extraterrestrial radiation (MJ/m²/day) for a given day/lat."""
    doy = pd.Timestamp(time).day_of_year - 1
    dist = 1 + (0.033 * np.cos((2 * np.pi * doy) / 365))
    dec = np.radians(-23.44 * np.cos(np.radians((360 / 365) * (doy + 10))))
    ang = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    e_rad = (ang * np.sin(lat) * np.sin(dec)) + (
        np.cos(lat) * np.cos(dec) * np.sin(ang)
    )
    return 37.5 * dist * e_rad


def validate_tmin_tmax(tmin, tmax):
    """Test that tmin smaller or equal tmax at every point and time."""
    comparison = np.where(~np.isnan(tmin + tmax), tmin <= tmax, True)
    if not bool(np.all(np.asarray(comparison))):
        raise ValueError("Tmin not always less or equal to tmax.")


def pet_calculator(
    # tavg: np.ndarray,
    lat: np.ndarray,
    time: datetime,
    stat_freq: str,
    method: str = "oudin",
    l_heat: float = 2.26,  # latent heat of vaporization (MJ/kg) or compatible with e_rad units
    w_density: float = 977.0,  # water density (kg/m3); used in your original scaling
    **kwargs,
) -> np.ndarray:
    """
    Calculate PET/ET0 using several formulations (as in your figure).

    Required inputs depend on `method`:
      - 'oudin' (default): uses tavg, lat, time (via e_rad_calculator)
      - 'hargreaves_samani': needs tmin, tmax
      - 'mcguinness_bordne': needs Ta (or uses tavg if not given)
      - 'hamon': needs k (optional, default 1.0); uses DL and es computed internally
      - 'baier_robertson': needs tmin, tmax; uses Re (extraterrestrial radiation) from e_rad_calculator
      - 'blaney_criddle': uses DL computed internally
      - 'thornthwaite': needs I and k (heat index + exponent), and typically monthly tavg; uses DL
      - 'jensen_haise': uses Re and Tavg (tavg)
      - 'priestley_taylor': needs delta, rn, g, gamma
      - 'milly_dunne': needs rn, g
      - 'penman_monteith': needs delta, rn, g, gamma, u2, es, ea (or will compute es from tavg if not provided)
      - 'penman_monteith_co2': same as PM plus co2 (ppm)

    Notes
    -----
      - This function assumes your e_rad_calculator(time, lat) returns R_e consistent with l_heat & w_density scaling.
      - Output unit will match your original Oudin output (typically mm/day), then converted to hourly if needed.
    """
    method = method.lower()

    # Common radiation term from your existing pipeline
    # (You already have this function elsewhere.)
    e_rad = e_rad_calculator(time, lat)  # treated as R_e in the figure

    # Daylength for methods that need it
    DL = kwargs.get("DL", _daylength_hours(lat, time))  # hours

    # Convenience
    tavg = kwargs.get("tavg")
    tmin = kwargs.get("tmin")
    tmax = kwargs.get("tmax")

    if method == "oudin":
        pet = (e_rad / (l_heat * w_density)) * ((tavg + 5.0) / 100.0) * 1000.0
        pet = np.where(tavg < -5.0, 0.0, pet)

    elif method in {"hargreaves_samani", "hargreaves-samani", "HS"}:
        tmin = kwargs["tmin"]
        tmax = kwargs["tmax"]
        pet = (
            0.0023
            * (e_rad / (l_heat * w_density))
            * np.sqrt(np.maximum(tmax - tmin, 0.0))
            * (tavg + 17.8)
            * 1000.0
        )

    elif method in {"mcguinness_bordne", "mcguinness-bordne"}:
        # Figure uses (Ta + 5)/68
        pet = 1000.0 * (e_rad / (l_heat * w_density)) * ((tavg + 5.0) / 68.0)

    elif method == "hamon":
        # PET = k * 0.165 * 216.7 * (DL/12) * es/(tavg+273.3)
        k = float(kwargs.get("k", 1.0))
        es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
        pet = k * 0.165 * 216.7 * (DL / 12.0) * (es / (tavg + 273.3))

    # elif method in {"baier_robertson", "baier-robertson"}:
    #     tmin = kwargs["tmin"]
    #     tmax = kwargs["tmax"]
    #     pet = 0.157 * tmax + 0.158 * (tmax - tmin) + 0.109 * e_rad - 5.39

    elif method in {"blaney_criddle", "blaney-criddle"}:
        pet = 0.825 * (0.46 * tavg + 8.13) * ((100.0 * DL) / (365.0 * 12.0))

    # elif method == "thornthwaite":
    #     # PET = 16*(DL/360)*((10*tavg)/I)^k
    #     # Typically monthly; you must supply I and k.
    #     annual_heat_index = kwargs.get("I", None)
    #     if annual_heat_index is None:
    #         # alternative https://upcommons.upc.edu/server/api/core/bitstreams/487bda42-f690-4738-bb0c-1ae96b7c1adc/content
    #         mean_monthly_temperature = kwargs.get("mean_monthly_temperature")
    #         if mean_monthly_temperature is None:
    #             if hasattr(tavg, "resample") and "time" in tavg.dims:
    #                 mean_monthly_temperature = tavg.resample(time="MS").mean("time")
    #             else:
    #                 raise ValueError(
    #                     "Thornthwaite requires I or tavg with a time dimension."
    #                 )
    #         monthly_heat_index = np.power(mean_monthly_temperature / 5.0, 1.514)
    #         if isinstance(monthly_heat_index, xr.DataArray):
    #             annual_heat_index = monthly_heat_index.resample(time="Y").mean("time")
    #         else:
    #             annual_heat_index = np.sum(monthly_heat_index, axis=0) / 12
    #     kexp = kwargs["k"]
    #     base = (10.0 * tavg) / annual_heat_index
    #     base = np.where(base > 0, base, 0.0)
    #     pet = 16.0 * (DL / 360.0) * np.power(base, kexp)

    elif method in {"jensen_haise", "jensen-haise"}:
        pet = 1000.0 * (e_rad / (l_heat * w_density)) * (tavg / 40.0)

    # elif method in {"priestley_taylor", "priestley-taylor"}:
    #     # PET = 1.26*\Delta*(Rn-G) / (\lambda*\rho*(\Delta+\gamma))
    #     delta = kwargs["delta"]
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     gamma = kwargs["gamma"]
    #     pet = (1.26 * delta * (rn - g)) / (l_heat * w_density * (delta + gamma))

    # elif method in {"milly_dunne", "milly-dunne"}:
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     pet = 0.8 * (rn - g)

    elif method in {"penman_monteith", "penman-monteith"}:
        delta = kwargs["delta"]
        rn = kwargs["rn"]
        g = kwargs.get("g", 0.0)
        gamma = kwargs["gamma"]
        u2 = kwargs["u2"]
        es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
        ea = kwargs["ea"]
        num = 0.408 * delta * (rn - g) + gamma * (900.0 / (tavg + 273.0)) * u2 * (
            es - ea
        )
        den = delta + gamma * (1.0 + 0.34 * u2)
        pet = num / den

    # elif method in {
    #     "penman_monteith_co2",
    #     "penman-monteith[co2]",
    #     "penman_monteith[co2]",
    # }:
    #     delta = kwargs["delta"]
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     gamma = kwargs["gamma"]
    #     u2 = kwargs["u2"]
    #     co2 = kwargs["co2"]  # ppm
    #     es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
    #     ea = kwargs["ea"]
    #     num = 0.408 * delta * (rn - g) + gamma * (900.0 / (Tavg + 273.0)) * u2 * (
    #         es - ea
    #     )
    #     den = delta + gamma * (1.0 + 0.34 * (u2 + 2e-4 * (co2 - 300.0)))
    #     pet = num / den

    else:
        raise ValueError(f"Unknown method: {method}")

    # daily stays daily, otherwise convert to hourly
    return np.maximum(pet, 0.0) if stat_freq == "daily" else np.maximum(pet, 0.0) / 24.0
