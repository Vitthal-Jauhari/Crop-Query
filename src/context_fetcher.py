"""
CropQuery — Context Fetcher
B1: Weather API (Open-Meteo, free, no key required)
B2: Location + agro-climatic zone (IP-based geolocation fallback)
D1: Core irrigation advisory logic
"""

import json
import datetime
from typing import Optional, Dict, Any

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("[ContextFetcher] requests not available — install with pip install requests")

from .config import WEATHER_API_URL, GEOCODE_API_URL


# ── Agro-climatic zone lookup (simplified, India) ─────────────────────────
_STATE_ZONE = {
    "Punjab":           "Trans-Gangetic Plains",
    "Haryana":          "Trans-Gangetic Plains",
    "Uttar Pradesh":    "Upper Gangetic Plains",
    "Bihar":            "Middle Gangetic Plains",
    "West Bengal":      "Lower Gangetic Plains",
    "Maharashtra":      "Western Plateau and Hills",
    "Karnataka":        "Southern Plateau and Hills",
    "Tamil Nadu":       "East Coast Plains and Hills",
    "Andhra Pradesh":   "East Coast Plains and Hills",
    "Telangana":        "Southern Plateau and Hills",
    "Rajasthan":        "Western Dry Region",
    "Gujarat":          "Gujarat Plains and Hills",
    "Madhya Pradesh":   "Central Plateau and Hills",
    "Odisha":           "East Coast Plains and Hills",
    "Kerala":           "Western Ghats and Coastal Plains",
    "Assam":            "North-Eastern Hills",
    "Himachal Pradesh": "Western Himalayas",
    "Uttarakhand":      "Western Himalayas",
    "Jharkhand":        "Eastern Plateau and Hills",
    "Chhattisgarh":     "Eastern Plateau and Hills",
}


def _current_season() -> str:
    month = datetime.datetime.now().month
    if 6 <= month <= 10:
        return "Kharif"
    elif 11 <= month or month <= 3:
        return "Rabi"
    else:
        return "Zaid"


# ── Crop water requirements (mm/week) by growth stage ─────────────────────
_CROP_WATER_REQ: Dict[str, Dict[str, float]] = {
    "Tomato":   {"seedling": 15, "vegetative": 25, "flowering": 40, "fruiting": 45, "default": 35},
    "Potato":   {"seedling": 20, "vegetative": 35, "flowering": 40, "fruiting": 40, "default": 35},
    "Wheat":    {"seedling": 18, "vegetative": 28, "flowering": 35, "fruiting": 30, "default": 28},
    "Rice":     {"seedling": 40, "vegetative": 55, "flowering": 60, "fruiting": 50, "default": 50},
    "Corn":     {"seedling": 20, "vegetative": 30, "flowering": 45, "fruiting": 40, "default": 35},
    "Grape":    {"seedling": 25, "vegetative": 35, "flowering": 30, "fruiting": 40, "default": 33},
    "Apple":    {"seedling": 20, "vegetative": 30, "flowering": 35, "fruiting": 40, "default": 30},
    "Cashew":   {"seedling": 15, "vegetative": 20, "flowering": 25, "fruiting": 30, "default": 22},
    "Pea":      {"seedling": 15, "vegetative": 22, "flowering": 30, "fruiting": 28, "default": 24},
    "default":  {"default": 30},
}

# Soil saturation threshold: if rainfall in last 14 days exceeds this,
# warn farmer before advising more irrigation.
_SATURATION_THRESHOLD_MM = 100

# Fraction of recent rainfall still available as soil moisture carry-over.
_CARRYOVER_FACTOR = 0.30


def spray_timing_advisory(weather_ctx: dict) -> str:
    """
    Returns a spray timing message — warning if conditions are bad,
    positive confirmation if conditions are good.
    Always returns a non-empty string.
    """
    if not weather_ctx:
        return "Check local weather before spraying."

    wind     = weather_ctx.get("wind_kmh", 0) or 0
    rain_prob = weather_ctx.get("rain_prob_6h", 0) or 0
    warnings = []

    if wind > 15:
        warnings.append(
            f"⚠️ Wind is {wind:.0f} km/h — do not spray now. Wait for calm weather (<15 km/h)."
        )
    if rain_prob > 40:
        warnings.append(
            f"⚠️ Rain expected in next 6 hours ({rain_prob:.0f}% chance) — "
            f"spraying now will wash off the medicine. Wait."
        )

    if warnings:
        return "\n".join(warnings)

    # Positive advisory when conditions are clear
    return (
        f"✅ Good spraying conditions now — wind {wind:.0f} km/h, "
        f"rain probability {rain_prob:.0f}% in next 6 hours. Safe to spray."
    )


class ContextFetcher:
    """
    Fetches weather (B1) and location (B2) context, and produces
    irrigation advisory text (D1). All network calls fail gracefully.
    """

    def __init__(self):
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._location: Optional[dict] = None

    # ── B2: Location ──────────────────────────────────────────────────────
    def _resolve_location(self) -> Optional[dict]:
        if not _REQUESTS_OK:
            return None
        try:
            geo      = requests.get("https://ipapi.co/json/", timeout=5).json()
            lat      = float(geo.get("latitude", 0))
            lon      = float(geo.get("longitude", 0))
            state    = geo.get("region", "")
            district = geo.get("city", "")
            self._lat = lat
            self._lon = lon

            zone   = _STATE_ZONE.get(state, None)
            season = _current_season()

            return {
                "state":              state,
                "district":           district,
                "latitude":           lat,
                "longitude":          lon,
                # None means non-Indian location; handled gracefully downstream
                "agro_climatic_zone": zone,
                "current_season":     season,
            }
        except Exception as e:
            print(f"[ContextFetcher] Location failed: {e}")
            return None

    def get_location(self) -> dict:
        if self._location is None:
            self._location = self._resolve_location() or {}
        return self._location

    # ── B1: Weather ───────────────────────────────────────────────────────
    def get_weather(self) -> dict:
        if not _REQUESTS_OK:
            return {}
        loc = self.get_location()
        lat = loc.get("latitude") or 28.6
        lon = loc.get("longitude") or 77.2

        try:
            params = {
                "latitude":      lat,
                "longitude":     lon,
                "current":       "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "hourly":        "precipitation_probability",
                "daily":         "precipitation_sum,temperature_2m_max",
                "past_days":     14,
                "forecast_days": 7,
                "timezone":      "auto",
            }
            r = requests.get(WEATHER_API_URL, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()

            current = data.get("current", {})
            daily   = data.get("daily", {})
            hourly  = data.get("hourly", {})

            temp_c       = current.get("temperature_2m")
            humidity_pct = current.get("relative_humidity_2m")
            wind_kmh     = current.get("wind_speed_10m")

            precip_list  = daily.get("precipitation_sum", [])
            rainfall_14d = sum(p for p in precip_list[:14] if p) if precip_list else None

            rain_probs   = hourly.get("precipitation_probability", [])
            rain_prob_6h = max(rain_probs[:6]) if rain_probs else 0

            dates       = daily.get("time", [])
            rains       = daily.get("precipitation_sum", [])
            temps       = daily.get("temperature_2m_max", [])
            forecast_7d = [
                {"date": d, "rain_mm": r or 0, "temp_max_c": t}
                for d, r, t in zip(dates[-7:], rains[-7:], temps[-7:])
            ]

            return {
                "temp_c":          temp_c,
                "humidity_pct":    humidity_pct,
                "wind_kmh":        wind_kmh,
                "rainfall_14d_mm": round(rainfall_14d, 1) if rainfall_14d is not None else None,
                "rain_prob_6h":    rain_prob_6h,
                "forecast_7d":     forecast_7d,
            }

        except Exception as e:
            print(f"[ContextFetcher] Weather fetch failed: {e}")
            return {}

    # ── D1: Irrigation advisory ───────────────────────────────────────────
    def get_irrigation_advice(
        self,
        crop: Optional[str] = None,
        growth_stage: str = "default",
        weather_ctx: Optional[dict] = None,
    ) -> str:
        w            = weather_ctx or {}
        forecast_7d  = w.get("forecast_7d", [])
        rainfall_14d = w.get("rainfall_14d_mm") or 0

        crop_key  = crop if crop in _CROP_WATER_REQ else "default"
        stage_key = growth_stage if growth_stage in _CROP_WATER_REQ.get(crop_key, {}) else "default"
        water_need_mm = _CROP_WATER_REQ[crop_key].get(stage_key, 30)

        if not forecast_7d:
            return (
                f"[Irrigation] {crop or 'Crop'} at {growth_stage} stage needs "
                f"~{water_need_mm} mm/week. "
                "Weather forecast unavailable — irrigate based on soil moisture observation."
            )

        rain_next_7d = sum(d.get("rain_mm", 0) for d in forecast_7d[:7])

        # Soil moisture carry-over from recent heavy rain
        recent_carryover  = min(rainfall_14d * _CARRYOVER_FACTOR, water_need_mm)
        effective_supply  = rain_next_7d + recent_carryover
        deficit_mm        = max(0, water_need_mm - effective_supply)

        # Saturation warning
        saturation_note = ""
        if rainfall_14d > _SATURATION_THRESHOLD_MM:
            saturation_note = (
                f" ⚠️ Heavy rain in last 14 days ({rainfall_14d:.0f} mm) — "
                f"soil may be saturated. Check soil moisture before irrigating."
            )

        # Best irrigation window (lowest forecast rain day in next 4 days)
        best_day = sorted(forecast_7d[:4], key=lambda d: d.get("rain_mm", 0))[0]["date"]

        crop_label = crop or "Your crop"

        if deficit_mm <= 0:
            return (
                f"[Irrigation] {crop_label} at {growth_stage} stage needs "
                f"{water_need_mm} mm/week. "
                f"Between forecast rain ({rain_next_7d:.0f} mm) and recent soil moisture carry-over "
                f"({recent_carryover:.0f} mm), no additional irrigation needed this week."
                + saturation_note
            )

        return (
            f"[Irrigation] {crop_label} at {growth_stage} stage needs {water_need_mm} mm/week. "
            f"Forecast: {rain_next_7d:.0f} mm + ~{recent_carryover:.0f} mm soil carry-over "
            f"= {effective_supply:.0f} mm available — irrigate {deficit_mm:.0f} mm deficit. "
            f"Best window: {best_day} (lowest rain probability). "
            f"Avoid irrigating on days with >40% rain probability."
            + saturation_note
        )
