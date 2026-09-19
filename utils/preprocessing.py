"""Input validation and model feature preparation."""
from typing import Mapping, Any
import pandas as pd

RANGES = {"vehicle_age_years": (0, 30), "mileage_km": (0, 500000), "engine_temp_c": (50, 160), "oil_quality_pct": (0, 100), "brake_wear_pct": (0, 100), "battery_voltage_v": (8, 16), "tire_pressure_psi": (10, 50), "service_gap_days": (0, 1000), "previous_repairs": (0, 50), "vibration_mm_s": (0, 20)}
FEATURES = list(RANGES)

def validate_vehicle(data: Mapping[str, Any]) -> list[str]:
    errors = []
    for name, (low, high) in RANGES.items():
        try:
            value = float(data[name])
            if not low <= value <= high:
                errors.append(f"{name} must be between {low} and {high}.")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name} must be a number.")
    return errors

def to_frame(data: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{key: float(data[key]) for key in FEATURES}], columns=FEATURES)
