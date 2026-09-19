"""Generate a reproducible synthetic educational vehicle dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["vehicle_age_years", "mileage_km", "engine_temp_c", "oil_quality_pct", "brake_wear_pct", "battery_voltage_v", "tire_pressure_psi", "service_gap_days", "previous_repairs", "vibration_mm_s"]

def generate_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vehicle_age_years": rng.uniform(0, 25, n), "mileage_km": rng.uniform(0, 400000, n),
        "engine_temp_c": rng.normal(95, 14, n).clip(60, 145), "oil_quality_pct": rng.uniform(15, 100, n),
        "brake_wear_pct": rng.uniform(0, 100, n), "battery_voltage_v": rng.normal(12.4, 1.0, n).clip(9, 15.5),
        "tire_pressure_psi": rng.normal(32, 5, n).clip(18, 46), "service_gap_days": rng.uniform(0, 900, n),
        "previous_repairs": rng.poisson(3, n).clip(0, 20), "vibration_mm_s": rng.gamma(2, 1.8, n).clip(0, 18),
    })
    score = (0.06 * df.vehicle_age_years + 0.000004 * df.mileage_km + 0.07 * (df.engine_temp_c - 95).clip(lower=0)
             + 0.035 * (70 - df.oil_quality_pct).clip(lower=0) + 0.05 * df.brake_wear_pct
             + 1.7 * (12.0 - df.battery_voltage_v).clip(lower=0) + 0.08 * (abs(df.tire_pressure_psi - 32) - 4).clip(lower=0)
             + 0.004 * df.service_gap_days + 0.25 * df.previous_repairs + 0.7 * df.vibration_mm_s + rng.normal(0, 2.0, n))
    threshold = float(np.quantile(score, 0.62))
    df["maintenance_required"] = (score >= threshold).astype(int)
    return df

if __name__ == "__main__":
    out = Path(__file__).with_name("vehicle_maintenance_dataset.csv")
    out.parent.mkdir(exist_ok=True)
    generate_dataset().to_csv(out, index=False)
    print(f"Generated {len(generate_dataset())} records at {out}")
