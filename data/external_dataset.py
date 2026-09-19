"""Optional external dataset integration with a safe local fallback."""
from __future__ import annotations
import io
import os
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from data.generate_dataset import generate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vehicle_maintenance_dataset.csv"
REQUIRED_COLUMNS = ["vehicle_age_years", "mileage_km", "engine_temp_c", "oil_quality_pct", "brake_wear_pct", "battery_voltage_v", "tire_pressure_psi", "service_gap_days", "previous_repairs", "vibration_mm_s", "maintenance_required"]

def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("External dataset returned no records.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"External dataset is missing required columns: {missing}")
    return frame[REQUIRED_COLUMNS].copy()

def _download_api() -> pd.DataFrame | None:
    load_dotenv(ROOT / ".env")
    url, key = os.getenv("DATASET_API_URL"), os.getenv("DATASET_API_KEY")
    if not url:
        return None
    headers = {"Accept": "application/json, text/csv"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    text = response.text
    if "json" in response.headers.get("Content-Type", "").lower() or text.lstrip().startswith(("{", "[")):
        payload = response.json()
        if isinstance(payload, dict):
            for key_name in ("data", "records", "results", "items"):
                if key_name in payload:
                    payload = payload[key_name]
                    break
        return _normalise(pd.DataFrame(payload))
    return _normalise(pd.read_csv(io.StringIO(text)))

def get_dataset_for_training() -> pd.DataFrame:
    """Use a configured API dataset when compatible; otherwise use local data."""
    try:
        external = _download_api()
        if external is not None and len(external) >= 2500:
            external.to_csv(DATA_PATH, index=False)
            return external
    except Exception as exc:
        print(f"External dataset unavailable; using fallback: {exc}")
    if DATA_PATH.exists():
        try:
            local = _normalise(pd.read_csv(DATA_PATH))
            if len(local) >= 2500:
                return local
        except Exception as exc:
            print(f"Local dataset invalid; regenerating fallback: {exc}")
    local = generate_dataset(n=3000, seed=42)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    local.to_csv(DATA_PATH, index=False)
    return local
