"""Agent tool functions. These are deterministic and auditable."""
from pathlib import Path
import joblib
from utils.preprocessing import to_frame

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "saved_model.pkl"

def predict_maintenance(parameters: dict, model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError("Trained model is missing. Run: python models/train_model.py")
    bundle = joblib.load(model_path)
    X = to_frame(parameters)
    prediction = int(bundle["model"].predict(X)[0])
    probability = float(bundle["model"].predict_proba(X)[0][1])
    return {"prediction": prediction, "risk_probability": probability, "model_name": bundle["model_name"]}

def diagnose_vehicle(p: dict) -> list[str]:
    concerns = []
    if p["engine_temp_c"] > 115: concerns.append("Possible high engine temperature; inspection recommended.")
    if p["oil_quality_pct"] < 35: concerns.append("Possible poor oil quality; service inspection recommended.")
    if p["brake_wear_pct"] > 75: concerns.append("Possible high brake wear; brake inspection recommended.")
    if p["battery_voltage_v"] < 11.5: concerns.append("Possible low battery voltage; electrical inspection recommended.")
    if not 27 <= p["tire_pressure_psi"] <= 36: concerns.append("Possible abnormal tire pressure; check and adjust pressure.")
    if p["service_gap_days"] > 365: concerns.append("Long service interval; preventive service is recommended.")
    if p["vibration_mm_s"] > 8: concerns.append("Possible excessive vibration; inspection recommended.")
    return concerns

def alert_tool(vehicle_id: str, risk: float, recommendation: dict) -> str:
    if recommendation["priority"] == "LOW": return f"No urgent alert: {vehicle_id} has low predicted maintenance risk."
    return f"Maintenance Alert: Vehicle {vehicle_id} has {recommendation['priority'].lower()} predicted maintenance risk. Recommended action: {recommendation['recommendation']}"
