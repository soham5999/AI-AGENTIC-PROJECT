"""Orchestrates validation, prediction, diagnosis, recommendation, memory and alert."""
from agent.tools import predict_maintenance, diagnose_vehicle, alert_tool
from agent.memory import save_prediction
from utils.preprocessing import validate_vehicle
from utils.recommendations import generate_recommendation
from agent.prompts import natural_language_explanation

class VehicleMaintenanceAgent:
    def analyze(self, vehicle_id: str, parameters: dict) -> dict:
        errors = validate_vehicle(parameters)
        if errors: raise ValueError(" ".join(errors))
        prediction = predict_maintenance(parameters)
        diagnosis = diagnose_vehicle(parameters)
        recommendation = generate_recommendation(prediction["risk_probability"], diagnosis)
        alert = alert_tool(vehicle_id, prediction["risk_probability"], recommendation)
        result = {"vehicle_id": vehicle_id, **prediction, "diagnosis": diagnosis, "recommendation": recommendation, "alert": alert}
        save_prediction(vehicle_id, parameters, prediction["risk_probability"], prediction["prediction"], diagnosis, recommendation)
        result["explanation"] = natural_language_explanation(result)
        return result
