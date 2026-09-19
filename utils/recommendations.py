"""Deterministic maintenance recommendation rules."""
def generate_recommendation(risk: float, concerns: list[str]) -> dict:
    if risk >= 0.70:
        priority = "HIGH"
        recommendation = "Schedule inspection/service as soon as practical."
    elif risk >= 0.40:
        priority = "MEDIUM"
        recommendation = "Schedule preventive maintenance soon."
    else:
        priority = "LOW"
        recommendation = "Continue monitoring and follow the normal service schedule."
    reason = "Possible concerns: " + "; ".join(concerns) if concerns else "No threshold-based concern was identified."
    return {"priority": priority, "recommendation": recommendation, "reason": reason}
