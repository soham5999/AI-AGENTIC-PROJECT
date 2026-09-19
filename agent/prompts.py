"""Optional LLM explanation layer; deterministic operation never depends on it."""
import os, json

def natural_language_explanation(result: dict) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key: return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = "Explain this vehicle maintenance decision-support result without diagnosing failure or inventing readings. Use the supplied structured data only:\n" + json.dumps(result, indent=2)
        response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role":"system", "content":"You are a cautious vehicle maintenance explanation assistant. Final decisions require a qualified technician."}, {"role":"user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content
    except Exception as exc:
        return f"Optional language layer unavailable ({type(exc).__name__}); deterministic results are shown."
