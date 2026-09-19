"""Modern multi-section Streamlit dashboard."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from agent.memory import count_summary, get_history
from agent.vehicle_agent import VehicleMaintenanceAgent
from data.external_dataset import get_dataset_for_training
from utils.preprocessing import FEATURES, RANGES

st.set_page_config(page_title="Vehicle Maintenance AI", page_icon="🚗", layout="wide")

@st.cache_data
def dataset() -> pd.DataFrame:
    return get_dataset_for_training()

@st.cache_resource
def agent() -> VehicleMaintenanceAgent:
    return VehicleMaintenanceAgent()

def header() -> None:
    st.title("🚗 Vehicle Maintenance AI Agent")
    st.caption("ML-powered maintenance risk analysis with agent tools, memory, recommendations, and alerts")
    st.warning("Synthetic educational data and decision-support prototype. It does not diagnose mechanical failure. A qualified technician must verify all recommendations.")

def dashboard() -> None:
    header()
    try:
        total, required, high, average = count_summary()
    except Exception as exc:
        st.error(f"Database error: {exc}")
        total = required = high = 0; average = 0.0
    a, b, c, d = st.columns(4)
    a.metric("Vehicles analyzed", total); b.metric("Maintenance predicted", required)
    c.metric("High-risk vehicles", high); d.metric("Average risk", f"{average:.1%}")
    data = dataset()
    st.subheader("System overview")
    a, b, c = st.columns(3)
    a.metric("Dataset records", len(data)); b.metric("Input features", len(FEATURES)); c.metric("Dataset target rate", f"{data.maintenance_required.mean():.1%}")
    st.subheader("Agent workflow")
    st.info("Vehicle input → validation → ML prediction → diagnostic analysis → recommendation → SQLite memory → alert → human technician review")
    st.success("Prediction engine ready. Optional API integration and offline fallback are enabled.")

def analysis() -> None:
    st.header("Vehicle Analysis")
    with st.form("analysis"):
        vehicle_id = st.text_input("Vehicle ID", "VH-001")
        defaults = [5, 80000, 98, 70, 40, 12.4, 32, 180, 2, 3]
        values = {}; columns = st.columns(2)
        for i, feature in enumerate(FEATURES):
            values[feature] = columns[i % 2].number_input(feature.replace("_", " ").title(), min_value=float(RANGES[feature][0]), max_value=float(RANGES[feature][1]), value=float(defaults[i]), step=0.1)
        submitted = st.form_submit_button("Run AI Analysis", type="primary")
    if not submitted: return
    if not vehicle_id.strip(): st.error("Vehicle ID is required."); return
    try:
        result = agent().analyze(vehicle_id.strip(), values)
        st.subheader("Structured ML result")
        a, b, c = st.columns(3)
        a.metric("Prediction", "MAINTENANCE REQUIRED" if result["prediction"] else "NOT CURRENTLY REQUIRED")
        b.metric("Risk probability", f"{result['risk_probability']:.1%}")
        c.metric("Priority", result["recommendation"]["priority"])
        st.progress(min(max(result["risk_probability"], 0.0), 1.0), text="Predicted maintenance risk")
        st.subheader("Agent analysis")
        st.write("**Selected model:**", result["model_name"])
        st.write("**Possible concerns:**")
        if result["diagnosis"]:
            for concern in result["diagnosis"]: st.write(f"- {concern}")
        else: st.write("No threshold-based concern identified.")
        st.write("**Recommended action:**", result["recommendation"]["recommendation"])
        st.write("**Reason:**", result["recommendation"]["reason"])
        st.info(result["alert"])
        if result.get("explanation"): st.write("**Optional natural-language explanation:**", result["explanation"])
        with st.expander("Validated input passed to the model"):
            st.json(values)
    except (ValueError, FileNotFoundError) as exc: st.error(str(exc))
    except Exception as exc: st.error(f"Analysis failed safely: {type(exc).__name__}: {exc}")

def agent_page() -> None:
    st.header("AI Agent")
    st.write("The agent does not replace the ML model. It orchestrates auditable tools around the model output.")
    for name, description in [("1. Validation", "Checks bounds and data types."), ("2. ML prediction", "Loads the selected trained classifier and calculates probability."), ("3. Diagnosis", "Identifies possible concerns from threshold rules."), ("4. Recommendation", "Assigns LOW, MEDIUM, or HIGH priority."), ("5. Memory", "Stores the structured result in SQLite."), ("6. Alert", "Creates a simulated notification."), ("7. Human review", "A technician verifies the recommendation.")]:
        st.markdown(f"### {name}"); st.write(description)
    st.warning("Possible concern and inspection recommended do not mean a component has definitely failed.")

def history() -> None:
    st.header("Maintenance History")
    vehicle_id = st.text_input("Vehicle ID", "VH-001", key="history_id")
    if st.button("Search history", type="primary"):
        rows = get_history(vehicle_id.strip())
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else: st.info("No stored predictions found for this vehicle.")

def performance() -> None:
    st.header("Model Performance")
    metrics_path = ROOT / "visualizations" / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best = metrics.loc[metrics.f1.idxmax()]
        st.success(f"Selected model: {best['model']} because it has the highest measured F1 score ({best['f1']:.3f}).")
        st.dataframe(metrics.style.format({x: "{:.3f}" for x in ["accuracy", "precision", "recall", "f1"]}), use_container_width=True)
    else: st.info("Run python models/train_model.py to generate metrics.")
    for name in ["model_comparison.png", "confusion_matrix.png", "feature_importance.png", "maintenance_distribution.png"]:
        path = ROOT / "visualizations" / name
        if path.exists(): st.image(str(path), caption=name.replace("_", " ").title())

def explorer() -> None:
    st.header("Dataset / Data Explorer")
    data = dataset(); st.write(f"Records: {len(data)} | Columns: {len(data.columns)}")
    st.dataframe(data.head(25), use_container_width=True)
    st.dataframe(data.describe().T, use_container_width=True)
    if os.getenv("DATASET_API_URL"): st.success("Source: configured external API, with local fallback.")
    else: st.info("Source: local synthetic educational dataset. Configure DATASET_API_URL only for a compatible provider.")

def alerts() -> None:
    st.header("Alerts / Recommendations")
    total, required, high, average = count_summary()
    st.metric("High-risk stored alerts", high); st.metric("Average stored risk", f"{average:.1%}")
    if high: st.warning("High-risk predictions exist. Schedule technician review as soon as practical.")
    else: st.success("No high-risk alerts are currently stored.")

def about() -> None:
    st.header("About the Project")
    st.markdown("This B.Tech academic prototype demonstrates ML classification plus agentic orchestration. It uses Logistic Regression, SVM, Random Forest, and Gradient Boosting; selects the highest-F1 model; stores results in SQLite; and provides optional API and LLM integrations.")
    st.caption("Academic disclaimer: synthetic data is not real-world vehicle data, and AI output is not a mechanical diagnosis.")

def main() -> None:
    page = st.sidebar.radio("Navigation", ["Dashboard", "Vehicle Analysis", "AI Agent", "Maintenance History", "Model Performance", "Dataset / Data Explorer", "Alerts / Recommendations", "About"])
    {"Dashboard": dashboard, "Vehicle Analysis": analysis, "AI Agent": agent_page, "Maintenance History": history, "Model Performance": performance, "Dataset / Data Explorer": explorer, "Alerts / Recommendations": alerts, "About": about}[page]()

if __name__ == "__main__": main()
