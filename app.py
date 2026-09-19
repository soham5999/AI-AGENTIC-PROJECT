"""Streamlit dashboard for the academic prototype."""
from pathlib import Path
import os, sys, pandas as pd, streamlit as st
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); load_dotenv(ROOT / ".env")
from agent.vehicle_agent import VehicleMaintenanceAgent
from agent.memory import count_summary, get_history
from utils.preprocessing import FEATURES, RANGES

st.set_page_config(page_title="AI Agent for Vehicle Maintenance Prediction", page_icon="🔧", layout="wide")
st.title("AI Agent for Vehicle Maintenance Prediction")
st.warning("Synthetic educational dataset and decision-support prototype. It does not diagnose actual mechanical failure. Final decisions must be verified by a qualified technician.")
try: total, required, high, avg = count_summary()
except Exception as exc: st.error(f"Database error: {exc}"); total=required=high=0; avg=0
c1,c2,c3,c4=st.columns(4); c1.metric("Total vehicles analyzed", total); c2.metric("Maintenance predicted", required); c3.metric("High-risk vehicles", high); c4.metric("Average predicted risk", f"{avg:.1%}")

@st.cache_resource
def get_agent(): return VehicleMaintenanceAgent()

st.header("Vehicle Analysis")
with st.form("analysis"):
    vehicle_id = st.text_input("Vehicle ID", "VH-001")
    cols = st.columns(2); values = {}
    defaults = [5, 80000, 98, 70, 40, 12.4, 32, 180, 2, 3]
    for i, feature in enumerate(FEATURES):
        values[feature] = cols[i % 2].number_input(feature.replace("_", " ").title(), min_value=float(RANGES[feature][0]), max_value=float(RANGES[feature][1]), value=float(defaults[i]))
    submitted = st.form_submit_button("Analyze Vehicle", type="primary")
if submitted:
    if not vehicle_id.strip(): st.error("Vehicle ID is required.")
    else:
        try:
            result = get_agent().analyze(vehicle_id.strip(), values)
            st.subheader("Analysis Result")
            a,b,c=st.columns(3); a.metric("Maintenance prediction", "REQUIRED" if result["prediction"] else "NOT CURRENTLY REQUIRED"); b.metric("Risk probability", f"{result['risk_probability']:.1%}"); c.metric("Priority", result["recommendation"]["priority"])
            st.write("**Possible concerns**"); st.write(result["diagnosis"] or "No threshold-based concern identified.")
            st.write("**Recommended action:**", result["recommendation"]["recommendation"]); st.write("**Reason:**", result["recommendation"]["reason"]); st.info(result["alert"])
            if result.get("explanation"): st.write("**Optional AI explanation:**", result["explanation"])
            st.caption(f"Selected model: {result['model_name']}")
        except (ValueError, FileNotFoundError) as exc: st.error(str(exc))
        except Exception as exc: st.error(f"Analysis failed safely: {type(exc).__name__}: {exc}. Train the model and check the dataset.")

st.header("Model Performance")
metrics_path = ROOT / "visualizations" / "model_metrics.csv"
if metrics_path.exists(): st.dataframe(pd.read_csv(metrics_path).style.format({c:"{:.3f}" for c in ["accuracy","precision","recall","f1"]}), use_container_width=True)
else: st.info("Run python models/train_model.py to generate metrics and charts.")
for name in ["model_comparison.png","confusion_matrix.png","feature_importance.png","maintenance_distribution.png"]:
    if (ROOT/"visualizations"/name).exists(): st.image(str(ROOT/"visualizations"/name), caption=name.replace("_", " ").title())

st.header("Maintenance History")
history_id = st.text_input("Vehicle ID to search", "VH-001", key="history")
if st.button("View History"):
    rows=get_history(history_id.strip()); st.dataframe(pd.DataFrame(rows), use_container_width=True) if rows else st.info("No history found for this vehicle.")
