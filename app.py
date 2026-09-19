"""Modern product-style Streamlit application."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.memory import count_summary, get_history
from agent.tools import predict_maintenance
from agent.vehicle_agent import VehicleMaintenanceAgent
from data.external_dataset import get_dataset_for_training
from data.upload import format_bytes, prepare_for_prediction, read_uploaded_file
from utils.preprocessing import FEATURES, RANGES

st.set_page_config(page_title="FleetCare AI", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#162033; --muted:#667085; --brand:#2563eb; --soft:#f4f7fb; }
.block-container { max-width: 1450px; padding-top: 2rem; }
[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf2; border-radius: 14px; padding: 16px; box-shadow: 0 4px 18px rgba(16,24,40,.04); }
[data-testid="stSidebar"] { border-right: 1px solid #e7ebf2; }
div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 600; transition: transform .15s ease, box-shadow .15s ease; }
div[data-testid="stButton"] > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(37,99,235,.18); }
.fleet-hero { background: linear-gradient(135deg,#102a56 0%,#2563eb 100%); color:white; padding:28px 32px; border-radius:18px; margin-bottom:20px; }
.fleet-hero h1 { margin:0; font-size:2.25rem; } .fleet-hero p { margin:.55rem 0 0; opacity:.86; }
.badge { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.78rem; font-weight:700; background:#dbeafe; color:#1d4ed8; }
.card { background:white; border:1px solid #e7ebf2; border-radius:14px; padding:18px; margin:8px 0; }
.small-muted { color:#667085; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def sample_dataset() -> pd.DataFrame:
    return get_dataset_for_training()

@st.cache_resource
def get_agent() -> VehicleMaintenanceAgent:
    return VehicleMaintenanceAgent()


def hero(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="fleet-hero"><span class="badge">FLEETCARE AI</span><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def dashboard() -> None:
    hero("Fleet maintenance intelligence", "Operational risk visibility powered by ML and a transparent agent workflow.")
    try:
        total, required, high, average = count_summary()
    except Exception:
        total = required = high = 0; average = 0.0
    a,b,c,d = st.columns(4)
    a.metric("Predictions recorded", total); b.metric("Maintenance flagged", required); c.metric("High-priority cases", high); d.metric("Average risk", f"{average:.1%}")
    st.markdown('<div class="card"><span class="badge">DEMO DATASET ACTIVE</span><h3>Ready for analysis</h3><p class="small-muted">The application is using the self-written sample dataset. Upload your organization data from Dataset / Upload when ready.</p></div>', unsafe_allow_html=True)
    data = sample_dataset()
    x,y = st.columns([1.35, 1])
    with x:
        st.subheader("Risk distribution")
        st.bar_chart(data["maintenance_required"].value_counts().rename({0:"No maintenance",1:"Maintenance"}))
    with y:
        st.subheader("Quick actions")
        if st.button("Start a prediction", use_container_width=True): st.session_state.page = "Prediction"
        if st.button("Explore sample data", use_container_width=True): st.session_state.page = "Dataset / Upload"
        if st.button("Review model metrics", use_container_width=True): st.session_state.page = "Model / AI"


def prediction() -> None:
    hero("Vehicle prediction", "Run a structured ML prediction, then review the agent's concerns and recommendation.")
    with st.form("prediction_form"):
        vehicle_id = st.text_input("Vehicle ID", "VH-001")
        defaults = [5, 80000, 98, 70, 40, 12.4, 32, 180, 2, 3]
        values = {}; columns = st.columns(2)
        for index, feature in enumerate(FEATURES):
            values[feature] = columns[index % 2].number_input(feature.replace("_", " ").title(), min_value=float(RANGES[feature][0]), max_value=float(RANGES[feature][1]), value=float(defaults[index]), step=0.1, help=f"Allowed range: {RANGES[feature][0]}–{RANGES[feature][1]}")
        run = st.form_submit_button("Run prediction", type="primary", use_container_width=True)
    if not run: return
    if not vehicle_id.strip(): st.error("Enter a vehicle ID before running a prediction."); return
    with st.spinner("Validating data and running ML + agent tools..."):
        try: result = get_agent().analyze(vehicle_id.strip(), values)
        except (ValueError, FileNotFoundError) as exc: st.error(str(exc)); return
        except Exception: st.error("The prediction could not be completed. Verify the model and input data."); return
    st.toast("Prediction completed and saved to maintenance history.", icon="✅")
    st.markdown("### Prediction result")
    a,b,c = st.columns(3)
    a.metric("Decision", "MAINTENANCE REQUIRED" if result["prediction"] else "NO CURRENT REQUIREMENT")
    b.metric("Risk probability", f"{result['risk_probability']:.1%}")
    c.metric("Priority", result["recommendation"]["priority"])
    st.progress(float(result["risk_probability"]), text="Model risk probability")
    left,right = st.columns(2)
    with left:
        st.markdown('<div class="card"><h4>Model information</h4><p>Selected model: <b>' + result["model_name"] + '</b><br>Prediction is based on the validated vehicle parameters.</p></div>', unsafe_allow_html=True)
        with st.expander("View validated input data"): st.dataframe(pd.DataFrame([values]), use_container_width=True)
    with right:
        st.markdown('<div class="card"><h4>Recommendation</h4><p>' + result["recommendation"]["recommendation"] + '</p><p class="small-muted">' + result["recommendation"]["reason"] + '</p></div>', unsafe_allow_html=True)
        st.info(result["alert"])
    st.subheader("Agent findings")
    if result["diagnosis"]:
        for concern in result["diagnosis"]: st.warning(concern)
    else: st.success("No threshold-based concern identified.")
    if result.get("explanation"): st.write(result["explanation"])
    st.caption("Decision support only. A qualified technician must verify the result.")


def dataset_upload() -> None:
    hero("Dataset & data upload", "Inspect the sample dataset or validate an organization's CSV, Excel, or DOCX table.")
    tabs = st.tabs(["Sample dataset", "Upload organization data"])
    with tabs[0]:
        data = sample_dataset(); st.success("Demo Dataset Active")
        a,b,c = st.columns(3); a.metric("Records", len(data)); b.metric("Columns", len(data.columns)); c.metric("Missing cells", int(data.isna().sum().sum()))
        st.dataframe(data.head(30), use_container_width=True)
        st.dataframe(data.describe().T, use_container_width=True)
    with tabs[1]:
        uploaded = st.file_uploader("Upload CSV, XLSX, XLS, or a DOCX table", type=["csv", "xlsx", "xls", "docx"], help="The first DOCX table is interpreted as a dataset.")
        if not uploaded:
            st.info("No organization file uploaded yet. Your sample dataset remains available.")
            return
        report = read_uploaded_file(uploaded)
        a,b,c = st.columns(3); a.metric("File", report.file_name); b.metric("Size", format_bytes(report.size_bytes)); c.metric("Format", report.extension.upper())
        if report.errors:
            for error in report.errors: st.error(error)
            return
        st.success(f"Upload validated: {len(report.frame):,} rows and {len(report.frame.columns):,} columns.")
        for warning in report.warnings: st.warning(warning)
        st.dataframe(report.frame.head(25), use_container_width=True)
        model_frame, errors = prepare_for_prediction(report.frame)
        if errors:
            st.info("Preview is available. Prediction is disabled until the uploaded data matches the model schema.")
            for error in errors: st.error(error)
        else:
            st.success("All model features are available and numeric. The file is ready for batch inference.")
            if st.button("Run batch predictions", type="primary"):
                try:
                    bundle = __import__("joblib").load(ROOT / "models" / "saved_model.pkl")
                    output = report.frame.copy(); output["predicted_maintenance"] = bundle["model"].predict(model_frame)
                    output["risk_probability"] = bundle["model"].predict_proba(model_frame)[:, 1]
                    st.dataframe(output, use_container_width=True); st.download_button("Download prediction results", output.to_csv(index=False), "maintenance_predictions.csv", "text/csv")
                except Exception: st.error("Batch prediction could not be completed. Train or verify the model first.")


def model_ai() -> None:
    hero("Model & AI transparency", "Understand what the model uses, how the agent works, and where the output should not be trusted.")
    metrics_path = ROOT / "visualizations" / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path); best = metrics.loc[metrics["f1"].idxmax()]
        st.success(f"Active model: {best['model']} — selected automatically by highest measured F1 score ({best['f1']:.3f}).")
        st.dataframe(metrics.style.format({x:"{:.3f}" for x in ["accuracy","precision","recall","f1"]}), use_container_width=True)
    else: st.warning("Metrics are not available. Run: python models/train_model.py")
    st.subheader("Agent architecture")
    st.info("Input → validation → ML prediction → diagnostic tools → recommendation → SQLite memory → alert → human review")
    st.subheader("Trust boundaries")
    st.write("The classifier provides the authoritative structured prediction. The agent organizes tools around it. An optional LLM may explain structured outputs but must not invent readings or diagnose failure.")
    st.warning("Feature importance indicates model association, not causation. Production use requires real validated data, calibration, monitoring, security review, and technician feedback.")
    for name in ["model_comparison.png", "confusion_matrix.png", "feature_importance.png", "maintenance_distribution.png"]:
        path = ROOT / "visualizations" / name
        if path.exists(): st.image(str(path), caption=name.replace("_", " ").title())


def analytics() -> None:
    hero("Analytics", "Review fleet history and risk patterns from completed predictions.")
    try: rows = get_history("")
    except Exception: rows = []
    if not rows:
        st.info("No prediction history yet. Run a vehicle prediction to populate analytics.")
        return
    frame = pd.DataFrame(rows); frame["risk_probability"] = pd.to_numeric(frame["risk_probability"])
    a,b,c = st.columns(3); a.metric("Stored events", len(frame)); b.metric("Unique vehicles", frame.vehicle_id.nunique()); c.metric("Average risk", f"{frame.risk_probability.mean():.1%}")
    st.bar_chart(frame.groupby("priority").size())
    st.dataframe(frame.head(50), use_container_width=True)


def settings_help() -> None:
    hero("Settings & help", "Configuration, operating guidance, and safety boundaries.")
    st.markdown("- Configure API credentials only through environment variables or Streamlit secrets.\n- Keep `.env` out of version control.\n- The local sample dataset is the reliable offline fallback.\n- Uploads are validated in memory and are not silently treated as production data.")
    st.warning("This application supports maintenance prioritization; it does not certify that a component has failed.")


def main() -> None:
    st.sidebar.markdown("## 🔧 FleetCare AI")
    st.sidebar.caption("Maintenance intelligence workspace")
    options = ["Dashboard", "Prediction", "Dataset / Upload", "Analytics", "Model / AI", "Settings / Help"]
    if "page" not in st.session_state: st.session_state.page = "Dashboard"
    page = st.sidebar.radio("Workspace", options, index=options.index(st.session_state.page))
    st.session_state.page = page
    st.sidebar.markdown("---")
    st.sidebar.caption("Demo Dataset Active · Human review required")
    {"Dashboard": dashboard, "Prediction": prediction, "Dataset / Upload": dataset_upload, "Analytics": analytics, "Model / AI": model_ai, "Settings / Help": settings_help}[page]()

if __name__ == "__main__": main()
