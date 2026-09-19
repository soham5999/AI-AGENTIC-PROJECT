# Project Report: AI Agent for Vehicle Maintenance Prediction

## 1. Title
AI Agent for Vehicle Maintenance Prediction

## 2. Abstract
This project presents an academic prototype that combines classification models with an agentic orchestration layer for vehicle maintenance decision support. A synthetic dataset is used to train four classifiers. The selected model predicts maintenance risk, while deterministic tools identify possible threshold-based concerns, recommend a priority, persist history and produce a simulated alert. Human technician review remains mandatory.

## 3. Introduction
Preventive maintenance can reduce unexpected downtime. Machine learning can identify patterns associated with maintenance need, while an agent can coordinate prediction, explanation, memory and notification tasks.

## 4. Problem Statement
Manual review of many vehicle parameters is repetitive. A safe educational assistant is needed to organize risk signals without claiming mechanical diagnosis.

## 5. Objectives
Generate data, compare classifiers, select by F1, build tool-based agent orchestration, validate inputs, store history and provide a usable dashboard.

## 6. Literature Survey
Predictive maintenance commonly uses condition indicators, classification/regression, anomaly detection and time-series monitoring. This prototype demonstrates the workflow without claiming production validity.

## 7. Existing System
Traditional schedules use fixed intervals and manual inspection. They may not prioritize vehicles using multiple condition indicators.

## 8. Proposed System
The proposed system combines tabular ML with rules-based diagnostic tools, SQLite memory, alerts and optional natural-language explanation.

## 9. System Architecture
Streamlit collects input; preprocessing validates bounds; the saved best-F1 pipeline predicts; tools generate findings and recommendations; SQLite stores the event; the UI presents a reviewable result.

## 10. Agent Architecture
`Input -> validation -> prediction tool -> diagnostic tool -> recommendation tool -> memory tool -> alert tool -> human review`.

## 11. Dataset
At least 2,500 synthetic records are generated with ten features and binary target `maintenance_required`. NumPy seed 42 makes generation reproducible. It must not be treated as real-world vehicle data.

## 12. Data Preprocessing
Features are placed in a fixed order. Numeric ranges reject impossible or suspicious UI values. Training uses an 80/20 stratified split.

## 13. Machine Learning Models
Logistic Regression and SVM use standardization. Random Forest and Gradient Boosting model nonlinear relationships. Metrics are accuracy, precision, recall and F1. The highest-F1 model is saved automatically.

## 14. Agentic Workflow
The agent coordinates independent functions and passes structured outputs between them. It does not let an LLM invent sensor values.

## 15. Tools
`predict_maintenance`, `diagnose_vehicle`, `generate_recommendation`, SQLite memory and `alert_tool` are separate reusable functions.

## 16. Memory
SQLite stores vehicle ID, timestamp, parameters, risk, prediction, diagnosis, recommendation and priority, enabling history lookup.

## 17. Automation
An analysis button executes the complete pipeline and generates an alert. Email/SMS are intentionally not sent.

## 18. Implementation
Python modules are separated into data, models, agent and utility directories. Streamlit provides the interface; joblib persists the selected model.

## 19. Results
Run `python models/train_model.py` to produce actual metrics. This report intentionally does not fabricate numbers; the generated CSV and charts are the source of truth.

## 20. Graphs
The training script generates model comparison, confusion matrix, feature importance and target distribution images in `visualizations/`.

## 21. Advantages
Modular design, reproducible training, multiple-model comparison, history, validation, safe language and optional LLM independence.

## 22. Limitations
Synthetic data and labels are not physical evidence. There is no real sensor streaming, external validation, calibration, authentication or technician feedback loop.

## 23. Future Scope
Use validated fleet data, temporal features, anomaly detection, calibration, SHAP, drift monitoring, role-based access, notifications and technician-confirmed outcomes.

## 24. Conclusion
The project demonstrates how ML and agentic orchestration can support maintenance prioritization while preserving human oversight and avoiding unsupported failure claims.

## 25. References
1. scikit-learn User Guide, https://scikit-learn.org/stable/user_guide.html
2. pandas Documentation, https://pandas.pydata.org/docs/
3. Streamlit Documentation, https://docs.streamlit.io/
4. SQLite Documentation, https://sqlite.org/docs.html
5. Géron, A., *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.
