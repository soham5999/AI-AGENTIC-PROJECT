"""Train, compare and save the best classifier by F1 score."""
from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data.generate_dataset import generate_dataset
from utils.preprocessing import FEATURES

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "vehicle_maintenance_dataset.csv"; OUT = ROOT / "visualizations"
def main() -> None:
    DATA.parent.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    if not DATA.exists(): generate_dataset().to_csv(DATA, index=False)
    df = pd.read_csv(DATA); X, y = df[FEATURES], df["maintenance_required"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    models = {"Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]), "SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=42))]), "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42), "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    metrics, cms = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr); pred = model.predict(Xte)
        metrics.append({"model": name, "accuracy": accuracy_score(yte,pred), "precision": precision_score(yte,pred,zero_division=0), "recall": recall_score(yte,pred,zero_division=0), "f1": f1_score(yte,pred,zero_division=0)})
        cms[name] = confusion_matrix(yte, pred).tolist()
    result = pd.DataFrame(metrics); best_name = result.sort_values("f1", ascending=False).iloc[0]["model"]
    joblib.dump({"model": models[best_name], "model_name": best_name, "features": FEATURES}, ROOT / "models" / "saved_model.pkl")
    result.to_csv(OUT / "model_metrics.csv", index=False); (OUT / "confusion_matrices.json").write_text(json.dumps(cms, indent=2))
    melted = result.melt("model", value_vars=["accuracy","precision","recall","f1"], var_name="metric", value_name="score")
    sns.barplot(data=melted, x="model", y="score", hue="metric"); plt.xticks(rotation=20); plt.ylim(0,1.05); plt.tight_layout(); plt.savefig(OUT/"model_comparison.png", dpi=150); plt.close()
    best_model = models[best_name]; importances = getattr(best_model, "feature_importances_", None)
    if importances is None and hasattr(best_model[-1], "feature_importances_"): importances = best_model[-1].feature_importances_
    if importances is None: importances = [0] * len(FEATURES)
    sns.barplot(x=importances, y=FEATURES); plt.title(f"Feature importance ({best_name})"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png", dpi=150); plt.close()
    sns.heatmap(cms[best_name], annot=True, fmt="d", cmap="Blues"); plt.title(f"Confusion matrix ({best_name})"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png", dpi=150); plt.close()
    df["maintenance_required"].value_counts().sort_index().plot(kind="bar", title="Synthetic target distribution"); plt.xlabel("Maintenance required"); plt.tight_layout(); plt.savefig(OUT/"maintenance_distribution.png", dpi=150); plt.close()
    print(result.to_string(index=False)); print(f"Selected by highest F1: {best_name}")
if __name__ == "__main__": main()
