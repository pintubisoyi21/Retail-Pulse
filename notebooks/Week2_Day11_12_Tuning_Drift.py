"""
=============================================================
RetailPulse – Week 2, Day 11 & 12
Optuna Hyperparameter Tuning + Evidently Drift Detection
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
import mlflow, joblib, warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)
from pathlib import Path

PLOTS_DIR  = Path("../reports/plots")
MODELS_DIR = Path("../models")
REPORTS_DIR = Path("../reports")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. LOAD CHURN DATA ────────────────────────────────────────
churn_df      = pd.read_parquet("../data/churn_predictions.parquet")
feature_cols  = joblib.load(MODELS_DIR/"churn_feature_cols.pkl")
X = churn_df[feature_cols].fillna(0)
y = churn_df["Churned"]

# ── 2. OPTUNA TUNING ─────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 1.0),
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }
    model = xgb.XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return score.mean()

print("Running Optuna optimization (30 trials) …")
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest AUC    : {study.best_value:.4f}")
print(f"Best params : {study.best_params}")

# ── 3. RETRAIN WITH BEST PARAMS ──────────────────────────────
best_params = study.best_params
best_params.update({"use_label_encoder": False, "eval_metric": "logloss",
                    "random_state": 42, "n_jobs": -1})

mlflow.set_experiment("churn_prediction")
with mlflow.start_run(run_name="xgboost_tuned_optuna"):
    tuned_model = xgb.XGBClassifier(**best_params)
    tuned_model.fit(X, y)
    y_proba = tuned_model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_proba)
    mlflow.log_params(best_params)
    mlflow.log_metric("train_auc", round(auc, 4))
    mlflow.xgboost.log_model(tuned_model, "xgboost_tuned")
    print(f"\nTuned model AUC: {auc:.4f}")

# Optuna optimization history
plt.figure(figsize=(8, 5))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optuna Optimization History")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"17_optuna_history.png", dpi=150)
plt.close()

# Hyperparameter importances
plt.figure(figsize=(8, 5))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Hyperparameter Importances")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"18_optuna_importances.png", dpi=150)
plt.close()

print("→ Saved: 17_optuna_history.png")
print("→ Saved: 18_optuna_importances.png")

# ── 4. DATA DRIFT DETECTION (Evidently AI) ────────────────────
print("\nGenerating Evidently drift report …")

# Simulate production drift by sampling recent vs older data
df_clean = pd.read_parquet("../data/clean_retail.parquet")
df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]
midpoint  = df_clean["InvoiceDate"].quantile(0.5)
reference = df_clean[df_clean["InvoiceDate"] <= midpoint][
    ["Quantity","UnitPrice","Revenue","Month","DayOfWeek","IsWeekend"]].sample(1000, random_state=42)
current   = df_clean[df_clean["InvoiceDate"] >  midpoint][
    ["Quantity","UnitPrice","Revenue","Month","DayOfWeek","IsWeekend"]].sample(1000, random_state=99)

drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
drift_report.run(reference_data=reference, current_data=current)
drift_report.save_html(str(REPORTS_DIR/"drift_report.html"))
print("→ Saved: reports/drift_report.html")

# ── 5. SAVE TUNED MODEL ──────────────────────────────────────
joblib.dump(tuned_model, MODELS_DIR/"xgboost_churn_tuned.pkl")
print("\n✅  Week 2, Day 11-12 COMPLETE – Tuning + Drift detection ready")
