"""
=============================================================
RetailPulse – Week 2, Day 13
Automated Retraining Pipeline (Airflow-ready)
=============================================================
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from prophet import Prophet
import mlflow, mlflow.xgboost, joblib, warnings, json
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

MODELS_DIR  = Path("../models")
REPORTS_DIR = Path("../reports")
PIPELINE_LOG = REPORTS_DIR / "pipeline_runs.json"

# ── PIPELINE TASKS ────────────────────────────────────────────
def task_data_validation():
    """Validate incoming data quality."""
    print("[TASK 1] Data Validation …")
    df = pd.read_parquet("../data/clean_retail.parquet")
    checks = {
        "row_count_ok":       len(df) > 100,
        "no_null_revenue":    df["Quantity"].isnull().sum() == 0,
        "positive_price":     (df["UnitPrice"] > 0).all(),
        "date_range_ok":      (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days > 30,
    }
    passed = all(checks.values())
    print(f"  Checks: {checks}")
    print(f"  PASS: {passed}")
    return passed

def task_feature_engineering():
    """Rebuild features from cleaned data."""
    print("[TASK 2] Feature Engineering …")
    df  = pd.read_parquet("../data/clean_retail.parquet")
    rfm = pd.read_parquet("../data/rfm_segmented.parquet")
    df["Revenue"]  = df["Quantity"] * df["UnitPrice"]
    snapshot       = df["InvoiceDate"].max()

    cust = df.groupby("CustomerID").agg(
    TotalRevenue      = ("Revenue", "sum"),
    TotalOrders       = ("InvoiceNo", "nunique"),
    TotalItems        = ("Quantity", "sum"),
    UniqueProducts    = ("StockCode", "nunique"),
    UniqueCountries   = ("Country", "nunique"),   # Added
    AvgOrderValue     = ("Revenue", "mean"),
    StdOrderValue     = ("Revenue", "std"),
    LastPurchaseDate  = ("InvoiceDate", "max"),
    FirstPurchaseDate = ("InvoiceDate", "min"),
).reset_index()

    cust["DaysSinceLastPurchase"] = (snapshot - cust["LastPurchaseDate"]).dt.days
    cust["CustomerLifetimeDays"]  = (cust["LastPurchaseDate"] - cust["FirstPurchaseDate"]).dt.days + 1
    cust["PurchaseFrequency"]     = cust["TotalOrders"] / cust["CustomerLifetimeDays"]
    cust["AvgDaysBetweenOrders"]  = cust["CustomerLifetimeDays"] / (cust["TotalOrders"] + 1)
    cust["RevenuePerDay"]         = cust["TotalRevenue"] / cust["CustomerLifetimeDays"]
    cust["StdOrderValue"]         = cust["StdOrderValue"].fillna(0)
    cust["Churned"]               = (cust["DaysSinceLastPurchase"] > 90).astype(int)

    seg_map = {"Champions":6,"Loyal Customers":5,"Potential Loyalists":4,
               "At Risk":3,"Hibernating":2,"Lost Customers":1}
    rfm["Segment_Code"] = rfm["Segment"].map(seg_map).fillna(0)
    cust = cust.merge(rfm[["CustomerID","R_Score","F_Score","M_Score",
                            "RFM_Score","Segment_Code"]], on="CustomerID", how="left")
    cust.to_parquet("../data/churn_features_latest.parquet", index=False)
    print(f"  Features saved: {cust.shape}")
    return True

def task_retrain_churn():
    """Retrain churn model and evaluate."""
    print("[TASK 3] Retraining Churn Model …")
    cust         = pd.read_parquet("../data/churn_features_latest.parquet")
    feature_cols = joblib.load(MODELS_DIR/"churn_feature_cols.pkl")

    missing_cols = [c for c in feature_cols if c not in cust.columns]

    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            cust[col] = 0

    X = cust[feature_cols].fillna(0)
    y = cust["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    best_params = joblib.load(MODELS_DIR/"xgboost_churn_tuned.pkl").get_params()
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"  Retrained AUC: {auc:.4f}")

    print("Features expected by model:")
    print(feature_cols)

    print("\nAvailable columns:")
    print(cust.columns.tolist())

    

    # Compare with baseline – only replace if better
    old_model = joblib.load(MODELS_DIR/"xgboost_churn.pkl")
    old_auc   = roc_auc_score(y_test, old_model.predict_proba(X_test)[:,1])

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_metrics({
            "new_auc": float(auc),
            "old_auc": float(old_auc),
            "improved": float(auc > old_auc)
        })

    if auc >= old_auc:
        joblib.dump(model, MODELS_DIR/"xgboost_churn.pkl")
        print(f"  ✅ Model UPDATED (AUC: {old_auc:.4f} → {auc:.4f})")
    else:
        print(f"  ⚠️  Model NOT updated (new={auc:.4f} < old={old_auc:.4f})")

    return {
        "new_auc": float(auc),
        "old_auc": float(old_auc),
        "updated": bool(auc >= old_auc)
    }

def task_retrain_forecast():
    """Retrain Prophet model on latest data."""
    print("[TASK 4] Retraining Forecast Model …")
    df = pd.read_parquet("../data/clean_retail.parquet")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    daily = (df.groupby(pd.Grouper(key="InvoiceDate", freq="D"))
               ["Revenue"].sum().reset_index()
               .rename(columns={"InvoiceDate":"ds","Revenue":"y"}))
    daily = daily[daily["y"] > 0]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    changepoint_prior_scale=0.05)
    model.add_country_holidays(country_name="GB")
    model.fit(daily)
    joblib.dump(model, MODELS_DIR/"prophet_model.pkl")
    print("  Prophet model retrained and saved")
    return True

def task_log_run(results):
    """Log pipeline run to JSON."""

    run = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "results": results,
    }

    history = []

    if PIPELINE_LOG.exists():
        try:
            with open(PIPELINE_LOG, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(run)

    with open(PIPELINE_LOG, "w") as f:
        json.dump(history[-50:], f, indent=2, default=str)

    print(f"[TASK 5] Run logged → {PIPELINE_LOG}")

# ── MAIN PIPELINE ─────────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print(f"RetailPulse Retraining Pipeline – {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 55)
    results = {}

    if not task_data_validation():
        print("❌  Data validation FAILED – aborting pipeline")
        return

    task_feature_engineering()
    results["churn"]    = task_retrain_churn()
    results["forecast"] = task_retrain_forecast()
    task_log_run(results)

    print("\n✅  Pipeline COMPLETE")
    return results

if __name__ == "__main__":
    run_pipeline()
