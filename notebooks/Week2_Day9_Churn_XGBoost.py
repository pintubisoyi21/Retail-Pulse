"""
=============================================================
RetailPulse – Week 2, Day 9
Churn Prediction: XGBoost + SHAP Explainability
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import mlflow, mlflow.xgboost, joblib, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PLOTS_DIR  = Path("../reports/plots")
MODELS_DIR = Path("../models")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. BUILD CHURN FEATURES ──────────────────────────────────
df  = pd.read_parquet("../data/clean_retail.parquet")
rfm = pd.read_parquet("../data/rfm_segmented.parquet")

df["Revenue"] = df["Quantity"] * df["UnitPrice"]
snapshot = df["InvoiceDate"].max()

# Customer-level feature engineering
cust_feats = df.groupby("CustomerID").agg(
    TotalRevenue      = ("Revenue",     "sum"),
    TotalOrders       = ("InvoiceNo",   "nunique"),
    TotalItems        = ("Quantity",    "sum"),
    UniqueProducts    = ("StockCode",   "nunique"),
    UniqueCountries   = ("Country",     "nunique"),
    AvgOrderValue     = ("Revenue",     "mean"),
    StdOrderValue     = ("Revenue",     "std"),
    LastPurchaseDate  = ("InvoiceDate", "max"),
    FirstPurchaseDate = ("InvoiceDate", "min"),
).reset_index()

cust_feats["DaysSinceLastPurchase"]  = (snapshot - cust_feats["LastPurchaseDate"]).dt.days
cust_feats["CustomerLifetimeDays"]   = (cust_feats["LastPurchaseDate"] - cust_feats["FirstPurchaseDate"]).dt.days + 1
cust_feats["PurchaseFrequency"]      = cust_feats["TotalOrders"] / cust_feats["CustomerLifetimeDays"]
cust_feats["AvgDaysBetweenOrders"]   = cust_feats["CustomerLifetimeDays"] / (cust_feats["TotalOrders"] + 1)
cust_feats["RevenuePerDay"]          = cust_feats["TotalRevenue"] / cust_feats["CustomerLifetimeDays"]
cust_feats["StdOrderValue"]          = cust_feats["StdOrderValue"].fillna(0)

# Churn label: no purchase in last 90 days
cust_feats["Churned"] = (cust_feats["DaysSinceLastPurchase"] > 90).astype(int)
print(f"Churn rate: {cust_feats['Churned'].mean()*100:.1f}%")
print(f"Total customers: {len(cust_feats):,}")

# Merge RFM scores
rfm_cols = ["CustomerID","R_Score","F_Score","M_Score","RFM_Score","Segment"]
cust_feats = cust_feats.merge(rfm[rfm_cols], on="CustomerID", how="left")

# Encode segment
seg_map = {"Champions":6,"Loyal Customers":5,"Potential Loyalists":4,
           "At Risk":3,"Hibernating":2,"Lost Customers":1}
cust_feats["Segment_Code"] = cust_feats["Segment"].map(seg_map).fillna(0)

# ── 2. FEATURE SELECTION ─────────────────────────────────────
drop_cols = ["CustomerID","LastPurchaseDate","FirstPurchaseDate","Churned","Segment"]
feature_cols = [c for c in cust_feats.columns if c not in drop_cols]

X = cust_feats[feature_cols].fillna(0)
y = cust_feats["Churned"]

print(f"\nFeature count: {len(feature_cols)}")
print(f"Class balance: {y.value_counts().to_dict()}")

# ── 3. TRAIN / TEST SPLIT + SMOTE ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {y_train_sm.value_counts().to_dict()}")

# ── 4. XGBOOST MODEL ─────────────────────────────────────────
mlflow.set_experiment("churn_prediction")
with mlflow.start_run(run_name="xgboost_v1"):
    params = {
        "n_estimators":       300,
        "max_depth":          6,
        "learning_rate":      0.05,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "min_child_weight":   3,
        "gamma":              0.1,
        "scale_pos_weight":   1,
        "use_label_encoder":  False,
        "eval_metric":        "logloss",
        "random_state":       42,
        "n_jobs":             -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_sm, y_train_sm,
              eval_set=[(X_test, y_test)], verbose=50)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc     = roc_auc_score(y_test, y_pred_proba)
    ap      = average_precision_score(y_test, y_pred_proba)

    # Precision@top20%
    n_top = int(0.2 * len(y_test))
    top_idx = np.argsort(y_pred_proba)[::-1][:n_top]
    prec_top20 = y_test.iloc[top_idx].mean()

    mlflow.log_params(params)
    mlflow.log_metrics({
        "AUC_ROC":     round(auc,4),
        "Avg_Precision": round(ap,4),
        "Prec_Top20":  round(prec_top20,4),
    })
    mlflow.xgboost.log_model(model, "xgboost_churn")

    print(f"\n=== XGBoost Churn Results ===")
    print(f"AUC-ROC          : {auc:.4f}  (target ≥ 0.88)")
    print(f"Avg Precision    : {ap:.4f}")
    print(f"Precision@Top20% : {prec_top20:.4f}  (target ≥ 0.75)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Active","Churned"]))

# ── 5. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC={auc:.3f}")
axes[0].plot([0,1],[0,1], "k--")
axes[0].set_title("ROC Curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend()

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(rec, prec, color="#3498db", lw=2, label=f"AP={ap:.3f}")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].legend()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Active","Churned"], yticklabels=["Active","Churned"])
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(PLOTS_DIR/"13_churn_metrics.png", dpi=150); plt.close()
print("→ Saved: 13_churn_metrics.png")

# ── 6. SHAP EXPLAINABILITY ───────────────────────────────────
print("\nGenerating SHAP values …")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False,
                  max_display=15, color="#e74c3c")
plt.title("SHAP Feature Importance – Churn Prediction")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"14_shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title("SHAP Beeswarm – Churn Prediction")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"15_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
print("→ Saved: 14_shap_importance.png, 15_shap_beeswarm.png")

# ── 7. SAVE ───────────────────────────────────────────────────
cust_feats["ChurnProba"] = model.predict_proba(
    cust_feats[feature_cols].fillna(0))[:, 1]
cust_feats["ChurnRisk"] = pd.cut(cust_feats["ChurnProba"],
    bins=[0,.3,.6,.8,1.0], labels=["Low","Medium","High","Critical"])
cust_feats.to_parquet("../data/churn_predictions.parquet", index=False)

joblib.dump(model,       MODELS_DIR/"xgboost_churn.pkl")
joblib.dump(feature_cols, MODELS_DIR/"churn_feature_cols.pkl")
print("\n✅  Week 2, Day 9 COMPLETE – Churn model ready")
