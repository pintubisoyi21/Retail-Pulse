"""
=============================================================
RetailPulse – Week 4
Automated Tests for all pipeline components
=============================================================
Run: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")

class TestDataQuality:
    def test_clean_data_exists(self):
        assert (DATA_DIR / "clean_retail.parquet").exists(), "Clean data missing"

    def test_clean_data_columns(self):
        df = pd.read_parquet(DATA_DIR / "clean_retail.parquet")
        required = {"InvoiceNo","StockCode","Description","Quantity",
                    "InvoiceDate","UnitPrice","CustomerID","Country","Revenue"}
        assert required.issubset(df.columns)

    def test_no_negative_quantity(self):
        df = pd.read_parquet(DATA_DIR / "clean_retail.parquet")
        assert (df["Quantity"] > 0).all()

    def test_no_negative_price(self):
        df = pd.read_parquet(DATA_DIR / "clean_retail.parquet")
        assert (df["UnitPrice"] > 0).all()

    def test_no_null_customer_id(self):
        df = pd.read_parquet(DATA_DIR / "clean_retail.parquet")
        assert df["CustomerID"].isnull().sum() == 0

    def test_rfm_scores_in_range(self):
        rfm = pd.read_parquet(DATA_DIR / "rfm_features.parquet")
        for col in ["R_Score","F_Score","M_Score"]:
            assert rfm[col].between(1,5).all(), f"{col} out of range"

class TestModels:
    def test_kmeans_model_exists(self):
        assert (MODELS_DIR / "kmeans_segmentation.pkl").exists()

    def test_churn_model_exists(self):
        assert (MODELS_DIR / "xgboost_churn.pkl").exists()

    def test_prophet_model_exists(self):
        assert (MODELS_DIR / "prophet_model.pkl").exists()

    def test_kmeans_predict(self):
        import joblib
        model  = joblib.load(MODELS_DIR / "kmeans_segmentation.pkl")
        scaler = joblib.load(MODELS_DIR / "rfm_scaler.pkl")
        dummy  = np.zeros((1, 6))
        scaled = scaler.transform(dummy)
        pred   = model.predict(scaled)
        assert 0 <= pred[0] < 6

    def test_churn_predict_proba(self):
        import joblib
        model     = joblib.load(MODELS_DIR / "xgboost_churn.pkl")
        feat_cols = joblib.load(MODELS_DIR / "churn_feature_cols.pkl")
        dummy     = pd.DataFrame(np.zeros((2, len(feat_cols))), columns=feat_cols)
        proba     = model.predict_proba(dummy)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()

class TestInventory:
    def test_inventory_file_exists(self):
        assert (DATA_DIR / "inventory_recommendations.parquet").exists()

    def test_eoq_positive(self):
        inv = pd.read_parquet(DATA_DIR / "inventory_recommendations.parquet")
        assert (inv["EOQ"] >= 0).all()

    def test_abc_valid_labels(self):
        inv = pd.read_parquet(DATA_DIR / "inventory_recommendations.parquet")
        assert set(inv["ABC"].unique()).issubset({"A","B","C"})

class TestChurnLogic:
    def test_churn_label_over_90_days(self):
        churn = pd.read_parquet(DATA_DIR / "churn_predictions.parquet")
        high  = churn[churn["DaysSinceLastPurchase"] > 90]
        assert (high["Churned"] == 1).all(), "Churn label logic broken"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
