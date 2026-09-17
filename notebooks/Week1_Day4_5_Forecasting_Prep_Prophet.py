"""
=============================================================
RetailPulse – Week 1, Day 4 & 5
Time-Series Preparation + Baseline Prophet Forecasting
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import mlflow, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PLOTS_DIR = Path("../reports/plots")
MODELS_DIR = Path("../models")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. BUILD DAILY AGGREGATE TIME SERIES ────────────────────
df = pd.read_parquet("../data/clean_retail.parquet")
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

daily = (df.groupby(pd.Grouper(key="InvoiceDate", freq="D"))
           .agg(Revenue=("Revenue","sum"), Transactions=("InvoiceNo","nunique"),
                UnitsSold=("Quantity","sum"))
           .reset_index())

# Fill missing dates with 0 (store closed / no sales)
date_range = pd.date_range(daily["InvoiceDate"].min(), daily["InvoiceDate"].max(), freq="D")
daily = daily.set_index("InvoiceDate").reindex(date_range).fillna(0).reset_index()
daily.rename(columns={"index":"InvoiceDate"}, inplace=True)
daily["InvoiceDate"] = pd.to_datetime(daily["InvoiceDate"])

print(f"Daily series length: {len(daily)} days")
print(daily.describe())

# ── 2. STATIONARITY TESTS ────────────────────────────────────
adf_result = adfuller(daily["Revenue"].dropna())
print(f"\nADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print("Stationary    :", "YES" if adf_result[1] < 0.05 else "NO")

# ── 3. SEASONAL DECOMPOSITION ────────────────────────────────
# Use multiplicative only if no zeros; fallback to additive
try:
    decomp = seasonal_decompose(daily["Revenue"], model="additive", period=7)
    fig = decomp.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle("Daily Revenue – Seasonal Decomposition (period=7)", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"08_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("→ Saved: 08_decomposition.png")
except Exception as e:
    print(f"Decomposition error: {e}")

# ── 4. PROPHET BASELINE MODEL ────────────────────────────────
prophet_df = daily[["InvoiceDate","Revenue"]].rename(
    columns={"InvoiceDate":"ds","Revenue":"y"})

# Train / test split  (last 30 days = test)
HORIZON = 30
train_df = prophet_df.iloc[:-HORIZON]
test_df  = prophet_df.iloc[-HORIZON:]

mlflow.set_experiment("demand_forecasting")
with mlflow.start_run(run_name="prophet_baseline"):
    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10,
    )
    model.add_country_holidays(country_name="GB")   # UK dataset
    model.fit(train_df)

    # Forecast
    future   = model.make_future_dataframe(periods=HORIZON, freq="D")
    forecast  = model.predict(future)
    test_pred = forecast.tail(HORIZON)["yhat"].values
    test_true = test_df["y"].values

    # Metrics
    mape  = np.mean(np.abs((test_true - test_pred) / (test_true + 1e-9))) * 100
    rmse  = np.sqrt(np.mean((test_true - test_pred)**2))
    mae   = np.mean(np.abs(test_true - test_pred))

    mlflow.log_params({
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "horizon_days": HORIZON,
    })
    mlflow.log_metrics({"MAPE": round(mape,2), "RMSE": round(rmse,2), "MAE": round(mae,2)})

    print(f"\n=== Prophet Baseline Results ===")
    print(f"MAPE : {mape:.2f}%  (target ≤ 12%)")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")

    # Plots
    fig1 = model.plot(forecast)
    fig1.set_size_inches(14, 5)
    plt.title("Prophet: Revenue Forecast with Actuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"09_prophet_forecast.png", dpi=150); plt.close()

    fig2 = model.plot_components(forecast)
    fig2.set_size_inches(14, 8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"10_prophet_components.png", dpi=150); plt.close()
    print("→ Saved: 09_prophet_forecast.png, 10_prophet_components.png")

# Save forecast and model
forecast.to_parquet("../data/prophet_forecast.parquet", index=False)
daily.to_parquet("../data/daily_sales.parquet", index=False)

import joblib
joblib.dump(model, MODELS_DIR/"prophet_model.pkl")
print("\n✅  Week 1, Day 4-5 COMPLETE")
