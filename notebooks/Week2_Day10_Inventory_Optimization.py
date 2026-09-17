"""
=============================================================
RetailPulse – Week 2, Day 10
Inventory Optimization using Forecasted Demand
=============================================================
EOQ, Safety Stock, Reorder Points, ABC Analysis
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PLOTS_DIR = Path("../reports/plots")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_parquet("../data/clean_retail.parquet")
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Product-level aggregation
product = df.groupby("StockCode").agg(
    Description       = ("Description", "first"),
    TotalQtySold      = ("Quantity",     "sum"),
    TotalRevenue      = ("Revenue",      "sum"),
    AvgUnitPrice      = ("UnitPrice",    "mean"),
    NumTransactions   = ("InvoiceNo",    "nunique"),
    NumDays           = ("InvoiceDate",  lambda x: (x.max()-x.min()).days + 1),
).reset_index()

product["DailyDemand"] = product["TotalQtySold"] / product["NumDays"].clip(lower=1)
product["DemandStd"]   = df.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="D")])\
                           ["Quantity"].sum().groupby("StockCode").std().fillna(0).values[:len(product)]

# ── 2. ABC ANALYSIS ──────────────────────────────────────────
product = product.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
product["CumRevPct"] = product["TotalRevenue"].cumsum() / product["TotalRevenue"].sum() * 100
product["ABC"] = "C"
product.loc[product["CumRevPct"] <= 80, "ABC"] = "A"
product.loc[(product["CumRevPct"] > 80) & (product["CumRevPct"] <= 95), "ABC"] = "B"

print("ABC Distribution:")
print(product["ABC"].value_counts())

# ── 3. EOQ, SAFETY STOCK, REORDER POINT ──────────────────────
ORDERING_COST    = 50.0      # £ per order
HOLDING_COST_PCT = 0.25      # 25% of unit price per year
LEAD_TIME_DAYS   = 7         # 7 day lead time
SERVICE_LEVEL_Z  = 1.65      # 95% service level

def calc_eoq(annual_demand, unit_price, ordering_cost, holding_pct):
    H = holding_pct * unit_price
    if H <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt((2 * annual_demand * ordering_cost) / H)

def calc_safety_stock(demand_std, lead_time, z):
    return z * demand_std * np.sqrt(lead_time)

def calc_reorder_point(daily_demand, lead_time, safety_stock):
    return daily_demand * lead_time + safety_stock

product["AnnualDemand"]  = product["DailyDemand"] * 365
product["EOQ"]           = product.apply(
    lambda r: calc_eoq(r["AnnualDemand"], r["AvgUnitPrice"],
                       ORDERING_COST, HOLDING_COST_PCT), axis=1).round(0)
product["SafetyStock"]   = calc_safety_stock(
    product["DemandStd"], LEAD_TIME_DAYS, SERVICE_LEVEL_Z).round(0)
product["ReorderPoint"]  = calc_reorder_point(
    product["DailyDemand"], LEAD_TIME_DAYS, product["SafetyStock"]).round(0)
product["MaxStock"]      = (product["ReorderPoint"] + product["EOQ"]).round(0)

# ── 4. FORECAST-ADJUSTED REORDER ─────────────────────────────
# Load hybrid forecast and adjust top products
try:
    hybrid_fc = pd.read_parquet("../data/hybrid_forecast.parquet")
    forecast_uplift = (hybrid_fc["Hybrid"].mean() / hybrid_fc["Actual"].mean())
    product.loc[product["ABC"]=="A", "EOQ"]          *= forecast_uplift
    product.loc[product["ABC"]=="A", "ReorderPoint"] *= forecast_uplift
    product.loc[product["ABC"]=="A", "MaxStock"]     *= forecast_uplift
    print(f"\nForecast uplift applied to A items: {forecast_uplift:.3f}x")
except Exception as e:
    print(f"Forecast file not yet available: {e}")

# ── 5. INVENTORY RISK FLAGS ───────────────────────────────────
product["OverstockRisk"]   = product["MaxStock"]   > product["EOQ"] * 3
product["StockoutRisk"]    = product["DailyDemand"] > product["ReorderPoint"] / max(LEAD_TIME_DAYS,1)
product["SlowMoving"]      = product["DailyDemand"] < 0.1
print(f"\nOverstock risk items  : {product['OverstockRisk'].sum()}")
print(f"Stockout risk items   : {product['StockoutRisk'].sum()}")
print(f"Slow-moving items     : {product['SlowMoving'].sum()}")

# ── 6. PLOTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ABC pie
abc_rev = product.groupby("ABC")["TotalRevenue"].sum()
colors  = ["#e74c3c","#f39c12","#2ecc71"]
axes[0,0].pie(abc_rev, labels=abc_rev.index, autopct="%1.1f%%",
              colors=colors, startangle=90)
axes[0,0].set_title("ABC Analysis – Revenue Share")

# EOQ distribution by ABC
for abc, color in zip(["A","B","C"], colors):
    data = product[product["ABC"]==abc]["EOQ"].clip(upper=500)
    axes[0,1].hist(data, bins=30, alpha=0.6, label=abc, color=color)
axes[0,1].set_title("EOQ Distribution by ABC Category")
axes[0,1].set_xlabel("EOQ (units)"); axes[0,1].legend()

# Safety Stock vs Daily Demand
scatter = axes[1,0].scatter(
    product["DailyDemand"].clip(upper=50),
    product["SafetyStock"].clip(upper=200),
    c=product["ABC"].map({"A":0,"B":1,"C":2}),
    cmap="RdYlGn", alpha=0.6, s=30)
axes[1,0].set_title("Safety Stock vs Daily Demand")
axes[1,0].set_xlabel("Daily Demand (units)"); axes[1,0].set_ylabel("Safety Stock")

# Top 20 products by revenue
top20 = product.head(20)
axes[1,1].barh(top20["StockCode"][::-1], top20["TotalRevenue"][::-1],
               color="#3498db", alpha=0.8)
axes[1,1].set_title("Top 20 Products by Revenue")
axes[1,1].set_xlabel("Total Revenue (£)")
axes[1,1].tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR/"16_inventory_analysis.png", dpi=150); plt.close()
print("→ Saved: 16_inventory_analysis.png")

# ── 7. SAVE RECOMMENDATIONS ──────────────────────────────────
recommendations = product[[
    "StockCode","Description","ABC","DailyDemand","AnnualDemand",
    "AvgUnitPrice","EOQ","SafetyStock","ReorderPoint","MaxStock",
    "OverstockRisk","StockoutRisk","SlowMoving"
]].round(2)

recommendations.to_parquet("../data/inventory_recommendations.parquet", index=False)
recommendations.head(100).to_csv("../reports/inventory_top100.csv", index=False)
print("\nTop 5 inventory recommendations:")
print(recommendations.head(5).to_string())
print("\n✅  Week 2, Day 10 COMPLETE – Inventory optimization ready")
