"""
=============================================================
RetailPulse – Week 1, Day 1 & 2
EDA + Data Cleaning + Feature Engineering
=============================================================
Dataset : Online Retail (UCI / Kaggle)
Columns : InvoiceNo, StockCode, Description, Quantity,
          InvoiceDate, UnitPrice, CustomerID, Country
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RAW_PATH   = Path("../data/Online_Retail.xlsx")
CLEAN_PATH = Path("../data/clean_retail.parquet")
PLOTS_DIR  = Path("../reports/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset …")
print("=" * 60)

df = pd.read_excel(RAW_PATH, dtype={"CustomerID": str})
df.columns = df.columns.str.strip()
print(f"Raw shape : {df.shape}")
print(df.head(3))
print("\nDtypes:\n", df.dtypes)

# ─────────────────────────────────────────────────────────────
# 2. INITIAL EDA – DAY 1
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Initial EDA")
print("=" * 60)

print(f"\nDate range  : {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"Customers   : {df['CustomerID'].nunique():,}")
print(f"Products    : {df['StockCode'].nunique():,}")
print(f"Countries   : {df['Country'].nunique()}")
print(f"Invoices    : {df['InvoiceNo'].nunique():,}")
print("\nMissing values:\n", df.isnull().sum())

# --- Plot 1: Distribution of Quantity (non-returns) --------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

pos_qty = df[df["Quantity"] > 0]["Quantity"].clip(upper=100)
axes[0].hist(pos_qty, bins=50, color="#2ecc71", edgecolor="white")
axes[0].set_title("Quantity Distribution (positive, clipped @100)")
axes[0].set_xlabel("Quantity")
axes[0].set_ylabel("Frequency")

pos_price = df[df["UnitPrice"] > 0]["UnitPrice"].clip(upper=50)
axes[1].hist(pos_price, bins=50, color="#3498db", edgecolor="white")
axes[1].set_title("Unit Price Distribution (clipped @50)")
axes[1].set_xlabel("Unit Price (£)")

top_countries = df["Country"].value_counts().head(10)
axes[2].barh(top_countries.index[::-1], top_countries.values[::-1], color="#e74c3c")
axes[2].set_title("Top 10 Countries by Transactions")
axes[2].set_xlabel("Number of Transactions")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_distributions.png", dpi=150)
plt.close()
print("→ Saved: 01_distributions.png")

# --- Plot 2: Monthly sales trend ---------------------------
df_temp = df.copy()
df_temp["YearMonth"] = df_temp["InvoiceDate"].dt.to_period("M")
monthly = df_temp.groupby("YearMonth").size()

fig, ax = plt.subplots(figsize=(14, 4))
monthly.plot(ax=ax, marker="o", linewidth=2, color="#9b59b6")
ax.set_title("Monthly Transaction Volume (2010–2011)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_monthly_trend.png", dpi=150)
plt.close()
print("→ Saved: 02_monthly_trend.png")

# ─────────────────────────────────────────────────────────────
# 3. DATA CLEANING – DAY 2
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data Cleaning")
print("=" * 60)

df_clean = df.copy()

# 3a. Remove cancelled invoices (InvoiceNo starting with 'C')
cancelled_mask = df_clean["InvoiceNo"].astype(str).str.startswith("C")
print(f"Cancelled invoices removed : {cancelled_mask.sum():,}")
df_clean = df_clean[~cancelled_mask]

# 3b. Remove negative or zero Quantity / UnitPrice
invalid_qty   = df_clean["Quantity"] <= 0
invalid_price = df_clean["UnitPrice"] <= 0
print(f"Invalid Quantity rows     : {invalid_qty.sum():,}")
print(f"Invalid UnitPrice rows    : {invalid_price.sum():,}")
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# 3c. Drop rows with missing CustomerID
missing_cid = df_clean["CustomerID"].isnull().sum()
print(f"Missing CustomerID rows   : {missing_cid:,}")
df_clean = df_clean.dropna(subset=["CustomerID"])

# 3d. Fill missing Description with StockCode
df_clean["Description"] = df_clean["Description"].fillna(df_clean["StockCode"])

# 3e. Remove test/anomaly stock codes
bad_codes = ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT"]
df_clean = df_clean[~df_clean["StockCode"].isin(bad_codes)]

print(f"\nClean shape : {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]:,} "
      f"({(1 - df_clean.shape[0]/df.shape[0])*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Engineering")
print("=" * 60)

# Revenue column
df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

# Date features
df_clean["Year"]       = df_clean["InvoiceDate"].dt.year
df_clean["Month"]      = df_clean["InvoiceDate"].dt.month
df_clean["DayOfWeek"]  = df_clean["InvoiceDate"].dt.dayofweek   # 0=Mon
df_clean["WeekOfYear"] = df_clean["InvoiceDate"].dt.isocalendar().week.astype(int)
df_clean["Hour"]       = df_clean["InvoiceDate"].dt.hour
df_clean["IsWeekend"]  = df_clean["DayOfWeek"].isin([5, 6]).astype(int)

# --- RFM SCORES ------------------------------------------------
snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = (
    df_clean.groupby("CustomerID")
    .agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    )
    .reset_index()
)

# Quintile scoring (1=worst, 5=best)
rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

print(rfm.describe())

# --- Plot 3: RFM Score distribution --------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                                 ["#e74c3c", "#3498db", "#2ecc71"]):
    rfm[col].clip(upper=rfm[col].quantile(0.99)).hist(bins=40, ax=ax, color=color, edgecolor="white")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_rfm_distributions.png", dpi=150)
plt.close()
print("→ Saved: 03_rfm_distributions.png")

# --- Rolling statistics per product (7-day & 30-day) ---------
daily_product = (
    df_clean.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="D")])
    ["Quantity"].sum()
    .reset_index()
    .rename(columns={"Quantity": "DailyQty"})
    .sort_values(["StockCode", "InvoiceDate"])
)

daily_product["Roll7"]  = (daily_product.groupby("StockCode")["DailyQty"]
                           .transform(lambda x: x.rolling(7,  min_periods=1).mean()))
daily_product["Roll30"] = (daily_product.groupby("StockCode")["DailyQty"]
                           .transform(lambda x: x.rolling(30, min_periods=1).mean()))

print("\nDaily product rolling stats (sample):")
print(daily_product.head(8))

# --- Correlation heatmap ------------------------------------
corr_cols = ["Quantity", "UnitPrice", "Revenue", "Month", "DayOfWeek", "IsWeekend", "Hour"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_clean[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png", dpi=150)
plt.close()
print("→ Saved: 04_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────
# 5. SAVE CLEANED DATA
# ─────────────────────────────────────────────────────────────
# Convert mixed object columns to string
for col in df_clean.select_dtypes(include=["object"]).columns:
    df_clean[col] = df_clean[col].astype(str)

for col in rfm.select_dtypes(include=["object"]).columns:
    rfm[col] = rfm[col].astype(str)

daily_product["StockCode"] = daily_product["StockCode"].astype(str)

df_clean.to_parquet(CLEAN_PATH, index=False)
rfm.to_parquet("../data/rfm_features.parquet", index=False)
daily_product.to_parquet("../data/daily_product.parquet", index=False)

print("\n" + "=" * 60)
print("✅ Week 1, Day 1-2 COMPLETE")
print("=" * 60)

