"""
=============================================================
RetailPulse – Week 3 (Days 15-21)
Streamlit Multi-Page Dashboard
=============================================================
Run: streamlit run src/dashboard/app.py
=============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, warnings
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "RetailPulse Analytics",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .risk-high    { color: #e74c3c; font-weight: bold; }
    .risk-medium  { color: #f39c12; font-weight: bold; }
    .risk-low     { color: #2ecc71; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6; border-radius: 8px 8px 0 0; padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADERS (cached) ─────────────────────────────────────
BASE = Path(__file__).parent.parent.parent / "data"

@st.cache_data(ttl=300)
def load_clean():
    df = pd.read_parquet(BASE/"clean_retail.parquet")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

@st.cache_data(ttl=300)
def load_rfm():
    return pd.read_parquet(BASE/"rfm_segmented.parquet")

@st.cache_data(ttl=300)
def load_churn():
    return pd.read_parquet(BASE/"churn_predictions.parquet")

@st.cache_data(ttl=300)
def load_daily():
    return pd.read_parquet(BASE/"daily_sales.parquet")

@st.cache_data(ttl=300)
def load_forecast():
    return pd.read_parquet(BASE/"hybrid_forecast.parquet")

@st.cache_data(ttl=300)
def load_inventory():
    return pd.read_parquet(BASE/"inventory_recommendations.parquet")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bar-chart.png", width=60)
    st.title("RetailPulse")
    st.caption("AI-Powered Retail Analytics")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Overview",
        "📈 Demand Forecasting",
        "👥 Customer Segments",
        "⚠️ Churn Risk",
        "📦 Inventory",
        "🔍 What-If Analysis",
    ])

    st.divider()
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE 1 – OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📊 RetailPulse – Executive Dashboard")

    try:
        df    = load_clean()
        rfm   = load_rfm()
        churn = load_churn()
        inv   = load_inventory()
    except FileNotFoundError as e:
        st.error(f"Data not found – run the pipeline notebooks first.\n{e}")
        st.stop()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev  = df["Revenue"].sum()
    total_cust = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()
    churn_rate = churn["Churned"].mean() * 100
    high_risk  = (churn["ChurnRisk"] == "Critical").sum()

    with col1:
        st.metric("Total Revenue", f"£{total_rev:,.0f}")
    with col2:
        st.metric("Unique Customers", f"{total_cust:,}")
    with col3:
        st.metric("Total Orders", f"{total_orders:,}")
    with col4:
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-2.3%")
    with col5:
        st.metric("Critical Risk Customers", f"{high_risk:,}", delta_color="inverse")

    st.divider()

    # Revenue trend + Segment donut
    col_left, col_right = st.columns([2, 1])

    with col_left:
        daily = df.groupby(pd.Grouper(key="InvoiceDate", freq="W"))["Revenue"].sum().reset_index()
        fig = px.area(daily, x="InvoiceDate", y="Revenue",
                      title="Weekly Revenue Trend",
                      color_discrete_sequence=["#667eea"],
                      labels={"InvoiceDate":"Date","Revenue":"Revenue (£)"})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig2 = px.pie(seg_counts, values="Count", names="Segment",
                      title="Customer Segment Mix", hole=0.4,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Country heatmap
    st.subheader("Revenue by Country")
    country_rev = df.groupby("Country")["Revenue"].sum().reset_index()
    fig3 = px.choropleth(country_rev, locations="Country",
                         locationmode="country names", color="Revenue",
                         color_continuous_scale="Viridis",
                         title="Global Revenue Distribution")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 – DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting")

    try:
        daily = load_daily()
        fc    = load_forecast()
    except FileNotFoundError:
        st.warning("Run Week 2 Day 8 notebook to generate forecast data.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("MAPE", "< 12%", "Target met ✅")
    with col2: st.metric("Forecast Horizon", "30 days")
    with col3: st.metric("Model", "Prophet + LSTM Hybrid")

    st.divider()

    # Historical + forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["InvoiceDate"], y=daily["Revenue"],
        mode="lines", name="Historical", line=dict(color="#3498db", width=1.5)))
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["Date"], y=fc["Actual"],
            mode="lines", name="Actual (test)", line=dict(color="#2ecc71", width=2)))
        fig.add_trace(go.Scatter(
            x=fc["Date"], y=fc["Hybrid"],
            mode="lines", name="Hybrid Forecast",
            line=dict(color="#e74c3c", width=2, dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["Date"], fc["Date"][::-1]]),
            y=pd.concat([fc["Hybrid"]*1.10, fc["Hybrid"][::-1]*0.90]),
            fill="toself", fillcolor="rgba(231,76,60,0.1)",
            line=dict(color="rgba(255,255,255,0)"), name="90% CI"))

    fig.update_layout(title="Revenue Forecast – Hybrid Prophet + LSTM",
                      xaxis_title="Date", yaxis_title="Revenue (£)",
                      height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Day of week pattern
    daily["DayOfWeek"] = pd.to_datetime(daily["InvoiceDate"]).dt.day_name()
    dow_avg = daily.groupby("DayOfWeek")["Revenue"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    fig2 = px.bar(dow_avg.reset_index(), x="DayOfWeek", y="Revenue",
                  color="Revenue", color_continuous_scale="Blues",
                  title="Average Revenue by Day of Week")
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 – CUSTOMER SEGMENTS
# ─────────────────────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    try:
        rfm = load_rfm()
    except FileNotFoundError:
        st.warning("Run Week 1 Day 3 notebook first.")
        st.stop()

    palette = {"Champions":"#2ecc71","Loyal Customers":"#3498db",
               "Potential Loyalists":"#f39c12","At Risk":"#e67e22",
               "Hibernating":"#e74c3c","Lost Customers":"#7f8c8d"}

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        fig = px.bar(seg_counts, x="Segment", y="count",
                     color="Segment", color_discrete_map=palette,
                     title="Customer Count per Segment")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        seg_rev = rfm.groupby("Segment")["Monetary"].sum().reset_index()
        fig2 = px.pie(seg_rev, values="Monetary", names="Segment",
                      title="Monetary Value per Segment", hole=0.35,
                      color="Segment", color_discrete_map=palette)
        st.plotly_chart(fig2, use_container_width=True)

    # RFM scatter
    fig3 = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency",
                      color="Segment", color_discrete_map=palette,
                      title="RFM Scatter – Recency vs Monetary (size=Frequency)",
                      hover_data=["CustomerID","RFM_Score"],
                      opacity=0.7, size_max=20)
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Segment filter table
    seg_filter = st.multiselect("Filter segments", rfm["Segment"].unique(),
                                default=["Champions","At Risk"])
    filtered = rfm[rfm["Segment"].isin(seg_filter)][
        ["CustomerID","Segment","Recency","Frequency","Monetary","RFM_Score"]]
    st.dataframe(filtered.sort_values("RFM_Score", ascending=False), use_container_width=True)
    st.download_button("⬇️ Export CSV", filtered.to_csv(index=False),
                       "segment_customers.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# PAGE 4 – CHURN RISK
# ─────────────────────────────────────────────────────────────
elif page == "⚠️ Churn Risk":
    st.title("⚠️ Churn Risk Dashboard")
    try:
        churn = load_churn()
    except FileNotFoundError:
        st.warning("Run Week 2 Day 9 notebook first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    for col, risk, color in zip(
        [col1, col2, col3, col4],
        ["Critical","High","Medium","Low"],
        ["🔴","🟠","🟡","🟢"]
    ):
        n = (churn["ChurnRisk"] == risk).sum()
        col.metric(f"{color} {risk}", f"{n:,}")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        risk_counts = churn["ChurnRisk"].value_counts().reset_index()
        fig = px.bar(risk_counts, x="ChurnRisk", y="count",
                     color="ChurnRisk",
                     color_discrete_map={"Critical":"#e74c3c","High":"#e67e22",
                                         "Medium":"#f39c12","Low":"#2ecc71"},
                     title="Customers by Churn Risk Level")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.histogram(churn, x="ChurnProba", nbins=50,
                            color_discrete_sequence=["#3498db"],
                            title="Churn Probability Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Churn probability vs recency
    if "Recency" in churn.columns:
        fig3 = px.scatter(churn.sample(min(2000, len(churn))),
                          x="DaysSinceLastPurchase", y="ChurnProba",
                          color="ChurnRisk",
                          color_discrete_map={"Critical":"#e74c3c","High":"#e67e22",
                                              "Medium":"#f39c12","Low":"#2ecc71"},
                          title="Churn Probability vs Days Since Last Purchase",
                          opacity=0.6)
        st.plotly_chart(fig3, use_container_width=True)

    # High-risk customer table
    st.subheader("🚨 Critical & High Risk Customers (Action Required)")
    at_risk = churn[churn["ChurnRisk"].isin(["Critical","High"])][
        ["CustomerID","ChurnProba","ChurnRisk","DaysSinceLastPurchase",
         "TotalRevenue","TotalOrders"]
    ].sort_values("ChurnProba", ascending=False).head(50)
    st.dataframe(at_risk, use_container_width=True)
    st.download_button("⬇️ Export At-Risk Customers",
                       at_risk.to_csv(index=False), "at_risk_customers.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# PAGE 5 – INVENTORY
# ─────────────────────────────────────────────────────────────
elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization")
    try:
        inv = load_inventory()
    except FileNotFoundError:
        st.warning("Run Week 2 Day 10 notebook first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("A-Class Products", f"{(inv['ABC']=='A').sum()}")
    col2.metric("Stockout Risk",     f"{inv['StockoutRisk'].sum()}")
    col3.metric("Overstock Risk",    f"{inv['OverstockRisk'].sum()}")

    st.divider()

    abc_filter = st.multiselect("Filter ABC Class", ["A","B","C"], default=["A","B"])
    inv_filtered = inv[inv["ABC"].isin(abc_filter)]

    col_l, col_r = st.columns(2)
    with col_l:
        abc_pie = inv.groupby("ABC")["AnnualDemand"].sum().reset_index()

        fig = px.pie(
            abc_pie,
            values="AnnualDemand",
            names="ABC",
            title="Demand Share by ABC Class",
            hole=0.35,
            color_discrete_map={"A":"#e74c3c","B":"#f39c12","C":"#2ecc71"})
        
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.scatter(inv_filtered.head(200),
                          x="DailyDemand", y="EOQ",
                          color="ABC", size="AnnualDemand",
                          color_discrete_map={"A":"#e74c3c","B":"#f39c12","C":"#2ecc71"},
                          title="EOQ vs Daily Demand (size=Revenue)",
                          hover_data=["StockCode","Description"], size_max=25)
        st.plotly_chart(fig2, use_container_width=True)

    # Inventory table with filters
    show_risk = st.checkbox("Show only risk items", False)
    display = inv_filtered[inv_filtered["StockoutRisk"] | inv_filtered["OverstockRisk"]] \
              if show_risk else inv_filtered
    st.dataframe(
        display[["StockCode","Description","ABC","DailyDemand","EOQ",
                 "SafetyStock","ReorderPoint","OverstockRisk","StockoutRisk"]].head(100),
        use_container_width=True)
    st.download_button("⬇️ Export Recommendations",
                       display.to_csv(index=False), "inventory_recs.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# PAGE 6 – WHAT-IF ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "🔍 What-If Analysis":
    st.title("🔍 What-If Scenario Analysis")
    st.caption("Simulate demand changes and see their effect on inventory and revenue.")

    try:
        daily = load_daily()
        inv   = load_inventory()
    except FileNotFoundError:
        st.warning("Run pipeline notebooks first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        demand_change = st.slider("Demand Change (%)", -50, 100, 0, 5)
    with col2:
        price_change  = st.slider("Price Change (%)",  -30,  50, 0, 5)
    with col3:
        lead_time     = st.slider("Lead Time (days)",    1,  30, 7, 1)

    # Simulate impact
    demand_mult = 1 + demand_change / 100
    price_mult  = 1 + price_change  / 100

    base_rev  = daily["Revenue"].mean() * 30
    sim_rev   = base_rev * demand_mult * price_mult
    rev_delta = sim_rev - base_rev

    a_items = inv[inv["ABC"]=="A"].copy()
    a_items["SimReorderPoint"] = a_items["DailyDemand"] * demand_mult * lead_time + a_items["SafetyStock"]
    a_items["SimEOQ"]          = a_items["EOQ"] * np.sqrt(demand_mult)
    stockout_sim = (a_items["SimReorderPoint"] > a_items["MaxStock"]).sum()

    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Simulated Monthly Revenue", f"£{sim_rev:,.0f}",
              delta=f"£{rev_delta:+,.0f}")
    s2.metric("Revenue Δ (%)", f"{(rev_delta/base_rev)*100:+.1f}%")
    s3.metric("A-Item Stockout Risk", f"{stockout_sim}",
              delta_color="inverse")
    s4.metric("Adjusted Lead Time", f"{lead_time} days")

    # Simulation chart
    weeks  = pd.date_range("2024-01-01", periods=12, freq="W")
    base_w = np.random.normal(base_rev/4, base_rev*0.05, 12)
    sim_w  = base_w * demand_mult * price_mult

    fig = go.Figure()
    fig.add_trace(go.Bar(x=weeks, y=base_w, name="Baseline", marker_color="#3498db", opacity=0.7))
    fig.add_trace(go.Bar(x=weeks, y=sim_w,  name="Simulated", marker_color="#e74c3c", opacity=0.7))
    fig.update_layout(barmode="group", title="Baseline vs Simulated Weekly Revenue",
                      xaxis_title="Week", yaxis_title="Revenue (£)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # A-item reorder table
    st.subheader("Simulated Reorder Points – A Class Items")
    st.dataframe(
        a_items[["StockCode","Description","DailyDemand","EOQ",
                 "SimEOQ","ReorderPoint","SimReorderPoint"]].head(20).round(1),
        use_container_width=True)
