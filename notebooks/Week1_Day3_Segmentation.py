"""
=============================================================
RetailPulse – Week 1, Day 3
Customer Segmentation : RFM + K-Means + DBSCAN
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import mlflow, mlflow.sklearn, joblib, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PLOTS_DIR  = Path("../reports/plots")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. LOAD RFM DATA ────────────────────────────────────────
rfm = pd.read_parquet("../data/rfm_features.parquet")
print(f"RFM customers: {len(rfm):,}")

features = ["R_Score","F_Score","M_Score","Recency","Frequency","Monetary"]
X = rfm[features].copy()
X["Monetary"] = np.log1p(X["Monetary"])

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. ELBOW / SILHOUETTE ────────────────────────────────────
inertias, sil_scores, db_scores = [], [], []
for k in range(2, 12):
    km     = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

fig, axes = plt.subplots(1,3,figsize=(16,4))
axes[0].plot(range(2,12), inertias,   "bo-"); axes[0].set_title("Elbow – Inertia")
axes[1].plot(range(2,12), sil_scores, "rs-"); axes[1].set_title("Silhouette Score")
axes[2].plot(range(2,12), db_scores,  "gD-"); axes[2].set_title("Davies-Bouldin")
for ax in axes: ax.set_xlabel("k")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"05_kmeans_selection.png", dpi=150); plt.close()

# ── 3. FINAL K-MEANS ─────────────────────────────────────────
BEST_K = 6
mlflow.set_experiment("customer_segmentation")
with mlflow.start_run(run_name=f"kmeans_k{BEST_K}"):
    km_final = KMeans(n_clusters=BEST_K, init="k-means++", n_init=20, random_state=42)
    rfm["KMeans_Cluster"] = km_final.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, rfm["KMeans_Cluster"])
    db  = davies_bouldin_score(X_scaled, rfm["KMeans_Cluster"])
    mlflow.log_params({"k": BEST_K})
    mlflow.log_metrics({"silhouette": round(sil,4), "davies_bouldin": round(db,4)})
    mlflow.sklearn.log_model(km_final, "kmeans_model")
    print(f"K-Means k={BEST_K} | Silhouette={sil:.4f} | DB={db:.4f}")

# ── 4. DBSCAN ────────────────────────────────────────────────
with mlflow.start_run(run_name="dbscan"):
    dbscan = DBSCAN(eps=0.8, min_samples=5, n_jobs=-1)
    rfm["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
    n_noise = (rfm["DBSCAN_Cluster"]==-1).sum()
    mlflow.log_metrics({"noise_pts": int(n_noise)})
    print(f"DBSCAN noise points: {n_noise}")

# ── 5. BUSINESS LABELLING ────────────────────────────────────
cluster_stats = rfm.groupby("KMeans_Cluster").agg(
    Count=("CustomerID","count"), Recency=("Recency","mean"),
    Frequency=("Frequency","mean"), Monetary=("Monetary","mean"),
    RFM_Score=("RFM_Score","mean")).round(1).sort_values("RFM_Score", ascending=False)

segment_map = {
    cluster_stats.index[0]: "Champions",
    cluster_stats.index[1]: "Loyal Customers",
    cluster_stats.index[2]: "Potential Loyalists",
    cluster_stats.index[3]: "At Risk",
    cluster_stats.index[4]: "Hibernating",
    cluster_stats.index[5]: "Lost Customers",
}
rfm["Segment"] = rfm["KMeans_Cluster"].map(segment_map)
print(rfm["Segment"].value_counts())

# ── 6. PLOTS ─────────────────────────────────────────────────
palette = {"Champions":"#2ecc71","Loyal Customers":"#3498db",
           "Potential Loyalists":"#f39c12","At Risk":"#e67e22",
           "Hibernating":"#e74c3c","Lost Customers":"#7f8c8d"}

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rfm["PCA1"], rfm["PCA2"] = X_pca[:,0], X_pca[:,1]

fig, axes = plt.subplots(1,2,figsize=(16,6))
for seg, col in palette.items():
    m = rfm["Segment"]==seg
    axes[0].scatter(rfm.loc[m,"PCA1"], rfm.loc[m,"PCA2"], label=seg, alpha=0.5, s=20, color=col)
axes[0].set_title("Customer Segments – PCA 2D"); axes[0].legend(fontsize=8)

seg_counts = rfm["Segment"].value_counts()
axes[1].barh(seg_counts.index, seg_counts.values,
             color=[palette.get(s,"#95a5a6") for s in seg_counts.index])
axes[1].set_title("Customers per Segment")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"06_customer_segments.png", dpi=150); plt.close()

# ── 7. SAVE ──────────────────────────────────────────────────
rfm.to_parquet("../data/rfm_segmented.parquet", index=False)
cluster_stats.to_csv("../reports/cluster_summary.csv")
joblib.dump(km_final, MODELS_DIR/"kmeans_segmentation.pkl")
joblib.dump(scaler,   MODELS_DIR/"rfm_scaler.pkl")
print("\n✅  Week 1, Day 3 COMPLETE")
