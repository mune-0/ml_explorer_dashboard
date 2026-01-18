import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Page title
st.title("🔍 Unsupervised Learning")
st.markdown("## Clustering and Dimensionality Reduction")

# Sidebar parameters
st.sidebar.header("Clustering Parameters")
n_clusters = st.sidebar.slider(
    "Number of Clusters (k)",
    min_value=2,
    max_value=10,
    value=3
)
n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

# ========================================
# DATA GENERATION
# ========================================

# Generate sample data (4 features for PCA demo)
X, true_labels = make_blobs(
    n_samples=n_samples,
    centers=5,
    n_features=4,
    random_state=42,
    cluster_std=1.0
)

# ========================================
# CLUSTERING
# ========================================

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Calculate silhouette score
silhouette = silhouette_score(X, cluster_labels)

# Perform PCA for visualization (4D -> 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ========================================
# DISPLAY RESULTS
# ========================================

st.subheader("📊 Clustering Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clusters (k)", n_clusters)

with col2:
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    quality = "Excellent!" if silhouette > 0.7 else "Good!" if silhouette > 0.5 else "Fair"
    st.caption(f"({quality})")

with col3:
    st.metric("Samples", n_samples)

# ========================================
# CLUSTER VISUALIZATION
# ========================================

# Visualize clusters
fig = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=cluster_labels.astype(str),
    title='K-Means Clustering (PCA Visualization)',
    labels={
        'x': 'First Principal Component (PC1)',
        'y': 'Second Principal Component (PC2)',
        'color': 'Cluster'
    },
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Add cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
fig.add_trace(go.Scatter(
    x=centers_pca[:, 0],
    y=centers_pca[:, 1],
    mode='markers',
    marker=dict(
        size=20,
        symbol='x',
        color='black',
        line=dict(width=2, color='white')
    ),
    name='Centroids',
    showlegend=True
))

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Educational info
st.info("""
💡 **Try this:**
- Move the "Number of Clusters" slider
- Watch how the groups reform
- Notice the silhouette score change
- Find the optimal k value!
""")
