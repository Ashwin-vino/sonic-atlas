# SONIC ATLAS — Machine Learning Clustering Report
## Phase 3D–3G: K-Means Optimization, Evaluation Metrics, PCA Projection & Taxonomy Alignment

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Algorithm**: K-Means ($K=6$, $n\_init=10$, random_state=42)  
> **Input Dimensions**: 10 standardized continuous audio features  
> **Data Scope**: 32,833 playlist observations (Experiment A) & 28,356 unique tracks (Experiment B)

---

## 1. Cluster Count Optimization ($K=2$ to $12$)

To determine the optimal cluster count, K-Means models were evaluated across $K \in [2, 12]$ using four complementary evaluation metrics:

| Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---|---|---|---|
| **K = 2** | 283,346.9 | 0.1745 | 2.2285 | 5,212.2 |
| **K = 3** | 255,753.1 | 0.1248 | 2.2023 | 4,658.2 |
| **K = 4** | 234,481.3 | 0.1333 | 1.9772 | 4,379.8 |
| **K = 5** | 216,335.3 | 0.1361 | 1.8399 | 4,248.7 |
| **K = 6 (Selected)** | **203,541.4** | **0.1380** | **1.7350** | **4,025.2** |
| **K = 7** | 193,537.2 | 0.1278 | 1.7642 | 3,810.4 |
| **K = 8** | 185,760.5 | 0.1223 | 1.7318 | 3,599.0 |
| **K = 9** | 178,768.8 | 0.1256 | 1.6767 | 3,432.7 |
| **K = 10** | 172,348.7 | 0.1290 | 1.6332 | 3,300.7 |
| **K = 11** | 167,635.3 | 0.1280 | 1.6261 | 3,146.3 |
| **K = 12** | 163,303.9 | 0.1221 | 1.6982 | 3,015.2 |

---

## 2. Selection of Final $K=6$ and Justification

### Multi-Metric Decision Matrix
1. **Silhouette Score**:
   While $K=2$ produces the highest global silhouette score ($0.1745$), $K=2$ merely separates music into a crude binary split ("Acoustic/Low-Energy" vs "Electronic/High-Energy"), failing to differentiate rap cadences, instrumental arrangements, or rhythmic Latin grooves. Within the actionable range ($K \ge 3$), **$K=6$ achieves a distinct local maximum ($0.1380$)**, surpassing $K=3, 4, 7, 8, 9, 10, 11, 12$.
2. **Davies-Bouldin Index**:
   $K=6$ drops the Davies-Bouldin score to **$1.7350$** (a 22% improvement in cluster separation compared to $K=2$).
3. **Elbow Method (Inertia)**:
   The rate of reduction in inertia exhibits a pronounced inflection point ("elbow") between $K=5$ and $K=7$. Past $K=6$, gains per additional cluster drop below 5%.
4. **Interpretability & Catalog Alignment**:
   $K=6$ provides six musically distinct, non-trivial sonic archetypes with balanced cluster populations, providing an intuitive, interpretable baseline for comparing unsupervised sonic clusters against human genre curation.

> **Important Analytical Distinction**: Unsupervised K-Means clustering is not a classification task; it optimizes geometric feature space density, not class accuracy. Evaluation reflects acoustic cohesion rather than label reproduction.

---

## 3. PCA Dimensionality Reduction & 2D Projection

To visualize the 10-dimensional feature space in two dimensions, Principal Component Analysis (PCA) was fitted to the standardized feature matrix:

- **PC1 (Intensity & Acoustic Power)**: Explains **21.52%** of total variance.
  - Dominant positive loadings: `energy` (+0.615), `loudness` (+0.554).
  - Dominant negative loadings: `acousticness` (-0.490).
- **PC2 (Rhythmic Groove & Mood Positivity)**: Explains **15.45%** of total variance.
  - Dominant positive loadings: `danceability` (+0.599), `valence` (+0.532), `speechiness` (+0.339).
  - Dominant negative loadings: `instrumentalness` (-0.310).
- **Combined 2D Explained Variance**: **36.97%**.

The PCA visualization ([`reports/figures/cluster_visualization_pca.png`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/reports/figures/cluster_visualization_pca.png)) confirms clear spatial separation between the high-energy clusters (Clusters 0, 1, 2) in positive PC1 territory and the acoustic/mellow cluster (Cluster 5) in negative PC1 territory.

---

## 4. Post-Hoc Ground-Truth Genre Alignment

Because genre labels were strictly excluded from model inputs, comparing the discovered clusters against ground truth super genres provides a genuine measure of how raw acoustic properties mirror human taxonomic curation:

| Evaluation Metric | Score | Scientific Meaning |
|---|---|---|
| **Overall Cluster Purity** | **44.82%** | Proportion of observations correctly matching their cluster's plurality genre. |
| **Adjusted Rand Index (ARI)** | **0.1341** | Pairwise agreement corrected for chance ($0.0 = \text{random}$). |
| **Adjusted Mutual Information (AMI)** | **0.1702** | Shared mutual information between acoustic clusters and genre labels. |
| **Homogeneity** | **0.1712** | Extent to which each cluster contains only members of a single genre. |
| **Completeness** | **0.1706** | Extent to which members of a given genre are assigned to the same cluster. |
| **V-Measure** | **0.1709** | Harmonic mean of homogeneity and completeness. |

### Interpretation
The moderate alignment scores confirm that **acoustic anatomy transcends human cultural genre labels**. For instance:
- **Cluster 0** brings together both heavy rock anthems and high-energy electronic tracks because both share extreme loudness, high tempo, and distorted frequency compression.
- **Cluster 3** pairs rhythmic Latin tracks with dance pop and upbeat R&B due to shared syncopation, high danceability, and high valence.
- **Cluster 4** isolates high-speechiness rap and hip-hop tracks.

---

## 5. Serialized Model Artifacts (`models/`)

1. `models/kmeans_model.pkl`: Serialized `sklearn.cluster.KMeans` model instance ($K=6$).
2. `models/feature_scaler.pkl`: Serialized `sklearn.preprocessing.StandardScaler` instance.
3. `models/clustering_metadata.json`: Machine-readable metadata including hyperparameter records, metrics, timestamps, and archetype definitions.
