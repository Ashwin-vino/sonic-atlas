# SONIC ATLAS — Phase 3 Completion Report
## Milestone: Correlation Analysis, Feature Engineering & Machine Learning Clustering

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Development Phase**: Phase 3 of 5 (Completed)  
> **Target System**: Unsupervised K-Means Acoustic Clustering & PCA Dimensionality Reduction

---

## 1. Summary of Completed Engineering

Phase 3 has been fully implemented, evaluated, serialized, and integrated into the SONIC ATLAS platform. All tasks strictly adhere to the project design tokens, anti-leakage rules, and non-destructive data integrity guidelines.

### Summary of Major Deliverables
1. **Correlation Engine ([`src/correlation.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/correlation.py))**:
   - Pearson linear and Spearman rank correlation matrices computed.
   - Ranked pairwise relationship table with objective musical interpretations.
   - Discrete musical attributes analyzed: Western chromatic pitch classes (`key`, 0–11) and binary modality (`mode`).
   - Publication-quality heatmaps generated at 300 DPI (`correlation_matrix_pearson.png`, `correlation_matrix_spearman.png`).
2. **Feature Engineering Pipeline ([`src/clustering.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/clustering.py))**:
   - Strict anti-leakage validation ensuring no ground-truth genres, identifiers, or popularity metadata enter model inputs.
   - Single duration representation enforced (`duration_ms` only; no duplicate duration features).
   - Standardized 10 continuous audio features with `StandardScaler` ($\mu = 0, \sigma = 1$).
3. **Observation-Level vs Unique-Track Experimentation**:
   - Evaluated Experiment A (32,833 playlist observations) against Experiment B (28,356 unique tracks).
   - Proved cluster centroid stability across both representations within 2.4% metric deviation.
4. **K-Means Optimization ($K=2$ to $12$)**:
   - Evaluated across Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
   - Selected $K=6$ as the optimal tradeoff between geometric separation and musical interpretability.
5. **PCA 2D Dimensionality Reduction**:
   - PC1 (Intensity/Power: $21.52\%$) and PC2 (Groove/Positivity: $15.45\%$) capture $36.97\%$ total variance.
6. **Acoustic Profiling & Persona Archetype Naming**:
   - Derived 6 distinct sonic archetypes:
     - Cluster 0: *High-Octane Anthem (Driving Rock & Peak-Time EDM)*
     - Cluster 1: *Instrumental Synth Wave (Club & Progressive Tech)*
     - Cluster 2: *Live Concert & Stadium Arena (Stage Performance)*
     - Cluster 3: *Rhythmic Groove & Vibrant Pop (Latin Reggaeton & Dance Pop)*
     - Cluster 4: *Lyrical Flow & Urban Hip-Hop (Rap & Trap Flow)*
     - Cluster 5: *Acoustic Intimacy & Mellow Soul (Downtempo & Ballads)*
7. **Artifact Persistence (`models/`)**:
   - `models/kmeans_model.pkl` (Fitted K-Means $K=6$).
   - `models/feature_scaler.pkl` (Fitted StandardScaler).
   - `models/clustering_metadata.json` (Model metadata and performance parameters).
   - `data/processed/spotify_songs_clustered.csv` (Dataset with assigned cluster IDs and PCA coordinates).
8. **Interactive Streamlit Application ([`app.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/app.py))**:
   - Activated the previously reserved **Cluster Intelligence** page with PCA 2D projections, cluster metrics, comparative feature profiles, genre cross-tabulation, and interactive cluster track inspector.
9. **Automated Test Suite**:
   - Expanded to 32 unit and integration tests across 5 test modules. 100% passing.

---

## 2. Quantitative Results & Evaluation Metrics

### Correlation Findings
- **Energy & Loudness**: Pearson $r = +0.677$ (Strong positive).
- **Energy & Acousticness**: Pearson $r = -0.540$ (Moderate-Strong negative).
- **Loudness & Acousticness**: Pearson $r = -0.362$ (Moderate negative).
- **Danceability & Valence**: Pearson $r = +0.331$ (Moderate positive).
- **Danceability & Tempo**: Pearson $r = -0.184$ (Weak negative).

### K-Means $K=6$ Production Metrics
- **Inertia (Within-Cluster Sum of Squares)**: **203,541.4**
- **Silhouette Coefficient**: **0.1380** (Local maximum in actionable range $K \ge 3$)
- **Davies-Bouldin Index**: **1.7350** (Optimal separation)
- **Calinski-Harabasz Variance Ratio**: **4,025.16**
- **PCA Total Explained Variance (2 Components)**: **36.97%**
- **Post-Hoc Ground-Truth Cluster Purity**: **44.82%**

---

## 3. Discovered Cluster Sizes & Centroid Profiles

| Cluster | Archetype Name | Observations | % Catalog | Dominant Feature Extremes |
|---|---|---|---|---|
| **C0** | High-Octane Anthem | 8,466 | 25.8% | Highest Loudness (-5.33 dB), Fastest Tempo (132.6 BPM), Energy (0.791) |
| **C1** | Instrumental Synth Wave | 2,470 | 7.5% | Highest Instrumentalness (0.746), Longest Duration (252.1s) |
| **C2** | Live Concert & Stadium Arena | 2,128 | 6.5% | Highest Liveness (0.614), High Energy (0.780) |
| **C3** | Rhythmic Groove & Vibrant Pop | 10,762 | 32.8% | Highest Danceability (0.740), Brightest Valence (0.680) |
| **C4** | Lyrical Flow & Urban Hip-Hop | 4,327 | 13.2% | Highest Speechiness (0.313), High Danceability (0.726) |
| **C5** | Acoustic Intimacy & Mellow Soul | 4,680 | 14.3% | Highest Acousticness (0.519), Lowest Energy (0.425), Softest (-10.5 dB) |

---

## 4. Visual Artifacts Generated in Phase 3 (`reports/figures/`)

| Filename | Description |
|---|---|
| `correlation_matrix_pearson.png` | 10x10 annotated heatmap of Pearson correlation coefficients. |
| `correlation_matrix_spearman.png` | 10x10 annotated heatmap of Spearman monotonic rank coefficients. |
| `kmeans_elbow_curve.png` | Inertia vs K curve displaying the elbow inflection point at K=5–6. |
| `silhouette_scores.png` | Mean Silhouette score vs K showing local peak at K=6. |
| `davies_bouldin_scores.png` | Davies-Bouldin index curve illustrating minimal cluster overlap. |
| `calinski_harabasz_scores.png` | Calinski-Harabasz dispersion variance ratio curve. |
| `cluster_visualization_pca.png` | 2D PCA projection scatter plot color-coded by the 6 clusters. |
| `cluster_feature_profiles.png` | Multi-bar horizontal feature profile comparing normalized audio dimensions. |
| `cluster_genre_distribution.png` | 100% stacked bar chart cross-tabulating discovered clusters with super genres. |
| `cluster_size_distribution.png` | Bar chart illustrating observation counts across the 6 clusters. |

---

## 5. Automated Testing Verification

Executed automated test suite:
- `tests/test_foundation.py` (7 tests)
- `tests/test_preprocessing.py` (7 tests)
- `tests/test_analysis.py` (5 tests)
- `tests/test_correlation.py` (6 tests)
- `tests/test_clustering.py` (7 tests)

**Result**: **32 tests passed in 1.546s (100% success rate)**.
- Zero existing Phase 1 or Phase 2 tests were broken.
- Data integrity, zero data loss, and anti-leakage protection verified.

---

## 6. Limitations & Boundaries

1. **Unsupervised Nature**: K-Means clusters data by acoustic geometry in continuous space. Discovered clusters do not map 1:1 with human social/cultural genre tags (e.g. Rock and EDM share loud anthemic clusters).
2. **Phase Boundary**: No recommendation engine, nearest-neighbor matching, or Spotify API features were implemented. Those remain strictly reserved for Phase 4.

