# SONIC ATLAS — Feature Engineering & Preparation Report
## Phase 3B: Feature Scaling, Standardization, Anti-Leakage Audit & Observation Comparison

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Source**: `data/processed/spotify_songs_cleaned.csv`  
> **Target Models**: K-Means Unsupervised Clustering & PCA 2D Dimensionality Reduction

---

## 1. Feature Preparation Audit

The feature engineering pipeline enforces strict data integrity, preventing data leakage, duplicate weightings, and numerical instability.

| Feature Name | Status | Feature Type | Reason for Inclusion / Exclusion | Missing Values | Infinite Values |
|---|---|---|---|---|---|
| `danceability` | **INCLUDED** | Continuous Audio | Core rhythmic and groove measurement ($0.0$ to $1.0$). | 0 | 0 |
| `energy` | **INCLUDED** | Continuous Audio | Core perceived activity and intensity ($0.0$ to $1.0$). | 0 | 0 |
| `loudness` | **INCLUDED** | Continuous Audio | Overall master sound level in decibels ($-46.45$ to $+1.28$ dB). | 0 | 0 |
| `speechiness` | **INCLUDED** | Continuous Audio | Vocal spoken word presence ($0.0$ to $1.0$). | 0 | 0 |
| `acousticness` | **INCLUDED** | Continuous Audio | Acoustic instrument confidence score ($0.0$ to $1.0$). | 0 | 0 |
| `instrumentalness` | **INCLUDED** | Continuous Audio | Vocal absence prediction ($0.0$ to $1.0$). | 0 | 0 |
| `liveness` | **INCLUDED** | Continuous Audio | Presence of audience / live stage ambience ($0.0$ to $1.0$). | 0 | 0 |
| `valence` | **INCLUDED** | Continuous Audio | Musical positiveness and mood ($0.0$ to $1.0$). | 0 | 0 |
| `tempo` | **INCLUDED** | Continuous Audio | Pacing pace in Beats Per Minute ($0.0$ to $239.44$ BPM). | 0 | 0 |
| `duration_ms` | **INCLUDED** | Continuous Audio | Song length in milliseconds ($4,000$ to $517,810$ ms). | 0 | 0 |
| `duration_sec` | **EXCLUDED** | Derived Duplicate | Duplicate duration representation (avoids double feature weighting). | — | — |
| `duration_min` | **EXCLUDED** | Derived Duplicate | Duplicate duration representation (avoids double feature weighting). | — | — |
| `key` | **EXCLUDED** | Discrete Musical | Nominal pitch class ($0$ to $11$). Continuous scaling would induce false distance metrics. | — | — |
| `mode` | **EXCLUDED** | Discrete Musical | Binary modality ($0$ or $1$). Kept for post-hoc musical characterization. | — | — |
| `track_popularity` | **EXCLUDED** | Catalog Metadata | Commercial streaming performance metric, not an acoustic sonic property. | — | — |
| `playlist_genre` | **EXCLUDED** | Target Label | **Anti-Leakage Protection**: Ground-truth genre reserved strictly for post-hoc evaluation. | — | — |
| `playlist_subgenre` | **EXCLUDED** | Target Label | **Anti-Leakage Protection**: Ground-truth subgenre reserved strictly for post-hoc evaluation. | — | — |
| `playlist_name` | **EXCLUDED** | Identifier | Metadata descriptor. | — | — |
| `playlist_id` | **EXCLUDED** | Identifier | Spotify playlist identifier. | — | — |
| `track_id` | **EXCLUDED** | Identifier | Spotify 22-character unique composition identifier. | — | — |
| `track_name` | **EXCLUDED** | Text Metadata | Song title. | — | — |
| `track_artist` | **EXCLUDED** | Text Metadata | Recording artist. | — | — |
| `track_album_id` | **EXCLUDED** | Identifier | Spotify album identifier. | — | — |
| `track_album_name` | **EXCLUDED** | Text Metadata | Album title. | — | — |
| `track_album_release_date` | **EXCLUDED** | Metadata | Release date string. | — | — |
| `release_year` | **EXCLUDED** | Temporal Feature | Historical year. Excluded to keep clustering focused on sonic timbre rather than release era. | — | — |
| `metadata_imputed` | **EXCLUDED** | Audit Flag | Provenance boolean flag. | — | — |

- **Final Feature Count**: **10 continuous audio features**.
- **Observations Used**: **32,833 playlist observations** (Experiment A) and **28,356 unique tracks** (Experiment B).
- **Missing Values in Feature Matrix**: **0**.
- **Infinite Values in Feature Matrix**: **0**.

---

## 2. Feature Standardization Methodology

### Why StandardScaler is Mandatory
K-Means clustering relies on Euclidean distance:
$$d(x, c) = \sqrt{\sum_{j=1}^{p} (x_j - c_j)^2}$$

Raw audio features exhibit vastly divergent scales:
- `duration_ms`: spans from $4,000$ to $517,810$ (order of $10^5$).
- `tempo`: spans from $0.0$ to $239.44$ (order of $10^2$).
- `loudness`: spans from $-46.45$ to $+1.28$ (order of $10^1$).
- `danceability`, `energy`, `valence`, etc.: span from $0.0$ to $1.0$ (order of $10^0$).

Without standardization, `duration_ms` would dominate Euclidean distance by a factor of 100,000x over `danceability`, rendering the clustering a simple duration partition.

`StandardScaler` transforms each feature $j$ to have zero mean and unit variance:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

- **Verification**: Post-scaling verification confirmed $\mu_j = 0.0000000$ and $\sigma_j = 1.0000000$ across all 10 features.
- **Persistence**: The fitted scaler is serialized at `models/feature_scaler.pkl` to transform future queries and tracks in Phase 4.

---

## 3. Experiment A vs Experiment B Comparison

| Comparison Metric | Experiment A: Observation-Level | Experiment B: Unique-Track Level | Delta / Deviation |
|---|---|---|---|
| **Total Rows Clustered** | **32,833** | **28,356** | -4,477 (-13.6%) |
| **Unique Track IDs** | 28,356 | 28,356 | Identical catalog compositions |
| **Selected Clusters ($K$)** | 6 | 6 | Concordant |
| **Inertia** | 203,541.4 | 175,539.3 | Scaled proportionally to sample size |
| **Silhouette Score** | **0.1380** | **0.1347** | $\Delta = 0.0033$ (2.4% variation) |
| **Davies-Bouldin Index** | **1.7350** | **1.7363** | $\Delta = 0.0013$ (0.07% variation) |
| **Calinski-Harabasz Index** | **4,025.2** | **3,489.1** | Proportionally aligned |
| **Cluster Centroid Topologies** | Stable across all 10 dimensions | Stable across all 10 dimensions | Substantially identical centroids |

### Conclusion on Playlist Repetitions
Repeated playlist appearances represent songs curated across multiple subgenres. The experimental comparison proves that repeated playlist appearances **do not distort** the cluster topology. Cluster centroids, feature distributions, and silhouette scores remain stable within 2.4% variance. 

The primary platform presentation utilizes **Experiment A (32,833 observations)** to preserve playlist context, with seamless capability to view unique tracks.
