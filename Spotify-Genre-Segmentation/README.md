# SONIC ATLAS — Music Intelligence Platform

> **Disclaimer**: *SONIC ATLAS is an independent, educational, and research-oriented music intelligence and data science project. It is not affiliated with, endorsed by, or sponsored by Spotify AB.*

---

## 1. Project Overview

**SONIC ATLAS** is a commercial-grade music intelligence and acoustic analytics platform engineered to explore, analyze, cluster, and recommend music based on high-dimensional audio feature representations.

Built on an empirical corpus of over 32,000 Spotify tracks across 6 major musical genres and 24 granular subgenres, SONIC ATLAS moves beyond metadata-only categorization to uncover the intrinsic acoustic anatomy of music: danceability, energy, acousticness, valence, speechiness, and harmonic characteristics.

---

## 2. Problem Statement

Modern music streaming catalogs house tens of millions of songs. However, traditional genre taxonomy is often subjective, inconsistent, and fragmented across record labels and cultural labels. Listeners and curators struggle to:
- Discover musically cohesive tracks when genre labels fail to capture sonic textures.
- Quantify what makes genres sonically distinct versus acoustically overlapping.
- Model multi-attribute music similarity without relying on external collaborative filtering metadata.

SONIC ATLAS solves this by leveraging acoustic feature vectors, unsupervised machine learning clustering, and vector similarity models to segment tracks based on raw musical properties.

---

## 3. Project Objectives

1. **Empirical Dataset Audit & Cleaning**: Rigorously validate, clean, and profile 32,833 playlist observations with zero data loss.
2. **Exploratory Acoustic Analysis**: Characterize audio feature distributions, skewness, moments, and cross-genre signatures.
3. **Unsupervised Genre Segmentation**: Discover data-driven sonic clusters using K-Means and dimensionality reduction (PCA / t-SNE) without human labeling bias (Phase 3).
4. **Content-Based Music Recommendation**: Formulate high-dimensional cosine similarity recommendations allowing users to find "acoustic twins" across catalog boundaries (Phase 4).
5. **Production Analytical Interface**: Deliver a clean, editorial dark-mode interface with restrained green accents, responsive typography, and reactive analytics (Phases 1–5).

---

## 4. Technology Stack

- **Core Language**: Python 3.9+
- **Data Manipulation & Analytics**: Pandas (`>=2.0.0`), NumPy (`>=1.24.0`)
- **Machine Learning & Modeling**: Scikit-Learn (`>=1.3.0`)
- **Visualization Engines**: Matplotlib (`>=3.7.0`), Seaborn (`>=0.12.0`)
- **Application Framework**: Streamlit (`>=1.28.0`)
- **Design System**: Custom CSS incorporating Manrope (Headings), Inter (Body), and JetBrains Mono (Technical telemetry)
- **Development & Versioning**: Jupyter Notebooks, Git

---

## 5. Project Folder Structure

```text
SONIC-ATLAS/
│
├── data/
│   ├── raw/
│   │   └── spotify_songs.csv           # Original dataset (32,833 records)
│   │
│   └── processed/
│       ├── spotify_songs_cleaned.csv   # Cleaned dataset (32,833 observations, 0 nulls)
│       ├── spotify_songs_clustered.csv # Clustered dataset with K=6 labels & 2D PCA
│       └── spotify_songs_recommendation_catalog.csv # Unique-track catalog (28,356 compositions)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Complete 14-section Phase 2 EDA notebook
│   ├── 02_correlation_and_clustering.ipynb # Phase 3: Collinearity, K-Means & PCA clustering
│   └── 03_recommendation_engine.ipynb  # Phase 4: Nearest neighbors, cross-genre & diversity
│
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Centralized paths, seeds, schema & design tokens
│   ├── data_loader.py                  # Robust CSV ingestion, fallback path search & summary
│   ├── preprocessing.py                # Cleaning pipeline, metadata fallbacks & validation
│   ├── analysis.py                     # Statistical distributions, rankings & figures
│   ├── correlation.py                  # (Phase 3) Pearson/Spearman collinearity & discrete analysis
│   ├── clustering.py                   # (Phase 3) StandardScaler, K-Means (K=6), 2D PCA & profiling
│   ├── evaluation.py                   # (Phase 3 & 4) Clustering & recommendation metrics
│   └── recommender.py                  # (Phase 4) Cosine NearestNeighbors recommendation engine
│
├── models/                             # Serialized Phase 3 & 4 ML Artifacts
│   ├── kmeans_model.pkl                # Trained K-Means estimator (K=6, random_state=42)
│   ├── feature_scaler.pkl              # Fitted StandardScaler (10 continuous audio features)
│   ├── clustering_metadata.json        # Hyperparameters, evaluation scores & archetype specs
│   ├── nearest_neighbors_model.pkl     # Fitted NearestNeighbors model (metric='cosine', 28,356 tracks)
│   └── recommendation_metadata.json    # Recommendation engine configuration and parameters
│
├── reports/
│   ├── EDA_SUMMARY.md                  # Comprehensive Phase 2 EDA & cleaning report
│   ├── CORRELATION_ANALYSIS.md         # Comprehensive Phase 3 correlation findings
│   ├── FEATURE_ENGINEERING_REPORT.md   # Scaling & anti-leakage feature preparation report
│   ├── CLUSTERING_REPORT.md            # K-Means multi-criteria evaluation & selection report
│   ├── CLUSTER_PROFILES.md             # Empirical feature fingerprints of the 6 archetypes
│   ├── PHASE_3_COMPLETION_REPORT.md    # Formal Phase 3 audit & sign-off report
│   ├── RECOMMENDATION_ENGINE_REPORT.md # Recommendation engine architecture & pipeline report
│   ├── RECOMMENDATION_EVALUATION.md    # Recommendation benchmarks, coverage & diversity report
│   ├── PHASE_4_COMPLETION_REPORT.md    # Formal Phase 4 audit & sign-off report
│   └── figures/                        # 24 publication-quality 300 DPI visualizations
│
├── tests/
│   ├── __init__.py
│   ├── test_foundation.py              # Phase 1 foundation test suite
│   ├── test_preprocessing.py           # Phase 2 cleaning & pipeline tests
│   ├── test_analysis.py                # Phase 2 analytics & figures tests
│   ├── test_correlation.py             # Phase 3 correlation & ranking tests
│   ├── test_clustering.py              # Phase 3 K-Means, scaling, PCA & serialization tests
│   └── test_recommender.py             # Phase 4 recommendation modes, filters, diversity & evaluation tests
│
├── assets/
│   ├── fonts/                          # Typography assets
│   └── images/                         # Graphic assets
│
├── project-prompts/
│   ├── 01-foundation.md                # Phase 1 master instructions
│   ├── 02-preprocessing-eda.md         # Phase 2 master instructions
│   ├── 03-correlation-clustering.md    # Phase 3 master instructions
│   ├── 04-recommendation-engine.md     # Phase 4 master instructions
│   └── 05-final-application.md         # Phase 5 master instructions
│
├── app.py                              # Streamlit application with Overview, Explorer, Genres, Insights & Clustering
├── config.py                           # Root configuration proxy
├── requirements.txt                    # Pinned production dependencies
├── README.md                           # Documentation & platform guide
└── .gitignore                          # Git exclusion rules
```

---

## 6. Preprocessing Methodology & Data Integrity Rules

The preprocessing pipeline ([`src/preprocessing.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/preprocessing.py)) executes strict, non-destructive transformations:

1. **Preserving Playlist Observations**:
   - The 32,833 rows represent playlist-associated song observations.
   - Repeated track IDs (4,477 occurrences) across different playlists are **preserved**, maintaining vital playlist context.
   - Exact 23-column duplicate rows: **0**.
   - Cleaned observation rows: **32,833 (100% catalog retention)**.
   - Unique composition view: A separate helper function (`create_unique_tracks_view()`) generates the 28,356 unique-track view for composition-only modeling.

2. **Non-Destructive Metadata Fallbacks**:
   - Exactly 5 records contained `NA` strings in `track_name`, `track_artist`, and `track_album_name`.
   - Records were **not deleted**. They received transparent display fallbacks (`"Unknown Track"`, `"Unknown Artist"`, `"Unknown Album"`).
   - An audit column `metadata_imputed` flags these 5 records for complete data provenance.

3. **Temporal Feature Extraction**:
   - Extracted 4-digit `release_year` (Integer) from `track_album_release_date` (1957 to 2020, 100% parsed).

4. **Duration Conversions**:
   - Added `duration_sec` and `duration_min` alongside `duration_ms`.

5. **Audio Feature Classification & Anti-Leakage Protection**:
   - **10 Continuous Features**: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.
   - **2 Discrete Musical Features**: `key` (0 to 11), `mode` (0 or 1).
   - **Popularity Measure**: `track_popularity` is treated as a commercial metadata metric, not an acoustic feature.
   - **Prohibited Clustering Inputs**: Ground-truth labels and identifiers (`playlist_genre`, `playlist_subgenre`, `playlist_name`, `playlist_id`, `track_id`, `track_name`, `track_artist`, `track_album_id`, `track_album_name`, `track_album_release_date`) are strictly isolated to prevent data leakage in Phase 3.

---

## 7. Exploratory Data Analysis (EDA) & Key Findings

1. **Taxonomic Balance**:
   - 6 super genres: EDM (18.41%), Rap (17.50%), Pop (16.77%), R&B (16.54%), Latin (15.70%), Rock (15.08%).
   - Exactly 4 balanced subgenres per super genre across 24 distinct classes.
2. **Acoustic Signatures**:
   - **EDM & Rock**: Highest perceived energy (means > 0.73).
   - **Rap & Latin**: Highest danceability (means > 0.71).
   - **Rap**: Uniquely high speechiness (mean = 0.198, median = 0.163).
   - **R&B**: Highest acousticness (mean = 0.225).
3. **Acoustic Collinearity**:
   - Energy and Loudness exhibit a strong positive correlation ($r \approx 0.68$).
4. **Temporal Evolution (1970–2020)**:
   - Acousticness declined precipitously from ~0.35 in the 1970s to <0.15 by the 2010s, while average energy rose to a sustained peak of ~0.72.

---

## 8. Phase 3 Machine Learning Clustering & Acoustic Topology

### 1. Collinearity Analysis
- **Strongest Positive Correlation**: `energy` and `loudness` ($r = +0.677$, $\rho = +0.656$).
- **Strongest Negative Correlation**: `energy` and `acousticness` ($r = -0.540$, $\rho = -0.491$).
- **Moderate Rhythmic Coupling**: `danceability` and `valence` ($r = +0.331$, $\rho = +0.331$).
- **Discrete Independence**: Musical key (0–11) and modality (major/minor) exhibit negligible linear correlation with continuous features and are properly analyzed as discrete musical dimensions.

### 2. Dataset Invariance: Experiment A vs. Experiment B
- **Experiment A (32,833 observations)** vs. **Experiment B (28,356 unique tracks)**.
- Maximum centroid delta across all 10 dimensions is **$< 2.4\%$**, and Silhouette scores match within $\pm 0.0006$ (0.1380 vs. 0.1386).
- Confirms that preserving multi-playlist song occurrences does **not** distort the geometric cluster manifold.

### 3. K-Means Model Selection ($K=6$)
- **Inertia (Elbow Method)**: Sharp curvature inflection at $K=6$.
- **Silhouette Score**: Optimal inflection point at $K=6$ ($0.1380$).
- **Davies-Bouldin Index**: Reaches a distinct minimum at $K=6$ ($1.7350$).
- **Calinski-Harabasz Index**: High dispersion ratio ($4,025.2$).

### 4. 2D Principal Component Analysis (PCA)
- **PC1 (21.52% Variance)**: Acoustic Intensity & Master Power (`loudness`, `energy` vs. `acousticness`).
- **PC2 (15.45% Variance)**: Groove & Harmonic Positivity (`danceability`, `valence` vs. `tempo`).
- **Total Variance Captured**: **36.97%** across 2 orthogonal dimensions.

### 5. Six Discovered Sonic Archetypes
1. **Cluster 0: High-Octane Anthem** (8,466 tracks, 25.8%, `#55D99A`) — Driving Rock & Peak-Time EDM. Loudest (-5.33 dB), fastest (132.6 BPM), energy (0.791).
2. **Cluster 1: Instrumental Synth Wave** (2,470 tracks, 7.5%, `#5BC0BE`) — Club & Progressive Tech. Instrumentalness (0.746), extended duration (252.1s).
3. **Cluster 2: Live Concert & Stadium Arena** (2,128 tracks, 6.5%, `#E9B86C`) — Stage Performance. Elevated liveness (0.614), energy (0.780).
4. **Cluster 3: Rhythmic Groove & Vibrant Pop** (10,762 tracks, 32.8%, `#D97755`) — Latin Reggaeton & Dance Pop. Danceability (0.740), valence (0.680).
5. **Cluster 4: Lyrical Flow & Urban Hip-Hop** (4,327 tracks, 13.2%, `#708A81`) — Rap, Trap & Spoken Cadence. Speechiness (0.313), danceability (0.726).
6. **Cluster 5: Acoustic Intimacy & Mellow Soul** (4,680 tracks, 14.3%, `#9AA79F`) — Organic Ballads & Ambient Soul. Acousticness (0.519), softest (-10.5 dB).

### 6. Anti-Leakage & Post-Hoc Alignment
- Ground-truth genre labels were withheld during clustering.
- Post-hoc alignment yields Adjusted Rand Index ($\text{ARI} = 0.1345$) and Cluster Purity ($0.4284$). This objectively demonstrates that unsupervised clustering grouped songs by **sonic texture rather than memorizing genre tags**, discovering genuine cross-genre affinities.

---

## 9. Publication-Quality Visualizations (`reports/figures/`)

Generated by [`src/analysis.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/analysis.py), [`src/correlation.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/correlation.py), and [`src/clustering.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/src/clustering.py) at 300 DPI:

### Phase 2: Exploratory Data Analysis
- `genre_distribution.png`: Dual-panel bar and donut chart illustrating super genre proportions.
- `songs_per_genre.png`: Standalone observation counts across super genres.
- `songs_per_subgenre.png`: Horizontal bar chart of all 24 subgenres, color-coded by genre.
- `genre_popularity_distribution.png`: Boxplot and mean bar chart analyzing popularity by genre.
- `top_artists.png`: Top 15 most represented recording artists in the catalog.
- `artist_popularity_distribution.png`: Observation counts vs average artist popularity.
- `top_playlists.png`: Top 12 curated playlists by track count.
- `audio_feature_distributions.png`: 3x3 histogram grid with KDEs, means, and medians for 9 continuous features.
- `genre_audio_features_comparison.png`: 2x3 boxplot matrix comparing Danceability, Energy, Valence, Acousticness, Speechiness, and Tempo across genres.
- `danceability_vs_energy.png`: Stratified scatter plot showing genre segregation across rhythm and intensity.
- `energy_vs_loudness.png`: Collinearity scatter plot illustrating energy vs mastering loudness.
- `valence_vs_energy.png`: Musical positiveness vs intensity scatter plot.
- `release_year_trends.png`: Temporal progression of acoustic properties from 1970 to 2020.
- `missing_values_audit.png`: Data completeness chart confirming 99.98% completeness.

### Phase 3: Correlation & Machine Learning Clustering
- `correlation_matrix_pearson.png`: 10x10 annotated heatmap of linear Pearson correlations.
- `correlation_matrix_spearman.png`: 10x10 annotated heatmap of rank-order Spearman correlations.
- `kmeans_elbow_curve.png`: Inertia curve across $K \in [2, 12]$ showing inflection at $K=6$.
- `silhouette_scores.png`: Silhouette curve across $K \in [2, 12]$ peaking at $K=6$ ($0.1380$).
- `davies_bouldin_scores.png`: Davies-Bouldin index minimizing at $K=6$ ($1.7350$).
- `calinski_harabasz_scores.png`: Calinski-Harabasz index showing strong separation at $K=6$ ($4,025.2$).
- `cluster_visualization_pca.png`: 2D PCA scatter plot showing the 6 cluster archetypes and centroids.
- `cluster_feature_profiles.png`: 2x3 grid of horizontal bar charts profiling the 10 features for each archetype.
- `cluster_genre_distribution.png`: 100% stacked bar chart revealing post-hoc genre crossover per cluster.
- `cluster_size_distribution.png`: Bar chart of observation counts across all 6 clusters.

---

## 10. Phase 4 Content-Based Music Recommendation Engine

### 1. High-Dimensional Cosine Similarity Engine
- **Vector Space**: 10 continuous standardized audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) scaled with the Phase 3 `StandardScaler`.
- **Zero Collaborative Filtering**: Purely content-driven, eliminating cold-start barriers, tracking history requirements, and commercial popularity bias.
- **Index Architecture**: `NearestNeighbors(metric='cosine', algorithm='brute')` indexing 28,356 unique tracks with sub-5ms query latency.

### 2. Four Recommendation Modes
1. **Acoustic Similar Tracks (Default)**: Uncovers the closest acoustic neighbors across the full catalog ($\overline{\text{Sim}} = 0.9377$).
2. **Acoustic Twins — Cross-Genre Discovery**: Strictly excludes the seed track's super genre, returning acoustically identical compositions from other genres ($\overline{\text{Sim}} = 0.9245$, Cross-Genre Rate $= 100.0\%$).
3. **Same Sonic Archetype**: Restricts recommendations strictly within the seed track's Phase 3 cluster ($\overline{\text{Sim}} = 0.9338$).
4. **Explore Beyond Your Cluster**: Discovers nearby acoustic territories in adjacent cluster personas ($\overline{\text{Sim}} = 0.8677$, Cross-Genre Rate $= 62.65\%$).

### 3. Diversity-Aware Reranking (Balanced Discovery)
- Deterministic greedy reranking ensures **100% unique recording artists** in recommendation lists with a negligible similarity difference of only $\Delta = 0.0012$ ($0.9377 \rightarrow 0.9365$).

### 4. Grounded Acoustic Explanation Engine
- Analyzes directional audio-feature deltas between seed and recommended tracks, explaining energy alignment, rhythmic synchronization, acoustic depth, and genre transitions.

---

## 11. Installation & Setup

### 1. Activate Virtual Environment
```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing Pipeline
```bash
python src/preprocessing.py
```

### 4. Run EDA & Analytics Figures
```bash
python src/analysis.py
```

### 5. Run Correlation Pipeline
```bash
python src/correlation.py
```

### 6. Run Clustering & Machine Learning Pipeline
```bash
python src/clustering.py
```

### 7. Run Recommendation Engine Pipeline
```bash
python src/recommender.py
```

### 8. Run Complete Automated Test Suite (44 Tests)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 9. Launch Streamlit Application
```bash
streamlit run app.py
```

### 10. Run Jupyter Notebooks
- **Phase 2 EDA**: `notebooks/01_exploratory_data_analysis.ipynb`
- **Phase 3 ML Clustering**: `notebooks/02_correlation_and_clustering.ipynb`
- **Phase 4 Recommendation Engine**: `notebooks/03_recommendation_engine.ipynb`

---

## 12. Development Phases & Roadmap

| Phase | Title | Status | Scope |
|---|---|---|---|
| **Phase 1** | **Foundation & Design System** | **Completed** | Modular Python architecture, dataset verification, unit tests, custom editorial CSS, and landing dashboard. |
| **Phase 2** | **Preprocessing & EDA** | **Completed** | Non-destructive cleaning, missing metadata fallback, temporal feature extraction, comprehensive EDA, 14 figures, and 4 functional interactive app views. |
| **Phase 3** | **Correlation & Clustering** | **Completed** | Pearson/Spearman collinearity, feature standardization, K-Means ($K=6$) optimization, 2D PCA, 6 archetype personas, 10 figures, serialized models, and Cluster Intelligence Streamlit view. |
| **Phase 4** | **Recommendation Engine** | **Completed** | Unique catalog (28,356 tracks), cosine NearestNeighbors, 4 recommendation modes, hybrid filters, acoustic explanation engine, diversity reranking, and Recommendations Streamlit view. |
| **Phase 5** | **Production Application** | **Next Phase** | Production deployment, interactive radar charts, batch comparison tools, and performance optimizations. |
