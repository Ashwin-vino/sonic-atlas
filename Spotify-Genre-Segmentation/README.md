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

SONIC ATLAS solves this by leveraging acoustic feature vectors, unsupervised machine learning clustering, vector similarity models, geometric multi-track comparison, and playlist acoustic profiling to segment tracks based on raw musical properties.

---

## 3. Project Objectives

1. **Empirical Dataset Audit & Cleaning**: Rigorously validate, clean, and profile 32,833 playlist observations with zero data loss.
2. **Exploratory Acoustic Analysis**: Characterize audio feature distributions, skewness, moments, and cross-genre signatures.
3. **Unsupervised Genre Segmentation**: Discover data-driven sonic clusters using K-Means ($K=6$) and dimensionality reduction (PCA) without human labeling bias (Phase 3).
4. **Content-Based Music Recommendation**: Formulate high-dimensional cosine similarity recommendations allowing users to find "acoustic twins" across catalog boundaries with sub-4ms retrieval latency (Phase 4).
5. **Advanced Music Intelligence & Analytics**: Deliver a Multi-Track Comparison Studio, Playlist Acoustic Profiler, and Catalog Macro Analytics (Phase 5).
6. **Production Deployment & Packaging**: Containerize with Docker, enforce zero external secret dependencies, and provide comprehensive evaluation decks and architectural documentation (Phase 5).

---

## 4. Technology Stack

- **Core Language**: Python 3.9+
- **Data Manipulation & Analytics**: Pandas (`>=2.0.0`), NumPy (`>=1.24.0`)
- **Machine Learning & Modeling**: Scikit-Learn (`>=1.3.0`), SciPy (`>=1.11.0`)
- **Visualization Engines**: Matplotlib (`>=3.8.0`), Seaborn (`>=0.13.0`), Altair (`>=5.0.0`)
- **Application Framework**: Streamlit (`>=1.30.0`) with Custom Editorial CSS Tokens
- **Presentation Deck Generation**: Python-pptx (`>=1.0.0`)
- **Containerization & Deployment**: Docker (Python 3.9-slim), Docker Compose
- **Design System**: Custom tokens incorporating Manrope (Headings), Inter (Body), and JetBrains Mono (Technical telemetry)
- **Development & Testing**: Python Unittest (76 automated tests), Jupyter Notebooks, Git

---

## 5. Project Folder Structure

```text
SONIC-ATLAS/
│
├── data/
│   ├── raw/
│   │   └── spotify_songs.csv           # Original dataset (32,833 records)
│   │
│   ├── processed/
│   │   ├── spotify_songs_cleaned.csv   # Cleaned dataset (32,833 observations, 0 nulls)
│   │   ├── spotify_songs_clustered.csv # Clustered dataset with K=6 labels & 2D PCA
│   │   └── spotify_songs_recommendation_catalog.csv # Unique-track catalog (28,356 compositions)
│   │
│   └── sample_playlists/               # Curated sample playlists for demonstration
│       ├── sample_festival_edm.csv     # High-energy EDM festival sample
│       ├── sample_acoustic_chill.csv   # Organic acoustic chill sample
│       └── sample_tracklist_simple.csv # Simple title + artist tracklist sample
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
│   ├── correlation.py                  # Pearson/Spearman collinearity & discrete analysis
│   ├── clustering.py                   # StandardScaler, K-Means (K=6), 2D PCA & profiling
│   ├── evaluation.py                   # Clustering & recommendation benchmark metrics
│   ├── recommender.py                  # Cosine NearestNeighbors recommendation engine
│   ├── comparison.py                   # (Phase 5A) Multi-track radar & pairwise similarity
│   ├── playlist_analyzer.py            # (Phase 5B) CSV upload, catalog matcher & profiling
│   ├── catalog_intelligence.py         # (Phase 5C) Macro heatmaps, artist profiler & filtering
│   └── generate_presentation.py        # (Phase 5F) Programmatic 18-slide PowerPoint generator
│
├── models/                             # Serialized ML Artifacts
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
│   ├── SYSTEM_ARCHITECTURE.md          # (Phase 5G) Complete 5-layer architectural blueprint
│   ├── PROJECT_DEMONSTRATION_GUIDE.md  # (Phase 5F) Step-by-step evaluator presentation guide
│   ├── SONIC_ATLAS_FINAL_PRESENTATION.pptx # (Phase 5F) Executive 18-slide dark deck
│   ├── PHASE_5_COMPLETION_REPORT.md    # (Phase 5I) Formal Phase 5 completion report
│   ├── SONIC_ATLAS_FINAL_PROJECT_REPORT.md # (Phase 5I) Comprehensive 5-phase project report
│   └── figures/                        # 24 publication-quality 300 DPI visualizations
│
├── tests/
│   ├── __init__.py
│   ├── test_foundation.py              # Phase 1 foundation test suite
│   ├── test_preprocessing.py           # Phase 2 cleaning & pipeline tests
│   ├── test_analysis.py                # Phase 2 analytics & figures tests
│   ├── test_correlation.py             # Phase 3 correlation & ranking tests
│   ├── test_clustering.py              # Phase 3 K-Means, scaling, PCA & serialization tests
│   ├── test_recommender.py             # Phase 4 recommendation modes, filters & evaluation tests
│   ├── test_comparison.py              # (Phase 5H) Multi-track radar & pairwise similarity tests
│   ├── test_playlist_analyzer.py       # (Phase 5H) Playlist upload, matching & profiling tests
│   └── test_catalog_intelligence.py    # (Phase 5H) Heatmaps, artist profiler & filtering tests
│
├── .streamlit/
│   └── config.toml                     # Streamlit server config, max upload size & dark theme
├── app.py                              # Unified Streamlit application (9 live production views)
├── requirements.txt                    # Pinned production dependencies
├── Dockerfile                          # Multi-stage production container build
├── docker-compose.yml                  # Local container orchestration
├── .dockerignore                       # Container build exclusion rules
├── README.md                           # Documentation & platform guide
└── .gitignore                          # Git exclusion rules
```

---

## 6. Preprocessing Methodology & Data Integrity Rules

The preprocessing pipeline ([`src/preprocessing.py`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/src/preprocessing.py)) executes strict, non-destructive transformations:

1. **Preserving Playlist Observations**:
   - The 32,833 rows represent playlist-associated song observations across 449 playlists.
   - Repeated track IDs (4,477 occurrences) across different playlists are **preserved**, maintaining vital playlist context.
   - Cleaned observation rows: **32,833 (100% catalog retention)**.
   - Cleaned unique track catalog: **28,356 unique compositions** deduplicated deterministically.

2. **Non-Destructive Metadata Fallbacks**:
   - Exactly 5 records contained `NA` strings in `track_name`, `track_artist`, and `track_album_name`.
   - Records were **not deleted**. They received transparent display fallbacks (`"Unknown Track"`, `"Unknown Artist"`, `"Unknown Album"`).

3. **Audio Feature Classification & Anti-Leakage Protection**:
   - **10 Continuous Features**: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.
   - **2 Discrete Musical Features**: `key` (0 to 11), `mode` (0 or 1).
   - **Popularity Measure**: `track_popularity` is treated as a commercial metadata metric, not an acoustic feature.
   - **Prohibited Clustering Inputs**: Ground-truth labels and identifiers are strictly isolated to prevent circular data leakage.

---

## 7. Machine Learning & The Six Sonic Archetypes

Unsupervised K-Means clustering ($K=6$, `random_state=42`) trained on the standardized 10-dimensional continuous space discovered six distinct musical personas:

1. **Cluster 0: High-Octane Anthem** (8,466 tracks, 25.8%, `#55D99A`) — Driving Rock & Festival EDM. Loudest (-5.33 dB), fastest (132.6 BPM), energy (0.791).
2. **Cluster 1: Instrumental Synth Wave** (2,470 tracks, 7.5%, `#5BC0BE`) — Club & Progressive Electronic. Instrumentalness (0.746), extended duration (252.1s).
3. **Cluster 2: Live Concert & Stadium Arena** (2,128 tracks, 6.5%, `#E9B86C`) — Stage Performance. Elevated liveness (0.614), energy (0.780).
4. **Cluster 3: Rhythmic Groove & Vibrant Pop** (6,778 tracks, 20.6%, `#D97755`) — Dance Pop & Latin Reggaeton. Danceability (0.740), valence (0.680).
5. **Cluster 4: Lyrical Flow & Urban Hip-Hop** (5,765 tracks, 17.6%, `#708A81`) — Rap, Trap & Spoken Cadence. Speechiness (0.313).
6. **Cluster 5: Acoustic Intimacy & Mellow Soul** (7,226 tracks, 22.0%, `#9AA79F`) — Organic Folk & Downtempo Ballads. Acousticness (0.519), soft loudness (-10.5 dB).

---

## 8. Content-Based Recommendation Engine

- **Model**: `NearestNeighbors(metric='cosine', algorithm='brute')` fitted on 28,356 unique compositions.
- **Latency**: Sub-4ms candidate querying; <0.05ms precomputed normalized vector fallback.
- **Four Core Recommendation Modes**:
  1. *Acoustic Similar Tracks* (Default, $\overline{\text{Sim}} = 0.9377$, 57.3% natural cross-genre matches).
  2. *Acoustic Twins — Cross-Genre Discovery* (Excludes seed genre, $\overline{\text{Sim}} = 0.9245$, 100.0% cross-genre rate).
  3. *Same Sonic Archetype* (Pins to seed cluster, $\overline{\text{Sim}} = 0.9338$, 100% cluster purity).
  4. *Explore Beyond Your Cluster* (Excludes seed cluster, $\overline{\text{Sim}} = 0.8677$, 62.6% cross-genre rate).
- **Diversity-Aware Reranking**: *Balanced Discovery* mode achieves **100% unique artists** with only a negligible $0.0012$ similarity penalty.

---

## 9. Advanced Music Intelligence (Phase 5 Modules)

### A. Track Comparison Studio (`src/comparison.py`)
- Allows users to select between 2 and 5 tracks from the 28,356 unique catalog.
- Renders multi-track polar radar charts across normalized continuous audio dimensions.
- Computes exact pairwise cosine similarity matrix in standardized 10D space.
- Generates grouped feature comparison bar charts in raw, interpretable measurements (dB, BPM, ms).
- Produces dynamic, purely empirical data-grounded comparative insights.

### B. Playlist Intelligence (`src/playlist_analyzer.py`)
- Supports dual CSV upload: **Input A** (dataset-compatible with audio features) and **Input B** (tracklists with track names and artists).
- Strictly validates file size ($\le 10$ MB), encoding, readability, and schema in isolated memory.
- Matches uploaded tracks against the local 28,356 catalog with zero external API calls.
- Classifies empirical playlist identities (e.g. *High-Octane Mainstage*, *Dancefloor Rhythm & Groove*, *Acoustic Intimacy*).
- Calculates mathematical diversity metrics: Artist Diversity, Genre Diversity (Simpson), Cluster Entropy, and Acoustic Variance.
- *Discover Tracks for This Playlist*: Recommends fresh tracks matching the playlist centroid while strictly excluding existing songs.
- 1-click export of Summary CSV, Matched Tracks CSV, Unmatched CSV, and Profile JSON.

### C. Catalog Intelligence (`src/catalog_intelligence.py`)
- Super Genre to Sonic Archetype topological mapping heatmap (6 Genres $\times$ 6 Clusters).
- Pairwise genre acoustic proximity heatmap.
- Cluster popularity boxplot distribution.
- Recording Artist Acoustic Profiler: Search any of the 10,692 artists to inspect unique tracks, playlist appearances, dominant archetype, and radar comparison against catalog baseline.
- Multi-dimensional catalog filtering workspace with CSV export.

---

## 10. Unified Streamlit Application (9 Production Views)

Run `streamlit run app.py` to access the full platform:

1. **Overview**: Executive dashboard, 5-phase status badges, dataset KPI cards, and platform architecture scorecard.
2. **Music Explorer**: Interactive 10D feature distributions, KDE histograms, box plots, and scatter plots.
3. **Genre Analysis**: Super-genre acoustic radar comparisons, genre distributions, and 24-subgenre taxonomy breakdown.
4. **Data Insights**: Temporal release date trends, top artist distributions, and curated playlist composition.
5. **Cluster Intelligence**: The Six Discovered Sonic Archetypes, interactive 2D/3D PCA projection scatter, and radar profiles.
6. **Recommendations**: Content-based recommendation studio with search autocomplete, 4 recommendation modes, hybrid filters, and directional explanations.
7. **Track Comparison**: Multi-track comparison studio (2–5 songs), polar radar geometry, pairwise cosine similarity matrix, and data-grounded insights.
8. **Playlist Intelligence**: Playlist CSV upload validation (Input A vs Input B), local catalog matching, empirical persona classifier, radar vs baseline, and playlist-level track recommendations.
9. **Catalog Intelligence**: Macro genre-to-cluster topological heatmaps, pairwise genre acoustic proximity, artist acoustic inspector, and multi-dimensional catalog filtering workspace.

---

## 11. Installation & Production Setup

### 1. Local Virtual Environment Setup
```bash
# Clone repository
git clone https://github.com/example/spotify-genre-segmentation.git
cd spotify-genre-segmentation

# Activate virtual environment
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Complete Automated Test Suite (76 Tests)
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Expected Output: `Ran 76 tests in ~3.2s — OK` (100% passing).*

### 3. Launch Local Streamlit Platform
```bash
streamlit run app.py
```
*Platform opens automatically at `http://localhost:8501`.*

### 4. Run via Docker Containerization
```bash
# Build Docker image
docker build -t sonic-atlas:latest .

# Run container
docker run -p 8501:8501 --name sonic_atlas_app sonic-atlas:latest

# Or launch via Docker Compose
docker compose up -d
```
*Verify container health: `curl -f http://localhost:8501/_stcore/health`.*

---

## 12. Development Phases & Roadmap

| Phase | Title | Status | Scope |
|---|---|---|---|
| **Phase 1** | **Foundation & Design System** | **Completed** | Modular Python architecture, dataset verification, unit tests, custom editorial CSS, and landing dashboard. |
| **Phase 2** | **Preprocessing & EDA** | **Completed** | Non-destructive cleaning, missing metadata fallback, temporal feature extraction, comprehensive EDA, 14 figures, and 4 functional interactive app views. |
| **Phase 3** | **Correlation & Clustering** | **Completed** | Pearson/Spearman collinearity, feature standardization, K-Means ($K=6$) optimization, 2D PCA, 6 archetype personas, 10 figures, serialized models, and Cluster Intelligence Streamlit view. |
| **Phase 4** | **Recommendation Engine** | **Completed** | Unique catalog (28,356 tracks), cosine NearestNeighbors, 4 recommendation modes, hybrid filters, acoustic explanation engine, diversity reranking, and Recommendations Streamlit view. |
| **Phase 5** | **Advanced Intelligence & Delivery** | **Completed** | Multi-Track Comparison Studio, Playlist Intelligence, Catalog Intelligence, Docker containerization, 18-slide PowerPoint deck, 76 unit tests, and production release. |

---

## 13. Project Deliverables Index

- **Final Presentation Deck**: [`reports/SONIC_ATLAS_FINAL_PRESENTATION.pptx`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/reports/SONIC_ATLAS_FINAL_PRESENTATION.pptx)
- **Demonstration Guide**: [`reports/PROJECT_DEMONSTRATION_GUIDE.md`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/reports/PROJECT_DEMONSTRATION_GUIDE.md)
- **System Architecture**: [`reports/SYSTEM_ARCHITECTURE.md`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/reports/SYSTEM_ARCHITECTURE.md)
- **Phase 5 Completion Report**: [`reports/PHASE_5_COMPLETION_REPORT.md`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/reports/PHASE_5_COMPLETION_REPORT.md)
- **Final Project Report**: [`reports/SONIC_ATLAS_FINAL_PROJECT_REPORT.md`](file:///Users/av/Documents/corizo%20Ai%20major%20project/Spotify-Genre-Segmentation/reports/SONIC_ATLAS_FINAL_PROJECT_REPORT.md)
