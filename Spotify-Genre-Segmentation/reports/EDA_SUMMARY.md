# SONIC ATLAS — Exploratory Data Analysis (EDA) & Preprocessing Summary Report
## Phase 2 Milestone: Comprehensive Audio Feature Audit, Taxonomy Analysis & Data Preprocessing

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Catalog Scope**: 32,833 playlist-associated Spotify track observations  
> **Output Artifact**: `data/processed/spotify_songs_cleaned.csv` (100% rows preserved, 0 missing values)

---

## 1. Empirical Dataset Dimensions & Integrity

All metrics below are derived directly from empirical calculations on the Spotify song corpus:

| Metric Category | Verified Count | Analytical Interpretation |
|---|---|---|
| **Total Song Observations** | **32,833** | Total observations curated across playlist taxonomies |
| **Cleaned Observation Rows** | **32,833** | 100% catalog preservation (0 records deleted) |
| **Unique Compositions (`track_id`)** | **28,356** | Distinct musical recordings in catalog |
| **Multi-Playlist Appearances** | **4,477** | Compositions appearing across multiple playlists/subgenres |
| **Verified Catalog Artists** | **10,692** | Unique recording artists |
| **Curated Playlists** | **449** | Unique playlist titles (across 471 unique `playlist_id` entries) |
| **Super Genres** | **6** | `edm`, `rap`, `pop`, `r&b`, `latin`, `rock` |
| **Granular Subgenres** | **24** | Exactly 4 balanced subgenres per super genre |
| **Total Attributes** | **27** | 23 raw attributes + 4 extracted engineering features |

---

## 2. Preprocessing & Cleaning Summary

### A. Non-Destructive Missing Metadata Handling
- **Observed Missing Values in Raw Data**: Exactly 5 records (0.0152% of dataset) contained `NA` strings in:
  - `track_name` (5 nulls)
  - `track_artist` (5 nulls)
  - `track_album_name` (5 nulls)
  - *All 20 other columns had 0 missing values.*
- **Decision & Method**: In accordance with the Project Foundation rules, these records were **NOT deleted**. Instead:
  - `track_name` received the transparent fallback label `"Unknown Track"`.
  - `track_artist` received the transparent fallback label `"Unknown Artist"`.
  - `track_album_name` received the transparent fallback label `"Unknown Album"`.
  - An audit boolean column `metadata_imputed` was introduced (True for the 5 rows, False for the remaining 32,828 rows) to ensure full provenance.
- **Records Removed**: **0 rows**.

### B. Duplicate Analysis & Observation Preservation
- **Exact Duplicate Rows (23-column match)**: **0 exact duplicate rows**.
- **Repeated Tracks Across Playlists**: **4,477 rows** share track IDs with another record.
- **Decision**: Repeated tracks represent legitimate appearances in distinct curated playlists or subgenres. Deleting repeated tracks would corrupt playlist-level and subgenre context.
- **Dual-Dataset Architecture**:
  1. Primary Cleaned Dataset (`data/processed/spotify_songs_cleaned.csv`): Retains all **32,833 observations**.
  2. Aggregated Unique Tracks View (`create_unique_tracks_view()`): Aggregates to **28,356 distinct compositions** with playlist occurrence counts for composition-level analyses.

### C. Feature Engineering & Type Casting
- **Temporal Extraction**: Extracted `release_year` (Integer) from `track_album_release_date`. 100% of dates successfully parsed (spanning 1957 to 2020).
- **Duration Conversions**: Added `duration_sec` (`duration_ms / 1000.0`) and `duration_min` (`duration_ms / 60000.0`).
- **Feature Typing**: Ensured 10 continuous audio features are strictly `float64`, discrete musical attributes (`key`, `mode`) are integers, and categorical columns are trimmed strings.

---

## 3. Feature Classification & Machine Learning Protection

To prevent data leakage during Phase 3 unsupervised clustering, features are strictly categorized:

| Category | Features Included | Role in ML & Platform Architecture |
|---|---|---|
| **Continuous Audio Features (10)** | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` | Primary inputs for feature scaling, dimensionality reduction, and K-Means clustering. |
| **Discrete Musical Attributes (2)** | `key` (0 to 11 Pitch Class), `mode` (0=Minor, 1=Major) | Harmonic properties; treated separately from continuous variables. |
| **Metadata Numeric Attributes (1)** | `track_popularity` (0 to 100) | Commercial streaming metric; excluded from acoustic clustering. |
| **Prohibited Clustering Columns** | `playlist_genre`, `playlist_subgenre`, `playlist_name`, `playlist_id`, `track_id`, `track_name`, `track_artist`, `track_album_id`, `track_album_name`, `track_album_release_date` | **STRICTLY EXCLUDED** from clustering inputs to avoid data leakage and false discovery claims. Retained for interpretation. |

---

## 4. Key Empirical Findings & Insights

### 1. Catalog Super Genre Distribution & Balance
The catalog exhibits remarkable taxonomic balance across 6 super genres:
- **EDM**: 6,043 tracks (18.41%)
- **Rap**: 5,746 tracks (17.50%)
- **Pop**: 5,507 tracks (16.77%)
- **R&B**: 5,431 tracks (16.54%)
- **Latin**: 5,155 tracks (15.70%)
- **Rock**: 4,951 tracks (15.08%)

### 2. Audio Feature Distribution Profiles
- **Danceability**: Mean = 0.655, Median = 0.672, Std = 0.145. Moderately left-skewed, reflecting streaming curation biased toward rhythmically engaging tracks.
- **Energy**: Mean = 0.699, Median = 0.721, Std = 0.183. Strongly left-skewed; over 50% of the catalog has an energy score > 0.70.
- **Loudness**: Mean = -6.72 dB, Median = -6.17 dB, Min = -46.45 dB, Max = 1.28 dB. Strongly clustered between -9 dB and -4 dB (standard commercial mastering range).
- **Acousticness**: Mean = 0.175, Median = 0.080, Std = 0.220. Highly right-skewed; ~75% of tracks have acousticness < 0.26, indicating predominantly electronic and amplified production.
- **Speechiness**: Mean = 0.107, Median = 0.062, Std = 0.101. Highly right-skewed; rap tracks form an isolated high-speechiness tail (> 0.25).
- **Valence (Positivity)**: Mean = 0.511, Median = 0.512, Std = 0.233. Bell-shaped symmetric distribution spanning the entire spectrum [0.0, 1.0].
- **Tempo**: Mean = 120.88 BPM, Median = 121.98 BPM, Std = 26.91 BPM. Peaks at standard dance/pop tempo anchors (100 BPM, 120-130 BPM).

### 3. Cross-Genre Acoustic Distinctions
- **Highest Energy**: EDM (Mean = 0.802) and Rock (Mean = 0.733).
- **Highest Danceability**: Rap (Mean = 0.718) and Latin (Mean = 0.713).
- **Highest Acousticness**: R&B (Mean = 0.225) and Pop (Mean = 0.210).
- **Highest Speechiness**: Rap (Mean = 0.198) — more than double any other genre.
- **Popularity Leader**: Pop tracks possess the highest median popularity (54), while EDM tracks have the lowest (38), reflecting niche subgenre specialization in EDM.

### 4. Bivariate Acoustic Relationships
- **Energy vs Loudness**: Exhibits a strong positive correlation ($r \approx 0.68$). Louder tracks consistently exhibit higher perceived acoustic energy.
- **Danceability vs Energy**: Demonstrates a mild positive correlation, but Latin and Rap occupy the high-danceability / moderate-energy quadrant, whereas EDM dominates the high-energy / variable-danceability quadrant.
- **Valence vs Energy**: Moderate positive correlation; tracks with higher musical energy tend to feel happier, though minor-key EDM and hard rock tracks present notable exceptions.

### 5. Temporal Evolution (1970 – 2020)
- **Acousticness Decline**: Average acousticness dropped dramatically from ~0.35 in the 1970s to <0.15 by the 2010s, reflecting synthesizer adoption and modern digital audio workstations.
- **Energy Surge**: Mean energy rose from ~0.60 in the 1970s to a sustained peak of ~0.72 across the 2000s and 2010s.

---

## 5. Limitations

1. **Streaming Catalog Bias**: The dataset comprises tracks curated into public Spotify playlists; it does not represent an unbiased sample of all music recorded in human history.
2. **Playlist Granularity**: Track popularity fluctuates dynamically on Spotify; the popularity metric represents a static snapshot at ingestion time.
3. **Collinearity**: The strong correlation between energy and loudness requires dimensionality reduction (e.g., PCA) or standardization in Phase 3 to avoid overweighting loudness-related variance in K-Means clustering.

---

## 6. Generated Visualizations Catalog (`reports/figures/`)

All figures have been saved at 300 DPI following the official editorial design palette:

| Figure Filename | Visual Description & Analytical Question Answered |
|---|---|
| `genre_distribution.png` | Dual-panel bar chart and donut chart illustrating the balance across the 6 ground-truth super genres. |
| `songs_per_genre.png` | Standalone bar chart detailing observation counts per genre. |
| `songs_per_subgenre.png` | Horizontal bar chart of all 24 subgenres, color-coded by super genre. |
| `genre_popularity_distribution.png` | Side-by-side boxplot and mean bar chart analyzing commercial popularity variation per genre. |
| `top_artists.png` | Horizontal bar chart ranking the top 15 most represented artists in the catalog. |
| `artist_popularity_distribution.png` | Scatter plot analyzing observation counts versus average artist popularity. |
| `top_playlists.png` | Horizontal bar chart of the top 12 curated playlists by track count. |
| `audio_feature_distributions.png` | 3x3 histogram grid with Kernel Density Estimates (KDE), means, and medians for 9 continuous audio features. |
| `genre_audio_features_comparison.png` | 2x3 boxplot matrix comparing Danceability, Energy, Valence, Acousticness, Speechiness, and Tempo across genres. |
| `danceability_vs_energy.png` | Stratified scatter plot demonstrating genre acoustic segregation on Danceability vs Energy. |
| `energy_vs_loudness.png` | Stratified scatter plot illustrating collinearity between perceived energy and mastering loudness. |
| `valence_vs_energy.png` | Stratified scatter plot analyzing musical positiveness against perceived intensity. |
| `release_year_trends.png` | Dual-panel time series tracking catalog distribution and acoustic evolution (1970–2020). |
| `missing_values_audit.png` | Data completeness audit chart confirming 99.98% completeness and transparent metadata handling. |

