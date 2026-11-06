# SONIC ATLAS — Content-Based Music Recommendation Engine Report
## Phase 4: Acoustic Proximity, Nearest Neighbors & Cross-Genre Discovery

> **Author**: SONIC ATLAS Engineering & Data Science Team  
> **Milestone**: Phase 4 Delivery  
> **Status**: Production-Ready  
> **Catalog Scope**: 28,356 Unique Compositions (`data/processed/spotify_songs_recommendation_catalog.csv`)  
> **Vector Dimension**: 10 Standardized Audio Features  

---

## 1. Architecture Overview

The **SONIC ATLAS Recommendation Engine** delivers high-precision, content-based music recommendations derived purely from empirical acoustic characteristics. Rather than relying on collaborative filtering, commercial streaming popularity biases, or subjective human tagging, the engine computes geometric proximity across 10 standardized audio dimensions:

$$\mathbf{x} = [\text{danceability}, \text{energy}, \text{loudness}, \text{speechiness}, \text{acousticness}, \text{instrumentalness}, \text{liveness}, \text{valence}, \text{tempo}, \text{duration\_ms}]^T$$

Each feature vector is standardized using the pre-fitted Phase 3 `StandardScaler` ($\mu=0, \sigma=1$) to ensure that large-scale attributes (such as tempo in BPM or duration in milliseconds) do not dominate unit-scale attributes (such as danceability or valence in $[0, 1]$).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDATION ENGINE PIPELINE ARCHITECTURE                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Seed Track Query ──► [O(1) Catalog Lookup] ──► Feature Extraction (10 Dims)          │
│                                                          │                             │
│                                                          ▼                             │
│                                            [Phase 3 StandardScaler]                    │
│                                                          │                             │
│                                                          ▼                             │
│                                      [NearestNeighbors (Cosine, Brute)]                │
│                                                          │                             │
│                                                          ▼                             │
│                                           Candidate Pool (250 Tracks)                  │
│                                                          │                             │
│                                      ┌───────────────────┴───────────────────┐         │
│                                      ▼                                       ▼         │
│                            [Recommendation Mode]                    [Hybrid Filters]   │
│                            - Acoustic Similar                       - Target Genre     │
│                            - Acoustic Twins (Cross-Genre)           - Target Subgenre  │
│                            - Same Sonic Archetype                   - Target Cluster   │
│                            - Explore Beyond Cluster                 - Popularity Range │
│                                      │                              - Artist Exclusion │
│                                      └───────────────────┬───────────────────┘         │
│                                                          │                             │
│                                                          ▼                             │
│                                              [Diversity-Aware Reranking]               │
│                                              - Highest Acoustic Similarity             │
│                                              - Balanced Discovery (Max 1/Artist)       │
│                                                          │                             │
│                                                          ▼                             │
│                                            [Acoustic Explanation Engine]               │
│                                            - Directional Delta Comparison              │
│                                            - Audio Feature Tabular Breakdown           │
│                                                          │                             │
│                                                          ▼                             │
│                                            Final Recommendations (Top K)               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Unique-Track Recommendation Catalog

The original Spotify dataset comprises **32,833 playlist observations** across 449 playlists. Because individual tracks recur across playlists, a naive recommendation engine would frequently recommend the same composition under different playlist names.

### Deterministic Aggregation Strategy
To solve this without modifying or discarding the underlying observation dataset, we constructed a dedicated **28,356 unique-track recommendation catalog**:
1. **Audio Feature Invariance**: Audio features, popularity, track name, artist, album, duration, release year, cluster assignment, and PCA coordinates are constant across all entries of the same `track_id`.
2. **Modal Genre Assignment**: Where a track is curated across multiple playlists categorized under different genres (1,687 tracks), its primary `playlist_genre` and `playlist_subgenre` are determined by the statistical mode across its playlist entries, with alphabetical tie-breaking for strict determinism.
3. **Multi-Playlist Telemetry**: Preserved `playlist_count` (number of curated playlists featuring the track) and `associated_genres` (all genres the song appeared in).

---

## 3. Four Core Recommendation Modes

The recommendation engine supports four distinct exploration modes:

### Mode 1: Acoustic Similar Tracks (Default)
- **Objective**: Identify songs with the closest acoustic fingerprint across all 10 audio dimensions.
- **Behavior**: Retrieves top nearest neighbors across the entire 28,356-track catalog without genre or cluster restrictions.
- **Empirical Metric**: Average cosine similarity = **0.9377**.

### Mode 2: Acoustic Twins — Cross-Genre Discovery
- **Objective**: Break out of musical silos by finding songs that sound acoustically identical but belong to a completely different super genre.
- **Behavior**: Automatically filters out the seed track's super genre, returning top nearest neighbors across the remaining 5 genres.
- **Empirical Metric**: Average cosine similarity = **0.9245**; Cross-genre rate = **100.0%**.

### Mode 3: Same Sonic Archetype
- **Objective**: Explore tracks belonging to the same Phase 3 unsupervised K-Means cluster ($K=6$).
- **Behavior**: Restricts candidate retrieval to tracks with the identical cluster ID and archetype persona as the seed track.
- **Empirical Metric**: Average cosine similarity = **0.9338**; Cluster purity = **100.0%**.

### Mode 4: Explore Beyond Your Cluster
- **Objective**: Discover nearby acoustic territories that Phase 3 separated into adjacent clusters.
- **Behavior**: Excludes tracks belonging to the seed track's cluster, encouraging adventurous listening while maintaining acoustic relevance.
- **Empirical Metric**: Average cosine similarity = **0.8677**; Cross-genre rate = **62.65%**.

---

## 4. Hybrid Constraint Filtering

Users can fine-tune recommendations using independent hybrid filters applied post-retrieval:
- **Genre Constraint**: Target a specific super genre (`edm`, `rap`, `pop`, `r&b`, `latin`, `rock`) or search across all genres.
- **Subgenre Constraint**: Dynamically filters across the 24 balanced subgenres based on the selected super genre.
- **Cluster Constraint**: Target any of the 6 Phase 3 sonic archetypes.
- **Popularity Range**: Filter tracks by commercial popularity ($[0, 100]$).
- **Artist Exclusion**: Exclude the seed track's recording artist to force new artist discovery.
- **Result Count**: 5, 10, 15, or 20 recommendations.

---

## 5. Acoustic Explanation Engine

For every recommended track, the engine generates an empirical explanation based on directional feature deltas:
- **Energy & Loudness**: Quantifies whether tracks share matched studio power or differ in mastering dynamic range.
- **Danceability & Tempo**: Analyzes rhythmic synchronization (BPM deltas).
- **Acousticness & Instrumentation**: Notes organic instrumentation vs. electronic production.
- **Vocal Delivery**: Analyzes speechiness deltas (spoken cadence vs. melodic singing).

A structured tabular feature comparison is generated on demand for any selected recommendation, displaying seed value, recommended value, absolute delta, and direction.

---

## 6. Diversity-Aware Reranking

Under **Balanced Discovery**, the engine executes a deterministic greedy reranking algorithm that restricts recommendations to a maximum of one track per recording artist.
- **Artist Diversity**: Improves from **0.9267** to a perfect **1.0000** (100% unique artists).
- **Similarity Impact**: Cosine similarity shifts only marginally from **0.9377** to **0.9365** ($\Delta = 0.0012$), proving that abundant acoustic twins exist across different recording artists.
