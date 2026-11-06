# SONIC ATLAS — Cluster Profiles & Acoustic Archetypes
## Phase 3G: Exhaustive Acoustic Profiles & Musical Interpretation of Discovered Clusters

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Model**: K-Means ($K=6$) fitted on 10 standardized continuous audio features  
> **Data Scope**: 32,833 playlist observations (28,356 unique tracks)

---

## 1. Overview of Discovered Sonic Archetypes

Unsupervised machine learning partitioned the 32,833 song observations into 6 coherent sonic archetypes based entirely on acoustic properties, without exposure to genre or playlist labels.

| Cluster ID | Archetype Persona Name | Core Sonic Identity | Observations | % Catalog | Unique Tracks | Top Ground-Truth Genres |
|---|---|---|---|---|---|---|
| **Cluster 0** | **High-Octane Anthem** | Driving Rock & Peak-Time EDM | 8,466 | 25.8% | 7,209 | EDM (2,349), Rock (2,226), Pop (1,774) |
| **Cluster 1** | **Instrumental Synth Wave** | Club, Tech & Progressive Electronic | 2,470 | 7.5% | 2,297 | EDM (1,434), Pop (290), Rock (233) |
| **Cluster 2** | **Live Concert & Stadium Arena** | Stage Performance & Audience Room Ambience | 2,128 | 6.5% | 1,863 | EDM (544), Rock (389), Rap (336) |
| **Cluster 3** | **Rhythmic Groove & Vibrant Pop** | Latin Reggaeton, Dance Pop & Upbeat Funk | 10,762 | 32.8% | 9,038 | Latin (2,825), Pop (2,113), R&B (1,990) |
| **Cluster 4** | **Lyrical Flow & Urban Hip-Hop** | Rap, Trap & Spoken Vocal Cadence | 4,327 | 13.2% | 3,780 | Rap (2,296), R&B (858), Latin (589) |
| **Cluster 5** | **Acoustic Intimacy & Mellow Soul** | Organic Instrumentation, Downtempo & Ballads | 4,680 | 14.3% | 4,169 | R&B (1,550), Rock (895), Pop (794) |

---

## 2. Exhaustive Cluster Breakdown

---

### Cluster 0: High-Octane Anthem
- **Subtitle**: *Driving Rock & Peak-Time EDM*
- **Acoustic Persona**: Characterized by maximal master sound levels, extreme driving intensity, and fast tempos. Tracks in this cluster rely on distorted electric guitars, aggressive synthesizer leads, and heavy four-on-the-floor percussion.
- **Population**: **8,466 observations** (25.8% of catalog; **7,209 unique tracks**).
- **Descriptive Popularity**: Median = **45.0** (0–100 scale).

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 0 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | 0.546 | 0.655 | -16.6% (Moderate/Driving) |
| **Energy** | **0.791** | 0.699 | **+13.2% (Very High)** |
| **Loudness (dB)** | **-5.328** | -6.719 | **+20.7% (Loudest Cluster)** |
| **Speechiness** | 0.071 | 0.107 | -33.6% (Standard Vocals) |
| **Acousticness** | 0.068 | 0.175 | -61.1% (Highly Synthetic/Electric) |
| **Instrumentalness** | 0.023 | 0.085 | -72.9% (Vocal Centric) |
| **Liveness** | 0.173 | 0.190 | -8.9% |
| **Valence** | 0.379 | 0.511 | -25.8% (Minor/Aggressive Mood) |
| **Tempo (BPM)** | **132.64** | 120.88 | **+9.7% (Fastest Cluster)** |
| **Duration (ms)** | 227,365 (3.79 min) | 225,800 | +0.7% |

- **Top Subgenres**: `hard rock` (984), `progressive electro house` (778), `pop edm` (623).
- **Representative Artists**: Martin Garrix (90), David Guetta (72), The Chainsmokers (64), Queen (58).

---

### Cluster 1: Instrumental Synth Wave
- **Subtitle**: *Club, Tech & Progressive Electronic*
- **Acoustic Persona**: Characterized by the near-complete absence of lead vocals, massive electronic synthesizer soundscapes, and extended club arrangements.
- **Population**: **2,470 observations** (7.5% of catalog; **2,297 unique tracks**).
- **Descriptive Popularity**: Median = **29.0** (Specialized electronic niche).

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 1 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | 0.663 | 0.655 | +1.2% |
| **Energy** | **0.783** | 0.699 | **+12.0% (High Energy)** |
| **Loudness (dB)** | -6.972 | -6.719 | -3.8% |
| **Speechiness** | 0.071 | 0.107 | -33.6% |
| **Acousticness** | 0.072 | 0.175 | -58.9% (Pure Synthetic) |
| **Instrumentalness** | **0.746** | **0.085** | **+777.6% (Extreme Non-Vocal Outlier)** |
| **Liveness** | 0.171 | 0.190 | -10.0% |
| **Valence** | 0.387 | 0.511 | -24.3% (Introspective/Club Mood) |
| **Tempo (BPM)** | 125.09 | 120.88 | +3.5% (Electronic House Pocket) |
| **Duration (ms)** | **252,107 (4.20 min)**| 225,800 | **+11.6% (Longest Cluster)** |

- **Top Subgenres**: `progressive electro house` (540), `electro house` (519), `big room` (303).
- **Representative Artists**: Dimitri Vegas & Like Mike (39), Martin Garrix (35), Bassjackers (24).

---

### Cluster 2: Live Concert & Stadium Arena
- **Subtitle**: *Stage Performance & Audience Ambience*
- **Acoustic Persona**: Captures recordings performed in front of live audiences with room reverberation, crowd cheering, and dynamic stage acoustics.
- **Population**: **2,128 observations** (6.5% of catalog; **1,863 unique tracks**).
- **Descriptive Popularity**: Median = **41.0**.

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 2 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | 0.610 | 0.655 | -6.9% |
| **Energy** | 0.780 | 0.699 | +11.6% |
| **Loudness (dB)** | -5.976 | -6.719 | +11.1% |
| **Speechiness** | 0.113 | 0.107 | +5.6% |
| **Acousticness** | 0.114 | 0.175 | -34.9% |
| **Instrumentalness** | 0.054 | 0.085 | -36.5% |
| **Liveness** | **0.614** | **0.190** | **+223.2% (Extreme Live Presence)** |
| **Valence** | 0.510 | 0.511 | -0.2% |
| **Tempo (BPM)** | 121.55 | 120.88 | +0.6% |
| **Duration (ms)** | 231,938 (3.87 min) | 225,800 | +2.7% |

- **Top Subgenres**: `progressive electro house` (169), `big room` (155), `album rock` (124).
- **Representative Artists**: Martin Garrix (27), Miguel Rios (25), Soda Stereo (22).

---

### Cluster 3: Rhythmic Groove & Vibrant Pop
- **Subtitle**: *Latin Reggaeton, Dance Pop & Upbeat Funk*
- **Acoustic Persona**: The largest cluster in the catalog, uniting bright melodic hooks, danceable percussive grooves, and high musical positivity across Latin and Pop.
- **Population**: **10,762 observations** (32.8% of catalog; **9,038 unique tracks**).
- **Descriptive Popularity**: Median = **51.0** (Highest commercial presence).

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 3 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | **0.740** | 0.655 | **+13.0% (Highest Danceability)** |
| **Energy** | 0.724 | 0.699 | +3.6% |
| **Loudness (dB)** | -6.212 | -6.719 | +7.5% |
| **Speechiness** | 0.075 | 0.107 | -29.9% |
| **Acousticness** | 0.142 | 0.175 | -18.9% |
| **Instrumentalness** | 0.014 | 0.085 | -83.5% |
| **Liveness** | 0.149 | 0.190 | -21.6% |
| **Valence** | **0.680** | 0.511 | **+33.1% (Brightest/Most Positive)** |
| **Tempo (BPM)** | 113.55 | 120.88 | -6.1% (Golden Groove Pocket) |
| **Duration (ms)** | 220,880 (3.68 min) | 225,800 | -2.2% |

- **Top Subgenres**: `latin hip hop` (896), `new jack swing` (716), `latin pop` (690), `dance pop` (686).
- **Representative Artists**: Don Omar (56), Ed Sheeran (54), Daddy Yankee (53), Ozuna (51).

---

### Cluster 4: Lyrical Flow & Urban Hip-Hop
- **Subtitle**: *Rap, Trap & Spoken Cadence*
- **Acoustic Persona**: Dominated by high vocal delivery rate, rhythmic syncopated speech, and heavy 808 sub-bass foundations characteristic of modern Hip-Hop and Trap.
- **Population**: **4,327 observations** (13.2% of catalog; **3,780 unique tracks**).
- **Descriptive Popularity**: Median = **50.0**.

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 4 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | **0.726** | 0.655 | **+10.8% (Very High)** |
| **Energy** | 0.663 | 0.699 | -5.1% |
| **Loudness (dB)** | -6.821 | -6.719 | -1.5% |
| **Speechiness** | **0.313** | **0.107** | **+192.5% (Highest Speechiness)** |
| **Acousticness** | 0.184 | 0.175 | +5.1% |
| **Instrumentalness** | 0.010 | 0.085 | -88.2% |
| **Liveness** | 0.174 | 0.190 | -8.4% |
| **Valence** | 0.544 | 0.511 | +6.5% |
| **Tempo (BPM)** | 122.59 | 120.88 | +1.4% |
| **Duration (ms)** | **215,457 (3.59 min)**| 225,800 | **-4.6% (Shortest Radio Format)** |

- **Top Subgenres**: `gangster rap` (800), `southern hip hop` (707), `hip hop` (465).
- **Representative Artists**: 50 Cent (49), The Notorious B.I.G. (36), Future (34), Drake (32).

---

### Cluster 5: Acoustic Intimacy & Mellow Soul
- **Subtitle**: *Organic Instrumentation & Downtempo Ballads*
- **Acoustic Persona**: Defined by organic, unamplified musical instruments (piano, acoustic guitar, woodwinds), low perceived dynamic energy, and contemplative, soulful moods.
- **Population**: **4,680 observations** (14.3% of catalog; **4,169 unique tracks**).
- **Descriptive Popularity**: Median = **46.0**.

#### Empirical Acoustic Profile (Original Feature Means)
| Feature Dimension | Cluster 5 Mean | Catalog Mean | Relative Deviation |
|---|---|---|---|
| **Danceability** | 0.605 | 0.655 | -7.6% |
| **Energy** | **0.425** | 0.699 | **-39.2% (Lowest Energy)** |
| **Loudness (dB)** | **-10.515** | -6.719 | **-56.5% (Softest Dynamic Range)** |
| **Speechiness** | 0.073 | 0.107 | -31.8% |
| **Acousticness** | **0.519** | **0.175** | **+196.6% (Highest Acousticness)** |
| **Instrumentalness** | 0.093 | 0.085 | +9.4% |
| **Liveness** | 0.148 | 0.190 | -22.1% |
| **Valence** | **0.393** | 0.511 | **-23.1% (Introspective/Soulful)** |
| **Tempo (BPM)** | **112.37** | 120.88 | **-7.0% (Slowest Tempo)** |
| **Duration (ms)** | 227,168 (3.79 min) | 225,800 | +0.6% |

- **Top Subgenres**: `neo soul` (668), `urban contemporary` (441), `indie poptimism` (419).
- **Representative Artists**: Queen (48), Frank Ocean (31), Billie Eilish (27), Khalid (23).

