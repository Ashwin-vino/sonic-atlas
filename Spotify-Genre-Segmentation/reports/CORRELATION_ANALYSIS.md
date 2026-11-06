# SONIC ATLAS — Correlation Analysis Report
## Phase 3A: Linear (Pearson) and Monotonic (Spearman) Acoustic Relationships

> **Platform**: SONIC ATLAS — Music Intelligence Platform  
> **Dataset**: `data/processed/spotify_songs_cleaned.csv` (32,833 playlist observations)  
> **Features Evaluated**: 10 continuous acoustic dimensions  
> **Discrete Attributes Analyzed**: Key (Pitch classes 0–11) and Mode (Binary: Major/Minor)

---

## 1. Executive Summary

This report delivers a rigorous correlation analysis of the 10 continuous audio measurements extracted from the Spotify song corpus. Pearson ($r$) and Spearman ($\rho$) correlation matrices were calculated to quantify both linear and non-linear monotonic dependencies across acoustic properties.

### Key Takeaways
1. **Strongest Collinearity**: **Energy and Loudness** exhibit a strong positive relationship ($r = +0.677, \rho = +0.656$). High-intensity acoustic energy is tightly coupled with higher master sound levels.
2. **Strongest Negative Tradeoff**: **Energy and Acousticness** exhibit a strong negative relationship ($r = -0.540, \rho = -0.491$). Unamplified acoustic instruments inherently possess lower perceived sound power.
3. **Danceability and Mood Positivity**: **Danceability and Valence** demonstrate a moderate positive correlation ($r = +0.331, \rho = +0.331$), demonstrating that rhythmic groove frequently aligns with brighter, happier compositions.
4. **Independent Dimensions**: **Duration** ($r \in [-0.09, +0.02]$) and **Liveness** ($r \in [-0.05, +0.16]$) operate largely independently from tempo and timbre.

---

## 2. Ranked Correlation Table

Below is the complete ranking of pairwise continuous audio feature relationships sorted by absolute Pearson correlation strength:

| Feature 1 | Feature 2 | Pearson $r$ | Spearman $\rho$ | Strength | Direction | Musical Interpretation |
|---|---|---|---|---|---|---|
| **energy** | **loudness** | **+0.677** | **+0.656** | Strong | Positive (+) | High-energy productions utilize heavier dynamic compression and louder master volume. |
| **energy** | **acousticness** | **-0.540** | **-0.491** | Moderate-Strong | Negative (-) | Acoustic instruments retain natural, unamplified dynamics with lower perceived energy. |
| **loudness** | **acousticness** | **-0.362** | **-0.284** | Moderate | Negative (-) | Acoustic recordings preserve broader dynamic range headroom and lower master loudness. |
| **danceability** | **valence** | **+0.331** | **+0.331** | Moderate | Positive (+) | Rhythmically engaging beats strongly correlate with brighter, happier musical moods. |
| **danceability** | **tempo** | **-0.184** | **-0.168** | Weak | Negative (-) | Very fast tempos (>140 BPM) become harder to dance to, favoring mid-tempo pockets (95–125 BPM). |
| **danceability** | **speechiness** | **+0.182** | **+0.261** | Weak | Positive (+) | Rhythmic vocal flow (rap/trap cadences) is closely paired with danceable percussion grooves. |
| **instrumentalness** | **valence** | **-0.175** | **-0.160** | Weak | Negative (-) | Instrumental electronic and ambient tracks frequently carry darker, introspective moods. |
| **energy** | **liveness** | **+0.161** | **+0.140** | Weak | Positive (+) | Live stadium arena recordings capture elevated crowd excitement and amplified room sound. |
| **energy** | **valence** | **+0.151** | **+0.123** | Weak | Positive (+) | Higher energy tracks lean slightly positive, though aggressive rock/EDM tracks introduce nuance. |
| **energy** | **tempo** | **+0.150** | **+0.177** | Weak | Positive (+) | Faster tempo productions often feature more driving percussive and harmonic energy. |
| **danceability** | **acousticness** | **-0.171** | **-0.198** | Weak | Negative (-) | Acoustic arrangements are less often engineered around syncopated dance rhythms. |
| **loudness** | **instrumentalness** | **-0.142** | **-0.171** | Weak | Negative (-) | Vocals provide high presence that drives commercial master loudness. |
| **acousticness** | **instrumentalness** | **+0.136** | **+0.168** | Weak | Positive (+) | Classical and ambient instrumental pieces feature unamplified acoustic instruments. |
| **danceability** | **loudness** | **+0.119** | **+0.162** | Weak | Positive (+) | Modern club and pop productions master rhythm tracks at high loudness. |
| **speechiness** | **tempo** | **-0.089** | **-0.063** | Negligible | Negative (-) | Rapid vocal flows often sit comfortably over downtempo and mid-tempo trap beats (70–90 / 140 BPM). |
| **duration_ms** | **valence** | **-0.088** | **-0.091** | Negligible | Negative (-) | Extended tracks lean slightly toward darker, complex arrangements. |
| **duration_ms** | **energy** | **-0.014** | **-0.021** | Negligible | Negative (-) | Track duration operates virtually independently of energy and intensity. |

---

## 3. Discrete Musical Attributes Analysis

### A. Key Analysis (Pitch Classes 0–11)
Western chromatic pitch classes are discrete nominal categories. They are not continuous scales and are properly excluded from Pearson linear correlation.

- **Most Common Keys in Catalog**:
  1. Key 1 ($C\sharp / D\flat$): **4,010 tracks (12.21%)** — Heavily favored in contemporary pop, trap, and electronic production due to synthesizer bass tuning.
  2. Key 0 ($C$): **3,454 tracks (10.52%)** — Standard natural major scale anchor.
  3. Key 7 ($G$): **3,352 tracks (10.21%)** — Common rock and guitar-based tuning.
  4. Key 9 ($A$): **3,027 tracks (9.22%)**
  5. Key 11 ($B$): **2,996 tracks (9.13%)**
- **Least Common Key**:
  - Key 3 ($D\sharp / E\flat$): **913 tracks (2.78%)** — Rarely utilized in popular streaming formats.

### B. Mode Analysis (Binary Modality)
- **Catalog Breakdown**:
  - **Major Mode ($mode = 1$)**: **18,574 tracks (56.57%)**
  - **Minor Mode ($mode = 0$)**: **14,259 tracks (43.43%)**
- **Acoustic Nuance**:
  - Mean Valence: Major ($0.511$) vs Minor ($0.510$). While classical music theory associates major keys with cheerfulness and minor keys with sadness, modern commercial pop, rap, and electronic music leverage minor progressions for groove, danceability, and coolness without reducing overall rhythmic valence.

---

## 4. Multicollinearity Implications for Clustering

1. **Loudness & Energy Coupling**:
   With $r = +0.677$, unweighted multi-feature distance calculations (such as Euclidean distance in K-Means) could overweight the intensity dimension twice if both features are included without proper standardization or dimensionality reduction.
2. **Mitigation Strategy in Phase 3**:
   - `StandardScaler` standardizes each feature to $\mu = 0, \sigma = 1$, preventing raw scale dominance (e.g. `duration_ms` in hundreds of thousands vs `danceability` in $[0, 1]$).
   - Principal Component Analysis (PCA) decomposes this collinearity into orthogonal axes: PC1 captures overall Intensity / Power ($energy + loudness - acousticness$), while PC2 captures Groove & Positivity ($danceability + valence$).

---

## 5. Visual Artifacts

The following heatmaps have been rendered at 300 DPI:
- [`reports/figures/correlation_matrix_pearson.png`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/reports/figures/correlation_matrix_pearson.png)
- [`reports/figures/correlation_matrix_spearman.png`](file:///Users/av/Documents/corizo%20Ai%20major%20project%20/Spotify-Genre-Segmentation/reports/figures/correlation_matrix_spearman.png)

