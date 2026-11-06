# SONIC ATLAS — Recommendation Quality Evaluation Report
## Phase 4: Content-Based Music Recommendation Engine Benchmarks

> **Document Version**: 1.0.0  
> **Evaluation Date**: November 2026  
> **Evaluation Sample**: 120 Stratified Seed Tracks across 6 Super Genres  
> **Catalog Scope**: 28,356 Unique Compositions (`data/processed/spotify_songs_recommendation_catalog.csv`)  
> **Feature Vector**: 10 Continuous Standardized Audio Dimensions  

---

## 1. Executive Summary

This report documents the empirical evaluation of the **SONIC ATLAS Content-Based Music Recommendation Engine**. Using high-dimensional cosine similarity across standardized acoustic vectors ($\mu=0, \sigma=1$), we systematically benchmarked recommendation quality, acoustic proximity, intra-list cohesion, artist diversity, cross-genre discovery rate, and catalog coverage.

Evaluation was performed on a balanced, stratified test set of **120 seed tracks** (20 per super genre: EDM, Latin, Pop, R&B, Rap, Rock). Each query generated $K=10$ recommendations across four distinct recommendation modes and two diversity-ranking strategies.

---

## 2. Evaluation Metric Definitions

To ensure measurable, mathematically sound evaluation without subjective or ungrounded claims, we formulated seven objective metrics:

1. **Mean Cosine Similarity**: Average cosine similarity ($\text{sim} = 1 - d_{\text{cosine}}$) between the seed track's standardized vector $\mathbf{s}$ and the top-$K$ recommended tracks $\mathbf{r}_i$:
   $$\overline{\text{Sim}} = \frac{1}{K} \sum_{i=1}^K \text{cosine\_similarity}(\mathbf{s}, \mathbf{r}_i)$$

2. **Intra-List Similarity (ILS)**: Average pairwise cosine similarity between all recommended tracks within a recommendation set $R = \{\mathbf{r}_1, \dots, \mathbf{r}_K\}$:
   $$\text{ILS}(R) = \frac{2}{K(K - 1)} \sum_{1 \le i < j \le K} \text{cosine\_similarity}(\mathbf{r}_i, \mathbf{r}_j)$$
   A high ILS indicates tightly cohesive acoustic curation; moderate ILS indicates stylistic breadth without dissonance.

3. **Artist Diversity**: Ratio of unique recording artists to the total number of recommendations:
   $$\text{Div}_{\text{artist}} = \frac{|\{ \text{artist}(\mathbf{r}_i) : i \in [1, K] \}|}{K}$$

4. **Genre Diversity**: Proportion of the 6 core super genres represented in the recommendation list:
   $$\text{Div}_{\text{genre}} = \frac{|\{ \text{genre}(\mathbf{r}_i) : i \in [1, K] \}|}{6}$$

5. **Cluster Diversity**: Proportion of the 6 unsupervised sonic archetypes represented in the recommendation list:
   $$\text{Div}_{\text{cluster}} = \frac{|\{ \text{cluster}(\mathbf{r}_i) : i \in [1, K] \}|}{6}$$

6. **Cross-Genre Discovery Rate**: Percentage of recommended tracks whose catalog genre differs from the seed track's genre:
   $$\text{Rate}_{\text{cross}} = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(\text{genre}(\mathbf{r}_i) \ne \text{genre}(\mathbf{s})) \times 100\%$$

7. **Catalog Coverage**: Proportion of the entire unique-track catalog ($N = 28,356$) reachable across the evaluated test queries:
   $$\text{Coverage} = \frac{|\bigcup_{q=1}^Q R_q|}{N} \times 100\%$$

---

## 3. Empirical Benchmark Results

The table below presents measured results across the 5 evaluation scenarios ($N=120$ seed queries, $K=10$ recommendations per query):

| Recommendation Mode | Diversity Mode | Mean Cosine Sim | Intra-List Sim (ILS) | Artist Diversity | Genre Diversity | Cluster Diversity | Cross-Genre Rate | Unique Tracks |
|---|---|---|---|---|---|---|---|---|
| **Acoustic Similar (Default)** | Highest Similarity | **0.9377** | 0.8977 | 0.9267 | 0.5847 | 0.2472 | **57.33%** | 1,171 |
| **Acoustic Twins (Cross-Genre)** | Highest Similarity | **0.9245** | 0.8760 | 0.9400 | 0.5903 | 0.2570 | **100.00%** | 1,169 |
| **Same Sonic Archetype** | Highest Similarity | **0.9338** | 0.8958 | 0.9300 | 0.5903 | 0.1667 | **57.67%** | 1,170 |
| **Explore Beyond Cluster** | Highest Similarity | **0.8677** | 0.8506 | 0.9531 | 0.5746 | 0.2429 | **62.65%** | 834 |
| **Balanced Discovery** | Diversity Aware | **0.9365** | 0.8946 | **1.0000** | 0.5972 | 0.2500 | **58.08%** | 1,172 |

### Global Catalog Reach
- **Total Unique Tracks Recommended Across 120 Queries**: **2,378 distinct tracks**.
- **Catalog Coverage Percentage**: **8.39%** of the entire 28,356 unique-track corpus reachable from just 120 seed tracks.

---

## 4. Key Analytical Insights

### 1. High Acoustic Fidelity Across All Modes
In default mode, the average cosine similarity of the top-10 recommended tracks is **0.9377**, validating that the 10-dimensional standardized feature space accurately clusters tracks in tight acoustic neighborhoods. Even when forcing the engine into **Cross-Genre Discovery** (Mode 2), the mean cosine similarity remains exceptionally high at **0.9245** (a minor divergence of only $1.3\%$).

### 2. Natural Cross-Genre Convergence (57.33%)
In standard acoustic similarity mode, **57.33%** of recommended tracks belong to a different super genre than the seed track. This is an essential empirical discovery:
- Without any forced filtering, more than half of the acoustically closest songs in the Spotify catalog cross genre boundaries.
- For example, high-intensity Rock tracks naturally match with peak-time EDM festival tracks, and upbeat Latin Pop naturally matches with R&B dance rhythms.
- This proves that catalog genre labels frequently disguise identical underlying musical textures.

### 3. Acoustic Twins Delivers Pure Cross-Genre Discovery (100%)
When the user selects **Acoustic Twins**, the engine strictly excludes the seed genre. The cross-genre rate reaches **100.0%** while preserving **0.9245** average similarity and **0.9400** artist diversity. This provides a genuine vehicle for discovery, helping listeners break out of genre silos without compromising musical feel.

### 4. Zero-Cost Artist Diversity via Balanced Discovery
Under **Balanced Discovery**, the artist diversity metric reaches a perfect **1.0000** (meaning every single recommendation in every list is by a completely distinct artist). 
Crucially, this diversity is achieved with virtually zero penalty to acoustic proximity:
- Highest Similarity: Mean cosine similarity = **0.9377**.
- Balanced Discovery: Mean cosine similarity = **0.9365** (a negligible delta of only $0.0012$).
- This demonstrates that the candidate pool of 250 nearest neighbors contains abundant high-similarity alternatives by different artists.

---

## 5. System Limitations & Scientific Guardrails

1. **Cosine Proximity vs. Subjective Preference**:
   Cosine similarity measures geometric closeness in standardized acoustic feature space. It does **not** represent a probabilistic prediction of user listening preference, cultural affinity, or lyrical resonance.
2. **Audio-Only Representation**:
   Recommendations are generated strictly from the 10 continuous audio features. Vocal timbre, lyrical sentiment, and cultural era are not directly modeled in the Euclidean space.
3. **No External Collaborative Filtering**:
   The engine operates completely without external user listening logs or collaborative filtering matrices, ensuring strict user privacy and zero data leakage.
