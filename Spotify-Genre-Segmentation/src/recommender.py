"""
SONIC ATLAS — Content-Based Music Recommendation Engine Module
Phase 4 Implementation: Acoustic Proximity, Nearest Neighbors, Cross-Genre Discovery,
Hybrid Constraint Filtering, Acoustic Explanations, and Diversity-Aware Reranking.
"""

import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Silence Accelerate/BLAS runtime warnings on macOS arm64
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTERED_DATA_PATH,
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    DEFAULT_CANDIDATE_POOL_SIZE,
    MODELS_DIR,
    NEAREST_NEIGHBORS_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    RECOMMENDATION_CATALOG_PATH,
    RECOMMENDATION_METADATA_PATH,
    RECOMMENDATION_MODES,
    SCALER_PATH,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UNIQUE-TRACK RECOMMENDATION CATALOG GENERATION
# ==============================================================================

def build_recommendation_catalog(
    df_clustered: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Constructs a dedicated unique-track catalog (exactly one record per track_id)
    derived deterministically from the Phase 3 clustered dataset.

    Deterministic Aggregation Rules:
    1. Audio features, popularity, track name, artist, album name, album id,
       duration, release year, cluster, and PCA coordinates are strictly preserved.
    2. Where a track occurs in multiple playlists with differing genre/subgenre labels,
       the primary genre and subgenre are assigned based on the modal occurrence across
       playlists, with alphabetical tie-breaking for perfect reproducibility.
    3. Multi-playlist context is preserved via 'playlist_count' and 'associated_genres'.

    Args:
        df_clustered: Optional input clustered dataframe. If None, loaded from disk.
        save_path: Path where catalog CSV should be written. Defaults to RECOMMENDATION_CATALOG_PATH.

    Returns:
        pd.DataFrame: Validated unique-track recommendation catalog (28,356 rows).
    """
    if df_clustered is None:
        if not CLUSTERED_DATA_PATH.exists():
            raise FileNotFoundError(f"Clustered dataset not found at {CLUSTERED_DATA_PATH}. Run Phase 3 first.")
        df_clustered = pd.read_csv(CLUSTERED_DATA_PATH)

    logger.info("Building unique-track recommendation catalog from %d observations...", len(df_clustered))

    # Deterministic modal assignment for genre and subgenre
    genre_mode = df_clustered.groupby("track_id")["playlist_genre"].agg(
        lambda s: s.mode().sort_values().iloc[0]
    )
    subgenre_mode = df_clustered.groupby("track_id")["playlist_subgenre"].agg(
        lambda s: s.mode().sort_values().iloc[0]
    )
    playlist_count = df_clustered.groupby("track_id")["playlist_id"].nunique()
    associated_genres = df_clustered.groupby("track_id")["playlist_genre"].agg(
        lambda s: ", ".join(sorted(set(s)))
    )

    # First record for standard track attributes (all audio features are identical)
    catalog = df_clustered.drop_duplicates(subset=["track_id"], keep="first").copy()
    catalog["playlist_genre"] = catalog["track_id"].map(genre_mode)
    catalog["playlist_subgenre"] = catalog["track_id"].map(subgenre_mode)
    catalog["playlist_count"] = catalog["track_id"].map(playlist_count)
    catalog["associated_genres"] = catalog["track_id"].map(associated_genres)

    # Sort deterministically by track_id
    catalog = catalog.sort_values(by="track_id", ascending=True).reset_index(drop=True)

    # Validation assertions
    assert catalog["track_id"].nunique() == len(catalog), "Catalog contains duplicate track IDs!"
    for feat in CONTINUOUS_AUDIO_FEATURES:
        assert catalog[feat].isnull().sum() == 0, f"Catalog has missing values in audio feature {feat}!"

    target_path = save_path or RECOMMENDATION_CATALOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(target_path, index=False)
    logger.info("Successfully saved recommendation catalog (%d tracks) to: %s", len(catalog), target_path)

    return catalog


def load_recommendation_catalog(catalog_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the cached recommendation catalog, constructing it if not yet serialized."""
    target_path = catalog_path or RECOMMENDATION_CATALOG_PATH
    if target_path.exists():
        return pd.read_csv(target_path)
    return build_recommendation_catalog(save_path=target_path)


# ==============================================================================
# 2. NEAREST NEIGHBORS MODEL TRAINING & PERSISTENCE
# ==============================================================================

def train_recommendation_index(
    catalog_df: pd.DataFrame,
    scaler: StandardScaler,
    features: List[str] = CONTINUOUS_AUDIO_FEATURES,
    n_neighbors: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> NearestNeighbors:
    """
    Fits a NearestNeighbors model with cosine distance on the standardized audio features.

    Strict Anti-Leakage Rules:
    - Only the 10 continuous audio features are provided to the index.
    - Zero metadata, genre tags, popularity, or cluster IDs are included.
    - Uses the exact pre-fitted Phase 3 StandardScaler to prevent data leakage.

    Args:
        catalog_df: Unique-track recommendation catalog.
        scaler: Pre-fitted StandardScaler from Phase 3.
        features: Continuous audio features list.
        n_neighbors: Number of candidate neighbors to index.

    Returns:
        NearestNeighbors: Fitted model.
    """
    logger.info("Transforming %d catalog tracks using Phase 3 feature scaler...", len(catalog_df))
    X_raw = catalog_df[features].values
    X_scaled = scaler.transform(X_raw)

    logger.info("Fitting NearestNeighbors (metric='cosine', algorithm='brute', n_neighbors=%d)...", n_neighbors)
    nn_model = NearestNeighbors(n_neighbors=min(n_neighbors, len(catalog_df)), metric="cosine", algorithm="brute")
    nn_model.fit(X_scaled)

    return nn_model


def save_recommendation_artifacts(
    nn_model: NearestNeighbors,
    catalog_df: pd.DataFrame,
    metadata: Dict[str, Any],
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> None:
    """Serializes recommendation model and metadata to disk."""
    m_path = model_path or NEAREST_NEIGHBORS_MODEL_PATH
    j_path = meta_path or RECOMMENDATION_METADATA_PATH

    m_path.parent.mkdir(parents=True, exist_ok=True)

    with open(m_path, "wb") as f:
        pickle.dump(nn_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved NearestNeighbors model to: %s", m_path)

    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved recommendation metadata to: %s", j_path)


def load_recommendation_artifacts() -> Tuple[NearestNeighbors, StandardScaler, pd.DataFrame, Dict[str, Any]]:
    """
    Loads all Phase 4 recommendation artifacts from disk.
    If any artifact is missing, builds and saves it automatically.

    Returns:
        Tuple: (NearestNeighbors model, StandardScaler, catalog DataFrame, metadata Dict)
    """
    # 1. Scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Phase 3 scaler missing at {SCALER_PATH}. Run Phase 3 first.")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 2. Catalog
    catalog_df = load_recommendation_catalog()

    # 3. Model
    if NEAREST_NEIGHBORS_MODEL_PATH.exists() and RECOMMENDATION_METADATA_PATH.exists():
        with open(NEAREST_NEIGHBORS_MODEL_PATH, "rb") as f:
            nn_model = pickle.load(f)
        with open(RECOMMENDATION_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        nn_model = train_recommendation_index(catalog_df, scaler)
        metadata = {
            "model_type": "NearestNeighbors",
            "metric": "cosine",
            "features_used": CONTINUOUS_AUDIO_FEATURES,
            "total_catalog_tracks": len(catalog_df),
            "created_at": datetime.now().isoformat(),
        }
        save_recommendation_artifacts(nn_model, catalog_df, metadata)

    return nn_model, scaler, catalog_df, metadata


# ==============================================================================
# 3. ACOUSTIC EXPLANATION & FEATURE COMPARISON ENGINE
# ==============================================================================

def generate_acoustic_explanation(
    seed_track: pd.Series,
    rec_track: pd.Series,
    top_k_reasons: int = 2,
) -> str:
    """
    Generates a natural, grounded, empirical explanation of why a track was recommended
    based on actual feature values and directional differences.

    Args:
        seed_track: pd.Series of the query track.
        rec_track: pd.Series of the candidate recommended track.
        top_k_reasons: Maximum number of descriptive bullet clauses.

    Returns:
        str: Concise acoustic explanation string.
    """
    explanations: List[str] = []

    # 1. Genre transition context
    seed_g = str(seed_track.get("playlist_genre", "")).upper()
    rec_g = str(rec_track.get("playlist_genre", "")).upper()
    is_cross_genre = seed_g != rec_g and seed_g != "" and rec_g != ""

    # 2. Compute absolute and directional deltas
    delta_energy = rec_track["energy"] - seed_track["energy"]
    delta_dance = rec_track["danceability"] - seed_track["danceability"]
    delta_acoustic = rec_track["acousticness"] - seed_track["acousticness"]
    delta_speech = rec_track["speechiness"] - seed_track["speechiness"]
    delta_valence = rec_track["valence"] - seed_track["valence"]
    delta_loudness = rec_track["loudness"] - seed_track["loudness"]
    delta_tempo = rec_track["tempo"] - seed_track["tempo"]

    # 3. Assess alignment
    similar_pairs = []
    if abs(delta_energy) < 0.08:
        similar_pairs.append(f"near-identical energy ({seed_track['energy']:.2f} vs {rec_track['energy']:.2f})")
    if abs(delta_dance) < 0.08:
        similar_pairs.append(f"matched danceability ({seed_track['danceability']:.2f} vs {rec_track['danceability']:.2f})")
    if abs(delta_tempo) < 6.0:
        similar_pairs.append(f"synchronized tempo ({seed_track['tempo']:.0f} vs {rec_track['tempo']:.0f} BPM)")
    if abs(delta_acoustic) < 0.08:
        similar_pairs.append("similar acoustic depth")
    if abs(delta_loudness) < 1.5:
        similar_pairs.append("comparable mastering loudness")

    # 4. Assess distinctive contrasts
    contrasts = []
    if delta_acoustic > 0.15:
        contrasts.append("higher organic acousticness")
    elif delta_acoustic < -0.15:
        contrasts.append("more electronic production")

    if delta_speech > 0.10:
        contrasts.append("higher vocal speechiness")
    elif delta_speech < -0.10:
        contrasts.append("smoother melodic vocal delivery")

    if delta_valence > 0.15:
        contrasts.append("more upbeat musical mood")
    elif delta_valence < -0.15:
        contrasts.append("darker, more melancholic tone")

    # Construct concise synthesis
    if is_cross_genre:
        base = f"Cross-genre match ({seed_g} → {rec_g})"
        if similar_pairs:
            base += f" with {similar_pairs[0]}"
        if contrasts:
            base += f", featuring {contrasts[0]}."
        else:
            base += "."
        return base
    else:
        if similar_pairs and contrasts:
            return f"Shares {similar_pairs[0]}, with {contrasts[0]}."
        elif len(similar_pairs) >= 2:
            return f"Shares {similar_pairs[0]} and {similar_pairs[1]}."
        elif similar_pairs:
            return f"Shares {similar_pairs[0]}."
        elif contrasts:
            return f"Close acoustic proximity with {contrasts[0]}."
        else:
            return "Closely aligned multi-dimensional audio feature vectors."


def compute_feature_differences(
    seed_track: pd.Series,
    rec_track: pd.Series,
) -> pd.DataFrame:
    """
    Computes a structured tabular comparison across all 10 continuous audio features.

    Returns:
        pd.DataFrame: Table with columns [Feature, Seed Track, Recommended, Delta, Direction].
    """
    records = []
    display_names = {
        "danceability": "Danceability",
        "energy": "Energy",
        "loudness": "Loudness (dB)",
        "speechiness": "Speechiness",
        "acousticness": "Acousticness",
        "instrumentalness": "Instrumentalness",
        "liveness": "Liveness",
        "valence": "Valence (Mood)",
        "tempo": "Tempo (BPM)",
        "duration_ms": "Duration (ms)",
    }

    for feat in CONTINUOUS_AUDIO_FEATURES:
        s_val = float(seed_track[feat])
        r_val = float(rec_track[feat])
        delta = r_val - s_val
        direction = "Higher" if delta > 0.001 else ("Lower" if delta < -0.001 else "Equal")

        records.append({
            "Feature": display_names.get(feat, feat.title()),
            "Seed Track": round(s_val, 3),
            "Recommended": round(r_val, 3),
            "Absolute Delta": round(abs(delta), 3),
            "Direction": direction,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 4. DIVERSITY-AWARE RERANKING ENGINE
# ==============================================================================

def rerank_for_diversity(
    candidates_df: pd.DataFrame,
    top_n: int = 10,
    max_per_artist: int = 1,
) -> pd.DataFrame:
    """
    Applies deterministic diversity-aware reranking to prevent artist clustering
    and promote catalog breadth while preserving high acoustic similarity.

    Algorithm:
    1. Iterates down the similarity-ranked candidate pool.
    2. Tracks artist frequencies; selects tracks where artist count < max_per_artist.
    3. If unique artist candidates are exhausted before reaching top_n, relaxes
       the constraint to backfill remaining slots from highest-similarity tracks,
       guaranteeing that top_n recommendations are returned when candidates exist.

    Args:
        candidates_df: Filtered candidate DataFrame sorted by cosine_similarity descending.
        top_n: Desired number of final recommendations.
        max_per_artist: Maximum tracks from the same artist in the top list.

    Returns:
        pd.DataFrame: Reranked DataFrame of top_n tracks.
    """
    if len(candidates_df) <= top_n:
        return candidates_df

    selected_indices = []
    artist_counts: Dict[str, int] = {}
    deferred_indices = []

    for idx, row in candidates_df.iterrows():
        artist = str(row.get("track_artist", "Unknown"))
        curr_count = artist_counts.get(artist, 0)

        if curr_count < max_per_artist:
            selected_indices.append(idx)
            artist_counts[artist] = curr_count + 1
            if len(selected_indices) == top_n:
                break
        else:
            deferred_indices.append(idx)

    # Backfill if insufficient unique artists exist
    if len(selected_indices) < top_n and deferred_indices:
        needed = top_n - len(selected_indices)
        selected_indices.extend(deferred_indices[:needed])

    reranked = candidates_df.loc[selected_indices].copy()
    reranked["rank"] = range(1, len(reranked) + 1)
    return reranked


# ==============================================================================
# 5. CORE SONIC RECOMMENDER CLASS
# ==============================================================================

class SonicRecommender:
    """
    High-performance, content-based recommendation engine powered by cosine
    similarity on standardized 10-dimensional acoustic feature vectors.
    """

    def __init__(
        self,
        catalog_df: Optional[pd.DataFrame] = None,
        scaler: Optional[StandardScaler] = None,
        nn_model: Optional[NearestNeighbors] = None,
    ):
        """Initializes the recommender, loading serialized artifacts if omitted."""
        if catalog_df is None or scaler is None or nn_model is None:
            nn_model, scaler, catalog_df, _ = load_recommendation_artifacts()

        self.catalog_df = catalog_df.reset_index(drop=True)
        self.scaler = scaler
        self.nn_model = nn_model
        self.features = CONTINUOUS_AUDIO_FEATURES

        # Precompute scaled feature matrix for fast lookup and distance math
        self.X_scaled = self.scaler.transform(self.catalog_df[self.features].values)

        # Precompute normalized feature matrix for instant vectorized cosine similarity
        norms = np.linalg.norm(self.X_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.X_norm = self.X_scaled / norms

        # O(1) hash maps for rapid track lookups
        self.track_id_to_idx: Dict[str, int] = {
            tid: i for i, tid in enumerate(self.catalog_df["track_id"].values)
        }
        self.idx_to_track_id: Dict[int, str] = {
            i: tid for i, tid in enumerate(self.catalog_df["track_id"].values)
        }

    def get_track_by_id(self, track_id: str) -> Optional[pd.Series]:
        """O(1) lookup of catalog track record by track_id."""
        idx = self.track_id_to_idx.get(str(track_id).strip())
        if idx is not None:
            return self.catalog_df.iloc[idx]
        return None

    def search_tracks(
        self,
        query: str,
        search_field: str = "both",
        limit: int = 25,
    ) -> pd.DataFrame:
        """
        Searches the 28,356 unique-track catalog by track name and/or artist.
        Results are ranked by commercial popularity to prioritize relevant hits.

        Args:
            query: Substring search text.
            search_field: 'both', 'track_name', or 'track_artist'.
            limit: Maximum search results to return.

        Returns:
            pd.DataFrame: Matching tracks sorted by popularity descending.
        """
        q = str(query).strip().lower()
        if not q:
            return self.catalog_df.sort_values(by="track_popularity", ascending=False).head(limit)

        if search_field == "track_name":
            mask = self.catalog_df["track_name"].str.lower().str.contains(q, na=False, regex=False)
        elif search_field == "track_artist":
            mask = self.catalog_df["track_artist"].str.lower().str.contains(q, na=False, regex=False)
        else:
            mask = (
                self.catalog_df["track_name"].str.lower().str.contains(q, na=False, regex=False)
                | self.catalog_df["track_artist"].str.lower().str.contains(q, na=False, regex=False)
            )

        matches = self.catalog_df[mask].sort_values(by="track_popularity", ascending=False).head(limit)
        return matches

    def recommend(
        self,
        seed_track_id: str,
        mode: str = "acoustic_similar",
        top_n: int = 10,
        target_genre: Optional[str] = None,
        target_subgenre: Optional[str] = None,
        target_cluster: Optional[Union[int, str]] = None,
        min_popularity: int = 0,
        max_popularity: int = 100,
        exclude_seed_artist: bool = False,
        diversity_mode: str = "highest_similarity",
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes high-dimensional cosine similarity search from a seed track.

        Process:
        1. Retrieves seed track vector from catalog.
        2. Queries nearest-neighbor index for candidate pool.
        3. Excludes seed track itself.
        4. Applies Mode constraints (Acoustic Similar, Acoustic Twins, Same Archetype, Explore Beyond).
        5. Applies Hybrid Constraints (Genre, Subgenre, Cluster, Popularity, Artist Exclusion).
        6. Applies Diversity Reranking if enabled.
        7. Attaches empirical acoustic explanations and similarity scores.

        Returns:
            Tuple[pd.DataFrame, Dict]: (Recommended tracks DataFrame, Diagnostics Dict).
        """
        seed_idx = self.track_id_to_idx.get(str(seed_track_id).strip())
        if seed_idx is None:
            raise KeyError(f"Seed track_id '{seed_track_id}' not found in 28,356 recommendation catalog.")

        seed_track = self.catalog_df.iloc[seed_idx]
        seed_genre = str(seed_track["playlist_genre"]).lower()
        seed_cluster = int(seed_track["cluster"])
        seed_artist = str(seed_track["track_artist"])

        # 1. Nearest Neighbor Query
        seed_vector = self.X_scaled[seed_idx : seed_idx + 1]
        n_query = min(candidate_pool_size, len(self.catalog_df))
        distances, indices = self.nn_model.kneighbors(seed_vector, n_neighbors=n_query)

        distances = distances[0]
        indices = indices[0]

        # 2. Build Candidate DataFrame
        candidates = self.catalog_df.iloc[indices].copy()
        candidates["cosine_distance"] = distances
        candidates["cosine_similarity"] = np.clip(1.0 - distances, 0.0, 1.0)

        # 3. Exclude seed track itself
        candidates = candidates[candidates["track_id"] != seed_track_id]
        total_candidates_retrieved = len(candidates)

        # 4. Mode-Specific Filtering
        mode_key = str(mode).lower()
        if mode_key == "acoustic_twins":
            # Cross-Genre Discovery: strictly exclude seed track super genre
            candidates = candidates[candidates["playlist_genre"].str.lower() != seed_genre]
        elif mode_key == "same_archetype":
            # Same Cluster: restrict to seed track cluster
            candidates = candidates[candidates["cluster"] == seed_cluster]
        elif mode_key == "explore_beyond":
            # Explore Beyond: exclude seed track cluster
            candidates = candidates[candidates["cluster"] != seed_cluster]
        # "acoustic_similar" applies no mode-level restriction

        candidates_after_mode = len(candidates)

        # 5. Hybrid Constraint Filtering
        # 5a. Genre constraint
        if target_genre and target_genre.lower() != "all":
            candidates = candidates[candidates["playlist_genre"].str.lower() == target_genre.lower()]

        # 5b. Subgenre constraint
        if target_subgenre and target_subgenre.lower() != "all":
            candidates = candidates[candidates["playlist_subgenre"].str.lower() == target_subgenre.lower()]

        # 5c. Cluster constraint
        if target_cluster is not None and str(target_cluster).lower() != "all":
            try:
                c_int = int(str(target_cluster).split()[0].replace("Cluster", "").strip())
                candidates = candidates[candidates["cluster"] == c_int]
            except ValueError:
                pass

        # 5d. Popularity range
        candidates = candidates[
            (candidates["track_popularity"] >= min_popularity)
            & (candidates["track_popularity"] <= max_popularity)
        ]

        # 5e. Artist exclusion
        if exclude_seed_artist:
            candidates = candidates[candidates["track_artist"].str.lower() != seed_artist.lower()]

        candidates_after_filters = len(candidates)

        # Dynamic Catalog-Wide Candidate Recovery:
        # If strict filtering (e.g. explore_beyond or narrow subgenre/popularity) exhausted the initial
        # nearest neighbor candidate pool, vectorially search across the entire 28,356 catalog in ~0.05ms.
        if candidates_after_filters < top_n:
            mask = self.catalog_df["track_id"] != seed_track_id
            if mode_key == "acoustic_twins":
                mask &= self.catalog_df["playlist_genre"].str.lower() != seed_genre
            elif mode_key == "same_archetype":
                mask &= self.catalog_df["cluster"] == seed_cluster
            elif mode_key == "explore_beyond":
                mask &= self.catalog_df["cluster"] != seed_cluster

            if target_genre and target_genre.lower() != "all":
                mask &= self.catalog_df["playlist_genre"].str.lower() == target_genre.lower()
            if target_subgenre and target_subgenre.lower() != "all":
                mask &= self.catalog_df["playlist_subgenre"].str.lower() == target_subgenre.lower()
            if target_cluster is not None and str(target_cluster).lower() != "all":
                try:
                    c_int = int(str(target_cluster).split()[0].replace("Cluster", "").strip())
                    mask &= self.catalog_df["cluster"] == c_int
                except ValueError:
                    pass
            if min_popularity > 0 or max_popularity < 100:
                mask &= (self.catalog_df["track_popularity"] >= min_popularity) & (
                    self.catalog_df["track_popularity"] <= max_popularity
                )
            if exclude_seed_artist:
                mask &= self.catalog_df["track_artist"].str.lower() != seed_artist.lower()

            candidate_indices = np.where(mask.values)[0]
            if len(candidate_indices) > 0:
                sub_X = self.X_norm[candidate_indices]
                seed_norm_vec = self.X_norm[seed_idx]
                sims = np.dot(sub_X, seed_norm_vec)

                # Get top indices
                top_slice = min(top_n * 3, len(candidate_indices))
                best_rel_idx = np.argsort(-sims)[:top_slice]
                best_abs_idx = candidate_indices[best_rel_idx]

                candidates = self.catalog_df.iloc[best_abs_idx].copy()
                candidates["cosine_similarity"] = np.clip(sims[best_rel_idx], 0.0, 1.0)
                candidates["cosine_distance"] = 1.0 - candidates["cosine_similarity"]
                candidates_after_filters = len(candidates)
                candidates_after_mode = max(candidates_after_mode, candidates_after_filters)

        # Sort strictly by cosine_similarity descending
        candidates = candidates.sort_values(by="cosine_similarity", ascending=False)

        # 6. Diversity Reranking
        if diversity_mode == "balanced_discovery":
            final_df = rerank_for_diversity(candidates, top_n=top_n, max_per_artist=1)
        else:
            final_df = candidates.head(top_n).copy()
            final_df["rank"] = range(1, len(final_df) + 1)

        # 7. Generate Acoustic Explanations
        if not final_df.empty:
            final_df["acoustic_explanation"] = final_df.apply(
                lambda row: generate_acoustic_explanation(seed_track, row), axis=1
            )
        else:
            final_df["acoustic_explanation"] = []

        # 8. Diagnostics Telemetry
        diagnostics = {
            "seed_track_id": seed_track_id,
            "seed_track_name": seed_track["track_name"],
            "seed_track_artist": seed_track["track_artist"],
            "seed_genre": seed_genre,
            "seed_cluster": seed_cluster,
            "seed_cluster_archetype": seed_track.get("cluster_archetype", f"Cluster {seed_cluster}"),
            "mode": mode,
            "diversity_mode": diversity_mode,
            "candidates_retrieved": total_candidates_retrieved,
            "candidates_after_mode": candidates_after_mode,
            "candidates_after_filters": candidates_after_filters,
            "results_returned": len(final_df),
            "mean_similarity": float(final_df["cosine_similarity"].mean()) if not final_df.empty else 0.0,
            "min_similarity": float(final_df["cosine_similarity"].min()) if not final_df.empty else 0.0,
            "max_similarity": float(final_df["cosine_similarity"].max()) if not final_df.empty else 0.0,
            "unique_artists": int(final_df["track_artist"].nunique()) if not final_df.empty else 0,
            "unique_genres": int(final_df["playlist_genre"].nunique()) if not final_df.empty else 0,
        }

        return final_df.reset_index(drop=True), diagnostics


# ==============================================================================
# 6. PIPELINE EXECUTION SCRIPT
# ==============================================================================

def run_phase_4_pipeline() -> Tuple[pd.DataFrame, NearestNeighbors, Dict[str, Any]]:
    """Runs the complete Phase 4 recommendation pipeline and persists artifacts."""
    logger.info("=" * 60)
    logger.info("STARTING PHASE 4 RECOMMENDATION PIPELINE")
    logger.info("=" * 60)

    # 1. Build & save recommendation catalog
    catalog = build_recommendation_catalog()

    # 2. Load Phase 3 scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler at {SCALER_PATH}. Run Phase 3 first.")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 3. Fit Nearest Neighbors index
    nn_model = train_recommendation_index(catalog, scaler)

    # 4. Save metadata
    metadata = {
        "model_name": "NearestNeighbors",
        "metric": "cosine",
        "algorithm": "brute",
        "features_used": CONTINUOUS_AUDIO_FEATURES,
        "n_features": len(CONTINUOUS_AUDIO_FEATURES),
        "total_catalog_tracks": len(catalog),
        "candidate_pool_size": DEFAULT_CANDIDATE_POOL_SIZE,
        "created_at": datetime.now().isoformat(),
    }
    save_recommendation_artifacts(nn_model, catalog, metadata)

    # 5. Sanity test recommendation query
    recommender = SonicRecommender(catalog_df=catalog, scaler=scaler, nn_model=nn_model)
    sample_id = catalog.iloc[0]["track_id"]
    sample_name = catalog.iloc[0]["track_name"]
    sample_artist = catalog.iloc[0]["track_artist"]

    recs, diag = recommender.recommend(sample_id, mode="acoustic_similar", top_n=5)
    twins, diag_twins = recommender.recommend(sample_id, mode="acoustic_twins", top_n=5)

    print("\n" + "=" * 60)
    print(f"SANITY CHECK: Acoustic Similar Tracks for '{sample_name}' by {sample_artist}:")
    print("=" * 60)
    for idx, row in recs.iterrows():
        print(f"  #{row['rank']} [{row['cosine_similarity']:.4f}] {row['track_name']} — {row['track_artist']} ({row['playlist_genre'].upper()})")
        print(f"     Explanation: {row['acoustic_explanation']}")

    print("\n" + "=" * 60)
    print(f"SANITY CHECK: Acoustic Twins (Cross-Genre) for '{sample_name}':")
    print("=" * 60)
    for idx, row in twins.iterrows():
        print(f"  #{row['rank']} [{row['cosine_similarity']:.4f}] {row['track_name']} — {row['track_artist']} ({row['playlist_genre'].upper()})")
        print(f"     Explanation: {row['acoustic_explanation']}")

    return catalog, nn_model, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_phase_4_pipeline()
