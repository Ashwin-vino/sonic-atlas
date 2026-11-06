"""
SONIC ATLAS — Unit & Integration Tests for Recommendation Engine
Phase 4 Implementation: Unique Catalog, Nearest Neighbors, Recommendation Modes,
Hybrid Filtering, Acoustic Explanations, Diversity Reranking, and Evaluation Metrics.
"""

import unittest
import warnings
from pathlib import Path

# Silence Accelerate/BLAS runtime warnings on macOS arm64
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTERED_DATA_PATH,
    CONTINUOUS_AUDIO_FEATURES,
    RECOMMENDATION_CATALOG_PATH,
)
from src.evaluation import (
    compute_intra_list_similarity,
    compute_recommendation_list_metrics,
)
from src.recommender import (
    SonicRecommender,
    build_recommendation_catalog,
    compute_feature_differences,
    generate_acoustic_explanation,
    load_recommendation_catalog,
    rerank_for_diversity,
    train_recommendation_index,
)


class TestRecommendationCatalog(unittest.TestCase):
    """Tests catalog creation, deduplication, and metadata integrity."""

    @classmethod
    def setUpClass(cls):
        """Loads or builds catalog once for test suite."""
        cls.catalog = load_recommendation_catalog()

    def test_catalog_dimensions_and_uniqueness(self):
        """Validates catalog contains 28,356 tracks with zero duplicate track IDs."""
        self.assertEqual(len(self.catalog), 28356)
        self.assertEqual(self.catalog["track_id"].nunique(), 28356)

    def test_no_missing_audio_features(self):
        """Validates all 10 continuous audio features have zero nulls."""
        for feat in CONTINUOUS_AUDIO_FEATURES:
            self.assertIn(feat, self.catalog.columns)
            self.assertEqual(self.catalog[feat].isnull().sum(), 0, f"Nulls found in {feat}")

    def test_metadata_preservation(self):
        """Validates essential metadata columns are present in catalog."""
        expected_cols = [
            "track_id", "track_name", "track_artist", "track_popularity",
            "playlist_genre", "playlist_subgenre", "cluster", "cluster_archetype",
            "pca_1", "pca_2", "playlist_count", "associated_genres"
        ]
        for col in expected_cols:
            self.assertIn(col, self.catalog.columns)


class TestSonicRecommender(unittest.TestCase):
    """Tests recommendation retrieval, modes, filters, explanations, and diversity."""

    @classmethod
    def setUpClass(cls):
        """Initializes recommender once for test suite."""
        cls.recommender = SonicRecommender()
        cls.catalog = cls.recommender.catalog_df
        # Pick a well-populated test seed track
        cls.seed = cls.catalog.iloc[100]
        cls.seed_id = str(cls.seed["track_id"])
        cls.seed_genre = str(cls.seed["playlist_genre"]).lower()
        cls.seed_cluster = int(cls.seed["cluster"])
        cls.seed_artist = str(cls.seed["track_artist"])

    def test_track_search(self):
        """Validates fast substring search across track and artist."""
        results = self.recommender.search_tracks(self.seed["track_artist"][:4], limit=10)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)

    def test_mode_1_acoustic_similar(self):
        """Validates default acoustic recommendations return top similarity without seed."""
        recs, diag = self.recommender.recommend(self.seed_id, mode="acoustic_similar", top_n=10)
        self.assertEqual(len(recs), 10)
        # Seed track must never be in recommendations
        self.assertNotIn(self.seed_id, recs["track_id"].values)
        # Verify similarity is sorted descending
        sims = recs["cosine_similarity"].values
        self.assertTrue(all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1)))
        # Verify cosine similarity is bounded [0, 1]
        self.assertTrue(all(0.0 <= s <= 1.0 for s in sims))

    def test_mode_2_acoustic_twins_cross_genre(self):
        """Validates Acoustic Twins mode strictly excludes seed track super genre."""
        recs, diag = self.recommender.recommend(self.seed_id, mode="acoustic_twins", top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertNotIn(self.seed_id, recs["track_id"].values)
        # Every returned track must belong to a different super genre
        for genre in recs["playlist_genre"].str.lower():
            self.assertNotEqual(genre, self.seed_genre)

    def test_mode_3_same_archetype(self):
        """Validates Same Archetype mode restricts to seed track cluster."""
        recs, diag = self.recommender.recommend(self.seed_id, mode="same_archetype", top_n=10)
        self.assertEqual(len(recs), 10)
        for c in recs["cluster"]:
            self.assertEqual(int(c), self.seed_cluster)

    def test_mode_4_explore_beyond_cluster(self):
        """Validates Explore Beyond mode excludes seed track cluster."""
        recs, diag = self.recommender.recommend(self.seed_id, mode="explore_beyond", top_n=10)
        self.assertEqual(len(recs), 10)
        for c in recs["cluster"]:
            self.assertNotEqual(int(c), self.seed_cluster)

    def test_hybrid_filters(self):
        """Validates hybrid genre, cluster, popularity, and artist exclusion filters."""
        # Target genre filter
        target_g = "pop" if self.seed_genre != "pop" else "rock"
        recs_g, _ = self.recommender.recommend(
            self.seed_id, target_genre=target_g, top_n=5
        )
        for g in recs_g["playlist_genre"].str.lower():
            self.assertEqual(g, target_g)

        # Popularity filter
        recs_pop, _ = self.recommender.recommend(
            self.seed_id, min_popularity=40, max_popularity=80, top_n=5
        )
        for p in recs_pop["track_popularity"]:
            self.assertTrue(40 <= p <= 80)

        # Artist exclusion filter
        recs_art, _ = self.recommender.recommend(
            self.seed_id, exclude_seed_artist=True, top_n=5
        )
        for art in recs_art["track_artist"].str.lower():
            self.assertNotEqual(art, self.seed_artist.lower())

    def test_acoustic_explanation_engine(self):
        """Validates generated explanations and feature differences."""
        recs, _ = self.recommender.recommend(self.seed_id, top_n=2)
        rec_track = recs.iloc[0]

        # Explanation generation
        explanation = generate_acoustic_explanation(self.seed, rec_track)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)

        # Feature differences table
        diff_df = compute_feature_differences(self.seed, rec_track)
        self.assertEqual(len(diff_df), 10)
        self.assertIn("Feature", diff_df.columns)
        self.assertIn("Absolute Delta", diff_df.columns)
        self.assertIn("Direction", diff_df.columns)

    def test_diversity_aware_reranking(self):
        """Validates balanced discovery guarantees unique artists in top list."""
        recs_sim, _ = self.recommender.recommend(
            self.seed_id, top_n=10, diversity_mode="highest_similarity"
        )
        recs_div, _ = self.recommender.recommend(
            self.seed_id, top_n=10, diversity_mode="balanced_discovery"
        )

        self.assertEqual(len(recs_div), 10)
        # In balanced discovery, unique artists must equal 10
        self.assertEqual(recs_div["track_artist"].nunique(), 10)
        # Reranked result is deterministic
        recs_div_2, _ = self.recommender.recommend(
            self.seed_id, top_n=10, diversity_mode="balanced_discovery"
        )
        self.assertEqual(
            recs_div["track_id"].tolist(), recs_div_2["track_id"].tolist()
        )

    def test_evaluation_metrics_helpers(self):
        """Validates intra-list similarity and list metrics helper functions."""
        recs, _ = self.recommender.recommend(self.seed_id, top_n=5)
        ils = compute_intra_list_similarity(recs, scaler=self.recommender.scaler)
        self.assertTrue(0.0 <= ils <= 1.0)

        metrics = compute_recommendation_list_metrics(
            recs, seed_genre=self.seed_genre, seed_cluster=self.seed_cluster, scaler=self.recommender.scaler
        )
        self.assertIn("mean_similarity", metrics)
        self.assertIn("intra_list_similarity", metrics)
        self.assertIn("artist_diversity", metrics)
        self.assertIn("cross_genre_rate", metrics)
        self.assertTrue(0.0 <= metrics["mean_similarity"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
