"""
Unit and Integration Tests for SONIC ATLAS Playlist Intelligence Module (Phase 5B)
Tests CSV upload validation, catalog matching (Input A vs Input B), profile calculation,
algorithmic identity classification, diversity metrics, playlist recommendations, and export generation.
"""

import io
import json
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CONTINUOUS_AUDIO_FEATURES,
    RECOMMENDATION_CATALOG_PATH,
)
from src.playlist_analyzer import (
    calculate_playlist_diversity,
    calculate_playlist_profile,
    classify_playlist_identity,
    detect_playlist_input_type,
    export_playlist_analysis,
    match_playlist_to_catalog,
    plot_playlist_genre_and_cluster,
    plot_playlist_radar,
    recommend_for_playlist,
    validate_uploaded_csv,
)
from src.recommender import SonicRecommender


class TestPlaylistAnalyzer(unittest.TestCase):
    """Test suite for Playlist Intelligence module."""

    @classmethod
    def setUpClass(cls):
        """Loads catalog and initializes SonicRecommender."""
        if not RECOMMENDATION_CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog not found at {RECOMMENDATION_CATALOG_PATH}")
        cls.catalog = pd.read_csv(RECOMMENDATION_CATALOG_PATH)
        cls.recommender = SonicRecommender(catalog_df=cls.catalog)

        # Create sample matched DataFrame for tests
        cls.sample_tracks = cls.catalog.head(10).copy()

    def test_validate_uploaded_csv_valid(self):
        """Tests that a valid CSV string or bytes parses successfully."""
        csv_text = "track_name,track_artist\nPangarap,I Feel Alive\n"
        is_valid, df, msg = validate_uploaded_csv(csv_text.encode("utf-8"), filename="test.csv")
        self.assertTrue(is_valid)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)

    def test_validate_uploaded_csv_invalid_extension(self):
        """Tests that non-CSV extensions are rejected."""
        is_valid, df, msg = validate_uploaded_csv(b"content", filename="test.txt")
        self.assertFalse(is_valid)
        self.assertIn("Invalid file format", msg)

    def test_validate_uploaded_csv_empty(self):
        """Tests that empty CSV files are rejected."""
        is_valid, df, msg = validate_uploaded_csv(b"", filename="empty.csv")
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_validate_uploaded_csv_missing_schema(self):
        """Tests that CSVs without recognized track identifiers or audio features are rejected."""
        bad_csv = "col_a,col_b\n1,2\n"
        is_valid, df, msg = validate_uploaded_csv(bad_csv.encode("utf-8"), filename="bad.csv")
        self.assertFalse(is_valid)
        self.assertIn("Unsupported CSV schema", msg)

    def test_detect_playlist_input_type(self):
        """Tests differentiation between Input A (Audio Features) and Input B (Tracklist)."""
        input_a = self.sample_tracks[CONTINUOUS_AUDIO_FEATURES].copy()
        self.assertEqual(detect_playlist_input_type(input_a), "INPUT_A_AUDIO_FEATURES")

        input_b = pd.DataFrame({"track_name": ["Song 1"], "track_artist": ["Artist 1"]})
        self.assertEqual(detect_playlist_input_type(input_b), "INPUT_B_TRACK_LIST")

    def test_match_playlist_to_catalog_by_id(self):
        """Tests matching uploaded rows by exact track_id."""
        sample_ids = self.sample_tracks[["track_id"]].copy()
        res = match_playlist_to_catalog(sample_ids, self.catalog)
        self.assertEqual(res["unique_matched_count"], len(sample_ids))
        self.assertEqual(res["unmatched_count"], 0)

    def test_match_playlist_to_catalog_by_title_artist(self):
        """Tests matching uploaded rows by normalized track title and artist name."""
        sample_meta = self.sample_tracks[["track_name", "track_artist"]].copy()
        res = match_playlist_to_catalog(sample_meta, self.catalog)
        self.assertGreaterEqual(res["unique_matched_count"], 1)

    def test_match_playlist_to_catalog_unmatched(self):
        """Tests that unrepresented tracks are transparently classified as unmatched."""
        non_existent = pd.DataFrame({
            "track_name": ["Zzzz Nonexistent Song Alpha 99999"],
            "track_artist": ["Zzzz Unknown Band 99999"],
        })
        res = match_playlist_to_catalog(non_existent, self.catalog)
        self.assertEqual(res["unique_matched_count"], 0)
        self.assertEqual(res["unmatched_count"], 1)

    def test_calculate_playlist_profile(self):
        """Tests calculation of summary statistics and feature distributions for a playlist."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        self.assertEqual(profile["total_tracks"], 10)
        self.assertIn("feature_statistics", profile)
        for feat in CONTINUOUS_AUDIO_FEATURES:
            self.assertIn(feat, profile["feature_statistics"])
            self.assertIn("mean", profile["feature_statistics"][feat])
            self.assertIn("catalog_mean", profile["feature_statistics"][feat])

    def test_classify_playlist_identity(self):
        """Tests algorithmic assignment of rule-based playlist persona."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        identity = classify_playlist_identity(profile)
        self.assertIn("title", identity)
        self.assertIn("badge", identity)
        self.assertIn("summary", identity)
        self.assertTrue(len(identity["title"]) > 0)

    def test_calculate_playlist_diversity(self):
        """Tests mathematical diversity index calculations."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        div = calculate_playlist_diversity(self.sample_tracks, profile)
        self.assertIn("artist_diversity", div)
        self.assertIn("genre_diversity", div)
        self.assertIn("cluster_diversity", div)
        self.assertIn("acoustic_variance", div)
        self.assertTrue(0.0 <= div["artist_diversity"] <= 1.0)
        self.assertTrue(0.0 <= div["genre_diversity"] <= 1.0)

    def test_recommend_for_playlist_exclusion(self):
        """Tests that playlist recommendations strictly exclude tracks already in the playlist."""
        recs = recommend_for_playlist(
            matched_df=self.sample_tracks,
            recommender=self.recommender,
            top_n=5,
            discovery_mode="acoustic_proximity",
        )
        self.assertEqual(len(recs), 5)
        self.assertIn("similarity_score", recs.columns)
        self.assertIn("recommendation_reason", recs.columns)

        # Verify no track in sample_tracks appears in recommendations
        sample_ids = set(self.sample_tracks["track_id"])
        rec_ids = set(recs["track_id"])
        self.assertEqual(len(sample_ids.intersection(rec_ids)), 0)

    def test_export_playlist_analysis_generation(self):
        """Tests CSV and JSON export string generation."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        identity = classify_playlist_identity(profile)
        div = calculate_playlist_diversity(self.sample_tracks, profile)
        exports = export_playlist_analysis(profile, identity, div, self.sample_tracks, pd.DataFrame())

        self.assertIn("summary_csv", exports)
        self.assertIn("matched_csv", exports)
        self.assertIn("profile_json", exports)

        # Verify profile_json parses cleanly
        parsed_json = json.loads(exports["profile_json"])
        self.assertIn("identity", parsed_json)
        self.assertIn("profile", parsed_json)

    def test_plot_playlist_radar_generation(self):
        """Tests rendering of the playlist radar comparison chart."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        fig = plot_playlist_radar(profile)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_playlist_genre_and_cluster_generation(self):
        """Tests rendering of the genre and archetype distribution chart."""
        profile = calculate_playlist_profile(self.sample_tracks, self.catalog)
        fig = plot_playlist_genre_and_cluster(profile)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

