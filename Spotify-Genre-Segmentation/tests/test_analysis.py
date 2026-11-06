"""
SONIC ATLAS — Unit Tests for Analysis & Visualization Modules (Phase 2)
Tests genre distributions, artist metrics, playlist analysis, audio feature
moments, temporal trends, and figure generation.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis import (
    analyze_audio_features,
    analyze_genre_distribution,
    analyze_temporal_trends,
    analyze_top_artists,
    analyze_top_playlists,
    generate_all_figures,
)
from src.config import FIGURES_DIR
from src.data_loader import load_cleaned_dataset


class TestSonicAtlasAnalysis(unittest.TestCase):
    """Test suite for Phase 2 exploratory data analysis and figures."""

    @classmethod
    def setUpClass(cls):
        """Loads cleaned dataset once for testing."""
        cls.df = load_cleaned_dataset()

    def test_genre_distribution_analysis(self):
        """Verifies genre distribution returns exactly 6 super genres with correct catalog totals."""
        genre_df = analyze_genre_distribution(self.df)
        self.assertEqual(len(genre_df), 6)
        self.assertEqual(int(genre_df["song_count"].sum()), 32833)
        self.assertAlmostEqual(float(genre_df["catalog_percentage"].sum()), 100.0, places=1)
        # Verify 4 subgenres per genre
        self.assertTrue((genre_df["subgenre_count"] == 4).all())

    def test_top_artists_analysis(self):
        """Verifies top artist ranking computation and metric presence."""
        top_artists = analyze_top_artists(self.df, top_n=10)
        self.assertEqual(len(top_artists), 10)
        self.assertIn("song_observations", top_artists.columns)
        self.assertIn("unique_compositions", top_artists.columns)
        self.assertIn("mean_popularity", top_artists.columns)
        # Ensure rankings are strictly descending
        self.assertTrue(top_artists["song_observations"].is_monotonic_decreasing)

    def test_audio_features_analysis(self):
        """Verifies audio feature summary moments computation."""
        audio_stats = analyze_audio_features(self.df)
        self.assertGreaterEqual(len(audio_stats), 11)
        self.assertIn("mean", audio_stats.columns)
        self.assertIn("std", audio_stats.columns)
        self.assertIn("skewness", audio_stats.columns)

    def test_temporal_trends_analysis(self):
        """Verifies release year temporal aggregation."""
        yearly = analyze_temporal_trends(self.df)
        self.assertIn("release_year", yearly.columns)
        self.assertIn("mean_energy", yearly.columns)
        self.assertGreater(len(yearly), 50)

    def test_generated_figures_exist(self):
        """Verifies that all required publication-quality figures exist on disk in reports/figures/."""
        expected_figures = [
            "genre_distribution.png",
            "songs_per_genre.png",
            "songs_per_subgenre.png",
            "genre_popularity_distribution.png",
            "top_artists.png",
            "top_playlists.png",
            "audio_feature_distributions.png",
            "genre_audio_features_comparison.png",
            "danceability_vs_energy.png",
            "energy_vs_loudness.png",
            "valence_vs_energy.png",
            "release_year_trends.png",
            "missing_values_audit.png",
        ]
        for fig_name in expected_figures:
            target = FIGURES_DIR / fig_name
            self.assertTrue(target.is_file(), f"Expected figure {fig_name} missing from {FIGURES_DIR}")
            self.assertGreater(target.stat().st_size, 1000, f"Figure {fig_name} is unexpectedly small.")


if __name__ == "__main__":
    unittest.main()

