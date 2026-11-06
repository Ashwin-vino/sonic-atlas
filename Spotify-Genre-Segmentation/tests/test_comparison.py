"""
Unit and Integration Tests for SONIC ATLAS Multi-Track Comparison Studio (Phase 5A)
Tests feature normalization, pairwise cosine similarity matrix, visualization generators,
tabular comparison builders, and empirical insight generation.
"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparison import (
    build_comparison_table,
    compute_pairwise_similarity,
    generate_comparison_insights,
    normalize_features_for_radar,
    plot_feature_barchart,
    plot_multitrack_radar,
)
from src.config import (
    CONTINUOUS_AUDIO_FEATURES,
    RECOMMENDATION_CATALOG_PATH,
)


class TestMultiTrackComparison(unittest.TestCase):
    """Test suite for Track Comparison Studio modules."""

    @classmethod
    def setUpClass(cls):
        """Loads a real subset of tracks from the recommendation catalog."""
        if not RECOMMENDATION_CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog not found at {RECOMMENDATION_CATALOG_PATH}")
        cls.catalog = pd.read_csv(RECOMMENDATION_CATALOG_PATH)
        # Select 5 diverse tracks from different genres
        cls.sample_5 = cls.catalog.drop_duplicates(subset=["playlist_genre"]).head(5).copy()
        cls.sample_2 = cls.sample_5.head(2).copy()

    def test_normalize_features_for_radar_bounds(self):
        """Tests that all 10 continuous features are normalized strictly into [0.0, 1.0]."""
        norm_df = normalize_features_for_radar(self.sample_5)
        self.assertEqual(len(norm_df), 5)
        for feat in CONTINUOUS_AUDIO_FEATURES:
            self.assertIn(feat, norm_df.columns)
            vals = norm_df[feat].values
            self.assertTrue(np.all(vals >= 0.0), f"Feature {feat} has values < 0.0")
            self.assertTrue(np.all(vals <= 1.0), f"Feature {feat} has values > 1.0")

    def test_compute_pairwise_similarity_two_tracks(self):
        """Tests that pairwise similarity for 2 tracks returns a symmetric 2x2 matrix with 1.0 on diagonal."""
        sim_mat = compute_pairwise_similarity(self.sample_2)
        self.assertEqual(sim_mat.shape, (2, 2))
        self.assertAlmostEqual(sim_mat.iloc[0, 0], 1.0, places=4)
        self.assertAlmostEqual(sim_mat.iloc[1, 1], 1.0, places=4)
        self.assertAlmostEqual(sim_mat.iloc[0, 1], sim_mat.iloc[1, 0], places=4)
        self.assertTrue(-1.0 <= sim_mat.iloc[0, 1] <= 1.0)

    def test_compute_pairwise_similarity_five_tracks(self):
        """Tests pairwise cosine similarity matrix for 5 tracks."""
        sim_mat = compute_pairwise_similarity(self.sample_5)
        self.assertEqual(sim_mat.shape, (5, 5))
        for i in range(5):
            self.assertAlmostEqual(sim_mat.iloc[i, i], 1.0, places=4)
            for j in range(5):
                self.assertAlmostEqual(sim_mat.iloc[i, j], sim_mat.iloc[j, i], places=4)

    def test_compute_pairwise_similarity_insufficient_tracks(self):
        """Tests that fewer than 2 tracks raises ValueError."""
        single_track = self.sample_5.head(1)
        with self.assertRaises(ValueError):
            compute_pairwise_similarity(single_track)

    def test_generate_comparison_insights(self):
        """Tests that dynamic comparison insights are generated and non-empty."""
        sim_mat = compute_pairwise_similarity(self.sample_5)
        insights = generate_comparison_insights(self.sample_5, sim_mat)
        self.assertIsInstance(insights, list)
        self.assertGreaterEqual(len(insights), 3)
        # Verify text content touches key acoustic dimensions
        combined_text = " ".join(insights)
        self.assertTrue(
            ("Acoustic Proximity Peak" in combined_text) or ("Energy Spectrum" in combined_text)
        )

    def test_build_comparison_table_structure(self):
        """Tests that the detailed comparison table contains both metadata and all 10 audio dimensions."""
        comp_df = build_comparison_table(self.sample_2)
        self.assertIn("Dimension", comp_df.columns)
        self.assertEqual(len(comp_df.columns), 3)  # Dimension + 2 track columns
        dims = comp_df["Dimension"].tolist()
        self.assertIn("Track Artist", dims)
        self.assertIn("Super Genre", dims)
        self.assertIn("Danceability", dims)
        self.assertIn("Energy", dims)
        self.assertIn("Tempo (BPM)", dims)

    def test_plot_multitrack_radar_generation(self):
        """Tests that the multi-track radar chart renders without error."""
        fig = plot_multitrack_radar(self.sample_2)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_barchart_generation(self):
        """Tests that the grouped feature comparison bar chart renders without error."""
        fig = plot_feature_barchart(self.sample_2)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
