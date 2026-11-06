"""
Unit and Integration Tests for SONIC ATLAS Catalog Intelligence Module (Phase 5C)
Tests macro topological heatmaps, pairwise genre acoustic proximity, artist acoustic profiling,
cluster popularity distributions, and multi-dimensional catalog filtering.
"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.catalog_intelligence import (
    compute_genre_acoustic_profiles,
    compute_genre_cluster_matrix,
    filter_catalog_records,
    get_artist_catalog_profile,
    plot_artist_radar_profile,
    plot_cluster_popularity_distribution,
    plot_genre_cluster_heatmap,
    plot_genre_similarity_heatmap,
)
from src.config import (
    CONTINUOUS_AUDIO_FEATURES,
    GENRE_COLUMNS,
    RECOMMENDATION_CATALOG_PATH,
)


class TestCatalogIntelligence(unittest.TestCase):
    """Test suite for Catalog Intelligence module."""

    @classmethod
    def setUpClass(cls):
        """Loads the unique-track recommendation catalog."""
        if not RECOMMENDATION_CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catalog not found at {RECOMMENDATION_CATALOG_PATH}")
        cls.catalog = pd.read_csv(RECOMMENDATION_CATALOG_PATH)

    def test_compute_genre_cluster_matrix(self):
        """Tests that the genre-to-cluster matrix has 6 rows and 6 columns, with rows summing to ~100%."""
        cross = compute_genre_cluster_matrix(self.catalog, normalize="genre")
        self.assertEqual(len(cross), 6)
        self.assertEqual(len(cross.columns), 6)
        row_sums = cross.sum(axis=1)
        for s in row_sums:
            self.assertAlmostEqual(s, 100.0, delta=0.5)

    def test_compute_genre_acoustic_profiles(self):
        """Tests genre mean audio features and 6x6 pairwise cosine similarity matrix."""
        means, sim_df = compute_genre_acoustic_profiles(self.catalog)
        self.assertEqual(len(means), 6)
        self.assertEqual(len(means.columns), len(CONTINUOUS_AUDIO_FEATURES))
        self.assertEqual(sim_df.shape, (6, 6))
        for g in sim_df.index:
            self.assertAlmostEqual(sim_df.loc[g, g], 1.0, places=4)

    def test_get_artist_catalog_profile_found(self):
        """Tests artist acoustic profiling for an existing artist."""
        profile = get_artist_catalog_profile(self.catalog, "Queen")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["artist_name"], "Queen")
        self.assertGreaterEqual(profile["unique_tracks"], 1)
        self.assertIn("dominant_archetype_name", profile)
        self.assertIn("feature_means", profile)
        for feat in CONTINUOUS_AUDIO_FEATURES:
            self.assertIn(feat, profile["feature_means"])

    def test_get_artist_catalog_profile_not_found(self):
        """Tests that a nonexistent artist query returns None."""
        profile = get_artist_catalog_profile(self.catalog, "ZzzNonexistentArtist99999")
        self.assertIsNone(profile)

    def test_filter_catalog_records(self):
        """Tests multi-dimensional catalog filtering workspace."""
        filtered = filter_catalog_records(
            self.catalog,
            genre="Rock",
            min_energy=0.7,
            max_energy=1.0,
            min_popularity=30,
            max_popularity=100,
        )
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertGreater(len(filtered), 0)
        self.assertTrue(all(filtered["playlist_genre"].str.lower() == "rock"))
        self.assertTrue(all(filtered["energy"] >= 0.7))
        self.assertTrue(all(filtered["track_popularity"] >= 30))

    def test_plot_genre_cluster_heatmap_generation(self):
        """Tests rendering of the genre-to-cluster heatmap."""
        cross = compute_genre_cluster_matrix(self.catalog, normalize="genre")
        fig = plot_genre_cluster_heatmap(cross)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_genre_similarity_heatmap_generation(self):
        """Tests rendering of the pairwise genre proximity heatmap."""
        _, sim_df = compute_genre_acoustic_profiles(self.catalog)
        fig = plot_genre_similarity_heatmap(sim_df)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cluster_popularity_distribution_generation(self):
        """Tests rendering of the cluster popularity boxplot."""
        fig = plot_cluster_popularity_distribution(self.catalog)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_artist_radar_profile_generation(self):
        """Tests rendering of the artist radar profile chart."""
        profile = get_artist_catalog_profile(self.catalog, "Queen")
        self.assertIsNotNone(profile)
        fig = plot_artist_radar_profile(profile, self.catalog)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
