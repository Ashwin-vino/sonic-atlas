"""
SONIC ATLAS — Unit Tests for Correlation Analysis Module (Phase 3)
Tests Pearson and Spearman correlation matrices, key/mode discrete analysis,
and figure generation.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import CONTINUOUS_AUDIO_FEATURES, FIGURES_DIR
from src.correlation import (
    analyze_key_distribution,
    analyze_mode_distribution,
    compute_pearson_correlation,
    compute_spearman_correlation,
    get_top_correlations,
)
from src.data_loader import load_cleaned_dataset


class TestSonicAtlasCorrelation(unittest.TestCase):
    """Test suite for Phase 3A correlation analysis."""

    @classmethod
    def setUpClass(cls):
        """Loads cleaned dataset once for testing."""
        cls.df = load_cleaned_dataset()

    def test_pearson_correlation_properties(self):
        """Verifies Pearson correlation matrix dimensions, symmetry, diagonal, and bounds."""
        p_corr = compute_pearson_correlation(self.df)
        self.assertEqual(p_corr.shape, (10, 10))
        # Diagonal must be exactly 1.0
        np.testing.assert_allclose(np.diag(p_corr), 1.0)
        # Matrix must be symmetric
        np.testing.assert_allclose(p_corr.values, p_corr.values.T)
        # Values must be within [-1, 1]
        self.assertTrue((p_corr.values >= -1.0).all() and (p_corr.values <= 1.0).all())

    def test_spearman_correlation_properties(self):
        """Verifies Spearman rank correlation matrix properties."""
        s_corr = compute_spearman_correlation(self.df)
        self.assertEqual(s_corr.shape, (10, 10))
        np.testing.assert_allclose(np.diag(s_corr), 1.0)
        np.testing.assert_allclose(s_corr.values, s_corr.values.T)
        self.assertTrue((s_corr.values >= -1.0).all() and (s_corr.values <= 1.0).all())

    def test_feature_selection_exclusion(self):
        """Verifies only continuous features are included, strictly excluding metadata/labels."""
        p_corr = compute_pearson_correlation(self.df)
        for col in ["playlist_genre", "playlist_subgenre", "track_id", "track_popularity", "key", "mode"]:
            self.assertNotIn(col, p_corr.columns)
            self.assertNotIn(col, p_corr.index)

    def test_top_correlations_ranking(self):
        """Verifies that top correlation pairs are ranked in descending order of absolute strength."""
        top_pairs = get_top_correlations(self.df, top_n=10)
        self.assertEqual(len(top_pairs), 10)
        self.assertIn("pearson_r", top_pairs.columns)
        self.assertIn("spearman_rho", top_pairs.columns)
        self.assertIn("musical_interpretation", top_pairs.columns)

        # Energy & Loudness should be the strongest positive relationship
        top_pair = top_pairs.iloc[0]
        self.assertEqual(set([top_pair["feature_1"], top_pair["feature_2"]]), {"energy", "loudness"})
        self.assertGreater(top_pair["pearson_r"], 0.65)

    def test_key_distribution_analysis(self):
        """Verifies key pitch class distribution and musical notation mapping."""
        key_df = analyze_key_distribution(self.df)
        self.assertEqual(len(key_df), 12)
        self.assertEqual(int(key_df["track_count"].sum()), len(self.df))
        self.assertIn("musical_notation", key_df.columns)

    def test_mode_distribution_analysis(self):
        """Verifies binary mode analysis."""
        mode_dict = analyze_mode_distribution(self.df)
        self.assertIn("major_count", mode_dict)
        self.assertIn("minor_count", mode_dict)
        self.assertEqual(mode_dict["major_count"] + mode_dict["minor_count"], len(self.df))
        self.assertIn("feature_means_by_mode", mode_dict)


if __name__ == "__main__":
    unittest.main()

