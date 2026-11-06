"""
SONIC ATLAS — Unit Tests for Clustering & Machine Learning Pipeline (Phase 3)
Tests feature scaling, anti-leakage validation, duplicate feature protection,
K-Means fitting, PCA transformation, cluster profiles, and artifact persistence.
"""

import sys
import unittest
import warnings
from pathlib import Path

# Suppress Apple Silicon Accelerate runtime warning in unit tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.clustering import (
    assign_archetype_names,
    build_cluster_profiles,
    compare_observation_vs_unique_clustering,
    compute_pca_projection,
    fit_final_kmeans,
    load_clustering_artifacts,
    prepare_clustering_features,
    validate_clustering_features,
)
from src.config import (
    CLUSTERING_METADATA_PATH,
    CONTINUOUS_AUDIO_FEATURES,
    KMEANS_MODEL_PATH,
    MODELS_DIR,
    SCALER_PATH,
)
from src.data_loader import load_cleaned_dataset


class TestSonicAtlasClustering(unittest.TestCase):
    """Test suite for Phase 3 machine learning clustering."""

    @classmethod
    def setUpClass(cls):
        """Loads cleaned dataset once for testing."""
        cls.df = load_cleaned_dataset()

    def test_feature_preparation_and_scaling(self):
        """Verifies StandardScaler standardizes features to zero mean and unit variance."""
        X_scaled, scaler, feature_names = prepare_clustering_features(self.df, fit_scaler=True)
        self.assertEqual(X_scaled.shape, (len(self.df), 10))
        self.assertEqual(len(feature_names), 10)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-7)

    def test_anti_leakage_validation(self):
        """Verifies that prohibited columns raise ValueError."""
        with self.assertRaises(ValueError):
            validate_clustering_features(["danceability", "playlist_genre"])

        with self.assertRaises(ValueError):
            validate_clustering_features(["danceability", "track_popularity"])

    def test_duplicate_duration_protection(self):
        """Verifies that multiple duration representations raise ValueError."""
        with self.assertRaises(ValueError):
            validate_clustering_features(["duration_ms", "duration_sec"])

    def test_reproducibility_kmeans(self):
        """Verifies K-Means produces identical assignments across runs with the same seed."""
        # Use small sample for fast test
        sample_df = self.df.head(200)
        X_sample, _, _ = prepare_clustering_features(sample_df)
        km1 = fit_final_kmeans(X_sample, n_clusters=4, random_state=42)
        km2 = fit_final_kmeans(X_sample, n_clusters=4, random_state=42)
        np.testing.assert_array_equal(km1.labels_, km2.labels_)

    def test_pca_projection(self):
        """Verifies PCA projection produces 2 components with positive explained variance."""
        sample_df = self.df.head(500)
        X_sample, _, _ = prepare_clustering_features(sample_df)
        X_pca, pca_model = compute_pca_projection(X_sample, n_components=2)
        self.assertEqual(X_pca.shape, (len(sample_df), 2))
        self.assertEqual(len(pca_model.explained_variance_ratio_), 2)
        self.assertGreater(pca_model.explained_variance_ratio_.sum(), 0.30)

    def test_cluster_profiles_and_archetypes(self):
        """Verifies build_cluster_profiles returns original scales and archetype personas."""
        sample_df = self.df.head(500).copy()
        sample_df["cluster"] = np.random.choice(6, size=len(sample_df))
        profiles = build_cluster_profiles(sample_df, cluster_col="cluster")
        self.assertEqual(len(profiles), 6)
        archetypes = assign_archetype_names(profiles)
        self.assertEqual(len(archetypes), 6)
        for c in range(6):
            self.assertIn("name", archetypes[c])
            self.assertIn("subtitle", archetypes[c])

    def test_artifacts_saved_and_loaded(self):
        """Verifies saved model artifacts can be reloaded and are functional."""
        self.assertTrue(KMEANS_MODEL_PATH.is_file(), f"Model file must exist at {KMEANS_MODEL_PATH}")
        self.assertTrue(SCALER_PATH.is_file(), f"Scaler file must exist at {SCALER_PATH}")
        self.assertTrue(CLUSTERING_METADATA_PATH.is_file(), f"Metadata file must exist at {CLUSTERING_METADATA_PATH}")

        model, scaler, metadata = load_clustering_artifacts()
        self.assertEqual(metadata["selected_k"], 6)
        self.assertIn("silhouette_score", metadata)
        self.assertIn("cluster_archetypes", metadata)

        # Test predicting on sample
        sample_X = self.df[CONTINUOUS_AUDIO_FEATURES].head(10).to_numpy(dtype=np.float64)
        sample_scaled = scaler.transform(sample_X)
        preds = model.predict(sample_scaled)
        self.assertEqual(len(preds), 10)
        self.assertTrue(((preds >= 0) & (preds < 6)).all())


if __name__ == "__main__":
    unittest.main()
