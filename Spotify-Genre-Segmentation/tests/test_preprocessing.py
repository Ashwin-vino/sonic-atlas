"""
SONIC ATLAS — Unit Tests for Preprocessing Pipeline (Phase 2)
Tests data cleaning, non-destructive metadata fallback imputation,
schema fidelity, release year extraction, and unique tracks view.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    CLEANED_DATASET_FILENAME,
    CONTINUOUS_AUDIO_FEATURES,
    DISCRETE_MUSICAL_FEATURES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.data_loader import load_cleaned_dataset, load_dataset
from src.preprocessing import (
    clean_dataset,
    clean_text_fields,
    create_unique_tracks_view,
    extract_temporal_features,
    handle_missing_metadata,
    inspect_duplicates,
    inspect_missing_values,
)


class TestSonicAtlasPreprocessing(unittest.TestCase):
    """Test suite for Phase 2 data cleaning and preprocessing."""

    @classmethod
    def setUpClass(cls):
        """Loads raw dataset once for the test suite."""
        cls.raw_df = load_dataset()

    def test_raw_dataset_integrity(self):
        """Verifies raw dataset remains unmodified at exactly 32,833 rows and 23 columns."""
        self.assertEqual(len(self.raw_df), 32833)
        self.assertEqual(self.raw_df.shape[1], 23)

    def test_missing_metadata_handling(self):
        """Verifies that missing metadata values are transparently imputed without row deletion."""
        cleaned, audit = handle_missing_metadata(self.raw_df)
        self.assertEqual(len(cleaned), 32833, "Row count must remain 32,833 (no rows deleted).")
        self.assertEqual(audit["records_removed"], 0)
        self.assertEqual(audit["affected_rows_count"], 5)
        self.assertEqual(cleaned["track_name"].isnull().sum(), 0)
        self.assertEqual(cleaned["track_artist"].isnull().sum(), 0)
        self.assertEqual(cleaned["track_album_name"].isnull().sum(), 0)
        self.assertIn("metadata_imputed", cleaned.columns)
        self.assertEqual(int(cleaned["metadata_imputed"].sum()), 5)

    def test_duplicate_inspection(self):
        """Verifies that exact row duplicates are 0, while multi-playlist occurrences are 4,477."""
        dup_report = inspect_duplicates(self.raw_df)
        self.assertEqual(dup_report["exact_duplicate_rows"], 0)
        self.assertEqual(dup_report["unique_identifiers"], 28356)
        self.assertEqual(dup_report["multi_occurrence_rows"], 4477)

    def test_temporal_feature_extraction(self):
        """Verifies release_year extraction from track_album_release_date."""
        df_temporal = extract_temporal_features(self.raw_df)
        self.assertIn("release_year", df_temporal.columns)
        self.assertEqual(df_temporal["release_year"].isnull().sum(), 0)
        self.assertGreaterEqual(int(df_temporal["release_year"].min()), 1957)
        self.assertLessEqual(int(df_temporal["release_year"].max()), 2020)

    def test_clean_dataset_execution(self):
        """Verifies that clean_dataset creates the processed CSV and matches integrity criteria."""
        cleaned_df = clean_dataset(self.raw_df, save_processed=True)
        self.assertEqual(len(cleaned_df), 32833)
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0, "No null values should remain in cleaned dataset.")
        self.assertTrue(PROCESSED_DATA_PATH.is_file(), f"Cleaned file must exist at {PROCESSED_DATA_PATH}")

    def test_unique_tracks_view_generation(self):
        """Verifies create_unique_tracks_view yields exactly 28,356 distinct compositions."""
        unique_df = create_unique_tracks_view(self.raw_df)
        self.assertEqual(len(unique_df), 28356)
        self.assertEqual(unique_df["track_id"].nunique(), 28356)
        self.assertIn("playlist_count", unique_df.columns)

    def test_audio_feature_bounds(self):
        """Verifies that all continuous and discrete audio features lie within expected theoretical ranges."""
        cleaned_df = load_cleaned_dataset()
        self.assertTrue((cleaned_df["danceability"] >= 0.0).all() and (cleaned_df["danceability"] <= 1.0).all())
        self.assertTrue((cleaned_df["energy"] >= 0.0).all() and (cleaned_df["energy"] <= 1.0).all())
        self.assertTrue((cleaned_df["valence"] >= 0.0).all() and (cleaned_df["valence"] <= 1.0).all())
        self.assertTrue((cleaned_df["acousticness"] >= 0.0).all() and (cleaned_df["acousticness"] <= 1.0).all())
        self.assertTrue((cleaned_df["tempo"] >= 0.0).all() and (cleaned_df["tempo"] <= 250.0).all())
        self.assertTrue((cleaned_df["key"] >= 0).all() and (cleaned_df["key"] <= 11).all())
        self.assertTrue(set(cleaned_df["mode"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

