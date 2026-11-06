"""
SONIC ATLAS — Preprocessing Module
Phase 2 Implementation: Complete, reusable data cleaning, validation,
metadata fallback imputation, feature classification, and dataset export.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    AUDIO_FEATURES,
    CONTINUOUS_AUDIO_FEATURES,
    DATASET_SCHEMA,
    DISCRETE_MUSICAL_FEATURES,
    GENRE_COLUMNS,
    METADATA_COLUMNS,
    METADATA_NUMERIC_FEATURES,
    NON_CLUSTERING_FEATURES,
    PROCESSED_DATA_PATH,
)

logger = logging.getLogger(__name__)


def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs detailed missing value auditing across all columns.

    Args:
        df: Input DataFrame to audit.

    Returns:
        pd.DataFrame: Table of columns with missing counts, percentages, and data types.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows) * 100

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pcts.round(4).values,
        "dtype": [str(d) for d in df.dtypes.values],
    })

    return report.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


def inspect_duplicates(
    df: pd.DataFrame,
    id_column: str = "track_id",
) -> Dict[str, Any]:
    """
    Audits exact duplicate rows and identifier-level duplicate occurrences.

    Distinguishes exact row duplicates (0) from multi-playlist occurrences (4,477 rows).
    The 32,833 rows represent playlist-associated song observations.

    Args:
        df: Input DataFrame.
        id_column: Primary identifier column to check for multi-playlist occurrences.

    Returns:
        Dict detailing exact row duplicates and identifier duplicates.
    """
    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())

    id_duplicates = 0
    unique_ids = 0
    duplicate_id_samples: List[str] = []

    if id_column in df.columns:
        unique_ids = int(df[id_column].nunique(dropna=True))
        id_duplicates = int(df.duplicated(subset=[id_column]).sum())
        dup_series = df[df.duplicated(subset=[id_column], keep=False)][id_column]
        duplicate_id_samples = list(dup_series.unique()[:5])

    return {
        "total_rows": total_rows,
        "exact_duplicate_rows": exact_duplicates,
        "unique_identifiers": unique_ids,
        "multi_occurrence_rows": id_duplicates,
        "multi_occurrence_samples": duplicate_id_samples,
    }


def validate_columns(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validates presence and fidelity of expected schema columns.

    Args:
        df: Input DataFrame.
        expected_columns: List of column names expected. Defaults to DATASET_SCHEMA.

    Returns:
        Dict indicating validity, missing columns, extra columns, and total count.
    """
    expected = expected_columns or DATASET_SCHEMA
    present_cols = list(df.columns)
    missing_cols = [c for c in expected if c not in present_cols]
    extra_cols = [c for c in present_cols if c not in expected]

    is_valid = len(missing_cols) == 0

    return {
        "is_valid": is_valid,
        "total_columns": len(present_cols),
        "expected_columns_count": len(expected),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "all_columns": present_cols,
    }


def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits data types across continuous audio features, discrete musical attributes,
    metadata, and categorical labels.

    Args:
        df: Input DataFrame.

    Returns:
        Dict separating valid numeric audio features and type anomalies.
    """
    audio_type_audit: Dict[str, str] = {}
    non_numeric_audio: List[str] = []

    for feature in AUDIO_FEATURES:
        if feature in df.columns:
            dtype_str = str(df[feature].dtype)
            audio_type_audit[feature] = dtype_str
            if not pd.api.types.is_numeric_dtype(df[feature]):
                non_numeric_audio.append(feature)

    genre_type_audit: Dict[str, str] = {}
    for genre_col in GENRE_COLUMNS:
        if genre_col in df.columns:
            genre_type_audit[genre_col] = str(df[genre_col].dtype)

    return {
        "audio_features_dtypes": audio_type_audit,
        "non_numeric_audio_features": non_numeric_audio,
        "genre_dtypes": genre_type_audit,
        "is_audio_types_valid": len(non_numeric_audio) == 0,
    }


def inspect_audio_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical bounds (min, 25%, 50%, 75%, max, mean, std) for audio features.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table of audio feature bounds.
    """
    features_present = [f for f in AUDIO_FEATURES if f in df.columns]
    if not features_present:
        return pd.DataFrame()

    stats_df = df[features_present].describe().transpose()
    stats_df = stats_df[["min", "25%", "50%", "75%", "max", "mean", "std"]]
    stats_df = stats_df.round(4).reset_index().rename(columns={"index": "feature"})
    return stats_df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trims leading/trailing whitespace and normalizes string fields while
    preserving original artistic capitalization and punctuation.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Copy of DataFrame with trimmed text fields.
    """
    df_clean = df.copy()
    text_columns = [
        "track_name",
        "track_artist",
        "track_album_name",
        "playlist_name",
        "playlist_genre",
        "playlist_subgenre",
    ]
    for col in text_columns:
        if col in df_clean.columns:
            # Strip whitespace safely for string entries
            df_clean[col] = df_clean[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    return df_clean


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts release year from track_album_release_date.
    Every release date in the dataset starts with a 4-digit year (1957 to 2020).

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added 'release_year' column.
    """
    df_clean = df.copy()
    if "track_album_release_date" in df_clean.columns:
        # Extract the first 4 characters safely
        years = df_clean["track_album_release_date"].astype(str).str[:4]
        df_clean["release_year"] = pd.to_numeric(years, errors="coerce").astype("Int64")
    return df_clean


def handle_missing_metadata(
    df: pd.DataFrame,
    fallback_value: str = "Unknown",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles the 5 records with missing track metadata (track_name, track_artist, track_album_name).
    Per data integrity rules:
    - Does NOT delete records.
    - Does NOT invent fake artist or track names.
    - Applies a clear, transparent fallback ('Unknown Track', 'Unknown Artist', 'Unknown Album').
    - Adds a boolean audit column 'metadata_imputed' to preserve original missingness information.

    Args:
        df: Input DataFrame.
        fallback_value: Fallback label prefix.

    Returns:
        Tuple of (Cleaned DataFrame, Audit dictionary).
    """
    df_clean = df.copy()

    # Identify rows with missing metadata before fallback
    missing_mask = (
        df_clean["track_name"].isnull()
        | df_clean["track_artist"].isnull()
        | df_clean["track_album_name"].isnull()
    )
    affected_rows = int(missing_mask.sum())
    affected_track_ids = list(df_clean.loc[missing_mask, "track_id"].unique())

    df_clean["metadata_imputed"] = missing_mask

    # Apply transparent fallbacks
    df_clean["track_name"] = df_clean["track_name"].fillna(f"{fallback_value} Track")
    df_clean["track_artist"] = df_clean["track_artist"].fillna(f"{fallback_value} Artist")
    df_clean["track_album_name"] = df_clean["track_album_name"].fillna(f"{fallback_value} Album")

    audit = {
        "affected_rows_count": affected_rows,
        "affected_track_ids": affected_track_ids,
        "fallback_strategy": "Non-destructive fallback labels ('Unknown Track/Artist/Album')",
        "records_removed": 0,
    }
    logger.info("Handled missing metadata: %d rows updated with transparent fallbacks.", affected_rows)
    return df_clean, audit


def clean_dataset(
    df: pd.DataFrame,
    save_processed: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Executes the complete, non-destructive data preprocessing pipeline for Phase 2:
    1. Preserves all 32,833 playlist-associated song observations.
    2. Trims text whitespace while preserving artistic punctuation.
    3. Transparently imputes the 5 missing metadata fields without deleting rows.
    4. Extracts release year and duration conversions.
    5. Validates audio feature data types.
    6. Persists cleaned dataset to data/processed/spotify_songs_cleaned.csv.

    Args:
        df: Raw Spotify songs DataFrame.
        save_processed: Whether to save to disk.
        output_path: Custom output Path. Defaults to PROCESSED_DATA_PATH.

    Returns:
        pd.DataFrame: Cleaned 32,833 observation-level DataFrame.
    """
    logger.info("Initiating Phase 2 preprocessing on %d rows...", len(df))

    # 1. Text cleaning
    df_clean = clean_text_fields(df)

    # 2. Missing metadata handling
    df_clean, missing_audit = handle_missing_metadata(df_clean)

    # 3. Temporal feature extraction
    df_clean = extract_temporal_features(df_clean)

    # 4. Duration conversions
    if "duration_ms" in df_clean.columns:
        df_clean["duration_sec"] = (df_clean["duration_ms"] / 1000.0).round(2)
        df_clean["duration_min"] = (df_clean["duration_ms"] / 60000.0).round(2)

    # 5. Type casting
    for feat in CONTINUOUS_AUDIO_FEATURES:
        if feat in df_clean.columns:
            df_clean[feat] = pd.to_numeric(df_clean[feat], errors="coerce")

    for feat in DISCRETE_MUSICAL_FEATURES:
        if feat in df_clean.columns:
            df_clean[feat] = pd.to_numeric(df_clean[feat], errors="coerce").astype(int)

    if "track_popularity" in df_clean.columns:
        df_clean["track_popularity"] = pd.to_numeric(df_clean["track_popularity"], errors="coerce").astype(int)

    # 6. Persistence
    if save_processed:
        target_file = output_path or PROCESSED_DATA_PATH
        target_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(target_file, index=False)
        logger.info("Saved cleaned dataset (%d rows) to: %s", len(df_clean), target_file)

    return df_clean


def create_unique_tracks_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an aggregated unique-track view (28,356 unique tracks) for analyses
    where individual composition uniqueness is prioritized over playlist repetition.

    Preserves the original playlist observations dataset intact.

    Args:
        df: Cleaned or raw DataFrame.

    Returns:
        pd.DataFrame: Deduplicated DataFrame with 1 row per unique track_id.
    """
    if "track_id" not in df.columns:
        return df

    # Retain the first occurrence of track metadata and audio features
    # Collect count of playlist appearances
    playlist_counts = df.groupby("track_id")["playlist_id"].nunique().rename("playlist_count")
    genres_per_track = df.groupby("track_id")["playlist_genre"].agg(lambda x: list(x.unique())).rename("genres_list")

    df_unique = df.drop_duplicates(subset=["track_id"]).copy()
    df_unique = df_unique.merge(playlist_counts, on="track_id", how="left")
    df_unique = df_unique.merge(genres_per_track, on="track_id", how="left")

    logger.info("Generated unique tracks view: %d distinct tracks.", len(df_unique))
    return df_unique


def run_preliminary_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes a complete preliminary audit of the dataset without mutating data.

    Returns:
        Dict consolidating schema validation, missing values, duplicates, and feature ranges.
    """
    col_audit = validate_columns(df)
    dup_audit = inspect_duplicates(df)
    type_audit = check_data_types(df)
    missing_report = inspect_missing_values(df)
    feature_ranges = inspect_audio_feature_ranges(df)

    cols_with_nulls = missing_report[missing_report["missing_count"] > 0]

    return {
        "schema_valid": col_audit["is_valid"],
        "column_audit": col_audit,
        "duplicate_audit": dup_audit,
        "type_audit": type_audit,
        "columns_with_missing": cols_with_nulls.to_dict(orient="records"),
        "total_missing_values": int(missing_report["missing_count"].sum()),
        "audio_feature_stats": feature_ranges.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SONIC ATLAS — Preprocessing Pipeline Execution")
    print("=" * 60)
    from src.data_loader import load_dataset
    raw_df = load_dataset()
    cleaned_df = clean_dataset(raw_df, save_processed=True)
    unique_df = create_unique_tracks_view(cleaned_df)
    print(f"Raw observations:       {len(raw_df):,}")
    print(f"Cleaned observations:   {len(cleaned_df):,}")
    print(f"Unique track catalog:   {len(unique_df):,}")
    print(f"Missing values after:   {cleaned_df.isnull().sum().sum()}")
    print(f"Processed file exists:  {PROCESSED_DATA_PATH.is_file()}")
    print("=" * 60)
