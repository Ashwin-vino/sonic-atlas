"""
SONIC ATLAS — Data Loader Module
Provides robust, reusable utilities for loading, validating, and inspecting
both raw and cleaned Spotify songs datasets with comprehensive error handling.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.config import (
    AUDIO_FEATURES,
    CLEANED_DATASET_FILENAME,
    DATASET_FILENAME,
    DATASET_SCHEMA,
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_dataset_path(custom_path: Optional[str] = None, is_cleaned: bool = False) -> Path:
    """
    Resolves the dataset path, supporting explicit paths or fallback search
    across known project directories for either raw or cleaned data.

    Args:
        custom_path: Optional custom file path string or Path object.
        is_cleaned: Whether looking for the cleaned dataset (default False for raw).

    Returns:
        Resolved Path object pointing to the existing dataset.

    Raises:
        FileNotFoundError: If the dataset cannot be located in any known candidate location.
    """
    target_filename = CLEANED_DATASET_FILENAME if is_cleaned else DATASET_FILENAME
    default_path = PROCESSED_DATA_PATH if is_cleaned else RAW_DATA_PATH

    if custom_path:
        candidate = Path(custom_path).resolve()
        if candidate.is_file():
            return candidate

    # Standard configured path
    if default_path.is_file():
        return default_path

    # Search potential relative locations
    candidate_locations = [
        default_path,
        PROJECT_ROOT / "data" / ("processed" if is_cleaned else "raw") / target_filename,
        PROJECT_ROOT.parent / "Spotify-Genre-Segmentation" / "data" / ("processed" if is_cleaned else "raw") / target_filename,
        PROJECT_ROOT / target_filename,
        Path("data") / ("processed" if is_cleaned else "raw") / target_filename,
        Path("Spotify-Genre-Segmentation/data") / ("processed" if is_cleaned else "raw") / target_filename,
    ]

    for cand in candidate_locations:
        if cand.is_file():
            logger.info("Located dataset at candidate path: %s", cand)
            return cand.resolve()

    raise FileNotFoundError(
        f"Spotify dataset '{target_filename}' not found. "
        f"Expected location: {default_path}"
    )


def load_dataset(
    filepath: Optional[str] = None,
    validate_schema: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Loads the raw Spotify song CSV dataset into a pandas DataFrame with rigorous validation.

    Args:
        filepath: Optional path to the CSV file. If None, resolves to RAW_DATA_PATH.
        validate_schema: If True, checks that expected schema columns exist in the loaded data.
        **kwargs: Additional keyword arguments passed to pandas.read_csv().

    Returns:
        pd.DataFrame: Loaded and validated Spotify songs DataFrame.
    """
    target_path = resolve_dataset_path(filepath, is_cleaned=False)
    logger.info("Loading raw dataset from: %s", target_path)

    try:
        df = pd.read_csv(target_path, **kwargs)
    except pd.errors.EmptyDataError as err:
        logger.error("Dataset file is completely empty: %s", target_path)
        raise ValueError(f"The dataset file at '{target_path}' is empty.") from err
    except pd.errors.ParserError as err:
        logger.error("Parser error encountered while reading CSV: %s", err)
        raise
    except PermissionError as err:
        logger.error("Permission denied when accessing file: %s", target_path)
        raise
    except Exception as err:
        logger.error("Unexpected error loading dataset: %s", err)
        raise

    if df.empty:
        raise ValueError(f"The dataset loaded from '{target_path}' contains zero rows.")

    if validate_schema:
        missing_expected = [col for col in DATASET_SCHEMA if col not in df.columns]
        if missing_expected:
            logger.warning(
                "Dataset schema warning: %d expected columns not found: %s",
                len(missing_expected),
                missing_expected,
            )

    logger.info("Successfully loaded dataset: %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def load_cleaned_dataset(
    filepath: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Loads the cleaned and processed Spotify dataset from data/processed/.

    Args:
        filepath: Optional path to the cleaned CSV file.
        **kwargs: Additional keyword arguments passed to pandas.read_csv().

    Returns:
        pd.DataFrame: Processed Spotify songs DataFrame.
    """
    target_path = resolve_dataset_path(filepath, is_cleaned=True)
    logger.info("Loading cleaned dataset from: %s", target_path)
    return pd.read_csv(target_path, **kwargs)


def get_dataset_dimensions(df: pd.DataFrame) -> Tuple[int, int]:
    """Returns the dimensions (rows, columns) of the DataFrame."""
    return int(df.shape[0]), int(df.shape[1])


def inspect_columns(df: pd.DataFrame) -> List[str]:
    """Returns a list of column names present in the dataset."""
    return list(df.columns)


def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Returns a dictionary mapping column names to their pandas data types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reports missing value counts and percentages for each column in the dataset.

    Returns:
        pd.DataFrame: Table sorted by missing count descending.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.round(4).values,
        "data_type": [str(d) for d in df.dtypes.values],
    })
    return report.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates genuine, comprehensive dataset statistics for the SONIC ATLAS platform.

    Returns:
        Dictionary containing verified metrics across dimensions, uniqueness, and features.
    """
    total_rows, total_cols = get_dataset_dimensions(df)
    exact_duplicates = int(df.duplicated().sum())

    unique_tracks = int(df["track_id"].nunique()) if "track_id" in df.columns else None
    unique_artists = int(df["track_artist"].nunique()) if "track_artist" in df.columns else None
    unique_genres = int(df["playlist_genre"].nunique()) if "playlist_genre" in df.columns else None
    unique_subgenres = int(df["playlist_subgenre"].nunique()) if "playlist_subgenre" in df.columns else None
    unique_playlists = int(df["playlist_name"].nunique()) if "playlist_name" in df.columns else None
    unique_playlist_ids = int(df["playlist_id"].nunique()) if "playlist_id" in df.columns else None

    # Identify numeric vs categorical columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # Available audio features
    avail_audio = [f for f in AUDIO_FEATURES if f in df.columns]

    missing_df = get_missing_values(df)
    cols_with_missing = missing_df[missing_df["missing_count"] > 0].to_dict(orient="records")

    genre_distribution = (
        df["playlist_genre"].value_counts().to_dict()
        if "playlist_genre" in df.columns
        else {}
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "exact_duplicates": exact_duplicates,
        "unique_tracks": unique_tracks,
        "unique_artists": unique_artists,
        "unique_genres": unique_genres,
        "unique_subgenres": unique_subgenres,
        "unique_playlists": unique_playlists,
        "unique_playlist_ids": unique_playlist_ids,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "available_audio_features": avail_audio,
        "columns_with_missing_values": cols_with_missing,
        "genre_distribution": genre_distribution,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SONIC ATLAS — Dataset Verification & Inspection")
    print("=" * 60)
    try:
        dataset = load_dataset()
        summary = get_dataset_summary(dataset)
        print(f"Total observations:     {summary['total_rows']:,}")
        print(f"Total columns:          {summary['total_columns']}")
        print(f"Exact duplicate rows:   {summary['exact_duplicates']}")
        print(f"Unique track IDs:       {summary['unique_tracks']:,}")
        print(f"Unique artists:         {summary['unique_artists']:,}")
        print(f"Unique playlist names:  {summary['unique_playlists']:,}")
        print(f"Unique playlist IDs:    {summary['unique_playlist_ids']:,}")
        print(f"Unique playlist genres: {summary['unique_genres']}")
        print(f"Unique subgenres:       {summary['unique_subgenres']}")
        print("\nGenre distribution:")
        for genre, count in summary["genre_distribution"].items():
            pct = (count / summary["total_rows"]) * 100
            print(f"  • {genre:8s}: {count:,} ({pct:.2f}%)")
        print("\nColumns with missing values:")
        if summary["columns_with_missing_values"]:
            for item in summary["columns_with_missing_values"]:
                print(f"  • {item['column']}: {item['missing_count']} rows ({item['missing_percentage']:.4f}%)")
        else:
            print("  • None detected.")
        print("=" * 60)
    except Exception as error:
        print(f"Verification failed: {error}")
