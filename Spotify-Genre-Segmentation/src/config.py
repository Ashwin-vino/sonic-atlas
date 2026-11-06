"""
SONIC ATLAS — Configuration & Design Tokens
Centralizes project paths, dataset definitions, reproducibility seeds,
feature classifications, and professional design system tokens.
"""

from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Primary dataset filenames and paths
DATASET_FILENAME = "spotify_songs.csv"
CLEANED_DATASET_FILENAME = "spotify_songs_cleaned.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / CLEANED_DATASET_FILENAME

# Model, report, asset, and prompt directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
IMAGES_DIR = ASSETS_DIR / "images"
PROMPTS_DIR = PROJECT_ROOT / "project-prompts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Serialized Phase 3 Model & Pipeline Artifacts
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
CLUSTERING_METADATA_PATH = MODELS_DIR / "clustering_metadata.json"
CLUSTERED_DATA_PATH = PROCESSED_DATA_DIR / "spotify_songs_clustered.csv"

# Serialized Phase 4 Recommendation Artifacts
RECOMMENDATION_CATALOG_FILENAME = "spotify_songs_recommendation_catalog.csv"
RECOMMENDATION_CATALOG_PATH = PROCESSED_DATA_DIR / RECOMMENDATION_CATALOG_FILENAME
NEAREST_NEIGHBORS_MODEL_PATH = MODELS_DIR / "nearest_neighbors_model.pkl"
RECOMMENDATION_METADATA_PATH = MODELS_DIR / "recommendation_metadata.json"

RECOMMENDATION_MODES: Dict[str, str] = {
    "acoustic_similar": "Acoustic Similar Tracks",
    "acoustic_twins": "Acoustic Twins — Cross-Genre Discovery",
    "same_archetype": "Same Sonic Archetype",
    "explore_beyond": "Explore Beyond Your Cluster",
}

RECOMMENDATION_TOP_K_OPTIONS: List[int] = [5, 10, 15, 20]
DEFAULT_CANDIDATE_POOL_SIZE: int = 250

# ==============================================================================
# 2. REPRODUCIBILITY & DATASET SCHEMA
# ==============================================================================

RANDOM_SEED: int = 42

# Expected column names for Spotify Songs dataset
DATASET_SCHEMA: List[str] = [
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Continuous or numerical acoustic measurements (10 features)
CONTINUOUS_AUDIO_FEATURES: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Discrete or categorical musical attributes (2 features)
DISCRETE_MUSICAL_FEATURES: List[str] = [
    "key",   # 0 to 11 Pitch Class notation (0 = C, 1 = C♯/D♭, ..., 11 = B)
    "mode",  # Modality (0 = Minor, 1 = Major)
]

# All 12 intrinsic audio features
AUDIO_FEATURES: List[str] = CONTINUOUS_AUDIO_FEATURES + DISCRETE_MUSICAL_FEATURES

# Metadata numeric attributes (catalog popularity metrics, NOT an acoustic property)
METADATA_NUMERIC_FEATURES: List[str] = [
    "track_popularity",
]

# Metadata columns (identifiers and descriptions)
METADATA_COLUMNS: List[str] = [
    "track_id",
    "track_name",
    "track_artist",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "track_popularity",
]

# Categorical ground truth labels
GENRE_COLUMNS: List[str] = [
    "playlist_genre",
    "playlist_subgenre",
]

# Prohibited clustering inputs: MUST NOT be fed to K-Means to prevent data leakage
NON_CLUSTERING_FEATURES: List[str] = [
    "playlist_genre",
    "playlist_subgenre",
    "playlist_name",
    "playlist_id",
    "track_id",
    "track_name",
    "track_artist",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
]

# ==============================================================================
# 3. PROFESSIONAL DESIGN SYSTEM TOKENS
# ==============================================================================

COLOR_PALETTE: Dict[str, str] = {
    "main_background": "#090D0B",
    "sidebar_background": "#0D1310",
    "main_surface": "#141C17",
    "elevated_surface": "#1B261F",
    "primary_green": "#55D99A",
    "secondary_green": "#23865B",
    "primary_text": "#F2F5F3",
    "secondary_text": "#9AA79F",
    "muted_text": "#69776E",
    "borders": "#26352C",
    "warning": "#E9B86C",
    "error": "#E47777",
}

# Cohesive, restrained palette for the 6 core musical genres
# (Strictly adheres to dark-mode editorial principles; no neon, no purple/violet)
GENRE_PALETTE: Dict[str, str] = {
    "edm": "#55D99A",    # Primary green
    "rap": "#E9B86C",    # Amber gold
    "pop": "#5BC0BE",    # Clean cyan/teal
    "r&b": "#23865B",    # Forest green
    "latin": "#D97755",  # Warm terracotta
    "rock": "#708A81",   # Slate sage
}

# Dedicated harmonious palette for the 6 unsupervised sonic clusters
CLUSTER_PALETTE: Dict[int, str] = {
    0: "#55D99A",  # High-Octane Anthem (Primary Green)
    1: "#5BC0BE",  # Instrumental Synth Wave (Clean Teal)
    2: "#E9B86C",  # Live Concert & Stadium Arena (Amber Gold)
    3: "#D97755",  # Rhythmic Groove & Vibrant Pop (Warm Terracotta)
    4: "#708A81",  # Lyrical Flow & Urban Hip-Hop (Slate Sage)
    5: "#9AA79F",  # Acoustic Intimacy & Mellow Soul (Muted Sage/Silver)
}

# Empirically derived sonic archetype personas discovered via K-Means (K=6)
CLUSTER_ARCHETYPES: Dict[int, Dict[str, str]] = {
    0: {
        "name": "High-Octane Anthem",
        "subtitle": "Driving Rock & Peak-Time EDM",
        "description": "Aggressive dynamic intensity, maximal master loudness (-5.3 dB), and driving peak tempo (132.6 BPM). Dominates mainstage festival sets.",
        "color": "#55D99A",
    },
    1: {
        "name": "Instrumental Synth Wave",
        "subtitle": "Club, Tech & Progressive Electronic",
        "description": "Heavy synthesizer instrumentation (0.746), extended progressive arrangement (252.1s duration), and virtually absent vocals.",
        "color": "#5BC0BE",
    },
    2: {
        "name": "Live Concert & Stadium Arena",
        "subtitle": "Stage Performance & Audience Ambience",
        "description": "Elevated crowd presence and room ambience (liveness 0.614), amplified live energy (0.780), and wide dynamic swing.",
        "color": "#E9B86C",
    },
    3: {
        "name": "Rhythmic Groove & Vibrant Pop",
        "subtitle": "Latin Reggaeton, Dance Pop & Upbeat Funk",
        "description": "Infectious syncopated rhythm (danceability 0.740), elevated harmonic valence (0.680), and high commercial radio appeal.",
        "color": "#D97755",
    },
    4: {
        "name": "Lyrical Flow & Urban Hip-Hop",
        "subtitle": "Rap, Trap & Spoken Cadence",
        "description": "Elevated vocal density and spoken syllables (speechiness 0.313) set over syncopated 808 beats and modern trap rhythms.",
        "color": "#708A81",
    },
    5: {
        "name": "Acoustic Intimacy & Mellow Soul",
        "subtitle": "Organic Instrumentation & Downtempo Ballads",
        "description": "Dominant acoustic and organic instruments (0.519), soft master loudness (-10.5 dB), and relaxed, intimate tempo.",
        "color": "#9AA79F",
    },
}

TYPOGRAPHY: Dict[str, str] = {
    "heading_font": "Manrope, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "body_font": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "mono_font": "'JetBrains Mono', 'SF Mono', Consolas, Menlo, monospace",
}

# 8px Spacing System
SPACING: Dict[str, str] = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
}

BORDER_RADIUS: Dict[str, str] = {
    "sm": "6px",
    "md": "10px",
    "lg": "12px",
    "pill": "9999px",
}
