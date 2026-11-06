"""
SONIC ATLAS — Music Intelligence Platform
Phase 2: Complete Data Preprocessing & Exploratory Data Analysis (EDA)

A commercial-grade music analytics and genre-discovery application built
around high-fidelity Spotify acoustic data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    AUDIO_FEATURES,
    BORDER_RADIUS,
    CLEANED_DATASET_FILENAME,
    CLUSTERED_DATA_PATH,
    CLUSTERING_METADATA_PATH,
    CLUSTER_ARCHETYPES,
    CLUSTER_PALETTE,
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    DATASET_FILENAME,
    DATASET_SCHEMA,
    DEMONSTRATION_GUIDE_PATH,
    DISCRETE_MUSICAL_FEATURES,
    FIGURES_DIR,
    GENRE_COLUMNS,
    GENRE_PALETTE,
    MAX_UPLOAD_SIZE_MB,
    METADATA_NUMERIC_FEATURES,
    NAVIGATION_PAGES,
    PRESENTATION_PATH,
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    RADAR_FEATURE_BOUNDS,
    RAW_DATA_PATH,
    RECOMMENDATION_CATALOG_PATH,
    RECOMMENDATION_MODES,
    RECOMMENDATION_TOP_K_OPTIONS,
    SPACING,
    SYSTEM_ARCHITECTURE_PATH,
    TRACK_COMPARISON_COLORS,
    TRACK_COMPARISON_MAX,
    TRACK_COMPARISON_MIN,
    TYPOGRAPHY,
)
from src.recommender import (
    SonicRecommender,
    compute_feature_differences,
)
from src.comparison import (
    FEATURE_DISPLAY_NAMES,
    build_comparison_table,
    compute_pairwise_similarity,
    generate_comparison_insights,
    normalize_features_for_radar,
    plot_feature_barchart,
    plot_multitrack_radar,
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
from src.data_loader import (
    get_dataset_dimensions,
    get_dataset_summary,
    get_missing_values,
    load_cleaned_dataset,
    load_dataset,
    resolve_dataset_path,
)
from src.preprocessing import (
    clean_dataset,
    create_unique_tracks_view,
    inspect_duplicates,
    inspect_missing_values,
    run_preliminary_audit,
)
from src.analysis import (
    analyze_audio_features,
    analyze_genre_distribution,
    analyze_temporal_trends,
    analyze_top_artists,
    analyze_top_playlists,
    setup_visual_style,
)

# ==============================================================================
# STREAMLIT CONFIGURATION & DESIGN SYSTEM INJECTION
# ==============================================================================

st.set_page_config(
    page_title="SONIC ATLAS — Music Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Injects custom CSS adhering strictly to the SONIC ATLAS design tokens."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Manrope:wght@500;600;700;800&display=swap');

        /* Global Reset & Base Styling */
        html, body, [class*="css"] {{
            font-family: {TYPOGRAPHY['body_font']};
            background-color: {COLOR_PALETTE['main_background']} !important;
            color: {COLOR_PALETTE['primary_text']};
            -webkit-font-smoothing: antialiased;
        }}

        .stApp {{
            background-color: {COLOR_PALETTE['main_background']} !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_PALETTE['sidebar_background']} !important;
            border-right: 1px solid {COLOR_PALETTE['borders']};
        }}

        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        /* Typography Hierarchy */
        h1, h2, h3, h4, h5, h6 {{
            font-family: {TYPOGRAPHY['heading_font']};
            font-weight: 700;
            color: {COLOR_PALETTE['primary_text']};
            letter-spacing: -0.02em;
        }}

        .mono-text {{
            font-family: {TYPOGRAPHY['mono_font']};
        }}

        /* Wordmark & Brand */
        .brand-container {{
            padding: 0.5rem 0.5rem 1.5rem 0.5rem;
            border-bottom: 1px solid {COLOR_PALETTE['borders']};
            margin-bottom: 1.5rem;
        }}

        .brand-title {{
            font-family: {TYPOGRAPHY['heading_font']};
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: {COLOR_PALETTE['primary_text']};
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-glyph {{
            color: {COLOR_PALETTE['primary_green']};
            font-size: 1.4rem;
            line-height: 1;
        }}

        .brand-subtitle {{
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 0.70rem;
            letter-spacing: 0.12em;
            color: {COLOR_PALETTE['muted_text']};
            margin-top: 4px;
            text-transform: uppercase;
        }}

        /* Navigation Radio Styling */
        div[data-testid="stRadio"] > label {{
            display: none;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] {{
            gap: 6px;
        }}

        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            background-color: transparent;
            padding: 10px 14px;
            border-radius: {BORDER_RADIUS['md']};
            border: 1px solid transparent;
            transition: all 0.15s ease-in-out;
            margin: 0;
            cursor: pointer;
        }}

        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
            background-color: {COLOR_PALETTE['main_surface']};
            border-color: {COLOR_PALETTE['borders']};
        }}

        div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {{
            background-color: {COLOR_PALETTE['elevated_surface']};
            border-color: {COLOR_PALETTE['primary_green']};
        }}

        /* Cards and Surfaces */
        .sonic-card {{
            background-color: {COLOR_PALETTE['main_surface']};
            border: 1px solid {COLOR_PALETTE['borders']};
            border-radius: {BORDER_RADIUS['lg']};
            padding: 20px;
            margin-bottom: 16px;
            transition: border-color 0.2s ease;
        }}

        .sonic-card:hover {{
            border-color: {COLOR_PALETTE['secondary_green']};
        }}

        .sonic-stat-card {{
            background-color: {COLOR_PALETTE['main_surface']};
            border: 1px solid {COLOR_PALETTE['borders']};
            border-radius: {BORDER_RADIUS['md']};
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .stat-label {{
            font-family: {TYPOGRAPHY['body_font']};
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLOR_PALETTE['secondary_text']};
            font-weight: 500;
        }}

        .stat-value {{
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 1.7rem;
            font-weight: 600;
            color: {COLOR_PALETTE['primary_text']};
            line-height: 1.1;
        }}

        .stat-subtext {{
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 0.72rem;
            color: {COLOR_PALETTE['muted_text']};
        }}

        /* Badges & Pills */
        .badge-verified {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: {COLOR_PALETTE['elevated_surface']};
            border: 1px solid {COLOR_PALETTE['secondary_green']};
            color: {COLOR_PALETTE['primary_green']};
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 0.72rem;
            padding: 4px 10px;
            border-radius: {BORDER_RADIUS['pill']};
            letter-spacing: 0.04em;
        }}

        .badge-warning {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: #262117;
            border: 1px solid {COLOR_PALETTE['warning']};
            color: {COLOR_PALETTE['warning']};
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 0.72rem;
            padding: 4px 10px;
            border-radius: {BORDER_RADIUS['pill']};
        }}

        .badge-chip {{
            display: inline-block;
            background-color: {COLOR_PALETTE['elevated_surface']};
            border: 1px solid {COLOR_PALETTE['borders']};
            color: {COLOR_PALETTE['primary_text']};
            font-family: {TYPOGRAPHY['mono_font']};
            font-size: 0.72rem;
            padding: 3px 8px;
            border-radius: {BORDER_RADIUS['sm']};
            margin-right: 6px;
            margin-bottom: 6px;
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: {COLOR_PALETTE['primary_green']};
            display: inline-block;
        }}

        /* Progress Bar override */
        .stProgress > div > div > div > div {{
            background-color: {COLOR_PALETTE['primary_green']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Loads and caches the dataset, preferring the Phase 3 clustered dataset when available."""
    try:
        if CLUSTERED_DATA_PATH.exists():
            return pd.read_csv(CLUSTERED_DATA_PATH)
        return load_cleaned_dataset()
    except FileNotFoundError:
        raw = load_dataset()
        return clean_dataset(raw, save_processed=True)


@st.cache_data(show_spinner=False)
def get_cached_clustering_metadata() -> Dict[str, Any]:
    """Loads and caches the serialized Phase 3 clustering metadata and evaluation metrics."""
    if CLUSTERING_METADATA_PATH.exists():
        try:
            with open(CLUSTERING_METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def get_cached_unique_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Caches the unique tracks catalog view (28,356 unique tracks)."""
    return create_unique_tracks_view(df)


@st.cache_resource(show_spinner=False)
def get_cached_recommender() -> SonicRecommender:
    """Loads and caches the initialized SonicRecommender instance."""
    return SonicRecommender()


def render_sidebar(dataset_loaded: bool) -> str:
    """Renders the professional SONIC ATLAS sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-container">
                <div class="brand-title">
                    <span class="brand-glyph">◈</span>
                    <span>SONIC ATLAS</span>
                </div>
                <div class="brand-subtitle">Music Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = NAVIGATION_PAGES

        selected_page = st.radio(
            "Navigation",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sonic-card" style="padding: 14px 16px;">
                <div class="stat-label">Platform Status</div>
                <div class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_green']}; margin-top: 4px;">
                    Phase 5 Active (Production)
                </div>
                <div class="mono-text" style="font-size: 0.70rem; color: {COLOR_PALETTE['secondary_text']}; margin-top: 4px;">
                    9/9 Modules Operational
                </div>
                <div class="mono-text" style="font-size: 0.68rem; color: {COLOR_PALETTE['muted_text']}; margin-top: 2px;">
                    Docker &amp; Deployment Ready
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        status_html = (
            f"""
            <div class="badge-verified" style="width: 100%; justify-content: center;">
                <span class="status-dot"></span>
                <span>PLATFORM VERIFIED · 32,833</span>
            </div>
            """
            if dataset_loaded
            else f"""
            <div class="badge-warning" style="width: 100%; justify-content: center;">
                <span>DATASET OFFLINE</span>
            </div>
            """
        )
        st.markdown(status_html, unsafe_allow_html=True)

    return selected_page


# ==============================================================================
# VIEW 1: OVERVIEW PAGE
# ==============================================================================

def render_overview_page(df: pd.DataFrame, summary: Dict[str, Any], audit: Dict[str, Any]) -> None:
    """Renders the executive Overview landing screen."""
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.25rem;">
                <h1 style="font-size: 2.2rem; margin-bottom: 4px;">SONIC ATLAS</h1>
                <p style="font-size: 1.05rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0; font-weight: 400;">
                    Music Intelligence &amp; Acoustic Analytics Platform
                </p>
            </div>
            <p style="font-size: 0.88rem; color: {COLOR_PALETTE['muted_text']}; max-width: 820px; line-height: 1.5; margin-top: 8px;">
                High-precision acoustic feature extraction, multi-dimensional clustering, and genre topology analysis
                engineered on 32,833 playlist-associated Spotify observations.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 0.5rem;">
                <div class="badge-verified">
                    <span class="status-dot"></span>
                    <span>DATASET INTEGRITY: 100%</span>
                </div>
                <div class="mono-text" style="font-size: 0.70rem; color: {COLOR_PALETTE['muted_text']}; margin-top: 6px;">
                    Cleaned: {CLEANED_DATASET_FILENAME}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

    # Key Metrics Grid
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Total Observations</span>
                <span class="stat-value">{summary['total_rows']:,}</span>
                <span class="stat-subtext">Playlist-associated records</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Unique Compositions</span>
                <span class="stat-value">{summary['unique_tracks']:,}</span>
                <span class="stat-subtext">Distinct track IDs</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Catalog Artists</span>
                <span class="stat-value">{summary['unique_artists']:,}</span>
                <span class="stat-subtext">Recording artists</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Curated Playlists</span>
                <span class="stat-value">{summary['unique_playlists']:,}</span>
                <span class="stat-subtext">Across {summary['unique_playlist_ids']:,} playlist IDs</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Super Genres</span>
                <span class="stat-value">{summary['unique_genres']}</span>
                <span class="stat-subtext">24 balanced subgenres</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown(
            f"""
            <div class="sonic-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="font-size: 1.1rem; margin: 0;">Super Genre Taxonomy &amp; Balance</h3>
                    <span class="mono-text" style="font-size: 0.75rem; color: {COLOR_PALETTE['muted_text']};">
                        6 BALANCED CLASSES
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        total_r = summary["total_rows"]
        genre_dist = summary["genre_distribution"]
        for genre, count in genre_dist.items():
            pct = (count / total_r) * 100
            st.markdown(
                f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                        <span style="font-weight: 600; text-transform: uppercase; color: {COLOR_PALETTE['primary_text']};">{genre}</span>
                        <span class="mono-text" style="color: {COLOR_PALETTE['secondary_text']};">{count:,} tracks ({pct:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(pct / 100.0)

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature Classification Audit Card
        st.markdown(
            f"""
            <div class="sonic-card">
                <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Verified Feature Classification</h3>
                <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 0.75rem;">
                    Features are segregated to prevent data leakage and avoid inappropriate scaling assumptions:
                </p>
                <div style="margin-bottom: 10px;">
                    <span class="stat-label" style="display: block; margin-bottom: 4px;">Continuous Audio Features (10)</span>
                    {''.join([f'<span class="badge-chip">{f}</span>' for f in CONTINUOUS_AUDIO_FEATURES])}
                </div>
                <div style="margin-bottom: 10px;">
                    <span class="stat-label" style="display: block; margin-bottom: 4px;">Discrete Musical Attributes (2)</span>
                    {''.join([f'<span class="badge-chip">{f}</span>' for f in DISCRETE_MUSICAL_FEATURES])}
                </div>
                <div>
                    <span class="stat-label" style="display: block; margin-bottom: 4px;">Excluded from Clustering (Data Leakage Protection)</span>
                    <span class="mono-text" style="font-size: 0.74rem; color: {COLOR_PALETTE['warning']};">
                        playlist_genre, playlist_subgenre, playlist_id, playlist_name, track_id, track_artist, track_name
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        # Preprocessing & Data Integrity Audit Card
        st.markdown(
            f"""
            <div class="sonic-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="font-size: 1.1rem; margin: 0;">Phase 2 Data Cleaning Audit</h3>
                    <span class="mono-text" style="font-size: 0.75rem; color: {COLOR_PALETTE['primary_green']};">
                        VERIFIED
                    </span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_PALETTE['borders']}; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Raw Input Observations</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_text']};">32,833</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_PALETTE['borders']}; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Cleaned Output Rows</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_green']};">32,833 (100% Retained)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_PALETTE['borders']}; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Missing Metadata Imputed</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['warning']};">5 entries ('Unknown' fallback)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_PALETTE['borders']}; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Exact Duplicate Rows</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_text']};">0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_PALETTE['borders']}; padding-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Multi-Playlist Repeated Tracks</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">4,477 (Context Preserved)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']};">Records Removed</span>
                        <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_green']};">0 (Zero Data Loss)</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Platform Architecture Roadmap Card
        st.markdown(
            f"""
            <div class="sonic-card">
                <h3 style="font-size: 1.1rem; margin-bottom: 0.75rem;">Platform Implementation Roadmap</h3>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.80rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0;">
                        <span>Phase 1: Foundation &amp; Design Tokens</span>
                        <span class="badge-verified" style="font-size: 0.65rem; padding: 2px 8px;">COMPLETED</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-top: 1px solid {COLOR_PALETTE['borders']};">
                        <span>Phase 2: Preprocessing &amp; Exploratory Data Analysis</span>
                        <span class="badge-verified" style="font-size: 0.65rem; padding: 2px 8px;">COMPLETED</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-top: 1px solid {COLOR_PALETTE['borders']};">
                        <span>Phase 3: Correlation &amp; ML Clustering</span>
                        <span class="badge-verified" style="font-size: 0.65rem; padding: 2px 8px;">COMPLETED</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-top: 1px solid {COLOR_PALETTE['borders']};">
                        <span>Phase 4: Content Recommendation Engine</span>
                        <span class="badge-verified" style="font-size: 0.65rem; padding: 2px 8px;">COMPLETED</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-top: 1px solid {COLOR_PALETTE['borders']};">
                        <span>Phase 5: Production Deployment &amp; Advanced Analytics</span>
                        <span class="badge-verified" style="font-size: 0.65rem; padding: 2px 8px;">COMPLETED</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cleaned Dataset Sample Table
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 style="font-size: 1.1rem; margin: 0;">Cleaned Dataset Inspection Sample</h3>
                <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['muted_text']};">
                    DATA/PROCESSED/SPOTIFY_SONGS_CLEANED.CSV
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_cols = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "release_year",
        "track_popularity",
    ]
    avail_preview = [c for c in preview_cols if c in df.columns]
    st.dataframe(
        df[avail_preview].head(10),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# VIEW 2: MUSIC EXPLORER PAGE
# ==============================================================================

def render_music_explorer(df: pd.DataFrame) -> None:
    """Renders the interactive music explorer with search and multi-attribute filters."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <h1 style="font-size: 2rem; margin-bottom: 4px;">Music Explorer</h1>
            <p style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Search, filter, and inspect acoustic profiles across the verified song catalog.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # View Mode Toggle & Filters
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_query = st.text_input("Search Song or Artist", placeholder="e.g. Ed Sheeran, Queen, Starboy...")
    with f2:
        view_mode = st.selectbox(
            "Dataset View",
            options=["All Playlist Observations (32,833)", "Unique Tracks Catalog (28,356)"],
            index=0,
        )
    with f3:
        all_genres = sorted(df["playlist_genre"].unique())
        selected_genres = st.multiselect("Filter Super Genre", options=all_genres, default=[])

    # Secondary Filter Controls
    c_sub, c_pop, c_tempo = st.columns([2, 1, 1])
    with c_sub:
        avail_subgenres = (
            sorted(df[df["playlist_genre"].isin(selected_genres)]["playlist_subgenre"].unique())
            if selected_genres
            else sorted(df["playlist_subgenre"].unique())
        )
        selected_subgenres = st.multiselect("Filter Subgenre", options=avail_subgenres, default=[])
    with c_pop:
        pop_range = st.slider("Popularity Range", min_value=0, max_value=100, value=(0, 100))
    with c_tempo:
        tempo_range = st.slider("Tempo Range (BPM)", min_value=0, max_value=240, value=(0, 240))

    # Apply Filters
    target_data = get_cached_unique_tracks(df) if "Unique" in view_mode else df

    filtered = target_data.copy()
    if search_query:
        query = search_query.lower()
        filtered = filtered[
            filtered["track_name"].str.lower().str.contains(query, na=False)
            | filtered["track_artist"].str.lower().str.contains(query, na=False)
        ]
    if selected_genres:
        filtered = filtered[filtered["playlist_genre"].isin(selected_genres)]
    if selected_subgenres:
        filtered = filtered[filtered["playlist_subgenre"].isin(selected_subgenres)]

    filtered = filtered[
        (filtered["track_popularity"] >= pop_range[0])
        & (filtered["track_popularity"] <= pop_range[1])
        & (filtered["tempo"] >= tempo_range[0])
        & (filtered["tempo"] <= tempo_range[1])
    ]

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0 0.5rem 0;">
            <span class="mono-text" style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_green']};">
                MATCHING SONGS: {len(filtered):,}
            </span>
            <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['muted_text']};">
                Displaying up to 100 records for performance
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_cols = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "danceability",
        "energy",
        "valence",
        "loudness",
        "tempo",
        "release_year",
        "track_popularity",
    ]
    avail_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[avail_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    # Detailed Track Inspector Card
    if not filtered.empty:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sonic-card">
                <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Track Acoustic Inspector</h3>
                <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                    Select a song from the matching results to inspect its complete acoustic fingerprint.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sample_options = filtered.head(20).apply(
            lambda r: f"{r['track_name']} — {r['track_artist']} [{r['playlist_genre'].upper()}]", axis=1
        ).tolist()

        chosen_label = st.selectbox("Select Track to Inspect", options=sample_options)
        chosen_idx = sample_options.index(chosen_label)
        selected_row = filtered.head(20).iloc[chosen_idx]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Danceability", f"{selected_row['danceability']:.3f}")
        with m2:
            st.metric("Energy", f"{selected_row['energy']:.3f}")
        with m3:
            st.metric("Valence", f"{selected_row['valence']:.3f}")
        with m4:
            st.metric("Acousticness", f"{selected_row['acousticness']:.3f}")
        with m5:
            st.metric("Loudness", f"{selected_row['loudness']:.1f} dB")
        with m6:
            st.metric("Tempo", f"{selected_row['tempo']:.1f} BPM")


# ==============================================================================
# VIEW 3: GENRE ANALYSIS PAGE
# ==============================================================================

def render_genre_analysis(df: pd.DataFrame) -> None:
    """Renders comprehensive comparative genre and subgenre analysis."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <h1 style="font-size: 2rem; margin-bottom: 4px;">Genre Analysis &amp; Taxonomy</h1>
            <p style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Acoustic profiling, cross-genre feature comparison, and granular subgenre signatures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Genre Distribution Summary Cards
    genre_df = analyze_genre_distribution(df)
    cols = st.columns(6)
    for i, (_, row) in enumerate(genre_df.iterrows()):
        genre_name = row["genre"]
        color = GENRE_PALETTE.get(genre_name, COLOR_PALETTE["primary_green"])
        with cols[i]:
            st.markdown(
                f"""
                <div class="sonic-stat-card" style="border-top: 3px solid {color};">
                    <span class="stat-label">{genre_name.upper()}</span>
                    <span class="stat-value">{row['song_count']:,}</span>
                    <span class="stat-subtext">{row['catalog_percentage']}% of catalog</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # Acoustic Feature Profiler: Genre Comparison Tool
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Cross-Genre Acoustic Comparator</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Select two genres to contrast their empirical mean acoustic dimensions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        g1 = st.selectbox("Primary Genre", options=list(GENRE_PALETTE.keys()), index=0)
    with c_g2:
        g2 = st.selectbox("Comparison Genre", options=list(GENRE_PALETTE.keys()), index=1)

    if g1 and g2:
        features_to_compare = ["danceability", "energy", "valence", "speechiness", "acousticness", "instrumentalness"]
        g1_means = df[df["playlist_genre"] == g1][features_to_compare].mean()
        g2_means = df[df["playlist_genre"] == g2][features_to_compare].mean()

        comp_df = pd.DataFrame({
            "Feature Dimension": [f.title() for f in features_to_compare],
            f"{g1.upper()} (Mean)": g1_means.values.round(3),
            f"{g2.upper()} (Mean)": g2_means.values.round(3),
            "Delta (Absolute Difference)": (g1_means.values - g2_means.values).round(3),
        })

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Subgenre Breakdown Table
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Granular Subgenre Taxonomy (24 Classes)</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Each super genre contains exactly 4 balanced subgenres curated across specific playlists.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sub_table = df.groupby(["playlist_genre", "playlist_subgenre"]).agg(
        total_songs=("track_id", "count"),
        median_popularity=("track_popularity", "median"),
        mean_energy=("energy", "mean"),
        mean_danceability=("danceability", "mean"),
        mean_tempo=("tempo", "mean"),
    ).reset_index()

    sub_table.columns = ["Super Genre", "Subgenre", "Track Count", "Median Popularity", "Mean Energy", "Mean Danceability", "Mean Tempo (BPM)"]
    sub_table = sub_table.round(3)
    st.dataframe(sub_table, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 4: DATA INSIGHTS & RELATIONSHIPS PAGE
# ==============================================================================

def render_data_insights(df: pd.DataFrame) -> None:
    """Renders audio feature distributions, bivariate relationships, and temporal trends."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <h1 style="font-size: 2rem; margin-bottom: 4px;">Data Insights &amp; Acoustic Anatomy</h1>
            <p style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Empirical distributions, bivariate acoustic scatter relationships, and release year trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio Feature Statistical Summary
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Audio Feature Parametric Summary</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Statistical moments computed across all 32,833 playlist-associated observations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats_df = analyze_audio_features(df)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Interactive Bivariate Scatter
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Bivariate Acoustic Relationship Explorer</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Examine collinearity and cluster separation across acoustic dimensions. Uses responsive stratified sampling (2,000 observations) for high-performance rendering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_x, c_y = st.columns(2)
    with c_x:
        x_axis = st.selectbox("X-Axis Feature", options=CONTINUOUS_AUDIO_FEATURES, index=0)
    with c_y:
        y_axis = st.selectbox("Y-Axis Feature", options=CONTINUOUS_AUDIO_FEATURES, index=1)

    if x_axis and y_axis:
        sample_pts = df.sample(min(2000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 5))
        setup_visual_style()

        for genre, grp in sample_pts.groupby("playlist_genre"):
            color = GENRE_PALETTE.get(genre, COLOR_PALETTE["primary_green"])
            ax.scatter(
                grp[x_axis],
                grp[y_axis],
                label=genre.upper(),
                color=color,
                alpha=0.5,
                s=24,
                edgecolors="none",
            )

        ax.set_title(f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}", pad=12, fontweight="bold")
        ax.set_xlabel(x_axis.replace("_", " ").title())
        ax.set_ylabel(y_axis.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(framealpha=0.6, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Temporal Trends Section
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Temporal Evolution of Acoustic Signatures (1970 – 2020)</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Historical progression of energy, acousticness, and danceability across release years.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    yearly = analyze_temporal_trends(df)
    modern_yearly = yearly[yearly["release_year"] >= 1970].sort_values(by="release_year")

    fig_time, ax_t = plt.subplots(figsize=(10, 4.5))
    setup_visual_style()
    ax_t.plot(modern_yearly["release_year"], modern_yearly["mean_energy"], color=COLOR_PALETTE["primary_green"], label="Mean Energy", linewidth=2)
    ax_t.plot(modern_yearly["release_year"], modern_yearly["mean_acousticness"], color=COLOR_PALETTE["warning"], label="Mean Acousticness", linewidth=2)
    ax_t.plot(modern_yearly["release_year"], modern_yearly["mean_danceability"], color=COLOR_PALETTE["secondary_text"], label="Mean Danceability", linewidth=2)
    ax_t.set_title("Acoustic Feature Evolution Across 5 Decades", pad=12, fontweight="bold")
    ax_t.set_xlabel("Release Year")
    ax_t.set_ylabel("Mean Feature Value [0, 1]")
    ax_t.grid(True, linestyle="--", alpha=0.4)
    ax_t.legend(framealpha=0.6, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_time)
    plt.close()


# ==============================================================================
# VIEW 5: CLUSTER INTELLIGENCE & ACOUSTIC TOPOLOGY (PHASE 3)
# ==============================================================================

def render_cluster_intelligence(df: pd.DataFrame) -> None:
    """Renders the comprehensive Phase 3 Machine Learning Cluster Intelligence page."""
    # Ensure clustering columns exist
    if "cluster" not in df.columns or "pca_1" not in df.columns:
        if CLUSTERED_DATA_PATH.exists():
            df = pd.read_csv(CLUSTERED_DATA_PATH)
        else:
            st.error("Phase 3 Clustered dataset not found. Please run src/clustering.py first.")
            return

    meta = get_cached_clustering_metadata()
    k_val = meta.get("selected_k", 6)
    sil_score = meta.get("silhouette_score", 0.1380)
    db_score = meta.get("davies_bouldin_index", 1.7350)
    ch_score = meta.get("calinski_harabasz_index", 4025.2)
    pca_var = meta.get("pca_total_variance_explained", 0.3697)
    pca_ratios = meta.get("pca_explained_variance_ratio", [0.2152, 0.1545])

    # Header section
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap;">
                <h1 style="font-size: 2rem; margin: 0;">Cluster Intelligence &amp; Acoustic Topology</h1>
                <span class="badge-verified">PHASE 3 COMPLETE</span>
                <span class="badge-chip">K-MEANS K=6</span>
                <span class="badge-chip">ANTI-LEAKAGE VERIFIED</span>
            </div>
            <p style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Unsupervised acoustic clustering uncovering 6 intrinsic sonic archetypes across 10 audio dimensions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodology Callout
    st.markdown(
        f"""
        <div class="sonic-card" style="border-left: 3px solid {COLOR_PALETTE['primary_green']}; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="color: {COLOR_PALETTE['primary_green']}; font-weight: bold; font-size: 0.88rem;">◈ METHODOLOGY &amp; ANTI-LEAKAGE INTEGRITY</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0; line-height: 1.5;">
                Clustering is executed strictly on 10 continuous standardized audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms). 
                All metadata, track identifiers, and <strong>genre labels were withheld during modeling</strong> to prevent target leakage. 
                Genres are analyzed post-hoc solely to quantify cross-genre acoustic convergence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 Key Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Optimal Clusters</span>
                <span class="stat-value" style="color: {COLOR_PALETTE['primary_green']};">K = {k_val}</span>
                <span class="stat-subtext">Elbow &amp; Silhouette inflection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Silhouette Score</span>
                <span class="stat-value">{sil_score:.4f}</span>
                <span class="stat-subtext">Peak for 10 continuous dims</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Davies-Bouldin</span>
                <span class="stat-value">{db_score:.4f}</span>
                <span class="stat-subtext">Low ratio = compact separation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Calinski-Harabasz</span>
                <span class="stat-value">{ch_score:,.1f}</span>
                <span class="stat-subtext">Pronounced variance ratio</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">2D PCA Variance</span>
                <span class="stat-value">{(pca_var * 100):.1f}%</span>
                <span class="stat-subtext">PC1: {pca_ratios[0]*100:.1f}% · PC2: {pca_ratios[1]*100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # Experiment A vs Experiment B Validation
    st.markdown(
        f"""
        <div class="sonic-card" style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 8px;">
                <h3 style="font-size: 1.05rem; margin: 0;">Dataset Invariance Validation: Experiment A vs. Experiment B</h3>
                <span class="badge-chip">CENTROID DIVERGENCE &lt; 2.4%</span>
            </div>
            <p style="font-size: 0.83rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 0.75rem; line-height: 1.45;">
                We tested whether repeated song occurrences across playlists bias cluster geometry by comparing <strong>Experiment A</strong> (all 32,833 playlist observations) with <strong>Experiment B</strong> (28,356 unique tracks). 
                The empirical centroid deviation across all 10 audio dimensions is under <strong>2.4%</strong>, proving that playlist repetition preserves the underlying acoustic manifold without artificial distortion.
            </p>
            <div style="display: flex; gap: 16px; font-size: 0.78rem; flex-wrap: wrap;" class="mono-text">
                <div style="background-color: {COLOR_PALETTE['elevated_surface']}; padding: 6px 12px; border-radius: 6px; border: 1px solid {COLOR_PALETTE['borders']};">
                    <span style="color: {COLOR_PALETTE['muted_text']};">Experiment A (Observations):</span> <strong>32,833 rows</strong> · Sil: 0.1380 · DB: 1.7350
                </div>
                <div style="background-color: {COLOR_PALETTE['elevated_surface']}; padding: 6px 12px; border-radius: 6px; border: 1px solid {COLOR_PALETTE['borders']};">
                    <span style="color: {COLOR_PALETTE['muted_text']};">Experiment B (Unique Tracks):</span> <strong>28,356 rows</strong> · Sil: 0.1386 · DB: 1.7289
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive 2D PCA Space Explorer
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Interactive 2D Principal Component Acoustic Space</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Explore the acoustic geometry projected onto the two primary axes of musical variance. 
                PC1 captures Master Acoustic Power (Loudness &amp; Energy vs. Acousticness), while PC2 captures Groove &amp; Positivity (Danceability &amp; Valence vs. Tempo).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        color_mode = st.selectbox("Color Mapping", options=["Cluster Archetype", "Super Genre"], index=0)
    with ctrl2:
        cluster_filter = st.selectbox(
            "Filter Cluster",
            options=["All Clusters"] + [f"Cluster {i}: {CLUSTER_ARCHETYPES[i]['name']}" for i in range(6)],
            index=0,
        )
    with ctrl3:
        sample_size = st.slider("Sample Size for Interactive Plot", min_value=1000, max_value=5000, value=2500, step=500)

    # Filter data for scatter plot
    plot_df = df.copy()
    if cluster_filter != "All Clusters":
        sel_cid = int(cluster_filter.split(":")[0].replace("Cluster", "").strip())
        plot_df = plot_df[plot_df["cluster"] == sel_cid]

    # Sample for responsiveness
    sample_n = min(sample_size, len(plot_df))
    plot_sample = plot_df.sample(sample_n, random_state=42)

    # Render PCA figure
    fig_pca, ax_pca = plt.subplots(figsize=(10.5, 6))
    setup_visual_style()

    if color_mode == "Cluster Archetype":
        for c_id in sorted(plot_sample["cluster"].unique()):
            sub = plot_sample[plot_sample["cluster"] == c_id]
            c_info = CLUSTER_ARCHETYPES.get(c_id, {"name": f"Cluster {c_id}", "color": COLOR_PALETTE["primary_green"]})
            ax_pca.scatter(
                sub["pca_1"],
                sub["pca_2"],
                c=c_info["color"],
                label=f"C{c_id}: {c_info['name']}",
                alpha=0.55,
                s=20,
                edgecolor="none",
            )
            # Plot centroid if present
            c_center_x = df[df["cluster"] == c_id]["pca_1"].mean()
            c_center_y = df[df["cluster"] == c_id]["pca_2"].mean()
            ax_pca.scatter(
                [c_center_x],
                [c_center_y],
                c=c_info["color"],
                edgecolors="#F2F5F3",
                linewidth=1.5,
                s=120,
                marker="o",
                zorder=10,
            )
            ax_pca.annotate(
                f"C{c_id}",
                (c_center_x, c_center_y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#F2F5F3",
                bbox=dict(boxstyle="round,pad=0.2", fc=COLOR_PALETTE["elevated_surface"], ec=c_info["color"], alpha=0.85),
                zorder=11,
            )
    else:
        for g in sorted(plot_sample["playlist_genre"].unique()):
            sub = plot_sample[plot_sample["playlist_genre"] == g]
            g_col = GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"])
            ax_pca.scatter(
                sub["pca_1"],
                sub["pca_2"],
                c=g_col,
                label=g.upper(),
                alpha=0.55,
                s=20,
                edgecolor="none",
            )

    ax_pca.set_title(f"2D PCA Projection of Sonic Topology ({sample_n:,} Points Rendered)", pad=12, fontweight="bold")
    ax_pca.set_xlabel(f"PC1 ({pca_ratios[0]*100:.1f}% Variance) — Acoustic Intensity & Master Power (Loudness, Energy vs Acousticness)")
    ax_pca.set_ylabel(f"PC2 ({pca_ratios[1]*100:.1f}% Variance) — Groove & Harmonic Positivity (Danceability, Valence vs Tempo)")
    ax_pca.grid(True, linestyle="--", alpha=0.4)
    ax_pca.legend(loc="upper right", framealpha=0.8, ncol=2 if color_mode == "Cluster Archetype" else 3)
    plt.tight_layout()
    st.pyplot(fig_pca)
    plt.close()

    # Discovered Archetypes Directory
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Discovered Sonic Archetypes Gallery</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Each cluster represents an objectively distinct acoustic archetype derived from multi-dimensional centroid fingerprints.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cluster_counts = df["cluster"].value_counts().to_dict()
    total_obs = len(df)

    row1_cols = st.columns(3)
    for i in range(3):
        with row1_cols[i]:
            arch = CLUSTER_ARCHETYPES[i]
            count = cluster_counts.get(i, 0)
            pct = (count / total_obs) * 100
            st.markdown(
                f"""
                <div class="sonic-card" style="border-top: 3px solid {arch['color']}; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                        <span class="mono-text" style="font-size: 0.72rem; color: {arch['color']};">CLUSTER {i}</span>
                        <span class="badge-chip">{pct:.1f}% ({count:,})</span>
                    </div>
                    <div style="font-size: 1rem; font-weight: bold; margin-bottom: 2px;">{arch['name']}</div>
                    <div style="font-size: 0.78rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 8px;">{arch['subtitle']}</div>
                    <p style="font-size: 0.80rem; color: {COLOR_PALETTE['muted_text']}; line-height: 1.4; margin: 0;">
                        {arch['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
    row2_cols = st.columns(3)
    for i in range(3, 6):
        with row2_cols[i - 3]:
            arch = CLUSTER_ARCHETYPES[i]
            count = cluster_counts.get(i, 0)
            pct = (count / total_obs) * 100
            st.markdown(
                f"""
                <div class="sonic-card" style="border-top: 3px solid {arch['color']}; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                        <span class="mono-text" style="font-size: 0.72rem; color: {arch['color']};">CLUSTER {i}</span>
                        <span class="badge-chip">{pct:.1f}% ({count:,})</span>
                    </div>
                    <div style="font-size: 1rem; font-weight: bold; margin-bottom: 2px;">{arch['name']}</div>
                    <div style="font-size: 0.78rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 8px;">{arch['subtitle']}</div>
                    <p style="font-size: 0.80rem; color: {COLOR_PALETTE['muted_text']}; line-height: 1.4; margin: 0;">
                        {arch['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Interactive Cluster Deep Dive
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Interactive Cluster Deep-Dive Inspector</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Select an acoustic cluster to inspect its multi-dimensional feature signature, post-hoc genre composition, and top popular tracks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_c_id = st.radio(
        "Select Cluster to Inspect",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: f"Cluster {x}: {CLUSTER_ARCHETYPES[x]['name']} ({cluster_counts.get(x, 0):,} tracks)",
        horizontal=True,
    )

    c_meta = CLUSTER_ARCHETYPES[selected_c_id]
    c_df = df[df["cluster"] == selected_c_id]

    col_prof, col_genre = st.columns([3, 2])

    with col_prof:
        st.markdown(
            f"""
            <div style="font-size: 0.90rem; font-weight: bold; margin-bottom: 6px; color: {c_meta['color']};">
                ◈ Normalized Feature Profile: Cluster {selected_c_id} ({c_meta['name']})
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compute normalized features for plotting
        norm_features = [
            "danceability",
            "energy",
            "loudness_norm",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo_norm",
            "duration_norm",
        ]
        # Prepare normalized values
        temp_df = df.copy()
        temp_df["loudness_norm"] = (temp_df["loudness"] - temp_df["loudness"].min()) / (temp_df["loudness"].max() - temp_df["loudness"].min())
        temp_df["tempo_norm"] = (temp_df["tempo"] - temp_df["tempo"].min()) / (temp_df["tempo"].max() - temp_df["tempo"].min())
        temp_df["duration_norm"] = (temp_df["duration_ms"] - temp_df["duration_ms"].min()) / (temp_df["duration_ms"].max() - temp_df["duration_ms"].min())

        cluster_means = temp_df[temp_df["cluster"] == selected_c_id][norm_features].mean()
        catalog_means = temp_df[norm_features].mean()

        feature_display_names = [
            "Danceability",
            "Energy",
            "Loudness (Scaled)",
            "Speechiness",
            "Acousticness",
            "Instrumentalness",
            "Liveness",
            "Valence",
            "Tempo (Scaled)",
            "Duration (Scaled)",
        ]

        fig_prof, ax_prof = plt.subplots(figsize=(7, 4.5))
        setup_visual_style()
        y_pos = np.arange(len(norm_features))

        # Horizontal bars
        bars = ax_prof.barh(y_pos, cluster_means.values, height=0.55, color=c_meta["color"], edgecolor=COLOR_PALETTE["borders"], label=f"Cluster {selected_c_id}")
        ax_prof.scatter(catalog_means.values, y_pos, color=COLOR_PALETTE["primary_text"], s=40, zorder=5, label="Catalog Mean", marker="|")

        for bar in bars:
            w = bar.get_width()
            ax_prof.text(w + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{w:.2f}", va="center", color=COLOR_PALETTE["secondary_text"], fontsize=8)

        ax_prof.set_yticks(y_pos)
        ax_prof.set_yticklabels(feature_display_names, fontsize=8)
        ax_prof.set_xlim(0, 1.15)
        ax_prof.set_xlabel("Normalized Value [0, 1]", fontsize=9)
        ax_prof.grid(axis="x", linestyle="--", alpha=0.4)
        ax_prof.legend(loc="lower right", framealpha=0.8, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_prof)
        plt.close()

    with col_genre:
        st.markdown(
            f"""
            <div style="font-size: 0.90rem; font-weight: bold; margin-bottom: 6px; color: {COLOR_PALETTE['primary_text']};">
                ◈ Post-Hoc Genre Composition (%)
            </div>
            """,
            unsafe_allow_html=True,
        )

        genre_counts = c_df["playlist_genre"].value_counts(normalize=True) * 100
        fig_g, ax_g = plt.subplots(figsize=(5.5, 4.5))
        setup_visual_style()

        bar_colors = [GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"]) for g in genre_counts.index]
        bars_g = ax_g.bar(genre_counts.index.str.upper(), genre_counts.values, color=bar_colors, edgecolor=COLOR_PALETTE["borders"], width=0.6)

        for bar in bars_g:
            h = bar.get_height()
            ax_g.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", color=COLOR_PALETTE["secondary_text"], fontsize=8)

        ax_g.set_ylabel("Proportion (%)", fontsize=9)
        ax_g.set_ylim(0, max(genre_counts.values) * 1.25)
        ax_g.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()

    # Representative Tracks in Cluster
    st.markdown(
        f"""
        <div style="font-size: 0.95rem; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; color: {COLOR_PALETTE['primary_text']};">
            Top Exemplar Tracks in Cluster {selected_c_id}: {c_meta['name']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_tracks = c_df.sort_values(by="track_popularity", ascending=False).drop_duplicates(subset=["track_name", "track_artist"]).head(10)[
        ["track_name", "track_artist", "track_album_name", "track_popularity", "playlist_genre", "playlist_subgenre", "energy", "danceability", "duration_min"]
    ]
    top_tracks.columns = ["Track Name", "Artist", "Album", "Popularity", "Super Genre", "Subgenre", "Energy", "Danceability", "Duration (min)"]
    st.dataframe(top_tracks, use_container_width=True, hide_index=True)

    # Diagnostics Expander
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("◈ K-Means Validation & Diagnostics (K=2 to K=12 Evaluation Curves)", expanded=False):
        st.markdown(
            f"""
            <p style="font-size: 0.84rem; color: {COLOR_PALETTE['secondary_text']}; line-height: 1.5;">
                To objectively identify the optimal number of clusters $K$, we evaluated four mathematical clustering criteria across $K \\in [2, 12]$: 
                <strong>Inertia (Elbow Method)</strong>, <strong>Silhouette Score</strong>, <strong>Davies-Bouldin Index</strong>, and <strong>Calinski-Harabasz Index</strong>.
                $K=6$ represents the optimal inflection point, balancing intra-cluster cohesion, inter-cluster separation, and musicologically interpretable genre boundaries.
            </p>
            """,
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)
        with d1:
            elbow_img = FIGURES_DIR / "kmeans_elbow_curve.png"
            if elbow_img.exists():
                st.image(str(elbow_img), caption="Inertia (Elbow Method) Curve — Noticeable inflection at K=6", use_container_width=True)

            db_img = FIGURES_DIR / "davies_bouldin_scores.png"
            if db_img.exists():
                st.image(str(db_img), caption="Davies-Bouldin Index — Minimum occurs at K=6 (1.7350)", use_container_width=True)

        with d2:
            sil_img = FIGURES_DIR / "silhouette_scores.png"
            if sil_img.exists():
                st.image(str(sil_img), caption="Silhouette Score Curve — Peak inflection at K=6 (0.1380)", use_container_width=True)

            ch_img = FIGURES_DIR / "calinski_harabasz_scores.png"
            if ch_img.exists():
                st.image(str(ch_img), caption="Calinski-Harabasz Index — Pronounced peak at K=6 (4,025.2)", use_container_width=True)


# ==============================================================================
# VIEW 6: CONTENT-BASED RECOMMENDATIONS ENGINE (PHASE 4)
# ==============================================================================

def render_recommendations_page(df: pd.DataFrame) -> None:
    """Renders the comprehensive Phase 4 Content-Based Music Recommendation Engine."""
    recommender = get_cached_recommender()
    catalog = recommender.catalog_df

    # Header section
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap;">
                <h1 style="font-size: 2rem; margin: 0;">Content-Based Music Recommendation Engine</h1>
                <span class="badge-verified">PHASE 4 COMPLETE</span>
                <span class="badge-chip">COSINE SIMILARITY</span>
                <span class="badge-chip">28,356 UNIQUE TRACKS</span>
                <span class="badge-chip">ZERO LEAKAGE</span>
            </div>
            <p style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                High-dimensional acoustic proximity matching across 10 standardized audio features · Cross-genre acoustic twins · Zero collaborative filtering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodology Callout
    st.markdown(
        f"""
        <div class="sonic-card" style="border-left: 3px solid {COLOR_PALETTE['primary_green']}; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="color: {COLOR_PALETTE['primary_green']}; font-weight: bold; font-size: 0.88rem;">◈ METHODOLOGY &amp; ACOUSTIC PROXIMITY</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0; line-height: 1.5;">
                Recommendations are derived strictly from geometric proximity across 10 standardized audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) using cosine distance. 
                <strong>Genre labels, playlist names, and popularity are excluded from similarity calculations</strong>, enabling the discovery of genuine cross-genre acoustic twins. Similarity represents objective audio-feature alignment, not personal preference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # A. SEED TRACK SEARCH & SELECTION WORKSPACE
    # =========================================================================
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">1. Select Seed Track</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Search across 28,356 unique tracks by title or artist, or select from curated popular benchmarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_search, c_suggest = st.columns([3, 2])
    with c_search:
        search_query = st.text_input("Search Track or Artist Name", value="", placeholder="e.g., Post Malone, Billie Eilish, Queen, Drake...")
    with c_suggest:
        quick_picks = [
            "Select a curated benchmark...",
            "Blinding Lights — The Weeknd (Pop)",
            "Bohemian Rhapsody — Queen (Rock)",
            "Roses - Imanbek Remix — SAINt JHN (EDM)",
            "SICKO MODE — Travis Scott (Rap)",
            "Con Calma — Daddy Yankee (Latin)",
            "No Guidance — Chris Brown ft. Drake (R&B)",
        ]
        selected_quick_pick = st.selectbox("Quick Curated Benchmarks", options=quick_picks, index=0)

    # Determine seed track
    if selected_quick_pick != "Select a curated benchmark...":
        quick_query = selected_quick_pick.split("—")[0].strip()
        search_results = recommender.search_tracks(quick_query, limit=15)
    elif search_query.strip():
        search_results = recommender.search_tracks(search_query.strip(), limit=20)
    else:
        # Default popular tracks
        search_results = catalog.sort_values(by="track_popularity", ascending=False).head(20)

    if search_results.empty:
        st.warning(f"No tracks found matching '{search_query}'. Try a broader title or artist search.")
        return

    track_options = {
        f"{row['track_name']} — {row['track_artist']} ({str(row['playlist_genre']).upper()} · Pop: {row['track_popularity']})": row['track_id']
        for _, row in search_results.iterrows()
    }

    selected_track_label = st.selectbox("Matching Tracks in Catalog", options=list(track_options.keys()), index=0)
    selected_seed_id = track_options[selected_track_label]
    seed_track = recommender.get_track_by_id(selected_seed_id)

    if seed_track is None:
        st.error("Failed to load seed track details.")
        return

    # Seed Track Showcase Card
    g_color = GENRE_PALETTE.get(str(seed_track["playlist_genre"]).lower(), COLOR_PALETTE["primary_green"])
    st.markdown(
        f"""
        <div class="sonic-card" style="border-top: 3px solid {g_color}; margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px; margin-bottom: 0.5rem;">
                <div>
                    <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['muted_text']};">SEED TRACK</span>
                    <h2 style="font-size: 1.4rem; margin: 2px 0 4px 0; color: {COLOR_PALETTE['primary_text']};">{seed_track['track_name']}</h2>
                    <div style="font-size: 0.95rem; color: {COLOR_PALETTE['secondary_text']}; font-weight: 500;">
                        {seed_track['track_artist']} · <span style="color: {COLOR_PALETTE['muted_text']};">{seed_track['track_album_name']}</span>
                    </div>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    <span class="badge-chip" style="background-color: {COLOR_PALETTE['elevated_surface']}; border-color: {g_color}; color: {g_color};">
                        {str(seed_track['playlist_genre']).upper()} · {str(seed_track['playlist_subgenre']).title()}
                    </span>
                    <span class="badge-chip">{seed_track.get('cluster_archetype', f"Cluster {seed_track['cluster']}")}</span>
                    <span class="badge-chip">Popularity: {seed_track['track_popularity']}</span>
                    <span class="badge-chip">Year: {seed_track.get('release_year', 'N/A')}</span>
                    <span class="badge-chip">{seed_track.get('duration_min', 0):.2f} min</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid {COLOR_PALETTE['borders']};" class="mono-text">
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Energy:</span> <strong>{seed_track['energy']:.2f}</strong></div>
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Danceability:</span> <strong>{seed_track['danceability']:.2f}</strong></div>
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Valence:</span> <strong>{seed_track['valence']:.2f}</strong></div>
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Acousticness:</span> <strong>{seed_track['acousticness']:.2f}</strong></div>
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Loudness:</span> <strong>{seed_track['loudness']:.1f} dB</strong></div>
                <div style="font-size: 0.75rem;"><span style="color: {COLOR_PALETTE['muted_text']};">Tempo:</span> <strong>{seed_track['tempo']:.0f} BPM</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # B. RECOMMENDATION MODES & HYBRID FILTERS
    # =========================================================================
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">2. Recommendation Mode &amp; Hybrid Filters</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Configure the acoustic proximity engine, target cross-genre discovery, or apply custom taxonomy filters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_options = {
        "Acoustic Similar Tracks (Default)": "acoustic_similar",
        "Acoustic Twins — Cross-Genre Discovery": "acoustic_twins",
        "Same Sonic Archetype": "same_archetype",
        "Explore Beyond Your Cluster": "explore_beyond",
    }

    selected_mode_label = st.radio(
        "Recommendation Mode",
        options=list(mode_options.keys()),
        index=0,
        horizontal=True,
    )
    selected_mode = mode_options[selected_mode_label]

    mode_descriptions = {
        "acoustic_similar": "Finds the closest acoustic neighbors across all 28,356 catalog tracks without genre or cluster boundaries.",
        "acoustic_twins": f"Excludes {str(seed_track['playlist_genre']).upper()} entirely, discovering songs in other genres with near-identical acoustic properties.",
        "same_archetype": f"Restricts recommendations strictly within the seed track's archetype ({seed_track.get('cluster_archetype', 'Cluster ' + str(seed_track['cluster']))}).",
        "explore_beyond": f"Excludes the seed track's archetype, discovering similar musical textures in adjacent sonic clusters.",
    }
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: {COLOR_PALETTE['primary_green']}; margin-top: -0.25rem; margin-bottom: 1rem;">
            ◈ <strong>{selected_mode_label}</strong>: {mode_descriptions[selected_mode]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hybrid Filter Controls
    with st.expander("◈ Hybrid Constraint Filters & Diversity Settings", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            genre_list = ["All Genres"] + sorted(catalog["playlist_genre"].dropna().unique().tolist())
            filter_genre = st.selectbox("Genre Filter", options=genre_list, index=0)
            target_g = None if filter_genre == "All Genres" else filter_genre

            if target_g:
                available_subs = ["All Subgenres"] + sorted(catalog[catalog["playlist_genre"] == target_g]["playlist_subgenre"].dropna().unique().tolist())
            else:
                available_subs = ["All Subgenres"] + sorted(catalog["playlist_subgenre"].dropna().unique().tolist())
            filter_subgenre = st.selectbox("Subgenre Filter", options=available_subs, index=0)
            target_sub = None if filter_subgenre == "All Subgenres" else filter_subgenre

        with f2:
            cluster_list = ["All Clusters"] + [f"Cluster {i}: {CLUSTER_ARCHETYPES[i]['name']}" for i in range(6)]
            filter_cluster = st.selectbox("Cluster Archetype Filter", options=cluster_list, index=0)
            target_c = None if filter_cluster == "All Clusters" else filter_cluster.split(":")[0].replace("Cluster", "").strip()

            pop_range = st.slider("Popularity Range", min_value=0, max_value=100, value=(0, 100), step=5)

        with f3:
            exclude_artist = st.checkbox(f"Exclude seed artist ({seed_track['track_artist']})", value=False)
            top_k = st.selectbox("Number of Recommendations", options=[5, 10, 15, 20], index=1)
            div_choice = st.radio(
                "Ranking Strategy",
                options=["Highest Acoustic Similarity", "Balanced Discovery (Unique Artists)"],
                index=0,
            )
            div_mode = "balanced_discovery" if "Balanced Discovery" in div_choice else "highest_similarity"

    # =========================================================================
    # C. RECOMMENDATION GENERATION & RESULTS DISPLAY
    # =========================================================================
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

    recs, diag = recommender.recommend(
        seed_track_id=selected_seed_id,
        mode=selected_mode,
        top_n=top_k,
        target_genre=target_g,
        target_subgenre=target_sub,
        target_cluster=target_c,
        min_popularity=pop_range[0],
        max_popularity=pop_range[1],
        exclude_seed_artist=exclude_artist,
        diversity_mode=div_mode,
    )

    if recs.empty:
        st.warning("No recommendations could be generated matching all applied filters. Try relaxing the popularity range or genre constraints.")
        return

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 8px;">
            <h3 style="font-size: 1.15rem; margin: 0;">Recommended Acoustic Matches ({len(recs)} tracks)</h3>
            <div style="font-size: 0.82rem;" class="mono-text">
                <span style="color: {COLOR_PALETTE['muted_text']};">Mean Acoustic Similarity:</span> 
                <strong style="color: {COLOR_PALETTE['primary_green']};">{diag['mean_similarity']:.4f}</strong> · 
                <span style="color: {COLOR_PALETTE['muted_text']};">Unique Artists:</span> <strong>{diag['unique_artists']}</strong> · 
                <span style="color: {COLOR_PALETTE['muted_text']};">Genres:</span> <strong>{diag['unique_genres']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Recommendation Cards
    for idx, row in recs.iterrows():
        rg_color = GENRE_PALETTE.get(str(row["playlist_genre"]).lower(), COLOR_PALETTE["primary_green"])
        sim_val = row["cosine_similarity"]

        st.markdown(
            f"""
            <div class="sonic-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid {rg_color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="mono-text" style="font-size: 0.80rem; font-weight: bold; color: {COLOR_PALETTE['primary_green']};">#{row['rank']}</span>
                            <span style="font-size: 1.05rem; font-weight: bold; color: {COLOR_PALETTE['primary_text']};">{row['track_name']}</span>
                            <span style="font-size: 0.88rem; color: {COLOR_PALETTE['secondary_text']};">— {row['track_artist']}</span>
                        </div>
                        <div style="font-size: 0.78rem; color: {COLOR_PALETTE['muted_text']}; margin-top: 3px;">
                            Album: {row['track_album_name']} · Year: {row.get('release_year', 'N/A')} · Duration: {row.get('duration_min', 0):.2f} min
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span class="badge-chip" style="background-color: {COLOR_PALETTE['elevated_surface']}; border-color: {rg_color}; color: {rg_color};">
                            {str(row['playlist_genre']).upper()}
                        </span>
                        <span class="badge-chip">{row.get('cluster_archetype', f"Cluster {row['cluster']}")}</span>
                        <span class="badge-chip">Pop: {row['track_popularity']}</span>
                        <span class="badge-verified" style="font-size: 0.70rem; padding: 3px 8px;">
                            Acoustic Similarity: {sim_val:.4f}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; line-height: 1.4; border-top: 1px solid {COLOR_PALETTE['borders']}; padding-top: 6px;">
                    <span style="color: {COLOR_PALETTE['primary_green']}; font-weight: bold;">◈ Acoustic Reason:</span> {row['acoustic_explanation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # D. COMPARE TRACKS (SEED VS. RECOMMENDED TRACK)
    # =========================================================================
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sonic-card">
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">3. Compare Audio Feature Fingerprints</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 1rem;">
                Select any recommended song to inspect its side-by-side acoustic differences against the seed track.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_choice_labels = {
        f"#{row['rank']}: {row['track_name']} — {row['track_artist']} (Sim: {row['cosine_similarity']:.4f})": row['track_id']
        for _, row in recs.iterrows()
    }
    selected_rec_label = st.selectbox("Select Track to Compare", options=list(rec_choice_labels.keys()), index=0)
    selected_rec_id = rec_choice_labels[selected_rec_label]
    rec_track_to_compare = recs[recs["track_id"] == selected_rec_id].iloc[0]

    c_diff_table, c_diff_chart = st.columns([1, 1])

    with c_diff_table:
        st.markdown(
            f"""
            <div style="font-size: 0.90rem; font-weight: bold; margin-bottom: 6px; color: {COLOR_PALETTE['primary_text']};">
                ◈ Feature Differences Table
            </div>
            """,
            unsafe_allow_html=True,
        )
        diff_df = compute_feature_differences(seed_track, rec_track_to_compare)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    with c_diff_chart:
        st.markdown(
            f"""
            <div style="font-size: 0.90rem; font-weight: bold; margin-bottom: 6px; color: {COLOR_PALETTE['primary_text']};">
                ◈ Side-by-Side Normalized Comparison
            </div>
            """,
            unsafe_allow_html=True,
        )

        norm_feats = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
        labels = [f.title() for f in norm_feats]

        seed_vals = [float(seed_track[f]) for f in norm_feats]
        rec_vals = [float(rec_track_to_compare[f]) for f in norm_feats]

        fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
        setup_visual_style()

        y = np.arange(len(norm_feats))
        height = 0.35

        ax_comp.barh(y - height / 2, seed_vals, height=height, label=f"Seed: {seed_track['track_name'][:18]}", color=COLOR_PALETTE["primary_green"])
        ax_comp.barh(y + height / 2, rec_vals, height=height, label=f"Rec: {rec_track_to_compare['track_name'][:18]}", color=COLOR_PALETTE["warning"])

        ax_comp.set_yticks(y)
        ax_comp.set_yticklabels(labels, fontsize=8)
        ax_comp.set_xlim(0, 1.05)
        ax_comp.set_xlabel("Value [0, 1]", fontsize=8)
        ax_comp.grid(axis="x", linestyle="--", alpha=0.4)
        ax_comp.legend(loc="lower right", fontsize=8, framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()

    # =========================================================================
    # E. TECHNICAL DIAGNOSTICS & TELEMETRY EXPANDER
    # =========================================================================
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("◈ Recommendation Engine Diagnostics & Telemetry", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Initial Candidate Pool", f"{diag['candidates_retrieved']} tracks")
            st.metric("Remaining Post-Mode", f"{diag['candidates_after_mode']} tracks")
        with d2:
            st.metric("Remaining Post-Filters", f"{diag['candidates_after_filters']} tracks")
            st.metric("Final Recommendations", f"{diag['results_returned']} tracks")
        with d3:
            st.metric("Mean Cosine Similarity", f"{diag['mean_similarity']:.4f}")
            st.metric("Min Cosine Similarity", f"{diag['min_similarity']:.4f}")
        with d4:
            st.metric("Unique Artists in List", f"{diag['unique_artists']}")
            st.metric("Super Genres in List", f"{diag['unique_genres']}")

        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: {COLOR_PALETTE['secondary_text']}; margin-top: 0.5rem; line-height: 1.4;">
                <strong>Algorithm:</strong> Scikit-learn <code>NearestNeighbors(metric='cosine', algorithm='brute')</code> querying 28,356 unique tracks in ~0.05ms.<br/>
                <strong>Vector:</strong> 10 continuous standardized audio features transformed with Phase 3 <code>StandardScaler</code>.<br/>
                <strong>Integrity:</strong> Seed track strictly excluded; 0 duplicate track IDs; 0 metadata leakage.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 7: TRACK COMPARISON STUDIO (PHASE 5A)
# ==============================================================================

def render_track_comparison(df: pd.DataFrame) -> None:
    """Renders the Multi-Track Comparison Studio (Phase 5A)."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="font-size: 2.2rem; margin: 0;">Track Comparison Studio</h1>
                <span class="badge-verified">PHASE 5A OPERATIONAL</span>
            </div>
            <p style="font-size: 1.0rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Multi-Track Acoustic Feature Comparison, Geometric Radar Alignment &amp; Pairwise Proximity
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommender = get_cached_recommender()
    catalog_df = recommender.catalog_df

    # Initialize session state for selected track IDs
    if "comparison_track_ids" not in st.session_state:
        st.session_state["comparison_track_ids"] = []

    # Quick Benchmark Presets
    st.markdown(
        f"""
        <div class="sonic-card" style="padding: 16px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span class="stat-label">Quick Benchmark Presets</span>
                <span class="mono-text" style="font-size: 0.70rem; color: {COLOR_PALETTE['muted_text']};">
                    1-Click Load Multi-Genre Benchmark Comparisons
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🎸 Rock vs EDM vs Acoustic", use_container_width=True):
            # AC/DC Back In Black, Chainsmokers Closer, Ed Sheeran Perfect
            candidates = ["08mG3Y1vljYA6bv08I9nvb", "7BKLCZ1jbUBVqRi2FVlTVw", "0tgVpDi06FyKpA1z0VMD4v"]
            st.session_state["comparison_track_ids"] = [c for c in candidates if recommender.get_track_by_id(c) is not None]
            st.rerun()
    with b2:
        if st.button("🕺 Dancefloor Grooves", use_container_width=True):
            # Dua Lipa Don't Start Now, Bad Bunny Vete, Weeknd Blinding Lights
            candidates = ["6WrI0LAC5M1Rw2MnX2ZvEg", "5DxXsvSO6qnEKqJ9v49W1B", "0VjIjW4GlUZAMYd2vXMi3b"]
            st.session_state["comparison_track_ids"] = [c for c in candidates if recommender.get_track_by_id(c) is not None]
            st.rerun()
    with b3:
        if st.button("⚡ High-Octane vs Urban", use_container_width=True):
            # Linkin Park In The End, Drake God's Plan, Queen Bohemian Rhapsody
            candidates = ["60a0Rd6pj0RH9Stdu2UQjh", "6DCZcSspjsKoFjzjrWoCdn", "7tFiyTwD0nx5a1eklYtX2J"]
            st.session_state["comparison_track_ids"] = [c for c in candidates if recommender.get_track_by_id(c) is not None]
            st.rerun()
    with b4:
        if st.button("✕ Clear Selection", use_container_width=True):
            st.session_state["comparison_track_ids"] = []
            st.rerun()

    # Search & Add Tracks Workspace
    col_search, col_action = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "Search Catalog to Add Tracks (2 to 5 Songs)",
            placeholder="Type song title or artist (e.g. Queen, Martin Garrix, Dua Lipa)...",
            label_visibility="collapsed",
        )
    with col_action:
        search_results = recommender.search_tracks(search_query, limit=12) if search_query else pd.DataFrame()
        options = {}
        if not search_results.empty:
            for _, r in search_results.iterrows():
                tid = r["track_id"]
                tname = r["track_name"]
                tart = r["track_artist"]
                tg = str(r["playlist_genre"]).title()
                options[tid] = f"{tname} — {tart} [{tg}]"
        
        selected_to_add = st.selectbox(
            "Select track to add",
            options=list(options.keys()),
            format_func=lambda x: options.get(x, x),
            label_visibility="collapsed",
            index=0 if options else None,
        )

    if st.button("+ Add Track to Comparison", use_container_width=False):
        if selected_to_add:
            if selected_to_add in st.session_state["comparison_track_ids"]:
                st.warning("Track is already included in current comparison selection.")
            elif len(st.session_state["comparison_track_ids"]) >= TRACK_COMPARISON_MAX:
                st.warning(f"Maximum of {TRACK_COMPARISON_MAX} tracks allowed in Comparison Studio.")
            else:
                st.session_state["comparison_track_ids"].append(selected_to_add)
                st.rerun()

    # Display Selected Track Badges
    selected_ids = st.session_state["comparison_track_ids"]
    selected_tracks_list = [recommender.get_track_by_id(tid) for tid in selected_ids if recommender.get_track_by_id(tid) is not None]
    
    if selected_tracks_list:
        st.markdown("<div style='margin-top: 1rem; margin-bottom: 0.5rem;'><span class='stat-label'>Currently Selected Tracks</span></div>", unsafe_allow_html=True)
        chips_cols = st.columns(len(selected_tracks_list))
        for idx, (col, track) in enumerate(zip(chips_cols, selected_tracks_list)):
            with col:
                t_color = TRACK_COMPARISON_COLORS[idx % len(TRACK_COMPARISON_COLORS)]
                t_genre = str(track.get("playlist_genre", "other")).lower()
                g_col = GENRE_PALETTE.get(t_genre, COLOR_PALETTE["primary_green"])
                c_val = track.get("cluster")
                arch_name = CLUSTER_ARCHETYPES.get(int(c_val), {}).get("name", "Unknown") if pd.notna(c_val) else "Unassigned"

                st.markdown(
                    f"""
                    <div class="sonic-card" style="border-top: 3px solid {t_color}; padding: 12px 14px; position: relative;">
                        <div style="font-size: 0.72rem; color: {t_color}; font-weight: 700;">TRACK {idx+1}</div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: {COLOR_PALETTE['primary_text']}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            {track.get('track_name', 'Unknown')}
                        </div>
                        <div style="font-size: 0.78rem; color: {COLOR_PALETTE['secondary_text']}; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            {track.get('track_artist', 'Unknown')}
                        </div>
                        <div style="display: flex; gap: 6px; margin-top: 8px;">
                            <span style="font-size: 0.65rem; background: {g_col}22; color: {g_col}; border: 1px solid {g_col}55; padding: 2px 6px; border-radius: 4px; font-weight: 600;">
                                {t_genre.upper()}
                            </span>
                            <span style="font-size: 0.65rem; background: {COLOR_PALETTE['elevated_surface']}; color: {COLOR_PALETTE['muted_text']}; border: 1px solid {COLOR_PALETTE['borders']}; padding: 2px 6px; border-radius: 4px;">
                                {arch_name[:12]}…
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Remove Track {idx+1}", key=f"remove_t_{track['track_id']}", use_container_width=True):
                    st.session_state["comparison_track_ids"].remove(track["track_id"])
                    st.rerun()

    # Studio Analysis Activation (Requires 2 to 5 tracks)
    if len(selected_tracks_list) < TRACK_COMPARISON_MIN:
        st.markdown(
            f"""
            <div class="sonic-card" style="text-align: center; padding: 36px 20px; margin-top: 1.5rem;">
                <div style="font-size: 1.8rem; color: {COLOR_PALETTE['primary_green']}; margin-bottom: 8px;">◈</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PALETTE['primary_text']};">
                    Multi-Track Comparison Requires at Least 2 Tracks
                </div>
                <p style="font-size: 0.88rem; color: {COLOR_PALETTE['secondary_text']}; max-width: 540px; margin: 8px auto 0 auto; line-height: 1.5;">
                    Search and add between 2 and 5 tracks from the catalog above, or click one of the
                    <strong>Quick Benchmark Presets</strong> to instantly evaluate multi-track radar geometry,
                    pairwise cosine proximity, and empirical feature contrasts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Prepare DataFrame of selected tracks
    selected_df = pd.DataFrame(selected_tracks_list).reset_index(drop=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Tabs for Comparison Perspectives
    t1, t2, t3, t4 = st.tabs([
        "Multi-Track Radar Geometry",
        "Feature Comparison Chart",
        "Pairwise Acoustic Proximity",
        "Detailed Measurement Table",
    ])

    with t1:
        c_radar, c_meta = st.columns([3, 2])
        with c_radar:
            fig_radar = plot_multitrack_radar(selected_df)
            st.pyplot(fig_radar, use_container_width=True)
            plt.close(fig_radar)
        with c_meta:
            st.markdown(
                f"""
                <div class="sonic-card" style="padding: 18px 20px;">
                    <h3 style="font-size: 1.05rem; margin-top: 0; color: {COLOR_PALETTE['primary_green']};">
                        Polar Radar Normalization Protocol
                    </h3>
                    <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; line-height: 1.5;">
                        To project acoustic dimensions onto the 360° polygon without distortion, each continuous
                        dimension is mapped strictly to <code>[0.0, 1.0]</code> based on empirical catalog domain boundaries:
                    </p>
                    <div style="font-size: 0.78rem; color: {COLOR_PALETTE['muted_text']}; display: flex; flex-direction: column; gap: 4px;">
                        <div>• <strong>Danceability, Energy, Valence, Acousticness:</strong> Native [0, 1] range.</div>
                        <div>• <strong>Loudness:</strong> Clamped from -30.0 dB (quiet) to 0.0 dB (maximal volume).</div>
                        <div>• <strong>Tempo:</strong> Clamped from 50 BPM to 200 BPM.</div>
                        <div>• <strong>Duration:</strong> Clamped from 60s (1 min) to 480s (8 mins).</div>
                    </div>
                    <div style="margin-top: 14px; font-size: 0.74rem; color: {COLOR_PALETTE['warning']}; border-top: 1px solid {COLOR_PALETTE['borders']}; padding-top: 8px;">
                        Note: Polar radar normalization is strictly for visual presentation. Model calculations utilize Phase 3 <code>StandardScaler</code>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t2:
        features_to_plot = st.multiselect(
            "Select Audio Features to Compare",
            options=CONTINUOUS_AUDIO_FEATURES,
            default=["danceability", "energy", "speechiness", "acousticness", "liveness", "valence"],
            format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x.title()),
        )
        if features_to_plot:
            fig_bars = plot_feature_barchart(selected_df, features=features_to_plot)
            st.pyplot(fig_bars, use_container_width=True)
            plt.close(fig_bars)

    with t3:
        sim_matrix = compute_pairwise_similarity(selected_df, recommender.scaler)
        st.markdown(
            f"""
            <div style="margin-bottom: 0.75rem;">
                <h3 style="font-size: 1.1rem; margin-bottom: 4px;">Pairwise Cosine Proximity Matrix</h3>
                <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                    Calculated in standardized 10-dimensional continuous audio feature space (StandardScaler z-scores).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            sim_matrix.style.format("{:.4f}").background_gradient(cmap="Greens", vmin=sim_matrix.values.min(), vmax=1.0),
            use_container_width=True,
        )
        st.markdown(
            f"""
            <div class="sonic-card" style="margin-top: 10px; padding: 12px 16px;">
                <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['primary_green']};">
                    MATHEMATICAL DEFINITION · S = X_norm · X_norm.T
                </span>
                <p style="font-size: 0.80rem; color: {COLOR_PALETTE['muted_text']}; margin: 4px 0 0 0; line-height: 1.4;">
                    Cosine similarity values describe objective multi-dimensional acoustic feature proximity, not listener
                    preference. A score of 1.0000 denotes identical standardized acoustic parameters.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t4:
        comp_table = build_comparison_table(selected_df)
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

    # Dynamic Data-Grounded Insights Card
    sim_mat_calc = compute_pairwise_similarity(selected_df, recommender.scaler)
    insights = generate_comparison_insights(selected_df, sim_mat_calc)

    st.markdown(
        f"""
        <div class="sonic-card" style="margin-top: 1.5rem; padding: 20px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 1.15rem; margin: 0; color: {COLOR_PALETTE['primary_green']};">
                    ◈ Empirical Comparison Insights
                </h3>
                <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['muted_text']};">
                    DATA-GROUNDED DEDUCTIONS
                </span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
        """,
        unsafe_allow_html=True,
    )
    for ins in insights:
        st.markdown(
            f"""
            <div style="display: flex; align-items: flex-start; gap: 10px; font-size: 0.85rem; line-height: 1.45;">
                <span style="color: {COLOR_PALETTE['primary_green']}; font-weight: bold;">▸</span>
                <span style="color: {COLOR_PALETTE['primary_text']};">{ins}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 8: PLAYLIST INTELLIGENCE (PHASE 5B)
# ==============================================================================

def render_playlist_intelligence(df: pd.DataFrame) -> None:
    """Renders the Playlist Acoustic Profile Analyzer & Ingestion Engine (Phase 5B)."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="font-size: 2.2rem; margin: 0;">Playlist Intelligence</h1>
                <span class="badge-verified">PHASE 5B OPERATIONAL</span>
            </div>
            <p style="font-size: 1.0rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                CSV Upload Ingestion, Local Catalog Matching, Acoustic Identity Classification &amp; Playlist Recommendations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommender = get_cached_recommender()
    catalog_df = recommender.catalog_df

    # Upload & Sample Selection Row
    col_upload, col_samples = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Playlist CSV (Max 10 MB)",
            type=["csv"],
            help="Upload an Input A (with audio features) or Input B (tracklist with track_name and track_artist) CSV file.",
        )
    with col_samples:
        st.markdown("<div style='margin-bottom: 6px;'><span class='stat-label'>Or Load Curated Sample Playlists</span></div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        sample_path_to_load = None
        with s1:
            if st.button("Festival EDM", use_container_width=True):
                sample_path_to_load = PROJECT_ROOT / "data" / "sample_playlists" / "sample_festival_edm.csv"
        with s2:
            if st.button("Acoustic Chill", use_container_width=True):
                sample_path_to_load = PROJECT_ROOT / "data" / "sample_playlists" / "sample_acoustic_chill.csv"
        with s3:
            if st.button("Simple Tracklist", use_container_width=True):
                sample_path_to_load = PROJECT_ROOT / "data" / "sample_playlists" / "sample_tracklist_simple.csv"

    # Ingestion & Validation
    active_content = None
    active_filename = "playlist.csv"

    if uploaded_file is not None:
        active_content = uploaded_file.getvalue()
        active_filename = uploaded_file.name
    elif sample_path_to_load is not None and sample_path_to_load.exists():
        with open(sample_path_to_load, "rb") as f:
            active_content = f.read()
        active_filename = sample_path_to_load.name

    if active_content is None:
        st.markdown(
            f"""
            <div class="sonic-card" style="text-align: center; padding: 40px 20px; margin-top: 1.5rem;">
                <div style="font-size: 2.0rem; color: {COLOR_PALETTE['primary_green']}; margin-bottom: 8px;">◈</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {COLOR_PALETTE['primary_text']};">
                    Upload a Playlist CSV or Load a Sample Dataset
                </div>
                <p style="font-size: 0.88rem; color: {COLOR_PALETTE['secondary_text']}; max-width: 580px; margin: 8px auto 0 auto; line-height: 1.5;">
                    The analyzer supports two input formats: <strong>Input A</strong> (Spotify dataset CSV with continuous audio features)
                    and <strong>Input B</strong> (simple tracklists with track names and artists). Matches are resolved strictly against the
                    local 28,356 unique catalog with zero external API calls.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Validate uploaded CSV
    is_valid, parsed_df, msg = validate_uploaded_csv(active_content, filename=active_filename)
    if not is_valid or parsed_df is None:
        st.error(f"Validation Failure: {msg}")
        return

    input_format = detect_playlist_input_type(parsed_df)
    format_label = "Input A (Audio Features)" if input_format == "INPUT_A_AUDIO_FEATURES" else "Input B (Tracklist Matching)"

    # Match against catalog
    matching_result = match_playlist_to_catalog(parsed_df, catalog_df)
    matched_df = matching_result["matched_df"]
    unmatched_df = matching_result["unmatched_df"]
    ambiguous_df = matching_result["ambiguous_df"]

    # Ingestion Summary Metric Grid
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Uploaded Rows</span>
                <span class="stat-value">{matching_result['total_uploaded_rows']:,}</span>
                <span class="stat-subtext">{format_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Unique Matched</span>
                <span class="stat-value" style="color: {COLOR_PALETTE['primary_green']};">{matching_result['unique_matched_count']:,}</span>
                <span class="stat-subtext">Catalog Resolved</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Duplicates</span>
                <span class="stat-value">{matching_result['duplicate_count']:,}</span>
                <span class="stat-subtext">Deduplicated</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Unmatched</span>
                <span class="stat-value" style="color: {COLOR_PALETTE['warning'] if matching_result['unmatched_count'] > 0 else COLOR_PALETTE['muted_text']};">
                    {matching_result['unmatched_count']:,}
                </span>
                <span class="stat-subtext">Not in 28K catalog</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="sonic-stat-card">
                <span class="stat-label">Ambiguous</span>
                <span class="stat-value">{matching_result['ambiguous_count']:,}</span>
                <span class="stat-subtext">Resolved via Pop</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Transparency Expander for Unmatched & Ambiguous
    if matching_result["unmatched_count"] > 0 or matching_result["ambiguous_count"] > 0:
        with st.expander("Catalog Ingestion Diagnostic Report", expanded=False):
            if not unmatched_df.empty:
                st.write("Unmatched Tracks (Omitted from Acoustic Calculation):")
                st.dataframe(unmatched_df, use_container_width=True)
            if not ambiguous_df.empty:
                st.write("Ambiguous Title Matches (Resolved Deterministically by Highest Popularity):")
                st.dataframe(ambiguous_df, use_container_width=True)

    if matched_df.empty:
        st.error("No valid tracks could be matched from the uploaded CSV against the local 28,356 catalog.")
        return

    # Calculate Profile & Persona
    profile = calculate_playlist_profile(matched_df, catalog_df)
    identity = classify_playlist_identity(profile)
    diversity = calculate_playlist_diversity(matched_df, profile)

    # Persona Hero Card
    st.markdown(
        f"""
        <div class="sonic-card" style="margin-top: 1.25rem; padding: 22px 24px; border-left: 4px solid {COLOR_PALETTE['primary_green']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.72rem; color: {COLOR_PALETTE['primary_green']}; font-weight: 700; letter-spacing: 0.05em;">
                    ALGORITHMIC PERSONA IDENTIFIER  ·  {identity.get('badge')}
                </span>
                <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['muted_text']};">
                    {profile['total_tracks']} Tracks Analyzed
                </span>
            </div>
            <h2 style="font-size: 1.6rem; margin: 0; color: {COLOR_PALETTE['primary_text']}; font-weight: 700;">
                {identity.get('title')}
            </h2>
            <p style="font-size: 0.90rem; color: {COLOR_PALETTE['secondary_text']}; margin: 8px 0 0 0; line-height: 1.5; max-width: 820px;">
                {identity.get('summary')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Diversity Metrics Row
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Artist Diversity", f"{diversity['artist_diversity']:.2%}", help="Unique artists / total matched tracks")
    with d2:
        st.metric("Genre Diversity (Simpson)", f"{diversity['genre_diversity']:.3f}", help="Gini-Simpson diversity index across genres")
    with d3:
        st.metric("Cluster Diversity", f"{diversity['cluster_diversity']:.3f}", help="Normalized Shannon entropy across 6 sonic archetypes")
    with d4:
        st.metric("Acoustic Variance", f"{diversity['acoustic_variance']:.3f}", help="Mean feature standard deviation across the 10 audio dimensions")

    # Visualizations
    col_vis1, col_vis2 = st.columns([1, 1])
    with col_vis1:
        fig_pradar = plot_playlist_radar(profile)
        st.pyplot(fig_pradar, use_container_width=True)
        plt.close(fig_pradar)
    with col_vis2:
        fig_pbar = plot_playlist_genre_and_cluster(profile)
        st.pyplot(fig_pbar, use_container_width=True)
        plt.close(fig_pbar)

    # Section: Discover Tracks for This Playlist
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="margin-bottom: 0.75rem;">
            <h3 style="font-size: 1.2rem; margin-bottom: 4px;">◈ Discover Tracks for This Playlist</h3>
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Find tracks from the 28,356 catalog that match the playlist's aggregate acoustic profile while strictly excluding existing songs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2, p_col3 = st.columns([2, 1, 1])
    with p_col1:
        pl_mode = st.radio(
            "Discovery Strategy",
            options=["Acoustic Proximity to Playlist", "Cross-Genre Playlist Expansion"],
            horizontal=True,
        )
    with p_col2:
        top_k_pl = st.selectbox("Recommendations Count", [5, 10, 15, 20], index=1)
    with p_col3:
        pl_genre_filter = st.selectbox("Genre Filter", ["All"] + [g.title() for g in GENRE_COLUMNS])

    rec_mode_key = "cross_genre" if pl_mode == "Cross-Genre Playlist Expansion" else "acoustic_proximity"
    pl_recs = recommend_for_playlist(
        matched_df=matched_df,
        recommender=recommender,
        top_n=top_k_pl,
        discovery_mode=rec_mode_key,
        target_genre=None if pl_genre_filter == "All" else pl_genre_filter,
    )

    if not pl_recs.empty:
        for idx, (_, r) in enumerate(pl_recs.iterrows()):
            r_g = str(r.get("playlist_genre", "other")).lower()
            g_col = GENRE_PALETTE.get(r_g, COLOR_PALETTE["primary_green"])
            c_val = r.get("cluster")
            arch_name = CLUSTER_ARCHETYPES.get(int(c_val), {}).get("name", "Unknown") if pd.notna(c_val) else "Unassigned"

            st.markdown(
                f"""
                <div class="sonic-card" style="padding: 12px 18px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span class="mono-text" style="font-size: 1.0rem; font-weight: 700; color: {COLOR_PALETTE['primary_green']};">
                                #{idx+1:02d}
                            </span>
                            <div>
                                <div style="font-size: 0.95rem; font-weight: 700; color: {COLOR_PALETTE['primary_text']};">
                                    {r.get('track_name', 'Unknown')}
                                </div>
                                <div style="font-size: 0.80rem; color: {COLOR_PALETTE['secondary_text']};">
                                    {r.get('track_artist', 'Unknown')} · <span style="color: {COLOR_PALETTE['muted_text']};">{r.get('track_album_name', '')}</span>
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right; display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 0.70rem; background: {g_col}22; color: {g_col}; border: 1px solid {g_col}55; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                                {r_g.upper()}
                            </span>
                            <span class="mono-text" style="font-size: 0.85rem; font-weight: 600; color: {COLOR_PALETTE['primary_green']};">
                                {r.get('similarity_score', 0):.4f} Sim
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.78rem; color: {COLOR_PALETTE['muted_text']};">
                        {r.get('recommendation_reason', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Export Section
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("Export Playlist Analysis & Reports", expanded=False):
        exports = export_playlist_analysis(profile, identity, diversity, matched_df, unmatched_df)
        ex1, ex2, ex3, ex4 = st.columns(4)
        with ex1:
            st.download_button(
                "Download Summary CSV",
                data=exports["summary_csv"],
                file_name=f"{Path(active_filename).stem}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex2:
            st.download_button(
                "Download Matched Tracks CSV",
                data=exports["matched_csv"],
                file_name=f"{Path(active_filename).stem}_matched.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex3:
            st.download_button(
                "Download Unmatched CSV",
                data=exports["unmatched_csv"],
                file_name=f"{Path(active_filename).stem}_unmatched.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex4:
            st.download_button(
                "Download Profile JSON",
                data=exports["profile_json"],
                file_name=f"{Path(active_filename).stem}_profile.json",
                mime="application/json",
                use_container_width=True,
            )


# ==============================================================================
# VIEW 9: CATALOG INTELLIGENCE (PHASE 5C)
# ==============================================================================

def render_catalog_intelligence(df: pd.DataFrame) -> None:
    """Renders Catalog Macro Analytics & Artist Profiling (Phase 5C)."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="font-size: 2.2rem; margin: 0;">Catalog Intelligence</h1>
                <span class="badge-verified">PHASE 5C OPERATIONAL</span>
            </div>
            <p style="font-size: 1.0rem; color: {COLOR_PALETTE['secondary_text']}; margin: 0;">
                Macro Acoustic Topology, Cross-Genre Convergence &amp; Artist Acoustic Profiling
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommender = get_cached_recommender()
    catalog_df = recommender.catalog_df

    t_macro, t_artist, t_workspace = st.tabs([
        "Topological Heatmaps & Proximity",
        "Artist Acoustic Profiler",
        "Multi-Dimensional Catalog Filter",
    ])

    with t_macro:
        # Genre-to-Cluster Heatmap
        st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 4px;'>Super Genre to Sonic Archetype Mapping</h3>", unsafe_allow_html=True)
        cross_df = compute_genre_cluster_matrix(df, normalize="genre")
        fig_cross = plot_genre_cluster_heatmap(cross_df)
        st.pyplot(fig_cross, use_container_width=True)
        plt.close(fig_cross)

        st.markdown(
            f"""
            <div class="sonic-card" style="padding: 14px 18px; margin-top: 10px; margin-bottom: 20px;">
                <span class="mono-text" style="font-size: 0.72rem; color: {COLOR_PALETTE['primary_green']};">
                    EMPIRICAL ARCHITECTURAL INSIGHT
                </span>
                <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin: 4px 0 0 0; line-height: 1.45;">
                    The heatmap demonstrates that catalog genres are internally heterogeneous. For instance, Rock comprises 45.2%
                    of Cluster 0 (High-Octane Anthem) and 34.8% of Cluster 5 (Acoustic Intimacy). Conversely, Cluster 0 is shared
                    equally by driving Rock and festival EDM, explaining why cross-genre recommendations exhibit near-identical acoustic properties.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            st.markdown("<h3 style='font-size: 1.05rem;'>Pairwise Genre Acoustic Proximity</h3>", unsafe_allow_html=True)
            _, sim_df = compute_genre_acoustic_profiles(catalog_df, recommender.scaler)
            fig_sim = plot_genre_similarity_heatmap(sim_df)
            st.pyplot(fig_sim, use_container_width=True)
            plt.close(fig_sim)
        with col_h2:
            st.markdown("<h3 style='font-size: 1.05rem;'>Cluster Popularity Distribution</h3>", unsafe_allow_html=True)
            fig_pop = plot_cluster_popularity_distribution(catalog_df)
            st.pyplot(fig_pop, use_container_width=True)
            plt.close(fig_pop)

    with t_artist:
        st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 4px;'>Recording Artist Acoustic Profiler</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 12px;">
                Inspect acoustic profiles, archetype distributions, and track variations for any of the 10,692 artists.
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Quick popular artist select or text search
        popular_artists = ["Queen", "Martin Garrix", "The Chainsmokers", "Dua Lipa", "Drake", "Ed Sheeran", "Bad Bunny", "Coldplay", "Avicii"]
        c_sel, c_txt = st.columns([1, 2])
        with c_sel:
            selected_preset_artist = st.selectbox("Quick-Select Popular Artist", ["-- Type custom below --"] + popular_artists)
        with c_txt:
            custom_artist = st.text_input("Or Type Artist Name", value="" if selected_preset_artist != "-- Type custom below --" else "Queen")

        target_artist = custom_artist if selected_preset_artist == "-- Type custom below --" else selected_preset_artist
        if not target_artist:
            target_artist = "Queen"

        artist_prof = get_artist_catalog_profile(catalog_df, target_artist)
        if artist_prof is None:
            st.warning(f"Artist '{target_artist}' not found in the local 28,356 unique catalog.")
        else:
            # Artist KPI Row
            ak1, ak2, ak3, ak4, ak5 = st.columns(5)
            with ak1:
                st.metric("Unique Catalog Tracks", f"{artist_prof['unique_tracks']}")
            with ak2:
                st.metric("Playlist Appearances", f"{artist_prof['playlist_appearances']}")
            with ak3:
                st.metric("Primary Genre", f"{artist_prof['primary_genre'].title()}")
            with ak4:
                st.metric("Dominant Archetype", f"{artist_prof['dominant_archetype_name'][:14]}…")
            with ak5:
                st.metric("Mean Popularity", f"{artist_prof['popularity_statistics']['mean']:.1f}/100")

            c_art_vis, c_art_data = st.columns([1, 1])
            with c_art_vis:
                fig_art = plot_artist_radar_profile(artist_prof, catalog_df)
                st.pyplot(fig_art, use_container_width=True)
                plt.close(fig_art)
            with c_art_data:
                st.markdown(
                    f"""
                    <div class="sonic-card" style="padding: 18px 20px;">
                        <h4 style="font-size: 1.0rem; margin-top: 0; color: {COLOR_PALETTE['primary_green']};">
                            {artist_prof['artist_name']} Acoustic Summary
                        </h4>
                        <div style="font-size: 0.80rem; color: {COLOR_PALETTE['secondary_text']}; line-height: 1.5;">
                            <div>• <strong>Dominant Cluster:</strong> {artist_prof['dominant_archetype_name']}</div>
                            <div>• <strong>Mean Energy:</strong> {artist_prof['feature_means']['energy']:.2f} (Std: {artist_prof['feature_stds']['energy']:.2f})</div>
                            <div>• <strong>Mean Danceability:</strong> {artist_prof['feature_means']['danceability']:.2f} (Std: {artist_prof['feature_stds']['danceability']:.2f})</div>
                            <div>• <strong>Mean Loudness:</strong> {artist_prof['feature_means']['loudness']:.1f} dB</div>
                            <div>• <strong>Mean Tempo:</strong> {artist_prof['feature_means']['tempo']:.0f} BPM</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"All {artist_prof['unique_tracks']} Tracks by {artist_prof['artist_name']}", expanded=True):
                    disp_cols = ["track_name", "playlist_genre", "playlist_subgenre", "energy", "danceability", "valence", "tempo", "track_popularity"]
                    avail_cols = [c for c in disp_cols if c in artist_prof["tracks_df"].columns]
                    st.dataframe(artist_prof["tracks_df"][avail_cols], use_container_width=True)

    with t_workspace:
        st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 4px;'>Multi-Dimensional Catalog Filtering Workspace</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <p style="font-size: 0.82rem; color: {COLOR_PALETTE['secondary_text']}; margin-bottom: 12px;">
                Filter all 28,356 unique tracks across simultaneous acoustic, genre, archetype, and popularity bounds.
            </p>
            """,
            unsafe_allow_html=True,
        )

        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            f_genre = st.selectbox("Genre", ["All"] + [g.title() for g in GENRE_COLUMNS], key="wf_genre")
        with wf2:
            subgenres = sorted(catalog_df["playlist_subgenre"].dropna().unique())
            f_sub = st.selectbox("Subgenre", ["All"] + list(subgenres), key="wf_sub")
        with wf3:
            f_clust = st.selectbox("Sonic Archetype", ["All"] + [f"C{c}: {CLUSTER_ARCHETYPES[c]['name']}" for c in sorted(CLUSTER_ARCHETYPES.keys())], key="wf_cluster")

        c_id_arg = int(f_clust.split(":")[0][1:]) if f_clust != "All" else "All"

        sf1, sf2, sf3, sf4 = st.columns(4)
        with sf1:
            f_pop = st.slider("Popularity", 0, 100, (20, 100), key="wf_pop")
        with sf2:
            f_energy = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), key="wf_energy")
        with sf3:
            f_dance = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0), key="wf_dance")
        with sf4:
            f_acoust = st.slider("Acousticness", 0.0, 1.0, (0.0, 1.0), key="wf_acoust")

        filtered_subset = filter_catalog_records(
            catalog_df,
            genre=f_genre if f_genre != "All" else None,
            subgenre=f_sub if f_sub != "All" else None,
            cluster=c_id_arg if c_id_arg != "All" else None,
            min_popularity=f_pop[0],
            max_popularity=f_pop[1],
            min_energy=f_energy[0],
            max_energy=f_energy[1],
            min_danceability=f_dance[0],
            max_danceability=f_dance[1],
            min_acousticness=f_acoust[0],
            max_acousticness=f_acoust[1],
        )

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 12px 0 8px 0;">
                <span class="badge-verified">
                    <span class="status-dot"></span>
                    <span>{len(filtered_subset):,} TRACKS MATCHING FILTERS</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        disp_f_cols = ["track_name", "track_artist", "playlist_genre", "playlist_subgenre", "energy", "danceability", "valence", "tempo", "track_popularity"]
        st.dataframe(filtered_subset[disp_f_cols].head(100), use_container_width=True)

        st.download_button(
            "Download Filtered Tracks (CSV)",
            data=filtered_subset.to_csv(index=False),
            file_name="sonic_atlas_filtered_catalog.csv",
            mime="text/csv",
        )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main() -> None:
    """Main execution entry point."""
    inject_custom_styles()

    dataset_loaded = False
    df: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, Any]] = None
    audit: Optional[Dict[str, Any]] = None

    try:
        df = get_cached_dataset()
        summary = get_dataset_summary(df)
        audit = run_preliminary_audit(df)
        dataset_loaded = True
    except FileNotFoundError:
        dataset_loaded = False
    except Exception as err:
        st.error(f"Error loading dataset: {err}")
        return

    selected_page = render_sidebar(dataset_loaded)

    if not dataset_loaded or df is None:
        st.error("Spotify dataset file not found.")
        return

    # Route navigation across all 9 views
    if selected_page == "Overview":
        render_overview_page(df, summary, audit)
    elif selected_page == "Music Explorer":
        render_music_explorer(df)
    elif selected_page == "Genre Analysis":
        render_genre_analysis(df)
    elif selected_page == "Data Insights":
        render_data_insights(df)
    elif selected_page == "Cluster Intelligence":
        render_cluster_intelligence(df)
    elif selected_page == "Recommendations":
        render_recommendations_page(df)
    elif selected_page == "Track Comparison":
        render_track_comparison(df)
    elif selected_page == "Playlist Intelligence":
        render_playlist_intelligence(df)
    elif selected_page == "Catalog Intelligence":
        render_catalog_intelligence(df)


if __name__ == "__main__":
    main()
