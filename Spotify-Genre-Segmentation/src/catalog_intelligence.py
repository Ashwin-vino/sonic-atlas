"""
SONIC ATLAS — Catalog Intelligence & Macro Acoustic Analytics Module
Phase 5C Implementation: Genre Acoustic Overlap, Subgenre Differentiation,
Genre-to-Cluster Topological Mapping Heatmaps, Artist Acoustic Profiling,
Cluster Popularity Distributions, and Multi-Dimensional Catalog Filtering.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTER_ARCHETYPES,
    CLUSTER_PALETTE,
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    GENRE_COLUMNS,
    GENRE_PALETTE,
    RADAR_FEATURE_BOUNDS,
    RECOMMENDATION_CATALOG_PATH,
    TYPOGRAPHY,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. GENRE-TO-CLUSTER TOPOLOGICAL MAPPING HEATMAP
# ==============================================================================

def compute_genre_cluster_matrix(
    df: pd.DataFrame,
    normalize: str = "genre",
) -> pd.DataFrame:
    """
    Computes the 6 Super-Genre x 6 Sonic Archetype cross-tabulation matrix.

    Args:
        df: Catalog or clustered dataset containing 'playlist_genre' and 'cluster'.
        normalize: 'genre' (rows sum to 100%), 'cluster' (columns sum to 100%), or 'all'.

    Returns:
        pd.DataFrame: Formatted percentage matrix (6 rows x 6 columns).
    """
    if "playlist_genre" not in df.columns or "cluster" not in df.columns:
        raise KeyError("DataFrame must contain 'playlist_genre' and 'cluster' columns.")

    clean_df = df.dropna(subset=["playlist_genre", "cluster"]).copy()
    clean_df["cluster"] = clean_df["cluster"].astype(int)

    norm_arg = "index" if normalize == "genre" else ("columns" if normalize == "cluster" else "all")
    cross = pd.crosstab(
        clean_df["playlist_genre"].str.title(),
        clean_df["cluster"],
        normalize=norm_arg,
    ) * 100.0

    # Rename cluster columns to archetype names
    cluster_cols = {}
    for c in cross.columns:
        arch_name = CLUSTER_ARCHETYPES.get(int(c), {}).get("name", f"Cluster {c}")
        cluster_cols[c] = f"C{c}: {arch_name}"
    cross = cross.rename(columns=cluster_cols)

    return cross.round(2)


def plot_genre_cluster_heatmap(cross_df: pd.DataFrame) -> plt.Figure:
    """
    Renders a heatmap displaying genre composition across the 6 sonic archetypes
    using the SONIC ATLAS dark editorial theme.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    # Create dark green gradient colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#141C17", "#1B261F", "#23865B", "#55D99A"]
    cmap = LinearSegmentedColormap.from_list("sonic_green_map", colors, N=256)

    sns.heatmap(
        cross_df,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar_kws={"label": "Share (%)"},
        linewidths=1.0,
        linecolor=COLOR_PALETTE["borders"],
        ax=ax,
        annot_kws={"size": 10.5, "weight": "bold", "color": COLOR_PALETTE["primary_text"]},
    )

    ax.set_title(
        "Genre-to-Archetype Distribution (Rows sum to 100%)",
        color=COLOR_PALETTE["primary_text"],
        fontsize=13,
        fontweight=700,
        pad=16,
    )
    ax.set_xlabel("Discovered Sonic Archetypes (Unsupervised K-Means)", color=COLOR_PALETTE["secondary_text"], fontsize=10, labelpad=10)
    ax.set_ylabel("Curated Super Genre", color=COLOR_PALETTE["secondary_text"], fontsize=10, labelpad=10)

    ax.tick_params(colors=COLOR_PALETTE["primary_text"], labelsize=9.5)
    plt.xticks(rotation=22, ha="right")
    plt.yticks(rotation=0)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=COLOR_PALETTE["secondary_text"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLOR_PALETTE["secondary_text"])
    cbar.outline.set_edgecolor(COLOR_PALETTE["borders"])

    plt.tight_layout()
    return fig


# ==============================================================================
# 2. GENRE ACOUSTIC DISTANCE & CONVERGENCE MATRIX
# ==============================================================================

def compute_genre_acoustic_profiles(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the 10-dimensional mean audio feature profile for each of the 6 super-genres,
    along with their pairwise cosine similarity matrix.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (genre_mean_features, genre_similarity_matrix)
    """
    clean_df = df.dropna(subset=["playlist_genre"]).copy()
    grouped = clean_df.groupby("playlist_genre")[CONTINUOUS_AUDIO_FEATURES].mean()
    grouped.index = grouped.index.str.title()

    # Standardize to compute true acoustic distance
    X_raw = grouped.values
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)
    else:
        # Standardize within genres
        std_scaler = StandardScaler()
        X_scaled = std_scaler.fit_transform(X_raw)

    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_scaled / norms

    sim_mat = np.dot(X_norm, X_norm.T)
    sim_df = pd.DataFrame(sim_mat, index=grouped.index, columns=grouped.index).round(4)

    return grouped.round(3), sim_df


def plot_genre_similarity_heatmap(sim_df: pd.DataFrame) -> plt.Figure:
    """Renders the pairwise acoustic similarity heatmap between genres."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#141C17", "#23865B", "#55D99A"]
    cmap = LinearSegmentedColormap.from_list("genre_sim_map", colors, N=256)

    sns.heatmap(
        sim_df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=sim_df.values.min(),
        vmax=1.0,
        linewidths=1.0,
        linecolor=COLOR_PALETTE["borders"],
        ax=ax,
        annot_kws={"size": 10, "weight": "600", "color": COLOR_PALETTE["primary_text"]},
    )

    ax.set_title("Pairwise Genre Acoustic Proximity (Cosine Similarity)", color=COLOR_PALETTE["primary_text"], fontsize=12, fontweight=700, pad=14)
    ax.tick_params(colors=COLOR_PALETTE["primary_text"], labelsize=10)

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor(COLOR_PALETTE["borders"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLOR_PALETTE["secondary_text"])

    plt.tight_layout()
    return fig


# ==============================================================================
# 3. CLUSTER POPULARITY DISTRIBUTION
# ==============================================================================

def plot_cluster_popularity_distribution(df: pd.DataFrame) -> plt.Figure:
    """Renders a violin/box plot of track popularity across the 6 sonic archetypes."""
    clean_df = df.dropna(subset=["cluster", "track_popularity"]).copy()
    clean_df["cluster"] = clean_df["cluster"].astype(int)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    archetype_labels = [
        f"C{c}\n{CLUSTER_ARCHETYPES[c]['name'][:14]}…" if len(CLUSTER_ARCHETYPES[c]['name']) > 15 else f"C{c}\n{CLUSTER_ARCHETYPES[c]['name']}"
        for c in sorted(CLUSTER_ARCHETYPES.keys())
    ]
    palette = [CLUSTER_PALETTE.get(c, COLOR_PALETTE["primary_green"]) for c in sorted(CLUSTER_ARCHETYPES.keys())]

    sns.boxplot(
        data=clean_df,
        x="cluster",
        y="track_popularity",
        hue="cluster",
        legend=False,
        palette=palette,
        ax=ax,
        linewidth=1.2,
        fliersize=2,
        flierprops={"markerfacecolor": COLOR_PALETTE["muted_text"], "markeredgecolor": "none"},
    )

    ax.set_xticks(range(len(archetype_labels)))
    ax.set_xticklabels(archetype_labels, color=COLOR_PALETTE["primary_text"], fontsize=9.5)
    ax.set_xlabel("Sonic Archetype", color=COLOR_PALETTE["secondary_text"], fontsize=10, labelpad=10)
    ax.set_ylabel("Track Popularity (0–100)", color=COLOR_PALETTE["secondary_text"], fontsize=10, labelpad=10)
    ax.set_title("Commercial Popularity Distribution by Sonic Archetype", color=COLOR_PALETTE["primary_text"], fontsize=12, fontweight=700, pad=12)

    ax.tick_params(colors=COLOR_PALETTE["secondary_text"])
    ax.grid(axis="y", color=COLOR_PALETTE["borders"], linestyle=":", alpha=0.7)

    for spine in ax.spines.values():
        spine.set_color(COLOR_PALETTE["borders"])

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. ARTIST ACOUSTIC PROFILER
# ==============================================================================

def get_artist_catalog_profile(
    df: pd.DataFrame,
    artist_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Computes an artist's comprehensive acoustic profile across all catalog tracks.

    Metrics:
    - Number of unique tracks.
    - Total playlist appearances (preserving playlist context).
    - Genre distribution.
    - Dominant sonic archetype.
    - Archetype distribution.
    - Mean and standard deviation across all 10 continuous audio features.
    - Popularity distribution.

    Args:
        df: Recommendation catalog (unique tracks) or clustered dataset.
        artist_name: Artist name query string.

    Returns:
        Optional[Dict[str, Any]]: Profile dictionary or None if artist not found.
    """
    mask = df["track_artist"].astype(str).str.strip().str.lower() == str(artist_name).strip().lower()
    artist_df = df[mask]

    if artist_df.empty:
        # Try substring match
        sub_mask = df["track_artist"].astype(str).str.lower().str.contains(str(artist_name).strip().lower(), regex=False)
        artist_df = df[sub_mask]
        if artist_df.empty:
            return None

    # Get exact casing of canonical artist name
    canonical_name = artist_df["track_artist"].value_counts().index[0]
    canonical_df = artist_df[artist_df["track_artist"] == canonical_name]

    unique_tracks = canonical_df["track_id"].nunique()
    playlist_appearances = int(canonical_df["playlist_count"].sum()) if "playlist_count" in canonical_df.columns else len(canonical_df)

    # Genre breakdown
    genre_dist = canonical_df["playlist_genre"].value_counts(normalize=True).round(3).to_dict()
    primary_genre = list(genre_dist.keys())[0] if genre_dist else "Unknown"

    # Cluster breakdown
    cluster_dist = {}
    dominant_cluster = None
    if "cluster" in canonical_df.columns:
        c_series = canonical_df["cluster"].dropna().astype(int)
        if not c_series.empty:
            cluster_dist = c_series.value_counts(normalize=True).round(3).to_dict()
            dominant_cluster = int(c_series.mode()[0])

    # Audio features
    feature_means = canonical_df[CONTINUOUS_AUDIO_FEATURES].mean().round(3).to_dict()
    feature_stds = canonical_df[CONTINUOUS_AUDIO_FEATURES].std(ddof=0).round(3).to_dict()

    # Popularity stats
    pop_series = canonical_df["track_popularity"].astype(float)
    pop_stats = {
        "mean": round(float(pop_series.mean()), 1),
        "median": round(float(pop_series.median()), 1),
        "min": int(pop_series.min()),
        "max": int(pop_series.max()),
    }

    return {
        "artist_name": canonical_name,
        "unique_tracks": unique_tracks,
        "playlist_appearances": playlist_appearances,
        "primary_genre": primary_genre,
        "genre_distribution": genre_dist,
        "dominant_cluster": dominant_cluster,
        "dominant_archetype_name": CLUSTER_ARCHETYPES.get(dominant_cluster, {}).get("name", "Unknown") if dominant_cluster is not None else "Unknown",
        "cluster_distribution": cluster_dist,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "popularity_statistics": pop_stats,
        "tracks_df": canonical_df,
    }


def plot_artist_radar_profile(
    artist_profile: Dict[str, Any],
    catalog_df: pd.DataFrame,
) -> plt.Figure:
    """Renders a radar chart comparing an artist's mean audio features against the catalog average."""
    features = CONTINUOUS_AUDIO_FEATURES
    num_vars = len(features)

    art_means = artist_profile.get("feature_means", {})
    cat_means = catalog_df[features].mean()

    art_vals = []
    cat_vals = []
    for f in features:
        low, high = RADAR_FEATURE_BOUNDS.get(f, (0.0, 1.0))
        a_v = art_means.get(f, 0.5)
        c_v = cat_means.get(f, 0.5)
        art_vals.append(np.clip((a_v - low) / (high - low), 0.0, 1.0))
        cat_vals.append(np.clip((c_v - low) / (high - low), 0.0, 1.0))

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    art_vals += art_vals[:1]
    cat_vals += cat_vals[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    labels = [f.title() if f != "duration_ms" else "Duration" for f in features]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=COLOR_PALETTE["primary_text"], fontsize=9.5, fontweight=600)

    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=COLOR_PALETTE["muted_text"], fontsize=7.5, fontfamily="monospace")
    ax.set_ylim(0, 1.05)

    ax.grid(color=COLOR_PALETTE["borders"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color(COLOR_PALETTE["borders"])

    # Catalog Mean
    ax.plot(angles, cat_vals, color=COLOR_PALETTE["secondary_text"], linewidth=1.5, linestyle="--", label="Catalog Baseline")
    ax.fill(angles, cat_vals, color=COLOR_PALETTE["secondary_text"], alpha=0.08)

    # Artist Mean
    art_name = artist_profile.get("artist_name", "Artist")
    ax.plot(angles, art_vals, color=COLOR_PALETTE["primary_green"], linewidth=2.2, marker="o", markersize=4, label=f"{art_name} (Mean)")
    ax.fill(angles, art_vals, color=COLOR_PALETTE["primary_green"], alpha=0.22)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.30, 1.10),
        facecolor=COLOR_PALETTE["elevated_surface"],
        edgecolor=COLOR_PALETTE["borders"],
        labelcolor=COLOR_PALETTE["primary_text"],
        fontsize=8.5,
    )

    plt.tight_layout()
    return fig


# ==============================================================================
# 5. MULTI-DIMENSIONAL CATALOG FILTERING WORKSPACE
# ==============================================================================

def filter_catalog_records(
    df: pd.DataFrame,
    genre: Optional[str] = None,
    subgenre: Optional[str] = None,
    cluster: Optional[Union[int, str]] = None,
    min_popularity: int = 0,
    max_popularity: int = 100,
    min_energy: float = 0.0,
    max_energy: float = 1.0,
    min_danceability: float = 0.0,
    max_danceability: float = 1.0,
    min_acousticness: float = 0.0,
    max_acousticness: float = 1.0,
    min_tempo: float = 0.0,
    max_tempo: float = 250.0,
) -> pd.DataFrame:
    """
    Applies multi-dimensional acoustic and metadata filtering to the catalog.
    """
    mask = (
        (df["track_popularity"] >= min_popularity)
        & (df["track_popularity"] <= max_popularity)
        & (df["energy"] >= min_energy)
        & (df["energy"] <= max_energy)
        & (df["danceability"] >= min_danceability)
        & (df["danceability"] <= max_danceability)
        & (df["acousticness"] >= min_acousticness)
        & (df["acousticness"] <= max_acousticness)
        & (df["tempo"] >= min_tempo)
        & (df["tempo"] <= max_tempo)
    )

    if genre and genre.lower() != "all":
        mask = mask & (df["playlist_genre"].str.lower() == genre.lower())

    if subgenre and subgenre.lower() != "all":
        mask = mask & (df["playlist_subgenre"].str.lower() == subgenre.lower())

    if cluster is not None and str(cluster).lower() != "all":
        mask = mask & (df["cluster"] == int(cluster))

    return df[mask].sort_values(by="track_popularity", ascending=False)
