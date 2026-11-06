"""
SONIC ATLAS — Audio Feature & Genre Analysis Module
Phase 2 Implementation: Comprehensive Exploratory Data Analysis (EDA),
statistical distributions, genre profiling, bivariate relationships, and
publication-grade visualization generator.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Safe writable matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    DISCRETE_MUSICAL_FEATURES,
    FIGURES_DIR,
    GENRE_COLUMNS,
    GENRE_PALETTE,
    METADATA_NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. DESIGN SYSTEM STYLING CONFIGURATION
# ==============================================================================

def setup_visual_style() -> None:
    """Configures Matplotlib and Seaborn to adhere strictly to the SONIC ATLAS design tokens."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": COLOR_PALETTE["main_background"],
        "axes.facecolor": COLOR_PALETTE["main_surface"],
        "axes.edgecolor": COLOR_PALETTE["borders"],
        "axes.labelcolor": COLOR_PALETTE["primary_text"],
        "xtick.color": COLOR_PALETTE["secondary_text"],
        "ytick.color": COLOR_PALETTE["secondary_text"],
        "text.color": COLOR_PALETTE["primary_text"],
        "grid.color": COLOR_PALETTE["borders"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "-apple-system", "BlinkMacSystemFont", "DejaVu Sans", "Arial"],
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.facecolor": COLOR_PALETTE["elevated_surface"],
        "legend.edgecolor": COLOR_PALETTE["borders"],
    })


# ==============================================================================
# 2. GENRE & TAXONOMY ANALYSIS
# ==============================================================================

def analyze_genre_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes song count, percentage of catalog, and subgenre count per genre.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Tabular summary of genres.
    """
    if "playlist_genre" not in df.columns:
        return pd.DataFrame()

    total = len(df)
    genre_counts = df["playlist_genre"].value_counts()
    subgenre_counts = df.groupby("playlist_genre")["playlist_subgenre"].nunique()

    report = pd.DataFrame({
        "genre": genre_counts.index,
        "song_count": genre_counts.values,
        "catalog_percentage": ((genre_counts.values / total) * 100).round(2),
        "subgenre_count": [subgenre_counts.get(g, 0) for g in genre_counts.index],
    })
    return report


def plot_genre_distributions(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generates songs_per_genre.png and genre_distribution.png.

    Args:
        df: Input DataFrame.
        output_dir: Directory to save figure.

    Returns:
        Path to saved figure.
    """
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    genre_df = analyze_genre_distribution(df)
    palette_colors = [GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"]) for g in genre_df["genre"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = ax1.bar(
        genre_df["genre"].str.upper(),
        genre_df["song_count"],
        color=palette_colors,
        edgecolor=COLOR_PALETTE["borders"],
        width=0.6,
    )
    ax1.set_title("Song Observations per Super Genre", pad=14, fontweight="bold")
    ax1.set_ylabel("Total Observations")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 80,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_PALETTE["secondary_text"],
        )

    # Donut chart
    wedges, texts, autotexts = ax2.pie(
        genre_df["song_count"],
        labels=genre_df["genre"].str.upper(),
        colors=palette_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor=COLOR_PALETTE["main_background"], linewidth=2),
    )
    plt.setp(texts, color=COLOR_PALETTE["primary_text"], fontsize=9)
    plt.setp(autotexts, color="#090D0B", fontsize=8.5, fontweight="bold")
    ax2.set_title("Catalog Genre Proportions", pad=14, fontweight="bold")

    plt.tight_layout()
    target_path = out_dir / "genre_distribution.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also save songs_per_genre.png
    fig_single, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        genre_df["genre"].str.upper(),
        genre_df["song_count"],
        color=palette_colors,
        edgecolor=COLOR_PALETTE["borders"],
        width=0.55,
    )
    ax.set_title("Songs per Genre", pad=12, fontweight="bold")
    ax.set_ylabel("Number of Songs")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "songs_per_genre.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved genre distribution figures to: %s", target_path)
    return target_path


def plot_subgenre_distribution(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates songs_per_subgenre.png detailing all 24 subgenres."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_counts = df.groupby(["playlist_genre", "playlist_subgenre"]).size().reset_index(name="count")
    sub_counts = sub_counts.sort_values(by=["playlist_genre", "count"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"]) for g in sub_counts["playlist_genre"]]

    y_pos = np.arange(len(sub_counts))
    ax.barh(y_pos, sub_counts["count"], color=colors, edgecolor=COLOR_PALETTE["borders"], height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_counts["playlist_subgenre"].str.title())
    ax.invert_yaxis()
    ax.set_xlabel("Song Observations")
    ax.set_title("Song Observations across 24 Subgenres (Color-coded by Genre)", pad=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    target_path = out_dir / "songs_per_subgenre.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved subgenre distribution figure to: %s", target_path)
    return target_path


def plot_genre_popularity(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates genre_popularity_distribution.png (boxplot and mean bar chart)."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    genres_ordered = df.groupby("playlist_genre")["track_popularity"].median().sort_values(ascending=False).index
    palette_list = [GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"]) for g in genres_ordered]

    # Boxplot
    sns.boxplot(
        data=df,
        x="playlist_genre",
        y="track_popularity",
        hue="playlist_genre",
        order=genres_ordered,
        palette=palette_list,
        legend=False,
        ax=ax1,
        fliersize=1.5,
        linewidth=1.2,
    )
    ax1.set_title("Track Popularity Distribution by Genre", pad=12, fontweight="bold")
    ax1.set_xlabel("Super Genre")
    ax1.set_ylabel("Track Popularity (0-100)")
    ax1.set_xticks(range(len(genres_ordered)))
    ax1.set_xticklabels([g.upper() for g in genres_ordered])

    # Mean Popularity Bar Chart
    mean_pop = df.groupby("playlist_genre")["track_popularity"].mean().loc[genres_ordered]
    bars = ax2.bar(
        [g.upper() for g in genres_ordered],
        mean_pop.values,
        color=palette_list,
        edgecolor=COLOR_PALETTE["borders"],
        width=0.55,
    )
    ax2.set_title("Mean Track Popularity by Genre", pad=12, fontweight="bold")
    ax2.set_ylabel("Average Popularity Score")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_PALETTE["secondary_text"],
        )

    plt.tight_layout()
    target_path = out_dir / "genre_popularity_distribution.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved genre popularity figure to: %s", target_path)
    return target_path


# ==============================================================================
# 3. ARTIST & PLAYLIST ANALYSIS
# ==============================================================================

def analyze_top_artists(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Computes top artists by observation appearances, unique tracks, and mean popularity.

    Args:
        df: Input DataFrame.
        top_n: Number of top artists.

    Returns:
        pd.DataFrame: Top artist ranking table.
    """
    # Exclude Unknown Artist if imputed
    valid_df = df[df["track_artist"] != "Unknown Artist"]

    grouped = valid_df.groupby("track_artist").agg(
        song_observations=("track_id", "count"),
        unique_compositions=("track_id", "nunique"),
        mean_popularity=("track_popularity", "mean"),
        genres_represented=("playlist_genre", lambda x: list(x.unique())),
    ).reset_index()

    top_df = grouped.sort_values(by="song_observations", ascending=False).head(top_n).reset_index(drop=True)
    top_df["mean_popularity"] = top_df["mean_popularity"].round(1)
    return top_df


def plot_top_artists(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates top_artists.png and artist_popularity_distribution.png."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    top_artists = analyze_top_artists(df, top_n=15)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(top_artists))

    bars = ax.barh(
        y_pos,
        top_artists["song_observations"],
        color=COLOR_PALETTE["primary_green"],
        edgecolor=COLOR_PALETTE["borders"],
        height=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_artists["track_artist"])
    ax.invert_yaxis()
    ax.set_xlabel("Catalog Appearances (Playlist Observations)")
    ax.set_title("Top 15 Most Represented Artists in Catalog", pad=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(w)}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLOR_PALETTE["secondary_text"],
        )

    plt.tight_layout()
    target_path = out_dir / "top_artists.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Artist Popularity Scatter
    fig_pop, ax_pop = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=top_artists,
        x="song_observations",
        y="mean_popularity",
        size="unique_compositions",
        sizes=(40, 250),
        color=COLOR_PALETTE["secondary_green"],
        edgecolor=COLOR_PALETTE["primary_green"],
        ax=ax_pop,
    )
    for _, row in top_artists.head(8).iterrows():
        ax_pop.annotate(
            row["track_artist"],
            (row["song_observations"], row["mean_popularity"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color=COLOR_PALETTE["primary_text"],
        )
    ax_pop.set_title("Top Artists: Observations vs Average Popularity", pad=12, fontweight="bold")
    ax_pop.set_xlabel("Song Observations in Catalog")
    ax_pop.set_ylabel("Mean Popularity Score (0-100)")
    ax_pop.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "artist_popularity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved top artists figures to: %s", target_path)
    return target_path


def analyze_top_playlists(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Computes top playlists by song count and genre distribution."""
    grouped = df.groupby(["playlist_name", "playlist_genre"]).size().reset_index(name="song_count")
    total_per_pl = df.groupby("playlist_name")["track_id"].count().rename("total_songs")
    merged = grouped.merge(total_per_pl, on="playlist_name")
    top_pl = merged.sort_values(by="total_songs", ascending=False).head(top_n).reset_index(drop=True)
    return top_pl


def plot_top_playlists(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates top_playlists.png."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    top_pl_series = df["playlist_name"].value_counts().head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_pl_series))

    bars = ax.barh(
        y_pos,
        top_pl_series.values,
        color=COLOR_PALETTE["elevated_surface"],
        edgecolor=COLOR_PALETTE["secondary_green"],
        height=0.6,
    )
    ax.set_yticks(y_pos)
    # Strip non-ASCII glyphs (e.g. emojis in playlist names) and truncate
    clean_labels = [
        "".join(c for c in name if ord(c) < 128)[:36] + ("..." if len(name) > 36 else "")
        for name in top_pl_series.index
    ]
    ax.set_yticklabels(clean_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Curated Tracks")
    ax.set_title("Top 12 Playlists by Track Count", pad=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(w)}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLOR_PALETTE["secondary_text"],
        )

    plt.tight_layout()
    target_path = out_dir / "top_playlists.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved top playlists figure to: %s", target_path)
    return target_path


# ==============================================================================
# 4. AUDIO FEATURE DISTRIBUTIONS
# ==============================================================================

def analyze_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical moments and distributions for all continuous and discrete audio features.

    Returns:
        pd.DataFrame: Table with mean, std, median, IQR, skewness, min, and max.
    """
    summary_list = []
    for feat in CONTINUOUS_AUDIO_FEATURES:
        if feat in df.columns:
            series = df[feat].dropna()
            summary_list.append({
                "feature": feat,
                "type": "Continuous",
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "median": round(float(series.median()), 4),
                "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
                "skewness": round(float(series.skew()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            })

    for feat in DISCRETE_MUSICAL_FEATURES:
        if feat in df.columns:
            series = df[feat].dropna()
            summary_list.append({
                "feature": feat,
                "type": "Discrete",
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "median": round(float(series.median()), 2),
                "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
                "skewness": round(float(series.skew()), 2),
                "min": int(series.min()),
                "max": int(series.max()),
            })

    return pd.DataFrame(summary_list)


def plot_audio_feature_distributions(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates audio_feature_distributions.png (histograms + KDEs for all continuous features)."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    features = [f for f in CONTINUOUS_AUDIO_FEATURES if f != "duration_ms"]  # duration plotted separately or in sec

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features[:9]):
        ax = axes[i]
        sns.histplot(
            df[feat].dropna(),
            kde=True,
            color=COLOR_PALETTE["primary_green"],
            edgecolor=COLOR_PALETTE["borders"],
            alpha=0.45,
            ax=ax,
            bins=30,
        )
        mean_val = df[feat].mean()
        median_val = df[feat].median()
        ax.axvline(mean_val, color=COLOR_PALETTE["warning"], linestyle="--", linewidth=1.2, label=f"Mean ({mean_val:.2f})")
        ax.axvline(median_val, color=COLOR_PALETTE["primary_text"], linestyle=":", linewidth=1.2, label=f"Median ({median_val:.2f})")

        ax.set_title(feat.replace("_", " ").title(), pad=8, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", framealpha=0.6)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Distributions of Continuous Audio Features (32,833 Observations)", y=1.01, fontsize=15, fontweight="bold")
    plt.tight_layout()
    target_path = out_dir / "audio_feature_distributions.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved audio feature distributions figure to: %s", target_path)
    return target_path


def plot_genre_feature_comparison(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates genre_audio_features_comparison.png (boxplots across genres for key acoustic dimensions)."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    key_features = ["danceability", "energy", "valence", "acousticness", "speechiness", "tempo"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    genres = sorted(df["playlist_genre"].unique())
    palette_list = [GENRE_PALETTE.get(g, COLOR_PALETTE["primary_green"]) for g in genres]

    for i, feat in enumerate(key_features):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x="playlist_genre",
            y=feat,
            hue="playlist_genre",
            order=genres,
            palette=palette_list,
            legend=False,
            ax=ax,
            fliersize=1.2,
            linewidth=1.1,
        )
        ax.set_title(feat.title(), pad=8, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticks(range(len(genres)))
        ax.set_xticklabels([g.upper() for g in genres])
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Comparison of Key Audio Features Across Super Genres", y=1.01, fontsize=15, fontweight="bold")
    plt.tight_layout()
    target_path = out_dir / "genre_audio_features_comparison.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved genre feature comparison figure to: %s", target_path)
    return target_path


# ==============================================================================
# 5. BIVARIATE RELATIONSHIP ANALYSIS
# ==============================================================================

def plot_bivariate_relationships(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    sample_size: int = 5000,
) -> List[Path]:
    """
    Generates meaningful bivariate scatter plots:
    - danceability_vs_energy.png
    - energy_vs_loudness.png
    - valence_vs_energy.png

    Uses responsive stratified sampling for plotting clarity and performance.
    """
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stratified sampling by genre for balanced rendering
    indices = []
    for _, grp in df.groupby("playlist_genre"):
        indices.extend(grp.sample(min(len(grp), sample_size // 6), random_state=42).index)
    sample_df = df.loc[indices].copy()

    saved_paths = []

    pairs = [
        ("danceability", "energy", "danceability_vs_energy.png", "Danceability vs Energy"),
        ("energy", "loudness", "energy_vs_loudness.png", "Energy vs Loudness (dB)"),
        ("valence", "energy", "valence_vs_energy.png", "Valence (Positivity) vs Energy"),
    ]

    for x_col, y_col, filename, title in pairs:
        fig, ax = plt.subplots(figsize=(9, 6))

        for genre, grp in sample_df.groupby("playlist_genre"):
            color = GENRE_PALETTE.get(genre, COLOR_PALETTE["primary_green"])
            ax.scatter(
                grp[x_col],
                grp[y_col],
                label=genre.upper(),
                color=color,
                alpha=0.45,
                edgecolors="none",
                s=20,
            )

        ax.set_title(f"{title} by Genre (Sample: {len(sample_df):,} tracks)", pad=12, fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(framealpha=0.6, loc="upper right")

        plt.tight_layout()
        save_target = out_dir / filename
        plt.savefig(save_target, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(save_target)

    logger.info("Saved %d bivariate relationship figures.", len(saved_paths))
    return saved_paths


# ==============================================================================
# 6. TIME-BASED ANALYSIS
# ==============================================================================

def analyze_temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Computes catalog track counts and mean popularity by release year."""
    if "release_year" not in df.columns:
        return pd.DataFrame()

    valid_years = df.dropna(subset=["release_year"])
    yearly = valid_years.groupby("release_year").agg(
        song_count=("track_id", "count"),
        mean_popularity=("track_popularity", "mean"),
        mean_energy=("energy", "mean"),
        mean_acousticness=("acousticness", "mean"),
        mean_danceability=("danceability", "mean"),
    ).reset_index()
    yearly["mean_popularity"] = yearly["mean_popularity"].round(2)
    return yearly


def plot_temporal_trends(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates release_year_trends.png showing temporal evolution of acoustic properties."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    yearly = analyze_temporal_trends(df)
    # Focus on years with meaningful sample sizes (e.g. >= 1970)
    yearly_filtered = yearly[yearly["release_year"] >= 1970].sort_values(by="release_year")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Catalog representation
    ax1.bar(
        yearly_filtered["release_year"],
        yearly_filtered["song_count"],
        color=COLOR_PALETTE["elevated_surface"],
        edgecolor=COLOR_PALETTE["borders"],
        width=0.8,
    )
    ax1.set_title("Catalog Representation by Release Year (1970 – 2020)", pad=10, fontweight="bold")
    ax1.set_ylabel("Track Count")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Evolution of Energy vs Acousticness vs Danceability
    ax2.plot(
        yearly_filtered["release_year"],
        yearly_filtered["mean_energy"],
        color=COLOR_PALETTE["primary_green"],
        linewidth=2,
        marker="o",
        markersize=3,
        label="Mean Energy",
    )
    ax2.plot(
        yearly_filtered["release_year"],
        yearly_filtered["mean_acousticness"],
        color=COLOR_PALETTE["warning"],
        linewidth=2,
        marker="s",
        markersize=3,
        label="Mean Acousticness",
    )
    ax2.plot(
        yearly_filtered["release_year"],
        yearly_filtered["mean_danceability"],
        color=COLOR_PALETTE["secondary_text"],
        linewidth=2,
        marker="^",
        markersize=3,
        label="Mean Danceability",
    )
    ax2.set_title("Evolution of Acoustic Signatures Over Time", pad=10, fontweight="bold")
    ax2.set_xlabel("Release Year")
    ax2.set_ylabel("Mean Feature Value [0, 1]")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", framealpha=0.6)

    plt.tight_layout()
    target_path = out_dir / "release_year_trends.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved release year trends figure to: %s", target_path)
    return target_path


def plot_missing_values_audit(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates missing_values_audit.png visualizing data completeness."""
    setup_visual_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_series = df.isnull().sum()
    fig, ax = plt.subplots(figsize=(12, 5))

    # All columns
    y_pos = np.arange(len(df.columns))
    missing_counts = [missing_series[c] for c in df.columns]
    colors = [COLOR_PALETTE["warning"] if c > 0 else COLOR_PALETTE["secondary_green"] for c in missing_counts]

    ax.barh(y_pos, missing_counts, color=colors, edgecolor=COLOR_PALETTE["borders"], height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.columns, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Missing Observations Count")
    ax.set_title("Missing Value Audit (99.98% Complete Catalog)", pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    target_path = out_dir / "missing_values_audit.png"
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved missing values audit figure to: %s", target_path)
    return target_path


# ==============================================================================
# 7. MASTER FIGURE GENERATOR
# ==============================================================================

def generate_all_figures(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Orchestrates the generation of all Phase 2 publication-grade figures.

    Args:
        df: Cleaned or raw DataFrame.
        output_dir: Output directory. Defaults to FIGURES_DIR.

    Returns:
        List of generated figure Path objects.
    """
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    generated.append(plot_genre_distributions(df, out_dir))
    generated.append(plot_subgenre_distribution(df, out_dir))
    generated.append(plot_genre_popularity(df, out_dir))
    generated.append(plot_top_artists(df, out_dir))
    generated.append(plot_top_playlists(df, out_dir))
    generated.append(plot_audio_feature_distributions(df, out_dir))
    generated.append(plot_genre_feature_comparison(df, out_dir))
    generated.extend(plot_bivariate_relationships(df, out_dir))
    generated.append(plot_temporal_trends(df, out_dir))
    generated.append(plot_missing_values_audit(df, out_dir))

    logger.info("Successfully generated %d figures in: %s", len(generated), out_dir)
    return generated


if __name__ == "__main__":
    from src.data_loader import load_cleaned_dataset, load_dataset
    print("=" * 60)
    print("SONIC ATLAS — Generating Exploratory Visualizations...")
    print("=" * 60)
    try:
        data = load_cleaned_dataset()
    except FileNotFoundError:
        data = load_dataset()

    paths = generate_all_figures(data)
    print(f"Generated {len(paths)} visualization figures:")
    for p in paths:
        print(f"  • {p.name}")
    print("=" * 60)
