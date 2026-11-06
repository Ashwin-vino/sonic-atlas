"""
SONIC ATLAS — Multi-Track Comparison Studio Module
Phase 5A Implementation: Multi-Track Acoustic Comparison, Pairwise Cosine Similarity,
Radar Chart Visualizations, Grouped Feature Bar Charts, and Data-Grounded Insights.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLUSTER_ARCHETYPES,
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    RADAR_FEATURE_BOUNDS,
    SCALER_PATH,
    TRACK_COMPARISON_COLORS,
    TRACK_COMPARISON_MAX,
    TRACK_COMPARISON_MIN,
    TYPOGRAPHY,
)

logger = logging.getLogger(__name__)

# User-facing readable feature display names
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness (dB)",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "valence": "Valence (Mood)",
    "tempo": "Tempo (BPM)",
    "duration_ms": "Duration",
}

# Radar axis labels (concise for angular display)
RADAR_AXIS_LABELS: Dict[str, str] = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumental",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo",
    "duration_ms": "Duration",
}


# ==============================================================================
# 1. FEATURE NORMALIZATION FOR RADAR VISUALIZATION
# ==============================================================================

def normalize_features_for_radar(
    df_tracks: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Normalizes the 10 continuous audio features strictly to [0.0, 1.0] for radar/spider
    visualization using predefined domain boundaries.

    Note: This normalization is strictly for visual presentation and multi-dimensional
    radar plotting; it is NOT model preprocessing (which uses Phase 3 StandardScaler).

    Args:
        df_tracks: DataFrame containing the 10 continuous audio features.
        bounds: Optional mapping of feature name to (min_val, max_val). Defaults to RADAR_FEATURE_BOUNDS.

    Returns:
        pd.DataFrame: DataFrame with normalized [0, 1] audio features.
    """
    if bounds is None:
        bounds = RADAR_FEATURE_BOUNDS

    norm_df = pd.DataFrame(index=df_tracks.index)

    for feat in CONTINUOUS_AUDIO_FEATURES:
        if feat in df_tracks.columns:
            low, high = bounds.get(feat, (0.0, 1.0))
            raw_vals = df_tracks[feat].astype(float)
            scaled = (raw_vals - low) / (high - low)
            norm_df[feat] = np.clip(scaled, 0.0, 1.0)
        else:
            norm_df[feat] = 0.5

    return norm_df


# ==============================================================================
# 2. PAIRWISE ACOUSTIC SIMILARITY COMPUTATION
# ==============================================================================

def compute_pairwise_similarity(
    df_tracks: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    """
    Calculates the exact pairwise cosine similarity matrix between selected tracks
    using the standardized 10-dimensional continuous audio feature space.

    Methodology:
    1. Features are standardized via the Phase 3 StandardScaler (models/feature_scaler.pkl).
    2. Feature vectors are normalized to unit L2 norm.
    3. The pairwise cosine similarity is computed as the dot product: S = X_norm @ X_norm.T.

    Args:
        df_tracks: DataFrame containing at least 2 tracks and the 10 continuous audio features.
        scaler: Fitted StandardScaler instance. If None, loaded from SCALER_PATH.

    Returns:
        pd.DataFrame: Symmetric N x N similarity matrix with track labels as index & columns.
    """
    if len(df_tracks) < 2:
        raise ValueError(f"Pairwise similarity requires at least 2 tracks, got {len(df_tracks)}.")

    if scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Feature scaler not found at {SCALER_PATH}.")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    # Extract 10 continuous features
    X_raw = df_tracks[CONTINUOUS_AUDIO_FEATURES].values
    X_scaled = scaler.transform(X_raw)

    # Normalize to unit length for cosine dot-product
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_scaled / norms

    # Pairwise cosine similarity matrix
    sim_matrix = np.dot(X_norm, X_norm.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    # Build human-readable track labels
    labels = []
    for _, row in df_tracks.iterrows():
        title = str(row.get("track_name", "Unknown Track"))
        artist = str(row.get("track_artist", "Unknown Artist"))
        if len(title) > 28:
            title = title[:26] + "…"
        labels.append(f"{title} ({artist})")

    return pd.DataFrame(sim_matrix, index=labels, columns=labels)


# ==============================================================================
# 3. DATA-GROUNDED COMPARATIVE INSIGHTS GENERATOR
# ==============================================================================

def generate_comparison_insights(
    df_tracks: pd.DataFrame,
    sim_matrix: pd.DataFrame,
) -> List[str]:
    """
    Generates dynamic, purely empirical, data-grounded observations comparing the
    selected tracks across their audio features, sonic archetypes, and acoustic proximity.

    Every statement is deterministically derived from measured values; no musical
    impressions are fabricated.

    Args:
        df_tracks: Selected tracks DataFrame (2 to 5 tracks).
        sim_matrix: Pairwise cosine similarity matrix calculated for df_tracks.

    Returns:
        List[str]: List of concise data-grounded insight strings.
    """
    insights: List[str] = []
    n = len(df_tracks)
    if n < 2:
        return ["Select at least two tracks to generate acoustic comparison insights."]

    tracks_list = df_tracks.to_dict(orient="records")

    # 1. Pairwise Similarity Extremes
    labels = list(sim_matrix.columns)
    pair_sims: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim_val = float(sim_matrix.iloc[i, j])
            pair_sims.append((sim_val, i, j))

    pair_sims.sort(key=lambda x: x[0], reverse=True)
    top_sim, top_i, top_j = pair_sims[0]
    t1_name = f"'{tracks_list[top_i].get('track_name', 'Track 1')}'"
    t2_name = f"'{tracks_list[top_j].get('track_name', 'Track 2')}'"

    insights.append(
        f"Acoustic Proximity Peak: {t1_name} and {t2_name} exhibit the closest acoustic proximity "
        f"with a pairwise cosine similarity of {top_sim:.4f}."
    )

    if n > 2:
        bot_sim, bot_i, bot_j = pair_sims[-1]
        b1_name = f"'{tracks_list[bot_i].get('track_name', 'Track A')}'"
        b2_name = f"'{tracks_list[bot_j].get('track_name', 'Track B')}'"
        insights.append(
            f"Greatest Acoustic Contrast: {b1_name} and {b2_name} demonstrate the highest divergence "
            f"with a similarity score of {bot_sim:.4f}."
        )

    # 2. Cross-Genre Acoustic Convergence
    cross_genre_aligned = []
    for sim_val, i, j in pair_sims:
        g1 = str(tracks_list[i].get("playlist_genre", "")).title()
        g2 = str(tracks_list[j].get("playlist_genre", "")).title()
        if g1 != g2 and sim_val >= 0.85:
            cross_genre_aligned.append((sim_val, i, j, g1, g2))

    if cross_genre_aligned:
        sim_val, i, j, g1, g2 = cross_genre_aligned[0]
        na1 = f"'{tracks_list[i].get('track_name', 'Track')}'"
        na2 = f"'{tracks_list[j].get('track_name', 'Track')}'"
        insights.append(
            f"Cross-Genre Acoustic Convergence: {na1} ({g1}) and {na2} ({g2}) share high acoustic similarity "
            f"({sim_val:.4f}) despite belonging to distinct catalog genres, demonstrating acoustic alignment across genre borders."
        )

    # 3. Energy & Loudness Extremes
    sorted_by_energy = sorted(tracks_list, key=lambda t: float(t.get("energy", 0)), reverse=True)
    max_energy_t = sorted_by_energy[0]
    min_energy_t = sorted_by_energy[-1]
    e_diff = float(max_energy_t.get("energy", 0)) - float(min_energy_t.get("energy", 0))

    if e_diff >= 0.15:
        insights.append(
            f"Energy Spectrum: '{max_energy_t.get('track_name')}' leads in dynamic intensity "
            f"(energy: {float(max_energy_t.get('energy')):.2f}, loudness: {float(max_energy_t.get('loudness')):.1f} dB), "
            f"while '{min_energy_t.get('track_name')}' has a more subdued acoustic output "
            f"(energy: {float(min_energy_t.get('energy')):.2f}, loudness: {float(min_energy_t.get('loudness')):.1f} dB)."
        )

    # 4. Danceability & Tempo Alignment
    sorted_by_dance = sorted(tracks_list, key=lambda t: float(t.get("danceability", 0)), reverse=True)
    top_dance_t = sorted_by_dance[0]
    insights.append(
        f"Rhythmic Profile: '{top_dance_t.get('track_name')}' is the most rhythmically danceable selection "
        f"(danceability: {float(top_dance_t.get('danceability')):.2f}, tempo: {float(top_dance_t.get('tempo')):.0f} BPM)."
    )

    # 5. Acousticness & Organic Presence
    sorted_by_acoust = sorted(tracks_list, key=lambda t: float(t.get("acousticness", 0)), reverse=True)
    top_acoust_t = sorted_by_acoust[0]
    if float(top_acoust_t.get("acousticness", 0)) >= 0.25:
        insights.append(
            f"Organic Instrumentation: '{top_acoust_t.get('track_name')}' exhibits the highest organic/acoustic depth "
            f"(acousticness: {float(top_acoust_t.get('acousticness')):.3f}), contrasting with synthetic studio production."
        )

    # 6. Sonic Archetype Composition
    clusters = [t.get("cluster") for t in tracks_list if "cluster" in t and pd.notna(t.get("cluster"))]
    if clusters:
        unique_clusters = set(int(c) for c in clusters)
        if len(unique_clusters) == 1:
            c_id = list(unique_clusters)[0]
            arch = CLUSTER_ARCHETYPES.get(c_id, {}).get("name", f"Cluster {c_id}")
            insights.append(
                f"Cluster Homogeneity: All selected tracks map to the same sonic archetype: '{arch}', "
                f"confirming shared structural musical properties."
            )
        else:
            names = [CLUSTER_ARCHETYPES.get(c, {}).get("name", f"Cluster {c}") for c in sorted(unique_clusters)]
            insights.append(
                f"Archetype Diversity: Selected tracks span {len(unique_clusters)} distinct sonic archetypes "
                f"({', '.join(names)})."
            )

    return insights


# ==============================================================================
# 4. MULTI-TRACK RADAR & GROUPED BAR CHARTS
# ==============================================================================

def plot_multitrack_radar(
    df_tracks: pd.DataFrame,
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Renders an interactive multi-track radar (spider) chart comparing 2 to 5 tracks
    across normalized audio dimensions adhering strictly to the SONIC ATLAS design system.

    Visual Specifications:
    - Background: #090D0B (Outer), #141C17 (Polar surface).
    - Gridlines: #26352C with subtle radial circles at 0.25, 0.50, 0.75, 1.0.
    - Labels: Manrope / Inter with color #F2F5F3.
    - Fills: Distinct restrained colors with alpha=0.15 fill and 2.0pt line width.
    """
    if colors is None:
        colors = TRACK_COMPARISON_COLORS

    norm_df = normalize_features_for_radar(df_tracks)
    features = CONTINUOUS_AUDIO_FEATURES
    num_vars = len(features)

    # Compute spoke angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    # Radial gridlines styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    labels = [RADAR_AXIS_LABELS.get(f, f.title()) for f in features]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        labels,
        color=COLOR_PALETTE["primary_text"],
        fontsize=10.5,
        fontfamily="sans-serif",
        fontweight=600,
    )

    # Radial ticks
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"],
        color=COLOR_PALETTE["muted_text"],
        fontsize=8,
        fontfamily="monospace",
    )
    ax.set_ylim(0, 1.05)

    ax.grid(color=COLOR_PALETTE["borders"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color(COLOR_PALETTE["borders"])
    ax.spines["polar"].set_linewidth(1.2)

    # Plot each track
    for idx, (_, row) in enumerate(df_tracks.iterrows()):
        color = colors[idx % len(colors)]
        values = [norm_df.loc[row.name, f] for f in features]
        values += values[:1]

        t_name = str(row.get("track_name", f"Track {idx+1}"))
        t_artist = str(row.get("track_artist", "Unknown"))
        if len(t_name) > 24:
            t_name = t_name[:22] + "…"
        label = f"{t_name} — {t_artist}"

        ax.plot(angles, values, color=color, linewidth=2.0, label=label, marker="o", markersize=4)
        ax.fill(angles, values, color=color, alpha=0.18)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.32, 1.12),
        facecolor=COLOR_PALETTE["elevated_surface"],
        edgecolor=COLOR_PALETTE["borders"],
        labelcolor=COLOR_PALETTE["primary_text"],
        fontsize=9,
        framealpha=0.95,
    )

    plt.tight_layout()
    return fig


def plot_feature_barchart(
    df_tracks: pd.DataFrame,
    features: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Renders a grouped bar chart comparing raw audio feature values across selected tracks.

    Args:
        df_tracks: Selected tracks DataFrame.
        features: Subset of continuous features to display. Defaults to [danceability, energy, speechiness, acousticness, valence].
        colors: Optional color list for tracks.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    if features is None:
        features = ["danceability", "energy", "speechiness", "acousticness", "liveness", "valence"]
    if colors is None:
        colors = TRACK_COMPARISON_COLORS

    n_tracks = len(df_tracks)
    n_features = len(features)
    x = np.arange(n_features)
    bar_width = 0.8 / max(n_tracks, 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    for i, (_, row) in enumerate(df_tracks.iterrows()):
        color = colors[i % len(colors)]
        t_name = str(row.get("track_name", f"Track {i+1}"))
        if len(t_name) > 18:
            t_name = t_name[:16] + "…"
        vals = [float(row.get(f, 0)) for f in features]
        offset = (i - (n_tracks - 1) / 2) * bar_width
        ax.bar(x + offset, vals, width=bar_width * 0.92, label=f"{t_name}", color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [FEATURE_DISPLAY_NAMES.get(f, f.title()) for f in features],
        color=COLOR_PALETTE["primary_text"],
        fontsize=10,
        fontweight=600,
    )
    ax.set_ylabel("Raw Measurement", color=COLOR_PALETTE["secondary_text"], fontsize=9.5)
    ax.tick_params(colors=COLOR_PALETTE["secondary_text"])
    ax.grid(axis="y", color=COLOR_PALETTE["borders"], linestyle=":", alpha=0.6)

    for spine in ax.spines.values():
        spine.set_color(COLOR_PALETTE["borders"])

    ax.legend(
        facecolor=COLOR_PALETTE["elevated_surface"],
        edgecolor=COLOR_PALETTE["borders"],
        labelcolor=COLOR_PALETTE["primary_text"],
        fontsize=9,
    )

    plt.tight_layout()
    return fig


# ==============================================================================
# 5. DETAILED COMPARISON TABLE GENERATION
# ==============================================================================

def format_duration(ms: float) -> str:
    """Formats duration in milliseconds to m:ss string."""
    try:
        total_sec = int(round(float(ms) / 1000.0))
        m = total_sec // 60
        s = total_sec % 60
        return f"{m}:{s:02d}"
    except Exception:
        return "-"


def build_comparison_table(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a detailed feature-by-feature comparison table with original, interpretable
    feature values for all selected tracks.

    Returns:
        pd.DataFrame: Table with features as rows and selected tracks as columns.
    """
    rows = []

    # Metadata rows
    track_names = [str(r.get("track_name", "Unknown")) for _, r in df_tracks.iterrows()]
    artists = [str(r.get("track_artist", "Unknown")) for _, r in df_tracks.iterrows()]
    albums = [str(r.get("track_album_name", "Unknown")) for _, r in df_tracks.iterrows()]
    genres = [str(r.get("playlist_genre", "Unknown")).title() for _, r in df_tracks.iterrows()]
    subgenres = [str(r.get("playlist_subgenre", "Unknown")).title() for _, r in df_tracks.iterrows()]
    popularities = [f"{int(r.get('track_popularity', 0))}/100" for _, r in df_tracks.iterrows()]

    archetypes = []
    for _, r in df_tracks.iterrows():
        c_val = r.get("cluster")
        if pd.notna(c_val):
            arch_info = CLUSTER_ARCHETYPES.get(int(c_val), {})
            archetypes.append(arch_info.get("name", f"Cluster {int(c_val)}"))
        else:
            archetypes.append("Unassigned")

    col_headers = [f"{name} ({art})" if len(name) <= 24 else f"{name[:22]}… ({art})" for name, art in zip(track_names, artists)]

    def make_row(label: str, vals: List[Any]) -> Dict[str, Any]:
        d = {"Dimension": label}
        for h, v in zip(col_headers, vals):
            d[h] = v
        return d

    rows.append(make_row("Track Artist", artists))
    rows.append(make_row("Album", albums))
    rows.append(make_row("Super Genre", genres))
    rows.append(make_row("Subgenre", subgenres))
    rows.append(make_row("Popularity", popularities))
    rows.append(make_row("Sonic Archetype", archetypes))

    # Audio feature rows (Original interpretable units)
    audio_formatters = {
        "danceability": lambda v: f"{float(v):.3f}",
        "energy": lambda v: f"{float(v):.3f}",
        "loudness": lambda v: f"{float(v):.2f} dB",
        "speechiness": lambda v: f"{float(v):.3f}",
        "acousticness": lambda v: f"{float(v):.3f}",
        "instrumentalness": lambda v: f"{float(v):.3f}",
        "liveness": lambda v: f"{float(v):.3f}",
        "valence": lambda v: f"{float(v):.3f}",
        "tempo": lambda v: f"{float(v):.1f} BPM",
        "duration_ms": lambda v: format_duration(v),
    }

    for feat in CONTINUOUS_AUDIO_FEATURES:
        display_label = FEATURE_DISPLAY_NAMES.get(feat, feat.title())
        fmt = audio_formatters.get(feat, lambda v: f"{v}")
        vals = [fmt(r.get(feat, 0)) for _, r in df_tracks.iterrows()]
        rows.append(make_row(display_label, vals))

    return pd.DataFrame(rows)
