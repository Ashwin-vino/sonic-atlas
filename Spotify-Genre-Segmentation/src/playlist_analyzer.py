"""
SONIC ATLAS — Playlist Acoustic Profile Analyzer Module
Phase 5B Implementation: Playlist CSV Upload Validation, Dual-Format Ingestion (Audio vs Tracklist),
Catalog Matching, Acoustic Profiling, Rule-Based Identity Classification, Diversity Metrics,
Playlist-Level Recommendations, and Multi-Format Exports.
"""

import io
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CLUSTER_ARCHETYPES,
    CLUSTER_PALETTE,
    COLOR_PALETTE,
    CONTINUOUS_AUDIO_FEATURES,
    GENRE_PALETTE,
    MAX_UPLOAD_SIZE_MB,
    RADAR_FEATURE_BOUNDS,
    RECOMMENDATION_CATALOG_PATH,
    TYPOGRAPHY,
)
from src.recommender import SonicRecommender

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UPLOAD VALIDATION & SCHEMA DETECTION
# ==============================================================================

def normalize_text_for_matching(text: Any) -> str:
    """Normalizes string for robust case-insensitive, punctuation-agnostic matching."""
    if pd.isna(text) or text is None:
        return ""
    s = str(text).strip().lower()
    # Remove punctuation, excess whitespace, and common metadata tags like (Remastered)
    s = re.sub(r"\(feat\..*?\)", "", s)
    s = re.sub(r"\(with.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def validate_uploaded_csv(
    file_content: Union[bytes, str, io.BytesIO, io.StringIO],
    filename: str = "uploaded_playlist.csv",
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Validates an uploaded playlist CSV file with strict security and schema checks.

    Validation Rules:
    1. Extension must be .csv.
    2. File size must not exceed max_size_mb (10 MB).
    3. Content must be readable as UTF-8 or Latin-1 text.
    4. Must not be empty and must contain valid headers.
    5. Must contain at least one recognized identifying column or audio feature set.

    Args:
        file_content: Uploaded file object, bytes, or string.
        filename: Name of the uploaded file.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        Tuple[bool, Optional[pd.DataFrame], str]: (is_valid, parsed_df, status_message)
    """
    # 1. File extension validation
    if not filename.lower().endswith(".csv"):
        return False, None, "Invalid file format: Only .csv files are supported."

    # 2. File size check
    raw_bytes: bytes
    if isinstance(file_content, bytes):
        raw_bytes = file_content
    elif isinstance(file_content, (io.BytesIO, io.StringIO)):
        pos = file_content.tell() if hasattr(file_content, "tell") else 0
        file_content.seek(0)
        data = file_content.read()
        raw_bytes = data.encode("utf-8") if isinstance(data, str) else data
        file_content.seek(pos)
    elif isinstance(file_content, str):
        raw_bytes = file_content.encode("utf-8")
    else:
        return False, None, "Unsupported file input type."

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, None, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB."

    if len(raw_bytes.strip()) == 0:
        return False, None, "Uploaded CSV file is empty (0 bytes)."

    # 3. Readability & Encoding
    df: Optional[pd.DataFrame] = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            break
        except Exception:
            continue

    if df is None or df.empty:
        return False, None, "Could not parse CSV content. File appears empty or corrupted."

    # Clean whitespace in column headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 4. Schema Compatibility Check
    has_track_id = "track_id" in df.columns
    has_track_meta = ("track_name" in df.columns) or ("title" in df.columns) or ("song" in df.columns)
    has_audio_features = sum(1 for feat in CONTINUOUS_AUDIO_FEATURES if feat in df.columns) >= 5

    if not (has_track_id or has_track_meta or has_audio_features):
        return (
            False,
            None,
            "Unsupported CSV schema: File must contain either track identifiers ('track_id'), "
            "song titles ('track_name'/'title'), or audio features (danceability, energy, etc.).",
        )

    return True, df, f"Successfully parsed {len(df):,} rows and {len(df.columns)} columns."


def detect_playlist_input_type(df: pd.DataFrame) -> str:
    """
    Detects whether an uploaded playlist is Input A (dataset-compatible with audio features)
    or Input B (track-list requiring catalog matching).

    Returns:
        str: 'INPUT_A_AUDIO_FEATURES' or 'INPUT_B_TRACK_LIST'
    """
    cols = set(df.columns)
    continuous_set = set(CONTINUOUS_AUDIO_FEATURES)
    audio_feature_count = len(cols.intersection(continuous_set))

    if audio_feature_count >= 8:
        return "INPUT_A_AUDIO_FEATURES"
    return "INPUT_B_TRACK_LIST"


# ==============================================================================
# 2. LOCAL CATALOG MATCHING ENGINE (NO EXTERNAL APIS)
# ==============================================================================

def match_playlist_to_catalog(
    uploaded_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Matches uploaded playlist rows strictly against the local unique-track catalog (28,356 tracks).

    Matching Strategy:
    1. Exact 'track_id' match (O(1) hash lookup).
    2. Fallback: Normalized 'track_name' + 'track_artist' string matching.
    3. Categorizes records into:
       - matched: exactly one unambiguous catalog track match.
       - ambiguous: multiple distinct catalog tracks match; deterministically selects the
         highest-popularity version while reporting ambiguous tracks.
       - unmatched: no reliable match found in local catalog.
       - duplicates: repeated occurrences within the uploaded file.

    Zero External APIs: Does not hallucinate or fetch external data.

    Returns:
        Dict[str, Any]: Comprehensive matching results dictionary.
    """
    if catalog_df is None:
        if not RECOMMENDATION_CATALOG_PATH.exists():
            raise FileNotFoundError(f"Recommendation catalog not found at {RECOMMENDATION_CATALOG_PATH}.")
        catalog_df = pd.read_csv(RECOMMENDATION_CATALOG_PATH)

    # Precompute catalog search lookups
    catalog_by_id = {str(tid): row for tid, row in zip(catalog_df["track_id"], catalog_df.to_dict(orient="records"))}
    
    # Title + Artist lookup dictionary
    catalog_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    catalog_by_title: Dict[str, List[Dict[str, Any]]] = {}

    for row in catalog_df.to_dict(orient="records"):
        norm_t = normalize_text_for_matching(row.get("track_name", ""))
        norm_a = normalize_text_for_matching(row.get("track_artist", ""))
        if norm_t and norm_a:
            pair_key = (norm_t, norm_a)
            catalog_by_pair.setdefault(pair_key, []).append(row)
        if norm_t:
            catalog_by_title.setdefault(norm_t, []).append(row)

    matched_records: List[Dict[str, Any]] = []
    unmatched_records: List[Dict[str, Any]] = []
    ambiguous_records: List[Dict[str, Any]] = []
    seen_track_ids: Set[str] = set()
    duplicate_rows = 0

    # Harmonize uploaded column names
    col_mapping = {
        "title": "track_name",
        "song": "track_name",
        "song_name": "track_name",
        "artist": "track_artist",
        "artist_name": "track_artist",
        "id": "track_id",
        "id_track": "track_id",
    }
    df_work = uploaded_df.rename(columns=col_mapping).copy()

    total_rows = len(df_work)

    for i, row in df_work.iterrows():
        matched_catalog_row: Optional[Dict[str, Any]] = None
        is_ambiguous = False
        candidates: List[Dict[str, Any]] = []

        # 1. Match by track_id
        if "track_id" in row and pd.notna(row["track_id"]):
            tid = str(row["track_id"]).strip()
            if tid in catalog_by_id:
                matched_catalog_row = catalog_by_id[tid]

        # 2. Match by normalized (track_name, track_artist)
        if matched_catalog_row is None and "track_name" in row and "track_artist" in row:
            norm_t = normalize_text_for_matching(row["track_name"])
            norm_a = normalize_text_for_matching(row["track_artist"])
            pair_key = (norm_t, norm_a)
            if pair_key in catalog_by_pair:
                candidates = catalog_by_pair[pair_key]
                if len(candidates) == 1:
                    matched_catalog_row = candidates[0]
                else:
                    is_ambiguous = True
                    # Deterministically pick highest popularity candidate
                    matched_catalog_row = max(candidates, key=lambda c: float(c.get("track_popularity", 0)))

        # 3. Match by track_name only (if single unique candidate)
        if matched_catalog_row is None and "track_name" in row:
            norm_t = normalize_text_for_matching(row["track_name"])
            if norm_t in catalog_by_title:
                candidates = catalog_by_title[norm_t]
                if len(candidates) == 1:
                    matched_catalog_row = candidates[0]
                elif len(candidates) > 1:
                    is_ambiguous = True
                    matched_catalog_row = max(candidates, key=lambda c: float(c.get("track_popularity", 0)))

        # 4. If Input A has audio features directly and no match was found, synthesize a catalog-like record
        if matched_catalog_row is None:
            has_features = sum(1 for feat in CONTINUOUS_AUDIO_FEATURES if feat in row and pd.notna(row[feat])) >= 8
            if has_features:
                synth = row.to_dict()
                if "track_id" not in synth or pd.isna(synth["track_id"]):
                    synth["track_id"] = f"uploaded_{i}"
                if "playlist_genre" not in synth or pd.isna(synth["playlist_genre"]):
                    synth["playlist_genre"] = "other"
                matched_catalog_row = synth

        # Categorize
        if matched_catalog_row is not None:
            matched_id = str(matched_catalog_row.get("track_id", f"row_{i}"))
            if matched_id in seen_track_ids:
                duplicate_rows += 1
            else:
                seen_track_ids.add(matched_id)
                matched_records.append(matched_catalog_row)

            if is_ambiguous:
                ambiguous_records.append({
                    "uploaded_row": i + 1,
                    "uploaded_track": row.get("track_name", "Unknown"),
                    "uploaded_artist": row.get("track_artist", "Unknown"),
                    "resolved_track_id": matched_catalog_row.get("track_id"),
                    "resolved_title": matched_catalog_row.get("track_name"),
                    "candidate_count": len(candidates),
                })
        else:
            unmatched_records.append({
                "uploaded_row": i + 1,
                "track_name": row.get("track_name", "Unknown"),
                "track_artist": row.get("track_artist", "Unknown"),
                "track_id": row.get("track_id", "None"),
            })

    matched_df = pd.DataFrame(matched_records) if matched_records else pd.DataFrame()
    unmatched_df = pd.DataFrame(unmatched_records) if unmatched_records else pd.DataFrame()
    ambiguous_df = pd.DataFrame(ambiguous_records) if ambiguous_records else pd.DataFrame()

    return {
        "total_uploaded_rows": total_rows,
        "unique_matched_count": len(matched_df),
        "duplicate_count": duplicate_rows,
        "unmatched_count": len(unmatched_df),
        "ambiguous_count": len(ambiguous_df),
        "matched_df": matched_df,
        "unmatched_df": unmatched_df,
        "ambiguous_df": ambiguous_df,
    }


# ==============================================================================
# 3. PLAYLIST ACOUSTIC PROFILE CALCULATION
# ==============================================================================

def calculate_playlist_profile(
    matched_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes summary metrics, feature distributions, archetype composition,
    and comparative baseline benchmarks for a matched playlist.

    Args:
        matched_df: DataFrame of matched playlist tracks.
        catalog_df: Full recommendation catalog for baseline comparison.

    Returns:
        Dict[str, Any]: Complete playlist acoustic profile.
    """
    if matched_df.empty:
        raise ValueError("Cannot calculate profile for empty playlist.")

    if catalog_df is None:
        if RECOMMENDATION_CATALOG_PATH.exists():
            catalog_df = pd.read_csv(RECOMMENDATION_CATALOG_PATH)
        else:
            catalog_df = matched_df

    # 1. Summary Counts
    total_tracks = len(matched_df)
    unique_artists = matched_df["track_artist"].nunique() if "track_artist" in matched_df.columns else 0

    # 2. Genre & Subgenre Distributions
    genre_dist = {}
    if "playlist_genre" in matched_df.columns:
        genre_dist = matched_df["playlist_genre"].value_counts(normalize=True).round(4).to_dict()

    subgenre_dist = {}
    if "playlist_subgenre" in matched_df.columns:
        subgenre_dist = matched_df["playlist_subgenre"].value_counts(normalize=True).round(4).to_dict()

    # 3. Cluster Archetype Distribution
    cluster_dist = {}
    if "cluster" in matched_df.columns:
        cluster_dist = (
            matched_df["cluster"].dropna().astype(int).value_counts(normalize=True).round(4).to_dict()
        )

    # 4. Continuous Feature Summary (Mean, Median, Std, Min, Max)
    features_stats: Dict[str, Dict[str, float]] = {}
    for feat in CONTINUOUS_AUDIO_FEATURES:
        if feat in matched_df.columns:
            vals = matched_df[feat].astype(float)
            cat_vals = catalog_df[feat].astype(float) if feat in catalog_df.columns else vals
            features_stats[feat] = {
                "mean": round(float(vals.mean()), 3),
                "median": round(float(vals.median()), 3),
                "std": round(float(vals.std(ddof=0)), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "catalog_mean": round(float(cat_vals.mean()), 3),
            }

    # 5. Popularity Stats
    pop_stats = {}
    if "track_popularity" in matched_df.columns:
        p_vals = matched_df["track_popularity"].astype(float)
        pop_stats = {
            "mean": round(float(p_vals.mean()), 1),
            "median": round(float(p_vals.median()), 1),
            "min": int(p_vals.min()),
            "max": int(p_vals.max()),
        }

    return {
        "total_tracks": total_tracks,
        "unique_artists": unique_artists,
        "genre_distribution": genre_dist,
        "subgenre_distribution": subgenre_dist,
        "cluster_distribution": cluster_dist,
        "feature_statistics": features_stats,
        "popularity_statistics": pop_stats,
    }


# ==============================================================================
# 4. RULE-BASED PLAYLIST IDENTITY & DIVERSITY METRICS
# ==============================================================================

def classify_playlist_identity(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Applies empirical, rule-based classification to characterize the musical identity
    of the playlist based on its measured audio features.

    Deterministic Rules:
    - High-Octane Mainstage: Energy > 0.72, Loudness > -6.5 dB, Tempo > 122 BPM.
    - Dancefloor Rhythm & Groove: Danceability > 0.70, Energy > 0.60, Valence > 0.52.
    - Acoustic Intimacy & Organic Chill: Acousticness > 0.40, Energy < 0.52, Loudness < -9.0 dB.
    - Lyrical Flow & Urban Cadence: Speechiness > 0.16, Danceability > 0.65.
    - Late-Night Ambient & Melancholic: Valence < 0.40, Energy < 0.55.
    - Progressive Synthesizer Electronic: Instrumentalness > 0.30.
    - Dynamic Eclectic Showcase: High acoustic standard deviation across genres.

    Returns:
        Dict[str, str]: Persona title, subtitle, and detailed acoustic description.
    """
    fstats = profile.get("feature_statistics", {})
    e_mean = fstats.get("energy", {}).get("mean", 0.5)
    d_mean = fstats.get("danceability", {}).get("mean", 0.5)
    l_mean = fstats.get("loudness", {}).get("mean", -8.0)
    a_mean = fstats.get("acousticness", {}).get("mean", 0.2)
    s_mean = fstats.get("speechiness", {}).get("mean", 0.08)
    i_mean = fstats.get("instrumentalness", {}).get("mean", 0.05)
    v_mean = fstats.get("valence", {}).get("mean", 0.5)
    t_mean = fstats.get("tempo", {}).get("mean", 120.0)

    # Assess primary sonic persona
    if i_mean >= 0.25:
        return {
            "title": "Progressive Synthesizer Electronic",
            "badge": "INSTRUMENTAL FOCUS",
            "summary": (
                f"Dominated by instrumental arrangements (instrumentalness {i_mean:.2f}) "
                f"and extended dynamic builds with minimal vocal interference."
            ),
        }
    elif a_mean >= 0.38 and e_mean <= 0.55:
        return {
            "title": "Acoustic Intimacy & Organic Chill",
            "badge": "ORGANIC ACOUSTIC",
            "summary": (
                f"Defined by warm organic acoustics (acousticness {a_mean:.2f}) and restrained mastering "
                f"(energy {e_mean:.2f}, loudness {l_mean:.1f} dB) suited for relaxed listening."
            ),
        }
    elif s_mean >= 0.15 and d_mean >= 0.62:
        return {
            "title": "Lyrical Flow & Urban Cadence",
            "badge": "VOCAL / HIP-HOP",
            "summary": (
                f"Features rhythmic, spoken-word cadence (speechiness {s_mean:.2f}) over syncopated beats "
                f"(danceability {d_mean:.2f}, tempo {t_mean:.0f} BPM)."
            ),
        }
    elif e_mean >= 0.72 and l_mean >= -6.8:
        return {
            "title": "High-Octane Mainstage",
            "badge": "DYNAMIC INTENSITY",
            "summary": (
                f"Maximal dynamic energy ({e_mean:.2f}) and loud mastering ({l_mean:.1f} dB) "
                f"delivering high sonic impact at a driving pace ({t_mean:.0f} BPM)."
            ),
        }
    elif d_mean >= 0.68 and v_mean >= 0.52:
        return {
            "title": "Dancefloor Rhythm & Vibrant Groove",
            "badge": "GROOVE & POP",
            "summary": (
                f"Engineered for danceability ({d_mean:.2f}) and positive musical tone "
                f"(valence {v_mean:.2f}), creating an uplifting rhythmic momentum."
            ),
        }
    elif v_mean <= 0.38 and e_mean <= 0.58:
        return {
            "title": "Late-Night Ambient & Melancholic",
            "badge": "ATMOSPHERIC / DOWNBEAT",
            "summary": (
                f"Features introspective, darker harmonic coloring (valence {v_mean:.2f}) "
                f"and tempered intensity ({e_mean:.2f}) for atmospheric contemplation."
            ),
        }
    else:
        return {
            "title": "Dynamic Eclectic Showcase",
            "badge": "MULTI-TEXTURE",
            "summary": (
                f"A balanced multi-genre curation balancing danceability ({d_mean:.2f}), "
                f"energy ({e_mean:.2f}), and organic acoustic textures ({a_mean:.2f})."
            ),
        }


def calculate_playlist_diversity(
    matched_df: pd.DataFrame,
    profile: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes mathematical diversity metrics for an analyzed playlist.

    Metrics:
    - Artist Diversity: unique artists / total tracks (0.0 to 1.0).
    - Genre Diversity: Simpson's Diversity Index (1 - sum(p_i^2)).
    - Cluster Diversity: Normalized Shannon Entropy of cluster distribution.
    - Acoustic Variance: Mean standard deviation across the 10 audio dimensions.

    Returns:
        Dict[str, float]: Diversity metrics.
    """
    total = len(matched_df)
    if total == 0:
        return {
            "artist_diversity": 0.0,
            "genre_diversity": 0.0,
            "cluster_diversity": 0.0,
            "acoustic_variance": 0.0,
        }

    # 1. Artist Diversity
    n_artists = matched_df["track_artist"].nunique() if "track_artist" in matched_df.columns else 1
    artist_div = round(n_artists / total, 4)

    # 2. Genre Diversity (Gini-Simpson)
    g_dist = profile.get("genre_distribution", {})
    if g_dist:
        simpson = 1.0 - sum(p ** 2 for p in g_dist.values())
        genre_div = round(float(simpson), 4)
    else:
        genre_div = 0.0

    # 3. Cluster Diversity (Normalized Entropy)
    c_dist = profile.get("cluster_distribution", {})
    if c_dist:
        # Shannon entropy normalized by log(6)
        probs = [p for p in c_dist.values() if p > 0]
        h = -sum(p * np.log(p) for p in probs)
        max_h = np.log(6)  # 6 clusters
        cluster_div = round(float(h / max_h), 4)
    else:
        cluster_div = 0.0

    # 4. Acoustic Variance (Mean std across 10 continuous features)
    fstats = profile.get("feature_statistics", {})
    stds = [fstats[feat].get("std", 0.0) for feat in CONTINUOUS_AUDIO_FEATURES if feat in fstats]
    acoust_var = round(float(np.mean(stds)), 4) if stds else 0.0

    return {
        "artist_diversity": artist_div,
        "genre_diversity": genre_div,
        "cluster_diversity": cluster_div,
        "acoustic_variance": acoust_var,
    }


# ==============================================================================
# 5. PLAYLIST ACOUSTIC VISUALIZATIONS
# ==============================================================================

def plot_playlist_radar(
    profile: Dict[str, Any],
) -> plt.Figure:
    """
    Renders a radar chart comparing the playlist's mean acoustic features
    against the full 28,356 unique catalog baseline.
    """
    fstats = profile.get("feature_statistics", {})
    features = CONTINUOUS_AUDIO_FEATURES
    num_vars = len(features)

    # Normalize values for radar
    pl_vals = []
    cat_vals = []
    for feat in features:
        low, high = RADAR_FEATURE_BOUNDS.get(feat, (0.0, 1.0))
        p_raw = fstats.get(feat, {}).get("mean", 0.5)
        c_raw = fstats.get(feat, {}).get("catalog_mean", 0.5)
        pl_vals.append(np.clip((p_raw - low) / (high - low), 0.0, 1.0))
        cat_vals.append(np.clip((c_raw - low) / (high - low), 0.0, 1.0))

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    pl_vals += pl_vals[:1]
    cat_vals += cat_vals[:1]

    fig, ax = plt.subplots(figsize=(8.0, 8.0), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    labels = [feat.title() if feat != "duration_ms" else "Duration" for feat in features]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        labels,
        color=COLOR_PALETTE["primary_text"],
        fontsize=9.5,
        fontweight=600,
    )

    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"],
        color=COLOR_PALETTE["muted_text"],
        fontsize=7.5,
        fontfamily="monospace",
    )
    ax.set_ylim(0, 1.05)

    ax.grid(color=COLOR_PALETTE["borders"], linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color(COLOR_PALETTE["borders"])

    # Catalog baseline (Muted grey/slate)
    ax.plot(angles, cat_vals, color=COLOR_PALETTE["secondary_text"], linewidth=1.5, linestyle="--", label="Catalog Baseline (Mean)")
    ax.fill(angles, cat_vals, color=COLOR_PALETTE["secondary_text"], alpha=0.08)

    # Playlist Profile (Primary Green)
    ax.plot(angles, pl_vals, color=COLOR_PALETTE["primary_green"], linewidth=2.5, marker="o", markersize=4, label="Uploaded Playlist (Mean)")
    ax.fill(angles, pl_vals, color=COLOR_PALETTE["primary_green"], alpha=0.22)

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


def plot_playlist_genre_and_cluster(
    profile: Dict[str, Any],
) -> plt.Figure:
    """Renders dual horizontal bar charts for genre and cluster archetype breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.patch.set_facecolor(COLOR_PALETTE["main_background"])

    for ax in (ax1, ax2):
        ax.set_facecolor(COLOR_PALETTE["main_surface"])
        ax.tick_params(colors=COLOR_PALETTE["secondary_text"])
        for spine in ax.spines.values():
            spine.set_color(COLOR_PALETTE["borders"])

    # Genre breakdown
    g_dist = profile.get("genre_distribution", {})
    if g_dist:
        genres = list(g_dist.keys())
        pcts = [g_dist[g] * 100 for g in genres]
        g_colors = [GENRE_PALETTE.get(g.lower(), COLOR_PALETTE["primary_green"]) for g in genres]
        ax1.barh(genres, pcts, color=g_colors, height=0.6, alpha=0.9)
        ax1.set_xlabel("Percentage (%)", color=COLOR_PALETTE["secondary_text"], fontsize=9)
        ax1.set_title("Genre Distribution", color=COLOR_PALETTE["primary_text"], fontsize=11, fontweight=600)
        ax1.grid(axis="x", color=COLOR_PALETTE["borders"], linestyle=":", alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "No genre labels", color=COLOR_PALETTE["muted_text"], ha="center")

    # Cluster breakdown
    c_dist = profile.get("cluster_distribution", {})
    if c_dist:
        c_ids = sorted(c_dist.keys())
        c_names = [CLUSTER_ARCHETYPES.get(int(cid), {}).get("name", f"Cluster {cid}") for cid in c_ids]
        c_pcts = [c_dist[cid] * 100 for cid in c_ids]
        c_colors = [CLUSTER_PALETTE.get(int(cid), COLOR_PALETTE["primary_green"]) for cid in c_ids]
        ax2.barh(c_names, c_pcts, color=c_colors, height=0.6, alpha=0.9)
        ax2.set_xlabel("Percentage (%)", color=COLOR_PALETTE["secondary_text"], fontsize=9)
        ax2.set_title("Sonic Archetype Composition", color=COLOR_PALETTE["primary_text"], fontsize=11, fontweight=600)
        ax2.grid(axis="x", color=COLOR_PALETTE["borders"], linestyle=":", alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No cluster assignments", color=COLOR_PALETTE["muted_text"], ha="center")

    plt.tight_layout()
    return fig


# ==============================================================================
# 6. PLAYLIST-LEVEL RECOMMENDATION ENGINE
# ==============================================================================

def recommend_for_playlist(
    matched_df: pd.DataFrame,
    recommender: SonicRecommender,
    top_n: int = 10,
    discovery_mode: str = "acoustic_proximity",
    target_genre: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates music recommendations for an uploaded playlist using the Phase 4 engine.

    Methodology:
    1. Aggregates the matched playlist into a 10D mean continuous audio vector.
    2. Excludes all track IDs present in the uploaded playlist.
    3. Queries the high-dimensional cosine similarity index.
    4. Attaches dynamic explanations showing why each track complements the playlist.

    Args:
        matched_df: DataFrame of matched playlist tracks.
        recommender: Active SonicRecommender instance.
        top_n: Number of recommendations to generate (5, 10, 15, 20).
        discovery_mode: 'acoustic_proximity' (closest matches) or 'cross_genre' (genres not in playlist).
        target_genre: Optional genre filter.

    Returns:
        pd.DataFrame: Recommended tracks with scores and explanations.
    """
    if len(matched_df) < 1:
        raise ValueError("Cannot recommend tracks for an empty playlist.")

    # 1. Compute playlist mean raw continuous vector
    mean_vector = matched_df[CONTINUOUS_AUDIO_FEATURES].mean().values.reshape(1, -1)

    # 2. Standardize using the Phase 3 scaler
    scaled_vector = recommender.scaler.transform(mean_vector)
    norm = np.linalg.norm(scaled_vector)
    if norm == 0:
        norm = 1e-10
    norm_vector = (scaled_vector / norm).flatten()

    # 3. Exclude existing track IDs
    existing_ids: Set[str] = set(matched_df["track_id"].astype(str).values)
    playlist_genres: Set[str] = set(matched_df["playlist_genre"].dropna().str.lower().unique()) if "playlist_genre" in matched_df.columns else set()

    # 4. Compute cosine similarity against all 28,356 tracks in catalog
    all_sims = np.dot(recommender.X_norm, norm_vector)

    catalog = recommender.catalog_df.copy()
    catalog["similarity_score"] = all_sims

    # Filter out existing tracks
    mask = ~catalog["track_id"].isin(existing_ids)

    # Apply discovery mode constraints
    if discovery_mode == "cross_genre" and playlist_genres:
        mask = mask & (~catalog["playlist_genre"].str.lower().isin(playlist_genres))

    if target_genre and target_genre.lower() != "all":
        mask = mask & (catalog["playlist_genre"].str.lower() == target_genre.lower())

    candidates = catalog[mask].sort_values(by="similarity_score", ascending=False)

    # Take top candidates with max 1 track per artist for balanced playlist additions
    seen_artists: Set[str] = set()
    selected_rows: List[Dict[str, Any]] = []

    # Calculate playlist energy/danceability/acousticness for explanations
    pl_e = float(mean_vector[0][CONTINUOUS_AUDIO_FEATURES.index("energy")])
    pl_d = float(mean_vector[0][CONTINUOUS_AUDIO_FEATURES.index("danceability")])
    pl_a = float(mean_vector[0][CONTINUOUS_AUDIO_FEATURES.index("acousticness")])

    for _, row in candidates.iterrows():
        artist = str(row.get("track_artist", "Unknown"))
        if artist in seen_artists:
            continue
        seen_artists.add(artist)

        # Dynamic playlist explanation
        rec_e = float(row.get("energy", 0))
        rec_d = float(row.get("danceability", 0))
        rec_a = float(row.get("acousticness", 0))
        rec_g = str(row.get("playlist_genre", "")).title()

        exp_parts = []
        if abs(rec_e - pl_e) < 0.08:
            exp_parts.append(f"mirrors the playlist's energy ({rec_e:.2f})")
        if abs(rec_d - pl_d) < 0.08:
            exp_parts.append(f"matching dance groove ({rec_d:.2f})")
        if rec_a > pl_a + 0.15:
            exp_parts.append("adds organic acoustic depth")
        elif rec_a < pl_a - 0.15:
            exp_parts.append("injects electronic production punch")

        if exp_parts:
            reason = f"Complements playlist: {', '.join(exp_parts)}."
        else:
            reason = f"Shares overall {rec_g} acoustic profile with the playlist (Sim: {row['similarity_score']:.3f})."

        record = row.to_dict()
        record["recommendation_reason"] = reason
        selected_rows.append(record)

        if len(selected_rows) >= top_n:
            break

    return pd.DataFrame(selected_rows)


# ==============================================================================
# 7. EXPORT DATA BUILDERS
# ==============================================================================

def export_playlist_analysis(
    profile: Dict[str, Any],
    identity: Dict[str, str],
    diversity: Dict[str, float],
    matched_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
) -> Dict[str, Union[str, bytes]]:
    """
    Builds clean, structured CSV and JSON strings for user export.

    Returns:
        Dict[str, str]: Map of export file identifier to CSV/JSON string.
    """
    # 1. Summary CSV
    summary_rows = [
        {"Metric": "Total Tracks Analyzed", "Value": str(profile.get("total_tracks", 0))},
        {"Metric": "Unique Artists", "Value": str(profile.get("unique_artists", 0))},
        {"Metric": "Playlist Persona", "Value": identity.get("title", "Unknown")},
        {"Metric": "Persona Summary", "Value": identity.get("summary", "")},
        {"Metric": "Artist Diversity", "Value": f"{diversity.get('artist_diversity', 0.0):.4f}"},
        {"Metric": "Genre Diversity (Simpson)", "Value": f"{diversity.get('genre_diversity', 0.0):.4f}"},
        {"Metric": "Cluster Diversity (Normalized)", "Value": f"{diversity.get('cluster_diversity', 0.0):.4f}"},
        {"Metric": "Acoustic Variance", "Value": f"{diversity.get('acoustic_variance', 0.0):.4f}"},
    ]

    fstats = profile.get("feature_statistics", {})
    for feat in CONTINUOUS_AUDIO_FEATURES:
        if feat in fstats:
            summary_rows.append({
                "Metric": f"Mean {feat.title()}",
                "Value": str(fstats[feat].get("mean")),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = summary_df.to_csv(index=False)

    # 2. Matched Tracks CSV
    matched_csv = matched_df.to_csv(index=False) if not matched_df.empty else "No tracks matched\n"

    # 3. Unmatched Tracks CSV
    unmatched_csv = unmatched_df.to_csv(index=False) if not unmatched_df.empty else "No unmatched tracks\n"

    # 4. Profile JSON
    export_json_dict = {
        "identity": identity,
        "diversity": diversity,
        "profile": profile,
    }
    profile_json = json.dumps(export_json_dict, indent=2)

    return {
        "summary_csv": summary_csv,
        "matched_csv": matched_csv,
        "unmatched_csv": unmatched_csv,
        "profile_json": profile_json,
    }

