"""
SONIC ATLAS — Correlation Analysis Module
Phase 3 Implementation: Pearson & Spearman correlation matrices,
relationship ranking, musical interpretations, discrete attribute handling,
and professional heatmap generators adhering to the dark design system.
"""

import logging
import os
import sys
import warnings
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
    GENRE_PALETTE,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CORRELATION MATRIX CALCULATIONS
# ==============================================================================

def compute_pearson_correlation(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes Pearson linear correlation matrix for continuous audio features.

    Excludes categorical identifiers, genre labels, and discrete pitch classes.

    Args:
        df: Input DataFrame.
        features: List of feature names. Defaults to CONTINUOUS_AUDIO_FEATURES.

    Returns:
        pd.DataFrame: Pearson correlation matrix (10x10).
    """
    target_features = features or CONTINUOUS_AUDIO_FEATURES
    valid_features = [f for f in target_features if f in df.columns]
    corr_matrix = df[valid_features].corr(method="pearson")
    return corr_matrix.round(4)


def compute_spearman_correlation(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes Spearman monotonic rank correlation matrix for continuous audio features.

    Robust to non-normal distributions and monotonic non-linearities.

    Args:
        df: Input DataFrame.
        features: List of feature names. Defaults to CONTINUOUS_AUDIO_FEATURES.

    Returns:
        pd.DataFrame: Spearman rank correlation matrix (10x10).
    """
    target_features = features or CONTINUOUS_AUDIO_FEATURES
    valid_features = [f for f in target_features if f in df.columns]
    corr_matrix = df[valid_features].corr(method="spearman")
    return corr_matrix.round(4)


def get_top_correlations(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extracts, ranks, and annotates pairwise continuous feature relationships.

    Provides musical interpretation and relationship strength classification.

    Args:
        df: Input DataFrame.
        features: Target continuous features.
        top_n: Number of top pairs to return.

    Returns:
        pd.DataFrame: Ranked correlation table.
    """
    target_features = features or CONTINUOUS_AUDIO_FEATURES
    p_corr = compute_pearson_correlation(df, target_features)
    s_corr = compute_spearman_correlation(df, target_features)

    rows = []
    cols = p_corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            p_val = float(p_corr.loc[f1, f2])
            s_val = float(s_corr.loc[f1, f2])
            abs_p = abs(p_val)

            # Strength categorization
            if abs_p >= 0.60:
                strength = "Strong"
            elif abs_p >= 0.30:
                strength = "Moderate"
            elif abs_p >= 0.15:
                strength = "Weak"
            else:
                strength = "Negligible"

            direction = "Positive (+)" if p_val > 0 else "Negative (-)"

            # Musical interpretation rule base
            interp = interpret_relationship(f1, f2, p_val)

            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "pearson_r": round(p_val, 3),
                "spearman_rho": round(s_val, 3),
                "strength": strength,
                "direction": direction,
                "musical_interpretation": interp,
            })

    ranked_df = pd.DataFrame(rows).sort_values(by="pearson_r", key=abs, ascending=False).reset_index(drop=True)
    return ranked_df.head(top_n)


def interpret_relationship(f1: str, f2: str, r: float) -> str:
    """Provides an objective, musical interpretation for feature pairs."""
    pair = {f1, f2}
    if pair == {"energy", "loudness"}:
        return "High-energy productions utilize heavier compression and higher master volume."
    elif pair == {"energy", "acousticness"}:
        return "Acoustic instruments possess natural unamplified dynamics with lower perceived energy."
    elif pair == {"loudness", "acousticness"}:
        return "Acoustic tracks retain broader dynamic range and lower mastering loudness."
    elif pair == {"danceability", "valence"}:
        return "Rhythmically engaging beats strongly correlate with brighter, happier musical moods."
    elif pair == {"danceability", "speechiness"}:
        return "Rhythmic vocal cadence (rap/trap flow) is closely paired with high danceability grooves."
    elif pair == {"danceability", "tempo"}:
        return "Excessively fast tempos become harder to dance to, favoring mid-tempo pockets."
    elif pair == {"instrumentalness", "valence"}:
        return "Instrumental synthesizer and ambient tracks tend to carry darker, contemplative tones."
    elif pair == {"energy", "liveness"}:
        return "Live concert recordings and stage arenas capture elevated crowd excitement and loudness."
    elif pair == {"energy", "valence"}:
        return "Higher energy music leans slightly positive, though aggressive rock/EDM creates nuance."
    elif pair == {"acousticness", "valence"}:
        return "Acoustic folk and ballads often explore melancholic or introspective harmonic themes."
    elif "duration_ms" in pair:
        return "Track length demonstrates negligible linear coupling with musical texture."
    else:
        direction_word = "co-occur" if r > 0 else "diverge"
        return f"Acoustic dimensions {direction_word} with a coefficient of {r:+.2f}."


# ==============================================================================
# 2. DISCRETE ATTRIBUTE ANALYSIS (KEY & MODE)
# ==============================================================================

def analyze_key_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the categorical distribution of pitch classes (key: 0 to 11).

    Converts pitch class integers to standard Western chromatic musical notations:
    0 = C, 1 = C♯/D♭, 2 = D, 3 = D♯/E♭, 4 = E, 5 = F, 6 = F♯/G♭,
    7 = G, 8 = G♯/A♭, 9 = A, 10 = A♯/B♭, 11 = B.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Table with key pitch class, musical notation, count, and percentage.
    """
    pitch_class_map = {
        0: "C",
        1: "C♯ / D♭",
        2: "D",
        3: "D♯ / E♭",
        4: "E",
        5: "F",
        6: "F♯ / G♭",
        7: "G",
        8: "G♯ / A♭",
        9: "A",
        10: "A♯ / B♭",
        11: "B",
    }

    if "key" not in df.columns:
        return pd.DataFrame()

    total = len(df)
    key_counts = df["key"].value_counts().sort_index()

    key_summary = pd.DataFrame({
        "pitch_class": key_counts.index,
        "musical_notation": [pitch_class_map.get(k, str(k)) for k in key_counts.index],
        "track_count": key_counts.values,
        "percentage": ((key_counts.values / total) * 100).round(2),
    })

    return key_summary.sort_values(by="track_count", ascending=False).reset_index(drop=True)


def analyze_mode_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the binary musical modality (mode: 0 = Minor, 1 = Major).

    Computes frequency of major vs minor and contrasts acoustic features across modalities.

    Args:
        df: Input DataFrame.

    Returns:
        Dict: Modality breakdown and comparative acoustic means.
    """
    if "mode" not in df.columns:
        return {}

    total = len(df)
    major_count = int((df["mode"] == 1).sum())
    minor_count = int((df["mode"] == 0).sum())

    # Contrast features between major and minor
    contrast_features = ["valence", "energy", "danceability", "acousticness", "loudness"]
    mode_comparison = df.groupby("mode")[contrast_features].mean().round(3).to_dict(orient="index")

    return {
        "major_count": major_count,
        "minor_count": minor_count,
        "major_percentage": round((major_count / total) * 100, 2),
        "minor_percentage": round((minor_count / total) * 100, 2),
        "feature_means_by_mode": {
            "Major (mode=1)": mode_comparison.get(1, {}),
            "Minor (mode=0)": mode_comparison.get(0, {}),
        },
    }


# ==============================================================================
# 3. HEATMAP VISUALIZATION GENERATORS
# ==============================================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generates a publication-grade annotated correlation heatmap adhering to the dark design tokens.

    Args:
        df: Input DataFrame.
        method: 'pearson' or 'spearman'.
        output_dir: Directory to save figure.

    Returns:
        Path: Location of saved 300 DPI figure.
    """
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if method.lower() == "spearman":
        corr = compute_spearman_correlation(df)
        title = "Spearman Rank Correlation Matrix (Continuous Audio Features)"
        filename = "correlation_matrix_spearman.png"
    else:
        corr = compute_pearson_correlation(df)
        title = "Pearson Linear Correlation Matrix (Continuous Audio Features)"
        filename = "correlation_matrix_pearson.png"

    # Friendly axis labels
    clean_labels = [f.replace("_", " ").title() for f in corr.columns]

    # Mask upper triangle for clean editorial presentation
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10.5, 8.5), facecolor=COLOR_PALETTE["main_background"])
    ax.set_facecolor(COLOR_PALETTE["main_surface"])

    # Editorial diverging colormap: Forest Green (#23865B) -> Surface (#141C17) -> Primary Green (#55D99A)
    # Strictly avoids purple/violet and neon
    cmap = sns.diverging_palette(140, 150, s=80, l=45, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-0.8,
        vmax=0.8,
        center=0.0,
        square=True,
        linewidths=0.75,
        linecolor=COLOR_PALETTE["borders"],
        cbar_kws={
            "shrink": 0.75,
            "label": f"{method.title()} Correlation Coefficient",
        },
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        annot_kws={"size": 8.5, "family": "sans-serif"},
        ax=ax,
    )

    ax.set_title(title, pad=16, fontsize=13, fontweight="bold", color=COLOR_PALETTE["primary_text"])
    plt.xticks(rotation=45, ha="right", color=COLOR_PALETTE["secondary_text"], fontsize=9)
    plt.yticks(rotation=0, color=COLOR_PALETTE["secondary_text"], fontsize=9)

    plt.tight_layout()
    target_path = out_dir / filename
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved %s correlation heatmap to: %s", method, target_path)
    return target_path


def plot_both_correlation_matrices(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generates both Pearson and Spearman heatmaps."""
    out_dir = output_dir or FIGURES_DIR
    p_path = plot_correlation_matrix(df, method="pearson", output_dir=out_dir)
    s_path = plot_correlation_matrix(df, method="spearman", output_dir=out_dir)
    return p_path, s_path


if __name__ == "__main__":
    from src.data_loader import load_cleaned_dataset
    print("=" * 60)
    print("SONIC ATLAS — Phase 3A: Correlation Analysis")
    print("=" * 60)
    df_clean = load_cleaned_dataset()
    top_corrs = get_top_correlations(df_clean, top_n=8)
    print("Top Feature Correlations:")
    print(top_corrs[["feature_1", "feature_2", "pearson_r", "spearman_rho", "strength"]].to_string(index=False))

    print("\nGenerating Correlation Heatmaps...")
    p_path, s_path = plot_both_correlation_matrices(df_clean)
    print(f"  • Pearson:  {p_path.name}")
    print(f"  • Spearman: {s_path.name}")
    print("=" * 60)
