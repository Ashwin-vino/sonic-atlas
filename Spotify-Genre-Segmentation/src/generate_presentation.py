"""
SONIC ATLAS — Executive Presentation Deck Generator
Phase 5F Implementation: Generates an 18-slide, 16:9 widescreen, dark-themed
PowerPoint presentation (reports/SONIC_ATLAS_FINAL_PRESENTATION.pptx) adhering
strictly to the SONIC ATLAS design system.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.config import PRESENTATION_PATH, REPORTS_DIR

# Design System Color Tokens in RGBColor
C_BG = RGBColor(9, 13, 11)           # Main background #090D0B
C_CARD = RGBColor(20, 28, 23)        # Card surface #141C17
C_CARD_HIGH = RGBColor(27, 38, 31)   # Elevated card surface #1B261F
C_GREEN = RGBColor(85, 217, 154)     # Primary green #55D99A
C_GREEN_DARK = RGBColor(35, 134, 91) # Secondary green #23865B
C_TEXT = RGBColor(242, 245, 243)     # Primary text #F2F5F3
C_MUTED = RGBColor(154, 167, 159)    # Secondary text #9AA79F
C_BORDER = RGBColor(38, 53, 44)      # Borders #26352C
C_GOLD = RGBColor(233, 184, 108)     # Warning / Gold #E9B86C
C_TEAL = RGBColor(91, 192, 190)      # Clean teal #5BC0BE


def create_deck() -> None:
    """Generates the comprehensive 18-slide SONIC ATLAS executive presentation deck."""
    prs = pptx.Presentation()
    # Configure 16:9 widescreen dimensions (13.333 x 7.5 inches)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.color.rgb = C_BG

    def add_header(slide, number: str, title: str, subtitle: str = ""):
        # Category / Number tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.4))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"SONIC ATLAS  ◈  SLIDE {number}  ·  MUSIC INTELLIGENCE PLATFORM"
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = C_GREEN

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = C_TEXT

        # Subtitle
        if subtitle:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = C_MUTED

    def add_card(slide, left, top, width, height, title="", border_color=C_BORDER, bg_color=C_CARD):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = C_GREEN
        return card

    # ==========================================================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent decorative banner
    banner = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.12), Inches(3.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = C_GREEN
    banner.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = "SONIC ATLAS"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = C_TEXT

    p2 = tf.add_paragraph()
    p2.text = "MUSIC INTELLIGENCE & ACOUSTIC PLATFORM"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = C_GREEN
    p2.space_before = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "Discovering the acoustic structure of music through high-dimensional data science, unsupervised machine learning, and content-based recommendation."
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_MUTED
    p3.space_before = Pt(18)

    p4 = tf.add_paragraph()
    p4.text = "Final Technical Evaluation & System Architecture Deck  ·  Phase 1–5 Comprehensive Delivery"
    p4.font.size = Pt(11)
    p4.font.color.rgb = C_BORDER
    p4.space_before = Pt(28)

    # Bottom Stat Pill
    add_card(s1, Inches(1.2), Inches(5.8), Inches(10.8), Inches(0.9), bg_color=C_CARD_HIGH)
    tb_stat = s1.shapes.add_textbox(Inches(1.5), Inches(6.0), Inches(10.2), Inches(0.5))
    tf_stat = tb_stat.text_frame
    p_s = tf_stat.paragraphs[0]
    p_s.text = "32,833 Observations  |  28,356 Unique Compositions  |  6 Super Genres  |  6 Sonic Archetypes  |  10D Continuous Acoustic Space"
    p_s.font.size = Pt(11)
    p_s.font.bold = True
    p_s.font.color.rgb = C_TEXT

    # ==========================================================================
    # SLIDE 2: PROBLEM STATEMENT
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "02", "Problem Statement", "The Failure of Traditional Genre Taxonomies in Modern Music Streaming")

    add_card(s2, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="The Limitations of Categorical Genres")
    tb = s2.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    points = [
        ("Static Metadata Labels", "Traditional cataloging relies on subjective, human-assigned genre tags that fail to capture raw acoustic texture."),
        ("Cross-Genre Blurring", "Modern festival EDM shares more sonic energy with driving Hard Rock than with ambient chill electronic tracks."),
        ("Curatorial Bias", "Streaming platforms group tracks based on playlist marketing rather than objective audio measurements."),
        ("Discovery Traps", "Listeners get pigeonholed into narrow genre silos despite enjoying matching acoustic textures across different genres."),
    ]
    for i, (hd, body) in enumerate(points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"• {hd}: {body}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    add_card(s2, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="The SONIC ATLAS Solution")
    tb2 = s2.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    sol_points = [
        ("Empirical Feature Space", "Extract 10 continuous acoustic measurements (danceability, energy, loudness, valence, tempo, etc.) directly from audio signals."),
        ("Unsupervised Clustering", "Let K-Means discover natural sonic archetypes rather than forcing artificial genre boundaries."),
        ("Acoustic Proximity Engine", "Calculate mathematical cosine similarity in standardized 10D space to find songs that truly sound alike."),
        ("Acoustic Twins Discovery", "Intentionally cross genre boundaries to reveal surprising, high-affinity musical matches."),
    ]
    for i, (hd, body) in enumerate(sol_points):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(10)

    # ==========================================================================
    # SLIDE 3: PROJECT OBJECTIVES
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "03", "Project Objectives & 5-Phase Roadmap", "Building an Independent Commercial-Grade Music Analytics Platform")

    phases = [
        ("Phase 1", "Foundation & Design System", "Strict data auditing, schema verification, and dark editorial design system tokens.", C_GREEN_DARK),
        ("Phase 2", "Data Cleaning & Deep EDA", "Zero-data-loss cleaning, missing metadata resolution, and 10D feature distributions.", C_GREEN_DARK),
        ("Phase 3", "ML Clustering & Archetypes", "Feature scaling, correlation analysis, K-Means (K=6), and PCA projections.", C_GREEN_DARK),
        ("Phase 4", "Recommendation Engine", "Content-based cosine similarity, 4 recommendation modes, and directional explanations.", C_GREEN_DARK),
        ("Phase 5", "Advanced Intelligence & Delivery", "Track Comparison Studio, Playlist Intelligence, Catalog Analytics & Production Deployment.", C_GREEN),
    ]

    for i, (p_id, p_title, p_desc, col) in enumerate(phases):
        top_pos = Inches(2.1 + i * 0.95)
        add_card(s3, Inches(0.8), top_pos, Inches(11.7), Inches(0.82), bg_color=C_CARD)
        # Left badge
        b = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top_pos, Inches(1.4), Inches(0.82))
        b.fill.solid()
        b.fill.fore_color.rgb = col
        b.line.fill.background()
        tf_b = b.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = p_id
        p_b.font.size = Pt(12)
        p_b.font.bold = True
        p_b.font.color.rgb = C_BG if col == C_GREEN else C_TEXT
        p_b.alignment = PP_ALIGN.CENTER

        tb = s3.shapes.add_textbox(Inches(2.4), top_pos + Inches(0.12), Inches(9.8), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = p_title
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = C_TEXT

        p2 = tf.add_paragraph()
        p2.text = p_desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 4: DATASET OVERVIEW
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "04", "Dataset Overview & Topology", "High-Fidelity Spotify Acoustic Observations and Curated Metadata")

    stats = [
        ("32,833", "Total Observations", "Full playlist-associated song records preserved with zero loss."),
        ("28,356", "Unique Compositions", "Unique track IDs deduplicated deterministically for recommendations."),
        ("10,692", "Distinct Artists", "Global spectrum of solo performers, producers, and bands."),
        ("449 / 471", "Playlists (Names/IDs)", "Curated Spotify editorial and user playlists across 6 genres."),
        ("6 Genres / 24 Subs", "Curated Taxonomy", "EDM, Latin, Pop, R&B, Rap, Rock with exactly 4 subgenres each."),
        ("10 Dimensions", "Continuous Features", "Continuous acoustic measurements capturing texture, tempo, and timbre."),
    ]

    for i, (val, label, sub) in enumerate(stats):
        col_idx = i % 3
        row_idx = i // 3
        left = Inches(0.8 + col_idx * 4.0)
        top = Inches(2.2 + row_idx * 2.3)
        add_card(s4, left, top, Inches(3.7), Inches(2.0), bg_color=C_CARD)

        tb = s4.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), Inches(3.1), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p_v = tf.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(24)
        p_v.font.bold = True
        p_v.font.color.rgb = C_GREEN

        p_l = tf.add_paragraph()
        p_l.text = label
        p_l.font.size = Pt(12)
        p_l.font.bold = True
        p_l.font.color.rgb = C_TEXT
        p_l.space_before = Pt(4)

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(9.5)
        p_s.font.color.rgb = C_MUTED
        p_s.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 5: TECHNOLOGY STACK
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "05", "Technology Stack & Architecture", "Python-Centric Data Science and Production Application Infrastructure")

    stack = [
        ("Core Language & Engine", "Python 3.9+", "High-performance vector operations, typed data modeling, and mathematical matrix computation.", C_GREEN),
        ("Data Wrangling & Processing", "Pandas & NumPy", "Vectorized DataFrame pipelines, modal multi-playlist aggregation, and deterministic deduplication.", C_TEAL),
        ("Machine Learning", "Scikit-learn", "K-Means clustering, StandardScaler, PCA dimensionality reduction, and NearestNeighbors cosine indexing.", C_GREEN),
        ("Visualization & Aesthetics", "Matplotlib & Seaborn", "Custom polar radar charts, correlation heatmaps, feature distributions, and design system integration.", C_GOLD),
        ("Application & UI Framework", "Streamlit 1.50+", "High-performance reactive frontend with custom CSS tokens, session state management, and file uploads.", C_GREEN),
        ("Production Deployment", "Docker & Compose", "Multi-stage containerization, healthchecks, zero external API dependencies, and reproducible runtime.", C_TEAL),
    ]

    for i, (hd, tool, desc, col) in enumerate(stack):
        col_idx = i % 2
        row_idx = i // 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(2.2 + row_idx * 1.6)
        add_card(s5, left, top, Inches(5.6), Inches(1.4), bg_color=C_CARD)

        tb = s5.shapes.add_textbox(left + Inches(0.3), top + Inches(0.15), Inches(5.0), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = f"{hd}  ·  {tool}"
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = C_TEXT
        p2.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 6: DATA PREPROCESSING AND EDA
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "06", "Data Preprocessing & Audit Integrity", "Zero Data Loss, Metadata Imputation, and Continuous Feature Separation")

    add_card(s6, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Data Integrity & Cleaning Protocol")
    tb = s6.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    c_points = [
        ("Observation Preservation", "Maintained all 32,833 playlist-associated song records. Did not blindly delete repeated tracks across playlists."),
        ("Missing Metadata Resolution", "Identified exactly 5 rows with missing track_name, artist, or album. Imputed with clean 'Unknown' fallbacks rather than discarding rows."),
        ("Zero Exact Duplicates", "Confirmed 0 exact full-row duplicate records in the raw dataset."),
        ("Feature Categorization", "Strictly separated 10 continuous audio measurements from discrete musical attributes (key, mode) and popularity metrics."),
    ]
    for i, (hd, body) in enumerate(c_points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    add_card(s6, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Key EDA Findings")
    tb2 = s6.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    eda_points = [
        ("Loudness Skewness", "Loudness is strongly negatively skewed (median -6.7 dB, range -46.5 to 1.3 dB), reflecting modern 'loudness wars' compression."),
        ("Danceability Distribution", "Symmetric Gaussian distribution centered at 0.655, reflecting high commercial radio dance focus."),
        ("Speechiness Sparsity", "Over 80% of songs have speechiness < 0.15; values > 0.30 exclusively isolate rap, hip-hop, and spoken verse."),
        ("Instrumentalness Bimodality", "Extremely polarized: 90% of songs have near-zero values (< 0.01), with a distinct spike near 0.70 isolating club synth electronic."),
    ]
    for i, (hd, body) in enumerate(eda_points):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"◈ {hd}: {body}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(8)

    # ==========================================================================
    # SLIDE 7: CORRELATION ANALYSIS
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "07", "Correlation Analysis & Acoustic Collinearity", "Empirical Feature Associations Across the 10 Audio Dimensions")

    corr_cards = [
        ("Energy & Loudness", "+0.68", "Strong Positive Collinearity", "Acoustic physics linkage: higher perceived energy strongly dictates heavily compressed master volume (-dB)."),
        ("Energy & Acousticness", "-0.54", "Strong Negative Inverse", "Structural dichotomy: organic acoustic instruments directly displace aggressive electronic and electric amplification."),
        ("Danceability & Valence", "+0.33", "Moderate Positive Coupling", "Rhythmic danceability correlates positively with musical optimism, happiness, and euphoric harmonic valence."),
        ("Loudness & Acousticness", "-0.36", "Inverse Dynamic Coupling", "Acoustic and classical/mellow recordings preserve wider dynamic headroom, resulting in softer master levels."),
    ]

    for i, (pair, coeff, desc, exp) in enumerate(corr_cards):
        col_idx = i % 2
        row_idx = i // 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(2.2 + row_idx * 2.2)
        add_card(s7, left, top, Inches(5.6), Inches(1.9), bg_color=C_CARD)

        tb = s7.shapes.add_textbox(left + Inches(0.3), top + Inches(0.15), Inches(5.0), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = f"{pair}  ·  r = {coeff}"
        p1.font.size = Pt(16)
        p1.font.bold = True
        p1.font.color.rgb = C_GREEN if "-" not in coeff else C_GOLD

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT
        p2.space_before = Pt(3)

        p3 = tf.add_paragraph()
        p3.text = exp
        p3.font.size = Pt(10)
        p3.font.color.rgb = C_MUTED
        p3.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 8: MACHINE LEARNING CLUSTERING
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "08", "Machine Learning Clustering Methodology", "Unsupervised K-Means Optimization on Standardized 10D Acoustic Feature Space")

    add_card(s8, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Clustering Pipeline & Scaling")
    tb = s8.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    pipe_steps = [
        ("10 Continuous Features", "danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms."),
        ("StandardScaler Normalization", "Features zero-mean standardized (mean=0, std=1) to prevent high-magnitude features (tempo, duration) from dominating Euclidean distance."),
        ("Strict Anti-Leakage Protocol", "Genre labels, playlist names, popularity, and artist identities strictly withheld from model inputs."),
        ("Reproducibility Anchor", "K-Means configured with fixed random_state=42, n_init=10, and max_iter=300."),
    ]
    for i, (hd, body) in enumerate(pipe_steps):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"• {hd}: {body}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    add_card(s8, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Optimal Cluster Selection (K=6)")
    tb2 = s8.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    opt_steps = [
        ("Elbow Method Analysis", "Tested K in [2, 10]. WCSS exhibited steady reduction with significant elbow curvature inflection around K=6."),
        ("Silhouette & Davies-Bouldin", "Evaluated cluster separation (Silhouette=0.1441, Davies-Bouldin=2.0628). Given natural acoustic continuity in music, K=6 balanced mathematical separation with musical interpretability."),
        ("PCA Dimensionality Reduction", "Fitted 2D and 3D PCA models. First 2 principal components capture 33.8% of variance; 3 components capture 44.4%."),
        ("Cluster Robustness", "Resulting 6 clusters yield balanced song counts (between 2,800 and 8,000 tracks each) without degenerate micro-clusters."),
    ]
    for i, (hd, body) in enumerate(opt_steps):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(8)

    # ==========================================================================
    # SLIDE 9: SIX SONIC ARCHETYPES
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "09", "The Six Discovered Sonic Archetypes", "Empirical Musical Personas Derived from Unsupervised K-Means Cluster Centers")

    archetypes_info = [
        ("C0: High-Octane Anthem", "Driving Rock & Festival EDM", "Max loudness (-5.3 dB), high energy (0.83), driving tempo (132.6 BPM). Mainstage crowd power.", C_GREEN),
        ("C1: Instrumental Synth Wave", "Progressive Electronic & Synth", "Heavy instrumentation (0.75), extended arrangements (252s), minimal vocal interference.", C_TEAL),
        ("C2: Live Concert & Arena", "Stage Performance & Crowd Ambience", "Elevated room presence (liveness 0.61), amplified dynamic swing (energy 0.78).", C_GOLD),
        ("C3: Rhythmic Groove & Pop", "Dance Pop & Latin Reggaeton", "Infectious syncopation (danceability 0.74), high harmonic valence (0.68), radio appeal.", RGBColor(217, 119, 85)),
        ("C4: Lyrical Flow & Hip-Hop", "Rap, Trap & Spoken Cadence", "Elevated speechiness (0.31) over syncopated 808 percussion and modern urban cadence.", RGBColor(112, 138, 129)),
        ("C5: Acoustic Intimacy", "Organic Folk & Mellow Soul", "Dominant acoustic instruments (0.52), softer loudness (-10.5 dB), relaxed tempo.", RGBColor(154, 167, 159)),
    ]

    for i, (title, sub, desc, col) in enumerate(archetypes_info):
        col_idx = i % 3
        row_idx = i // 3
        left = Inches(0.8 + col_idx * 4.0)
        top = Inches(2.2 + row_idx * 2.3)
        add_card(s9, left, top, Inches(3.7), Inches(2.0), bg_color=C_CARD)

        tb = s9.shapes.add_textbox(left + Inches(0.25), top + Inches(0.18), Inches(3.2), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(11.5)
        p1.font.bold = True
        p1.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(9.5)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT
        p2.space_before = Pt(3)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(9)
        p3.font.color.rgb = C_MUTED
        p3.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 10: CONTENT-BASED RECOMMENDATION ENGINE
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "10", "Content-Based Recommendation Engine", "High-Dimensional Cosine Similarity Index and Multi-Mode Discovery Architecture")

    add_card(s10, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Engine Pipeline & Indexing")
    tb = s10.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    rec_arch = [
        ("Deduplicated Catalog", "Indexed 28,356 unique tracks (one per track_id) with modal genre assignment across multi-playlist tracks."),
        ("Standardized Feature Vector", "Reuses Phase 3 StandardScaler across the 10 continuous features with zero retraining."),
        ("Cosine Similarity NearestNeighbors", "Fitted NearestNeighbors(metric='cosine', algorithm='brute') over normalized unit vectors."),
        ("Sub-Millisecond Fallback", "Precomputed X_norm enables vector dot-product querying (<0.05 ms) preventing candidate starvation under strict filters."),
        ("Directional Explanations", "Computes grounded natural language explanations contrasting seed vs recommended audio dimensions."),
    ]
    for i, (hd, body) in enumerate(rec_arch):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"◈ {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(7)

    add_card(s10, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Four Specialized Recommendation Modes")
    tb2 = s10.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    modes = [
        ("Mode 1: Acoustic Similar (Default)", "Searches the full 28,356 catalog for closest acoustic matches. Mean similarity: 0.9377."),
        ("Mode 2: Acoustic Twins (Cross-Genre)", "Excludes the seed track's super genre, surfacing acoustic twins in foreign genres. Cross-genre rate: 100.0%."),
        ("Mode 3: Same Sonic Archetype", "Pins recommendations strictly to the seed song's Phase 3 cluster. Cluster purity: 100.0%."),
        ("Mode 4: Explore Beyond Your Cluster", "Excludes the seed cluster, encouraging adjacent sonic exploration. Cross-genre rate: 62.65%."),
    ]
    for i, (hd, body) in enumerate(modes):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(8)

    # ==========================================================================
    # SLIDE 11: ACOUSTIC TWINS & CROSS-GENRE DISCOVERY
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "11", "Acoustic Twins & Cross-Genre Discovery", "Empirical Proof That Raw Acoustic Textures Transcend Categorical Genre Labels")

    add_card(s11, Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.2), title="Empirical Finding: 57.33% Natural Cross-Genre Convergence")
    tb = s11.shapes.add_textbox(Inches(1.1), Inches(2.7), Inches(11.1), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = (
        "In unconstrained acoustic similarity mode (Mode 1), 57.33% of the top nearest neighbors naturally belong to a "
        "different super genre than the seed track. This rigorously demonstrates that curated streaming tags frequently disguise "
        "identical underlying audio production, tempo, and dynamic properties."
    )
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT

    # Two columns showing real examples
    add_card(s11, Inches(0.8), Inches(4.7), Inches(5.6), Inches(2.0), title="Driving Festival Energy: Rock vs EDM")
    tb1 = s11.shapes.add_textbox(Inches(1.1), Inches(5.2), Inches(5.0), Inches(1.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "High-octane Hard Rock tracks frequently match festival EDM anthems with >0.94 cosine similarity, sharing identical maximal loudness (-5.1 dB), high tempo (128 BPM), and driving master compression."
    p1.font.size = Pt(10)
    p1.font.color.rgb = C_MUTED

    add_card(s11, Inches(6.8), Inches(4.7), Inches(5.6), Inches(2.0), title="Dance Syncopation: Latin Reggaeton vs R&B")
    tb2 = s11.shapes.add_textbox(Inches(7.1), Inches(5.2), Inches(5.0), Inches(1.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Latin reggaeton tracks share close acoustic proximity (>0.93 similarity) with contemporary R&B dance tracks, both anchored by high danceability (>0.74), syncopated rhythmic tempo (95-105 BPM), and elevated valence."
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 12: TRACK COMPARISON STUDIO
    # ==========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "12", "Track Comparison Studio (Phase 5A)", "Multi-Track Acoustic Feature Comparison, Radar Geometry & Pairwise Proximity")

    features_comp = [
        ("Multi-Track Selection (2–5 Songs)", "Allows users to search and select 2 to 5 songs from the 28,356 catalog, preventing duplicate selections and displaying full verified metadata."),
        ("Normalized Radar Charts", "Maps the 10 continuous audio dimensions into a polar polygon with distinct restrained colors (#55D99A, #E9B86C, #5BC0BE, #D97755, #B388FF)."),
        ("Pairwise Cosine Similarity Matrix", "Calculates exact pairwise acoustic proximity in standardized 10D space, clearly identifying the closest acoustic pair and greatest acoustic contrast."),
        ("Original Interpretable Units", "Provides a comprehensive side-by-side comparison table featuring actual raw measurements (dB, BPM, ms) without confusing visualization scaling."),
        ("Data-Grounded Insights", "Dynamically generates 5-7 empirical observations highlighting energy extremes, rhythmic alignment, organic acoustic depth, and archetype consensus."),
    ]

    for i, (hd, body) in enumerate(features_comp):
        top_pos = Inches(2.1 + i * 0.95)
        add_card(s12, Inches(0.8), top_pos, Inches(11.7), Inches(0.82), bg_color=C_CARD)
        tb = s12.shapes.add_textbox(Inches(1.1), top_pos + Inches(0.12), Inches(11.1), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = hd
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = C_GREEN

        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(10)
        p2.font.color.rgb = C_TEXT

    # ==========================================================================
    # SLIDE 13: PLAYLIST INTELLIGENCE
    # ==========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "13", "Playlist Intelligence & Upload Engine (Phase 5B)", "CSV Ingestion, Local Catalog Matching, Acoustic Profiling, and Playlist Discovery")

    add_card(s13, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Dual-Input Ingestion & Local Matching")
    tb = s13.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    pl_points = [
        ("Dual Ingestion Formats", "Supports Input A (dataset-compatible CSV with audio features) and Input B (tracklist with track_name & artist or track_id)."),
        ("Strict Security Validation", "Validates file extension, 10 MB size limit, UTF-8 encoding, and non-empty schema in isolated memory (no server overwrites)."),
        ("Local Catalog Matching", "Matches tracks against local 28,356 catalog via O(1) track_id lookup or normalized string matching. Zero external APIs."),
        ("Transparency Reporting", "Accurately categorizes and reports matched tracks, ambiguous matches, unmatched tracks, and duplicates."),
    ]
    for i, (hd, body) in enumerate(pl_points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"• {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(7)

    add_card(s13, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Profile Analytics & Playlist Discovery")
    tb2 = s13.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    pl_analytics = [
        ("Empirical Persona Classifier", "Assigns rule-based identities (e.g. 'High-Octane Mainstage', 'Dancefloor Groove', 'Acoustic Intimacy') based on measured feature means."),
        ("Mathematical Diversity Metrics", "Computes Artist Diversity (unique/total), Genre Diversity (Gini-Simpson), Cluster Diversity (Entropy), and Acoustic Variance."),
        ("Discover Tracks for Playlist", "Aggregates the playlist centroid vector, queries nearest neighbors, and recommends fresh tracks strictly excluding existing playlist songs."),
        ("Multi-Format Export", "Enables 1-click downloads of Summary CSV, Matched Tracks CSV, Unmatched Tracks CSV, and Profile JSON."),
    ]
    for i, (hd, body) in enumerate(pl_analytics):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(7)

    # ==========================================================================
    # SLIDE 14: SYSTEM ARCHITECTURE
    # ==========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "14", "Complete System Architecture", "Five-Layer Modular Architecture Connecting Raw Data to Production Delivery")

    arch_layers = [
        ("Data Layer", "Raw CSV (32,833 obs)  ·  Cleaned Observation Dataset  ·  Deduplicated Catalog (28,356 tracks)  ·  In-Memory Uploads", C_BORDER),
        ("Processing Layer", "Data Loader  ·  Preprocessing & Auditing  ·  Modal Genre Aggregator  ·  Feature Normalization & Filtering", C_CARD_HIGH),
        ("Machine Learning Layer", "StandardScaler (10D)  ·  K-Means (K=6)  ·  PCA 2D/3D  ·  Cosine NearestNeighbors  ·  Evaluation Engine", C_CARD_HIGH),
        ("Application Layer", "Streamlit Multi-Page Controller  ·  9 Interactive Views  ·  Visual Radar/Heatmaps  ·  Export Builders", C_CARD_HIGH),
        ("Deployment Layer", "Dockerfile (Python 3.9-slim)  ·  Docker Compose  ·  Streamlit Config & Theme  ·  Production Requirements", C_GREEN),
    ]

    for i, (layer_name, details, col) in enumerate(arch_layers):
        top_pos = Inches(2.1 + i * 0.95)
        add_card(s14, Inches(0.8), top_pos, Inches(11.7), Inches(0.82), bg_color=C_CARD)
        # Left badge
        b = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top_pos, Inches(2.2), Inches(0.82))
        b.fill.solid()
        b.fill.fore_color.rgb = col
        b.line.fill.background()
        tf_b = b.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = layer_name
        p_b.font.size = Pt(11)
        p_b.font.bold = True
        p_b.font.color.rgb = C_BG if col == C_GREEN else C_TEXT
        p_b.alignment = PP_ALIGN.CENTER

        tb = s14.shapes.add_textbox(Inches(3.2), top_pos + Inches(0.2), Inches(9.0), Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = details
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT

    # ==========================================================================
    # SLIDE 15: RESULTS AND EVALUATION
    # ==========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15)
    add_header(s15, "15", "Evaluation & Empirical Benchmarks", "120-Seed Stratified Benchmark Evaluation of Clustering and Recommendation Performance")

    add_card(s15, Inches(0.8), Inches(2.2), Inches(11.7), Inches(4.5), title="Benchmark Metrics Across Recommendation Modes")
    tb = s15.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(11.1), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True

    eval_table = [
        ("Mode 1: Acoustic Similar (Default)", "0.9377", "0.8977", "0.9267", "57.33%", "2.53"),
        ("Mode 2: Acoustic Twins (Cross-Genre)", "0.9245", "0.8872", "0.9250", "100.00%", "2.65"),
        ("Mode 3: Same Sonic Archetype", "0.9338", "0.9168", "0.9275", "57.67%", "1.00"),
        ("Mode 4: Explore Beyond Your Cluster", "0.8677", "0.8407", "0.9200", "62.65%", "2.68"),
        ("Balanced Discovery Mode (Reranked)", "0.9365", "0.8963", "1.0000", "58.00%", "2.54"),
    ]

    p_header = tf.paragraphs[0]
    p_header.text = f"{'Recommendation Mode':<35} | {'Mean Sim':<10} | {'ILS':<10} | {'Artist Div':<10} | {'Cross-Genre':<12} | {'Cluster Div':<10}"
    p_header.font.size = Pt(10)
    p_header.font.bold = True
    p_header.font.color.rgb = C_GREEN
    p_header.space_after = Pt(8)

    for mode, sim, ils, adiv, xg, cdiv in eval_table:
        p = tf.add_paragraph()
        p.text = f"{mode:<35} | {sim:<10} | {ils:<10} | {adiv:<10} | {xg:<12} | {cdiv:<10}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(6)

    p_note = tf.add_paragraph()
    p_note.text = (
        "Key Insights: Balanced Discovery achieves 100% unique artists with only a 0.0012 similarity penalty. "
        "Overall Catalog Coverage: 2,378 unique tracks (8.39% of catalog) reached from only 120 queries. "
        "Sub-4ms query latency across all 28,356 compositions."
    )
    p_note.font.size = Pt(9.5)
    p_note.font.color.rgb = C_MUTED
    p_note.space_before = Pt(12)

    # ==========================================================================
    # SLIDE 16: DEPLOYMENT AND FINAL APPLICATION
    # ==========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16)
    add_header(s16, "16", "Production Deployment Readiness", "Containerization, Security Auditing, Caching Strategies, and Streamlit Cloud Support")

    deploy_cards = [
        ("Containerization (Docker)", "Self-contained multi-stage Dockerfile built on python:3.9-slim. Includes healthchecks, curl diagnostics, and automated port mapping (8501)."),
        ("Streamlit Theme & Config", "Custom .streamlit/config.toml sets dark background (#090D0B), surface (#141C17), green accents (#55D99A), 10 MB upload limits, and headless mode."),
        ("In-Memory Caching Architecture", "@st.cache_data and @st.cache_resource ensure instantaneous page switching and zero repeated computation for data and models."),
        ("Zero Secret / API Footprint", "Entire system executes 100% locally with zero external API keys, tokens, or network reliance. Completely immune to API rate limits and token expirations."),
    ]

    for i, (title, desc) in enumerate(deploy_cards):
        col_idx = i % 2
        row_idx = i // 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(2.2 + row_idx * 2.2)
        add_card(s16, left, top, Inches(5.6), Inches(1.9), bg_color=C_CARD)

        tb = s16.shapes.add_textbox(left + Inches(0.3), top + Inches(0.18), Inches(5.0), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = C_GREEN

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = C_TEXT
        p2.space_before = Pt(6)

    # ==========================================================================
    # SLIDE 17: LIMITATIONS AND FUTURE SCOPE
    # ==========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    set_slide_background(s17)
    add_header(s17, "17", "Limitations and Future Scope", "Objective Engineering Evaluation of Platform Boundaries and Roadmap Possibilities")

    add_card(s17, Inches(0.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Current Platform Limitations")
    tb = s17.shapes.add_textbox(Inches(1.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    limits = [
        ("High-Level Audio Features", "Analysis relies on Spotify's precomputed acoustic summary vectors rather than raw spectrograms or audio waveforms."),
        ("Absence of Collaborative Filtering", "Engine is strictly content-based; does not model user co-listening graphs, session skip rates, or demographic preferences."),
        ("Fixed Acoustic Corpus", "Local recommendation catalog is fixed at 28,356 songs; new commercial releases require offline feature extraction and re-indexing."),
        ("No Direct Audio Playback", "Platform intentionally excludes proprietary Spotify OAuth and DRM streaming to preserve open, reliable local execution."),
    ]
    for i, (hd, body) in enumerate(limits):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"• {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(7)

    add_card(s17, Inches(6.8), Inches(2.2), Inches(5.6), Inches(4.5), title="Future Technical Horizons")
    tb2 = s17.shapes.add_textbox(Inches(7.1), Inches(2.8), Inches(5.0), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    future = [
        ("Raw Waveform Deep Learning", "Incorporate PyTorch/torchaudio models (e.g. MERT, CLMR) for direct end-to-end representation learning from audio waveforms."),
        ("Hybrid Recommendation System", "Combine content-based acoustic proximity with implicit collaborative filtering (Matrix Factorization / LightFM)."),
        ("Dynamic DJ Auto-Mixing Engine", "Utilize tempo, key, and harmonic mixing (Camelot wheel) to generate continuous transitions for uploaded playlists."),
        ("Vector Database Integration", "Deploy Milvus, Qdrant, or FAISS index for scaling the catalog from 28K to 100M+ real-time track vectors."),
    ]
    for i, (hd, body) in enumerate(future):
        p = tf2.add_paragraph() if i > 0 else tf2.paragraphs[0]
        p.text = f"✔ {hd}: {body}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_GREEN
        p.space_after = Pt(7)

    # ==========================================================================
    # SLIDE 18: CONCLUSION & FINAL SIGN-OFF
    # ==========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    set_slide_background(s18)

    banner_fin = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.12), Inches(4.5))
    banner_fin.fill.solid()
    banner_fin.fill.fore_color.rgb = C_GREEN
    banner_fin.line.fill.background()

    tb_fin = s18.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(4.5))
    tf_fin = tb_fin.text_frame
    tf_fin.word_wrap = True

    p = tf_fin.paragraphs[0]
    p.text = "SONIC ATLAS"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = C_TEXT

    p2 = tf_fin.add_paragraph()
    p2.text = "Phase 1–5 Technical Delivery Complete · Platform Production Ready"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = C_GREEN
    p2.space_before = Pt(6)

    p3 = tf_fin.add_paragraph()
    p3.text = (
        "SONIC ATLAS delivers an independent, commercial-grade music intelligence platform that bridges the gap between "
        "raw acoustic physics and music discovery. By combining rigorous exploratory data analysis, unsupervised K-Means clustering, "
        "sub-4ms content-based recommendation, multi-track comparison, and playlist intelligence, the platform proves that "
        "musical affinity is rooted in acoustic topology, not arbitrary genre labels."
    )
    p3.font.size = Pt(13)
    p3.font.color.rgb = C_MUTED
    p3.space_before = Pt(16)

    p4 = tf_fin.add_paragraph()
    p4.text = "✔ 32,833 Observations Verified  ·  ✔ 6 Discovered Archetypes  ·  ✔ 44 Automated Unit/Integration Tests Passing  ·  ✔ Docker Containerized"
    p4.font.size = Pt(11)
    p4.font.bold = True
    p4.font.color.rgb = C_TEXT
    p4.space_before = Pt(20)

    # Save presentation
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(PRESENTATION_PATH))
    print(f"Successfully generated 18-slide executive presentation at: {PRESENTATION_PATH}")


if __name__ == "__main__":
    create_deck()
