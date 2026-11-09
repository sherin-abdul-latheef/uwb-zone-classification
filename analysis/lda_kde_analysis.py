# analysis/lda_kde_analysis.py
#
# Research Question:
#   Does NQCM preprocessing improve zone separability in the
#   Fisher LDA projection of the 4D UWB feature space?
#
# Output:
#   reports/figures/lda_kde_raw_vs_nqcm.png  ← main figure
#   reports/tables/lda_eigenvalue_summary.csv ← quantified result
#
# Run from project root:
#   python analysis/lda_kde_analysis.py
# ══════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path

# ── Make project root importable (same pattern as run_custom_knn_cv.py) ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FIGURES_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
from src.data_loader import load_data

# ══════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
# FIGURES_DIR comes from config.settings
TABLES_DIR   = RESULTS_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# PLOT SETTINGS  — matches plots.py rcParams exactly
# ══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.size'      : 11,
    'axes.titlesize' : 12,
    'axes.labelsize' : 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family'    : 'Arial',
    'figure.dpi'     : 300,
})

# ── Zone colours — same palette used in fingerprint plots ─────────────
ZONE_COLORS = {1: '#185FA5', 2: '#3B6D11', 3: '#993C1D', 4: '#534AB7'}
ZONE_FILLS  = {1: '#E6F1FB', 2: '#EAF3DE', 3: '#FAECE7', 4: '#EEEDFE'}

# ══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def fit_lda(X, y):
    """
    Fit Fisher LDA on 4D feature matrix X with zone labels y.

    Returns
    -------
    X_proj    : (N, 2) projected coordinates in LD1-LD2 space
    lda       : fitted LinearDiscriminantAnalysis object
    explained : array of explained between-class variance ratios per axis
                e.g. [0.82, 0.14] means LD1 holds 82% of between-class info
    """
    lda       = LinearDiscriminantAnalysis(n_components=2)
    X_proj    = lda.fit_transform(X, y)
    explained = lda.explained_variance_ratio_
    return X_proj, lda, explained


def draw_kde_contours(ax, X_proj, y, zones,
                      levels=(0.25, 0.50, 0.75, 0.95)):
    """
    For each zone, fit a 2D KDE on the LDA-projected points,
    then draw filled + line contours at the requested density levels.

    levels are fractions of peak density:
        0.25 = outer extent  (25% of peak — the tail)
        0.50 = mid boundary
        0.75 = dense region
        0.95 = tight core
    """
    # Grid covering the full projection space with padding
    x_min = X_proj[:, 0].min() - 0.8
    x_max = X_proj[:, 0].max() + 0.8
    y_min = X_proj[:, 1].min() - 0.8
    y_max = X_proj[:, 1].max() + 0.8

    xx, yy      = np.mgrid[x_min:x_max:220j, y_min:y_max:220j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    for zone in zones:
        mask = y == zone
        pts  = X_proj[mask]

        if pts.shape[0] < 5:
            continue

        # Fit KDE with Scott's bandwidth — data-driven, no manual tuning
        kde = gaussian_kde(pts.T, bw_method='scott')
        zz  = kde(grid_coords).reshape(xx.shape)

        # Normalise so peak = 1.0 → levels are intuitive fractions
        zz /= zz.max()

        col  = ZONE_COLORS[zone]
        fill = ZONE_FILLS[zone]

        # Filled zone territory at outermost level (lightest shading)
        ax.contourf(xx, yy, zz,
                    levels=[levels[0], 1.0],
                    colors=[fill],
                    alpha=0.30,
                    zorder=1)

        # Contour lines — progressively thicker toward core
        ax.contour(xx, yy, zz,
                   levels=list(levels),
                   colors=[col],
                   linewidths=[0.5, 0.7, 1.0, 1.5],
                   linestyles=['dotted', 'dashed', 'solid', 'solid'],
                   zorder=3)

        # Scatter points — transparent, behind contours
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=col, s=4, alpha=0.25,
                   edgecolors='none', zorder=2)


def build_legend(zones):
    """Return legend patches — one per zone, matching fill + border colour."""
    return [
        mpatches.Patch(
            facecolor=ZONE_FILLS[z],
            edgecolor=ZONE_COLORS[z],
            linewidth=1.2,
            label=f'Zone {z}'
        )
        for z in zones
    ]


# ══════════════════════════════════════════════════════════════════════
# MAIN PLOT FUNCTION
# ══════════════════════════════════════════════════════════════════════

def plot_lda_kde(X_raw, X_nqcm, y):
    """
    Side-by-side LDA + KDE contour figure.

    Left panel  : Raw features   (k=11 — locally adaptive boundary context)
    Right panel : NQCM features  (k=37 — globally stable boundary context)

    Each panel uses its own LDA fit — both share the same zone labels y.
    """
    zones = sorted(np.unique(y))

    # ── Fit LDA independently on each feature set ─────────────────────
    print("Fitting LDA on raw features...")
    X_raw_proj,  _, ev_raw   = fit_lda(X_raw,  y)

    print("Fitting LDA on NQCM features...")
    X_nqcm_proj, _, ev_nqcm  = fit_lda(X_nqcm, y)

    # ── Figure layout ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18 / 2.54, 9 / 2.54))

    panels = [
        (axes[0], X_raw_proj,  ev_raw,  "Raw Features",  "k = 11"),
        (axes[1], X_nqcm_proj, ev_nqcm, "NQCM Features", "k = 37"),
    ]

    for ax, X_proj, ev, feat_label, k_note in panels:

        draw_kde_contours(ax, X_proj, y, zones)

        ax.set_xlabel(
            f"LD1  ({ev[0] * 100:.1f}% between-class variance)",
            fontsize=9
        )
        ax.set_ylabel(
            f"LD2  ({ev[1] * 100:.1f}% between-class variance)",
            fontsize=9
        )
        ax.set_title(
            f"LDA Projection — {feat_label}\n"
            f"KDE contours at 25 / 50 / 75 / 95% density  |  {k_note}",
            fontsize=9, fontweight='bold'
        )
        ax.legend(
            handles=build_legend(zones),
            fontsize=7,
            framealpha=0.85,
            loc='upper right'
        )
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.18, linewidth=0.5)

    plt.suptitle(
        "Fisher LDA — Zone Separability: Raw vs NQCM\n"
        "4D UWB Feature Space projected onto LD1 × LD2",
        fontsize=10, fontweight='bold', y=1.03
    )
    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────
    out_path = FIGURES_DIR / "lda_kde_raw_vs_nqcm.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure saved → {out_path}")
    plt.show()

    return ev_raw, ev_nqcm


# ══════════════════════════════════════════════════════════════════════
# TABLE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def save_eigenvalue_table(ev_raw, ev_nqcm):
    """
    Save discriminant axis summary to reports/tables/ as CSV.

    Columns
    -------
    Discriminant axis          : LD1, LD2
    Raw  — % between-class var : how much between-class variance LD captures before NQCM
    NQCM — % between-class var : same after NQCM
    Change (pp)                : percentage-point improvement
    Interpretation             : plain-English reading of the change
    """
    rows = []
    for i, (r, n) in enumerate(zip(ev_raw, ev_nqcm), start=1):
        change = (n - r) * 100
        if change > 2:
            interp = "Improved — NQCM concentrates separation"
        elif change < -2:
            interp = "Decreased — variance redistributed across axes"
        else:
            interp = "Stable — preprocessing preserved axis structure"

        rows.append({
            "Discriminant axis"           : f"LD{i}",
            "Raw  — % between-class var." : round(r * 100, 2),
            "NQCM — % between-class var." : round(n * 100, 2),
            "Change (pp)"                 : round(change, 2),
            "Interpretation"              : interp,
        })

    df = pd.DataFrame(rows)

    out_path = TABLES_DIR / "lda_eigenvalue_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Table  saved → {out_path}")
    print()
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("RESEARCH QUESTION: LDA ZONE SEPARABILITY — RAW vs NQCM")
    print("=" * 62)

    # ── Load data — same call as all other experiment files ───────────
    X_raw, _, X_gqcm, X_nqcm, y, _ = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )

    print(f"X_raw  shape : {X_raw.shape}   ← raw dist1–dist4")
    print(f"X_nqcm shape : {X_nqcm.shape}  ← NQCM dist1–dist4")
    print(f"y      shape : {y.shape}   ← zone labels 1–4")
    print(f"Zones        : {sorted(np.unique(y))}")
    print()

    # ── Generate figure ───────────────────────────────────────────────
    ev_raw, ev_nqcm = plot_lda_kde(X_raw, X_nqcm, y)

    # ── Save quantified result ────────────────────────────────────────
    print()
    save_eigenvalue_table(ev_raw, ev_nqcm)

    # ── Console interpretation ────────────────────────────────────────
    print()
    print("=" * 62)
    print("INTERPRETATION SUMMARY")
    print("=" * 62)
    print(f"Raw  LD1 captures : {ev_raw[0]  * 100:.1f}% of between-class variance")
    print(f"NQCM LD1 captures : {ev_nqcm[0] * 100:.1f}% of between-class variance")
    print(f"LD1 change        : {(ev_nqcm[0] - ev_raw[0]) * 100:+.1f} percentage points")
    print()

    if ev_nqcm[0] > ev_raw[0]:
        print("✅ NQCM concentrates more between-class variance into LD1.")
        print("   The single best discriminant axis becomes more powerful")
        print("   after preprocessing — zones are more linearly separable.")
    elif ev_nqcm[0] < ev_raw[0] - 0.02:
        print("ℹ️  LD1 ratio decreased — but check the plot.")
        print("   This can happen when NQCM makes zones so tight that")
        print("   less variance is needed to separate them — a good thing.")
    else:
        print("ℹ️  LD1 ratio is stable — inspect the plot for visual")
        print("   evidence of tighter, more separated zone blobs.")

    print()
    print(f"Outputs written to:")
    print(f"  Figure : reports/figures/lda_kde_raw_vs_nqcm.png")
    print(f"  Table  : reports/tables/lda_eigenvalue_summary.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()