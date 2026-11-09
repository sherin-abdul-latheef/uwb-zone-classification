# ══════════════════════════════════════════════════
# FIGURE 4.2 — Value Range Comparison (Raw vs NQCM)
# ══════════════════════════════════════════════════
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import save_figure
from config.settings import RAW_PATH, DATA_PATH

# ── Muted ColorBrewer-derived palette — matches fig_4_1_histogram.py ──
COLOR_RAW  = '#4c9884'   # muted teal
COLOR_NQCM = '#c17a3e'   # muted amber

# ── Uniform typography constants — matches fig_4_1_histogram.py ──────
FONT_TITLE = 9
FONT_LABEL = 8
FONT_TICK  = 7
FONT_ANNOT = 6.5

# ── Data ─────────────────────────────────────────────────────────────
df_raw = pd.read_excel(RAW_PATH)
knn_df = pd.read_excel(DATA_PATH)

# Both files use the SAME column names — confirmed by fig_4_1_histogram.py
FEATURES = ['dist1', 'dist2', 'dist3', 'dist4']


def plot_box_plot():

    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54))

    # ── Shared y-axis ceiling — computed ONCE across BOTH datasets ────
    raw_max  = max(df_raw[f].dropna().max() for f in FEATURES)
    nqcm_max = max(knn_df[f].dropna().max() for f in FEATURES)
    shared_max = max(raw_max, nqcm_max)

    shared_ylim  = shared_max * 1.20   # 20% headroom above tallest box/outlier
    shared_label_y = shared_max * 1.08 # fixed label height, same across BOTH panels

    # ── Left — RAW ────────────────────────────────────────────────
    raw_data = [df_raw[f].dropna().values for f in FEATURES]

    bp1 = axes[0].boxplot(
        raw_data,
        patch_artist = True,
        tick_labels  = FEATURES,   # "labels" was renamed in matplotlib 3.9
        medianprops  = dict(color='black', linewidth=2)
    )
    for patch in bp1['boxes']:
        patch.set_facecolor(COLOR_RAW)
        patch.set_alpha(0.85)

    axes[0].set_title("Raw Features", fontsize=FONT_TITLE, fontweight='bold')
    axes[0].set_ylabel("Distance (mm)", fontsize=FONT_LABEL)
    axes[0].set_xlabel("Feature", fontsize=FONT_LABEL)
    axes[0].tick_params(labelsize=FONT_TICK)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].set_ylim(0, shared_ylim)   # ← shared scale

    for i, feat in enumerate(FEATURES):
        vals = df_raw[feat].dropna()
        rng  = vals.max() - vals.min()
        axes[0].text(
            i + 1, shared_label_y,
            f"range\n{rng:.0f}mm",
            ha         = 'center',
            fontsize   = FONT_ANNOT,
            color      = 'black',
            fontweight = 'bold',
            clip_on    = False
        )

    # ── Right — NQCM ──────────────────────────────────────────────
    nqcm_data = [knn_df[f].dropna().values for f in FEATURES]

    bp2 = axes[1].boxplot(
        nqcm_data,
        patch_artist = True,
        tick_labels  = FEATURES,   # "labels" was renamed in matplotlib 3.9
        medianprops  = dict(color='black', linewidth=2)
    )
    for patch in bp2['boxes']:
        patch.set_facecolor(COLOR_NQCM)
        patch.set_alpha(0.85)

    axes[1].set_title("NQCM Features", fontsize=FONT_TITLE, fontweight='bold')
    axes[1].set_ylabel("Distance (mm)", fontsize=FONT_LABEL)
    axes[1].set_xlabel("Feature", fontsize=FONT_LABEL)
    axes[1].tick_params(labelsize=FONT_TICK)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].set_ylim(0, shared_ylim)   # ← same shared scale as Raw panel

    for i, feat in enumerate(FEATURES):
        vals = knn_df[feat].dropna()
        rng  = vals.max() - vals.min()
        axes[1].text(
            i + 1, shared_label_y,
            f"range\n{rng:.0f}mm",
            ha         = 'center',
            fontsize   = FONT_ANNOT,
            color      = 'black',
            fontweight = 'bold',
            clip_on    = False
        )

    
    plt.tight_layout()
    save_figure("fig_4_2_box_plot.png")


if __name__ == "__main__":
    plot_box_plot()