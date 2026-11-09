# ══════════════════════════════════════════════════
# FIGURE 4.1 — Univariate Feature Distributions (Raw vs NQCM)
# ══════════════════════════════════════════════════
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import save_figure
from config.settings import RAW_PATH, DATA_PATH

# ── Muted ColorBrewer-derived palette (desaturated Dark2) ────────
COLOR_RAW  = '#4c9884'   # muted teal
COLOR_NQCM = '#c17a3e'   # muted amber

# ── Uniform typography constants ──────────────────────────────────
FONT_TITLE = 9
FONT_LABEL = 8
FONT_TICK  = 7
FONT_ROW   = 9
FONT_ANNOT = 6.5

# ── Data ─────────────────────────────────────────────────────────
df_raw = pd.read_excel(RAW_PATH)
knn_df = pd.read_excel(DATA_PATH)

FEATURES = ['dist1', 'dist2', 'dist3', 'dist4']  # same column names in both files
FEATURE_LABELS = {'dist1': ' feature 1', 'dist2': ' feature 2', 'dist3': ' feature 3', 'dist4': ' feature 4'}  # "Distance" stated once, globally

MAX_BINS = 35  # caps fragmentation on features like dist3

def freedman_diaconis_bins(data, max_bins=MAX_BINS):
    """Bin width via Freedman-Diaconis rule, capped to avoid over-fragmented histograms."""
    data = np.asarray(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    if bin_width == 0:
        return 30, 0
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    n_bins = int(np.clip(n_bins, 10, max_bins))
    return n_bins, bin_width

# ── Pass 1: compute bins + find global x-max and y-max across ALL panels ──
all_vals = {}
bin_counts = {}
global_x_max = 0
global_y_max = 0

for feat in FEATURES:
    raw_vals  = df_raw[feat].dropna().values
    nqcm_vals = knn_df[feat].dropna().values
    all_vals[feat] = (raw_vals, nqcm_vals)

    n_bins_raw,  bw_raw  = freedman_diaconis_bins(raw_vals)
    n_bins_nqcm, bw_nqcm = freedman_diaconis_bins(nqcm_vals)
    bin_counts[feat] = (n_bins_raw, n_bins_nqcm, bw_raw, bw_nqcm)

    global_x_max = max(global_x_max, raw_vals.max(), nqcm_vals.max())

    raw_hist_counts, _  = np.histogram(raw_vals,  bins=n_bins_raw)
    nqcm_hist_counts, _ = np.histogram(nqcm_vals, bins=n_bins_nqcm)
    global_y_max = max(global_y_max, raw_hist_counts.max(), nqcm_hist_counts.max())

# round up for a clean axis ceiling
global_x_max = int(np.ceil(global_x_max / 500) * 500)
global_y_max = int(np.ceil(global_y_max / 50) * 50)

# ── Build figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16 / 2.54, 10 / 2.54))

for i, feat in enumerate(FEATURES):
    raw_vals, nqcm_vals = all_vals[feat]
    n_bins_raw, n_bins_nqcm, bw_raw, bw_nqcm = bin_counts[feat]
    label = FEATURE_LABELS[feat]

    # ── Top row — RAW ─────────────────────────────────────────────
    axes[0, i].hist(
        raw_vals, bins=n_bins_raw, color=COLOR_RAW, alpha=0.85,
        edgecolor='white', linewidth=0.3
    )
    axes[0, i].set_xlim(0, global_x_max)
    axes[0, i].set_ylim(0, global_y_max)
    axes[0, i].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

    zero_count = (df_raw[feat] == 0).sum()
    if zero_count > 0:
        axes[0, i].text(
            0.05, 0.90, f"n(0)={zero_count}",
            transform=axes[0, i].transAxes, fontsize=FONT_ANNOT,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.75)
        )

    axes[0, i].set_title(label, fontsize=FONT_TITLE, fontweight='bold')
    axes[0, i].tick_params(labelsize=FONT_TICK)
    axes[0, i].set_xticklabels([])  # top and bottom rows share the same x-range

    # ── Bottom row — NQCM ────────────────────────────────────────
    axes[1, i].hist(
        nqcm_vals, bins=n_bins_nqcm, color=COLOR_NQCM, alpha=0.85,
        edgecolor='white', linewidth=0.3
    )
    axes[1, i].set_xlim(0, global_x_max)
    axes[1, i].set_ylim(0, global_y_max)
    axes[1, i].tick_params(labelsize=FONT_TICK)
    # NOTE: no per-panel xlabel here — replaced by a single shared label below

    # ── Y-axis label + tick labels only on leftmost column ─────────
    if i == 0:
        axes[0, i].set_ylabel("Count", fontsize=FONT_LABEL)
        axes[1, i].set_ylabel("Count", fontsize=FONT_LABEL)
    else:
        axes[0, i].set_ylabel("")
        axes[1, i].set_ylabel("")
        axes[0, i].set_yticklabels([])
        axes[1, i].set_yticklabels([])

    print(f"{feat}: raw bin width={bw_raw:.1f}mm ({n_bins_raw} bins), "
          f"NQCM bin width={bw_nqcm:.1f}mm ({n_bins_nqcm} bins)")

print(f"\nShared x-axis range: 0–{global_x_max} mm")
print(f"Shared y-axis range: 0–{global_y_max} (count)")

# ── Row labels — placed above each row ─────────────────────────────
fig.text(0.5, 0.965, "Raw Data", va='top', ha='center',
          fontsize=FONT_ROW, fontweight='bold', color=COLOR_RAW)
fig.text(0.5, 0.475, "NQCM Data", va='top', ha='center',
          fontsize=FONT_ROW, fontweight='bold', color=COLOR_NQCM)

# ── Single shared x-axis label for the whole figure (stated once) ──
fig.text(0.52, 0.015, "Distance (mm)", ha='center', va='bottom', fontsize=FONT_LABEL)

plt.tight_layout(rect=[0.02, 0.04, 1, 0.94])  # extra bottom margin for shared x-label
plt.subplots_adjust(hspace=0.35, wspace=0.15)
save_figure("fig_4_1_histogram.png")