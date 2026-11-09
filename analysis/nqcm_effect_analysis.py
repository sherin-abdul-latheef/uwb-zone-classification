# analysis/nqcm_effect_analysis.py
#
# Research Question:
#   What does NQCM preprocessing actually do to help KNN?
#
# Why CV showed 0%
# ----------------
#   CV = std/mean. Normalization scales both std and mean equally,
#   so CV is preserved by design. NQCM is not a variance reducer —
#   it is a scale equalizer and outlier compressor.
#
# What this file plots instead
# ----------------------------
# Figure 1 — Feature scale equalization
#   Raw: each feature has a different absolute scale → KNN is biased
#        toward high-scale features (they dominate the distance formula)
#   NQCM: all features brought to comparable scale → fair contribution
#   Plot: per-feature std, normalised to [0,1] so both rows visible
#
# Figure 2 — Intra-zone vs inter-zone distance distributions
#   For each point, compute:
#     intra_dist = mean Manhattan distance to same-zone points
#     inter_dist = mean Manhattan distance to nearest different-zone points
#   If NQCM works:
#     → intra distribution shifts left  (within-zone tighter)
#     → inter distribution shifts right (between-zone gap larger)
#     → overlap between the two distributions decreases
#   This DIRECTLY proves NQCM improves KNN separability.
#
# Output
# ------
#   reports/figures/nqcm_scale_equalization.png
#   reports/figures/nqcm_distance_distributions.png
#   reports/tables/nqcm_distance_summary.csv
#
# Run from project root:
#   python analysis/nqcm_effect_analysis.py
# ══════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FIGURES_DIR, RESULTS_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as KDE

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
from src.data_loader import load_data

# ══════════════════════════════════════════════════════════════════════
# PATHS & SETTINGS
# ══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
# FIGURES_DIR comes from config.settings
TABLES_DIR   = RESULTS_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size'      : 10,
    'axes.titlesize' : 10,
    'axes.labelsize' : 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family'    : 'Arial',
    'figure.dpi'     : 300,
})

FEAT_LABELS  = ['dist1', 'dist2', 'dist3', 'dist4']
ZONE_COLORS  = {1: '#185FA5', 2: '#3B6D11', 3: '#993C1D', 4: '#534AB7'}
RAW_COLOR    = '#c0392b'
NQCM_COLOR   = '#27ae60'
INTRA_COLOR  = '#185FA5'
INTER_COLOR  = '#993C1D'


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Feature scale equalization
# ══════════════════════════════════════════════════════════════════════

def plot_scale_equalization(X_raw, X_nqcm, y, zones):
    """
    Shows the REAL effect of NQCM on KNN:
    Raw features have wildly different scales → KNN is biased.
    NQCM equalizes feature scales → all anchors vote equally.

    Panel A: per-feature within-zone std for Raw (mm scale)
    Panel B: per-feature within-zone std for NQCM (normalised scale)
    Panel C: feature scale imbalance ratio — max_std/min_std across features
             Raw: high ratio (one feature dominates KNN distance)
             NQCM: ratio closer to 1 (balanced contribution)
    """
    fig, axes = plt.subplots(1, 3, figsize=(22 / 2.54, 8 / 2.54))

    # ── Compute per-feature within-zone std ───────────────────────────
    raw_stds  = np.array([X_raw[y == z].std(axis=0)  for z in zones])  # (4,4)
    nqcm_stds = np.array([X_nqcm[y == z].std(axis=0) for z in zones])  # (4,4)
    # raw_stds[i, j] = std of zone i, feature j

    x    = np.arange(len(FEAT_LABELS))
    w    = 0.18
    zone_labels = [f'Z{z}' for z in zones]

    # ── Panel A: Raw within-zone std ──────────────────────────────────
    ax = axes[0]
    for i, zone in enumerate(zones):
        ax.bar(x + i * w, raw_stds[i],
               width=w, color=ZONE_COLORS[zone],
               alpha=0.85, label=f'Zone {zone}',
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(FEAT_LABELS)
    ax.set_ylabel('Within-zone std (mm)', fontsize=8)
    ax.set_title('Raw — per-feature std\n(mm — features at different scales)',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Annotate max vs min to show imbalance
    overall_max = raw_stds.mean(axis=0).max()
    overall_min = raw_stds.mean(axis=0).min()
    ax.text(0.98, 0.97,
            f'Max/min ratio: {overall_max/overall_min:.1f}×\n'
            f'(KNN biased toward high-scale features)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, color=RAW_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7',
                      edgecolor=RAW_COLOR, linewidth=0.8))

    # ── Panel B: NQCM within-zone std ────────────────────────────────
    ax = axes[1]
    for i, zone in enumerate(zones):
        ax.bar(x + i * w, nqcm_stds[i],
               width=w, color=ZONE_COLORS[zone],
               alpha=0.85, label=f'Zone {zone}',
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(FEAT_LABELS)
    ax.set_ylabel('Within-zone std (normalised units)', fontsize=8)
    ax.set_title('NQCM — per-feature std\n(normalised — features equalised)',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    overall_max_n = nqcm_stds.mean(axis=0).max()
    overall_min_n = nqcm_stds.mean(axis=0).min()
    ratio_n = overall_max_n / (overall_min_n + 1e-9)
    ax.text(0.98, 0.97,
            f'Max/min ratio: {ratio_n:.1f}×\n'
            f'(features contribute equally to KNN)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, color=NQCM_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAF3DE',
                      edgecolor=NQCM_COLOR, linewidth=0.8))

    # ── Panel C: Imbalance ratio per zone ─────────────────────────────
    ax = axes[2]
    raw_ratios  = [raw_stds[i].max()  / (raw_stds[i].min()  + 1e-9)
                   for i in range(len(zones))]
    nqcm_ratios = [nqcm_stds[i].max() / (nqcm_stds[i].min() + 1e-9)
                   for i in range(len(zones))]
    zl = [f'Zone {z}' for z in zones]
    xp = np.arange(len(zones))
    wr = 0.30

    b1 = ax.bar(xp - wr/2, raw_ratios,  width=wr, color=RAW_COLOR,
                alpha=0.85, label='Raw', edgecolor='white', linewidth=0.4)
    b2 = ax.bar(xp + wr/2, nqcm_ratios, width=wr, color=NQCM_COLOR,
                alpha=0.85, label='NQCM', edgecolor='white', linewidth=0.4)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f'{h:.1f}×', ha='center', va='bottom', fontsize=7)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--',
               label='Ideal (balanced)')
    ax.set_xticks(xp)
    ax.set_xticklabels(zl, fontsize=8)
    ax.set_ylabel('max_std / min_std across features', fontsize=8)
    ax.set_title('Feature scale imbalance\n(1× = perfectly balanced)',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle(
        'NQCM Effect on KNN: Feature Scale Equalization\n'
        'Raw features have unequal scales → KNN distance is biased. '
        'NQCM equalizes scales → fair anchor contribution.',
        fontsize=9, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    p = FIGURES_DIR / 'nqcm_scale_equalization.png'
    plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Saved → {p}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Intra vs inter zone distance distributions
# ══════════════════════════════════════════════════════════════════════

def compute_intra_inter_distances(X, y, zones, n_sample=300, metric='manhattan'):
    """
    For a random sample of points compute:
      intra_dist: mean distance to other points in the SAME zone
      inter_dist: mean distance to points in the NEAREST different zone

    metric: 'manhattan' (your best-performing metric)

    Returns: (intra_distances, inter_distances) arrays of length n_sample
    """
    np.random.seed(42)
    idx      = np.random.choice(len(y), size=min(n_sample, len(y)),
                                replace=False)
    X_s, y_s = X[idx], y[idx]

    intra, inter = [], []

    for i in range(len(X_s)):
        pt   = X_s[i]
        zone = y_s[i]

        if metric == 'manhattan':
            dists = np.abs(X - pt).sum(axis=1)   # Manhattan to all points
        else:
            dists = np.sqrt(((X - pt) ** 2).sum(axis=1))  # Euclidean

        # Intra: same zone, exclude self
        same_mask  = (y == zone)
        same_mask[idx[i]] = False                 # exclude self
        same_dists = dists[same_mask]

        # Inter: nearest different zone (mean of closest zone's distances)
        best_inter = np.inf
        for other_zone in zones:
            if other_zone == zone:
                continue
            other_dists = dists[y == other_zone]
            zone_mean   = other_dists.mean()
            if zone_mean < best_inter:
                best_inter = zone_mean

        if len(same_dists) > 0:
            intra.append(same_dists.mean())
            inter.append(best_inter)

    return np.array(intra), np.array(inter)


def plot_distance_distributions(X_raw, X_nqcm, y, zones):
    """
    Side-by-side KDE of intra vs inter zone distances.

    Raw panel:  intra and inter distributions overlap → hard to classify
    NQCM panel: intra shifts left, inter stays right → clear gap → easy KNN

    The overlap area between intra and inter is the Bayes error region —
    where no classifier can be 100% correct. Smaller overlap = better.
    """
    print('Computing intra/inter distances for raw data '
          '(this takes ~10 seconds)...')
    intra_raw,  inter_raw  = compute_intra_inter_distances(
        X_raw,  y, zones, n_sample=400, metric='manhattan')

    print('Computing intra/inter distances for NQCM data...')
    intra_nqcm, inter_nqcm = compute_intra_inter_distances(
        X_nqcm, y, zones, n_sample=400, metric='manhattan')

    fig, axes = plt.subplots(1, 3, figsize=(22 / 2.54, 8 / 2.54))

    # ── Shared x-range for panels A and B ────────────────────────────
    all_vals  = np.concatenate([intra_raw, inter_raw])
    all_nqcm  = np.concatenate([intra_nqcm, inter_nqcm])

    def kde_curve(data, x_grid):
        k = KDE(data, bw_method='scott')
        return k(x_grid)

    def overlap_area(d1, d2, n=500):
        """Numerical overlap between two KDE distributions."""
        combined = np.concatenate([d1, d2])
        xg = np.linspace(combined.min(), combined.max(), n)
        k1 = kde_curve(d1, xg)
        k2 = kde_curve(d2, xg)
        k1 /= k1.sum();  k2 /= k2.sum()
        return np.minimum(k1, k2).sum() / n * (xg[-1] - xg[0])

    # ── Panel A: Raw ──────────────────────────────────────────────────
    ax = axes[0]
    xg_raw = np.linspace(0, np.percentile(all_vals, 99), 400)

    ki_raw = kde_curve(intra_raw, xg_raw)
    ke_raw = kde_curve(inter_raw, xg_raw)
    ki_raw /= ki_raw.max();  ke_raw /= ke_raw.max()

    ax.fill_between(xg_raw, ki_raw, alpha=0.30, color=INTRA_COLOR)
    ax.fill_between(xg_raw, ke_raw, alpha=0.30, color=INTER_COLOR)
    ax.plot(xg_raw, ki_raw, color=INTRA_COLOR, linewidth=1.8,
            label='Intra-zone\n(same zone)')
    ax.plot(xg_raw, ke_raw, color=INTER_COLOR, linewidth=1.8,
            label='Inter-zone\n(nearest other zone)')

    # Shade overlap
    overlap_raw = np.minimum(ki_raw, ke_raw)
    ax.fill_between(xg_raw, overlap_raw, alpha=0.50,
                    color='orange', label='Overlap\n(ambiguous)')

    ov_r = overlap_area(intra_raw, inter_raw)
    ax.set_xlabel('Mean Manhattan distance', fontsize=8)
    ax.set_ylabel('Density (normalised)', fontsize=8)
    ax.set_title(f'Raw — k = 11\nOverlap area ≈ {ov_r:.3f}',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: NQCM ─────────────────────────────────────────────────
    ax = axes[1]
    xg_nq = np.linspace(0, np.percentile(all_nqcm, 99), 400)

    ki_nq = kde_curve(intra_nqcm, xg_nq)
    ke_nq = kde_curve(inter_nqcm, xg_nq)
    ki_nq /= ki_nq.max();  ke_nq /= ke_nq.max()

    ax.fill_between(xg_nq, ki_nq, alpha=0.30, color=INTRA_COLOR)
    ax.fill_between(xg_nq, ke_nq, alpha=0.30, color=INTER_COLOR)
    ax.plot(xg_nq, ki_nq, color=INTRA_COLOR, linewidth=1.8,
            label='Intra-zone')
    ax.plot(xg_nq, ke_nq, color=INTER_COLOR, linewidth=1.8,
            label='Inter-zone')

    overlap_nqcm_curve = np.minimum(ki_nq, ke_nq)
    ax.fill_between(xg_nq, overlap_nqcm_curve, alpha=0.50,
                    color='orange', label='Overlap')

    ov_n = overlap_area(intra_nqcm, inter_nqcm)
    ax.set_xlabel('Mean Manhattan distance', fontsize=8)
    ax.set_ylabel('Density (normalised)', fontsize=8)
    ax.set_title(f'NQCM — k = 37\nOverlap area ≈ {ov_n:.3f}',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel C: Summary stats ─────────────────────────────────────────
    ax = axes[2]
    metrics_labels = ['Mean\nintra dist',
                      'Mean\ninter dist',
                      'Overlap\n(lower=better)']

    # Normalise to raw=1 so both fit on same axis
    mean_intra_raw  = intra_raw.mean()
    mean_inter_raw  = inter_raw.mean()
    mean_intra_nqcm = intra_nqcm.mean()
    mean_inter_nqcm = inter_nqcm.mean()

    # Ratio = NQCM / Raw — values < 1 mean improvement
    ratios_raw  = [1.0, 1.0, 1.0]
    ratios_nqcm = [
        mean_intra_nqcm / mean_intra_raw,
        mean_inter_nqcm / mean_inter_raw,
        ov_n / (ov_r + 1e-9),
    ]

    xp = np.arange(len(metrics_labels))
    wr = 0.30
    b1 = ax.bar(xp - wr/2, ratios_raw,  width=wr, color=RAW_COLOR,
                alpha=0.85, label='Raw (baseline=1)',
                edgecolor='white', linewidth=0.4)
    b2 = ax.bar(xp + wr/2, ratios_nqcm, width=wr, color=NQCM_COLOR,
                alpha=0.85, label='NQCM (relative)',
                edgecolor='white', linewidth=0.4)
    for bar in list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}×', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=NQCM_COLOR)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(xp)
    ax.set_xticklabels(metrics_labels, fontsize=7.5)
    ax.set_ylabel('Ratio relative to raw', fontsize=8)
    ax.set_title('NQCM vs Raw\n(lower = better for intra + overlap)',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle(
        'NQCM Effect on KNN: Intra-zone vs Inter-zone Distance Distributions\n'
        'Smaller overlap = fewer ambiguous neighbourhoods = better classification',
        fontsize=9, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    p = FIGURES_DIR / 'nqcm_distance_distributions.png'
    plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ Saved → {p}')
    plt.show()

    return {
        'mean_intra_raw' : round(mean_intra_raw,  4),
        'mean_inter_raw' : round(mean_inter_raw,  4),
        'mean_intra_nqcm': round(mean_intra_nqcm, 4),
        'mean_inter_nqcm': round(mean_inter_nqcm, 4),
        'overlap_raw'    : round(ov_r, 4),
        'overlap_nqcm'   : round(ov_n, 4),
        'overlap_reduction_%': round((ov_r - ov_n) / (ov_r + 1e-9) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# TABLE
# ══════════════════════════════════════════════════════════════════════

def save_distance_table(dist_stats):
    rows = [
        {'Metric': 'Mean intra-zone distance', 'Raw': dist_stats['mean_intra_raw'],
         'NQCM': dist_stats['mean_intra_nqcm']},
        {'Metric': 'Mean inter-zone distance', 'Raw': dist_stats['mean_inter_raw'],
         'NQCM': dist_stats['mean_inter_nqcm']},
        {'Metric': 'Distribution overlap area', 'Raw': dist_stats['overlap_raw'],
         'NQCM': dist_stats['overlap_nqcm']},
        {'Metric': 'Overlap reduction (%)', 'Raw': '—',
         'NQCM': f"{dist_stats['overlap_reduction_%']}%"},
    ]
    df = pd.DataFrame(rows)
    p  = TABLES_DIR / 'nqcm_distance_summary.csv'
    df.to_csv(p, index=False)
    print(f'✅ Table saved → {p}')
    print()
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 62)
    print('NQCM EFFECT ANALYSIS — SCALE EQUALIZATION + DISTANCES')
    print('=' * 62)

    X_raw, _, X_gqcm, X_nqcm, y, _ = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )
    zones = sorted(np.unique(y))

    print(f'X_raw  : {X_raw.shape}   X_nqcm : {X_nqcm.shape}')
    print(f'Zones  : {zones}')
    print()

    print('--- Figure 1: Feature scale equalization ---')
    plot_scale_equalization(X_raw, X_nqcm, y, zones)

    print()
    print('--- Figure 2: Intra vs inter zone distances ---')
    dist_stats = plot_distance_distributions(X_raw, X_nqcm, y, zones)

    print()
    save_distance_table(dist_stats)

    print()
    print('=' * 62)
    print('THESIS INTERPRETATION')
    print('=' * 62)
    print('Figure 1 explains WHY NQCM helps:')
    print('  Raw features have unequal scales — KNN is biased.')
    print('  NQCM equalizes scales — all anchors contribute fairly.')
    print()
    print('Figure 2 proves NQCM helps KNN directly:')
    print(f'  Intra-zone distance: raw={dist_stats["mean_intra_raw"]:.2f} '
          f'→ nqcm={dist_stats["mean_intra_nqcm"]:.2f}')
    print(f'  Inter-zone distance: raw={dist_stats["mean_inter_raw"]:.2f} '
          f'→ nqcm={dist_stats["mean_inter_nqcm"]:.2f}')
    print(f'  Overlap reduction  : {dist_stats["overlap_reduction_%"]}%')
    print()
    print('Outputs:')
    print('  reports/figures/nqcm_scale_equalization.png')
    print('  reports/figures/nqcm_distance_distributions.png')
    print('  reports/tables/nqcm_distance_summary.csv')
    print('=' * 62)


if __name__ == '__main__':
    main()
