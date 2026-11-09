# experiments/run_variance_analysis.py
#
# Variance Analysis — Fold-level accuracy dispersion
# Part 1: Levene's test for equality of variance (Raw vs NQCM)
# Part 2: Heatmap of accuracy across (k, fold) combinations
#
# Builds on the exact verified logic from sensitivity_analysis.py
# Fold indices are loaded from fold_indices.pkl (frozen from run_ablation.py)
# so this script's folds are guaranteed identical to the ablation study's.
#
# Run     : python experiments/run_variance_analysis.py
import os
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, FIGURES_DIR, FOLD_INDICES_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene

from src.weights    import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities
from src.evaluator  import evaluate
from config.settings import RAW_PATH, GQCM_PATH

# ── FONT SETTINGS — matches thesis settings.py ────────────────
plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 9,
    'axes.labelsize' : 9,
    'axes.titlesize' : 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi'     : 300,
    'savefig.dpi'    : 300,
})

# ── NQCM PATH — update if different ───────────────────────────
NQCM_PATH = DATA_PATH

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
N_SPLITS = 5

OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)



# ══════════════════════════════════════════════════════════════
# RUN ONE CONDITION — RETURNS FULL FOLD-LEVEL ACCURACY MATRIX
# Shape: (len(K_VALUES), N_SPLITS)
# folds is now passed in — NOT built inside this function.
# ══════════════════════════════════════════════════════════════

def run_condition_full(X_knn, X_raw, X_gqcm, y, k_values, use_weights, label, folds):
    acc_matrix = np.zeros((len(k_values), N_SPLITS))

    for ki, k in enumerate(k_values):
        for fi, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_knn[train_idx], X_knn[test_idx]
            y_train, y_test = y[train_idx],     y[test_idx]
            X_raw_train  = X_raw[train_idx]
            X_gqcm_train = X_gqcm[train_idx]

            weight_matrix = compute_weight_matrix(X_raw_train, X_gqcm_train, y_train)
            train_weights = get_train_weights(weight_matrix, y_train)

            if not use_weights:
                train_weights = np.ones_like(train_weights)

            weighted_md = compute_weighted_MD(X_test, X_train, train_weights)
            sorted_idx  = np.argsort(weighted_md, axis=1)
            top_k_labels, top_k_dist = get_top_k(
                sorted_idx, weighted_md, y_train, k=k
            )
            P, zones    = compute_probabilities(top_k_labels, top_k_dist)
            predictions = zones[np.argmax(P, axis=1)]
            acc = evaluate(y_test, predictions)['accuracy']
            acc_matrix[ki, fi] = acc * 100

        print(f"  [{label}] k={k:2d} done.")

    return acc_matrix


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():

    print("Loading Raw data...")
    raw_df  = pd.read_excel(RAW_PATH)
    gqcm_df = pd.read_excel(GQCM_PATH)
    X_raw   = raw_df[['dist1', 'dist2', 'dist3', 'dist4']].values
    X_gqcm  = gqcm_df[['gqcm_dist1', 'gqcm_dist2',
                         'gqcm_dist3', 'gqcm_dist4']].values
    y       = raw_df['zone_label'].values

    print("Loading NQCM data...")
    nqcm_df = pd.read_excel(NQCM_PATH)
    X_nqcm  = nqcm_df[['dist1', 'dist2', 'dist3', 'dist4']].values
    y_nqcm  = nqcm_df['zone_label'].values

    assert np.array_equal(y, y_nqcm), "ERROR: Zone labels do not match!"
    print(f"Labels aligned. Total samples: {len(y)}\n")

    # ── Load frozen fold indices — same folds used by run_ablation.py ──
    print(f"Loading fold indices from {FOLD_INDICES_PATH} ...")
    with open(FOLD_INDICES_PATH, "rb") as f:
        folds = pickle.load(f)

    max_test_idx = max(test_idx.max() for _, test_idx in folds)
    assert max_test_idx < len(y), (
        f"Fold indices go up to row {max_test_idx}, but this data has {len(y)} rows — "
        f"NQCM file row count doesn't match the ablation study's data."
    )
    print(f"Loaded {len(folds)} folds. Indices verified against row count.\n")

    print("-- Running Raw Condition (full fold matrix) --")
    raw_matrix = run_condition_full(
        X_raw, X_raw, X_gqcm, y, K_VALUES, use_weights=False, label="Raw", folds=folds
    )

    print("\n-- Running NQCM Weighted Condition (full fold matrix) --")
    nqcm_matrix = run_condition_full(
        X_nqcm, X_raw, X_gqcm, y, K_VALUES, use_weights=True, label="NQCM", folds=folds
    )

    # ════════════════════════════════════════════════════════
    # PART 1 — LEVENE'S TEST
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("LEVENE'S TEST — Equality of Variance")
    print("=" * 60)

    raw_flat  = raw_matrix.flatten()
    nqcm_flat = nqcm_matrix.flatten()

    stat, p_value = levene(raw_flat, nqcm_flat)

    print(f"Levene statistic : {stat:.4f}")
    print(f"p-value          : {p_value:.6f}")

    if p_value < 0.05:
        print("Result: REJECT null hypothesis (p < 0.05)")
        print("  -> Variance of Raw and NQCM fold accuracies")
        print("     differs SIGNIFICANTLY.")
    else:
        print("Result: FAIL TO REJECT null hypothesis (p >= 0.05)")
        print("  -> No significant difference in variance detected.")

    print(f"\nRaw  variance  : {np.var(raw_flat):.4f}")
    print(f"NQCM variance  : {np.var(nqcm_flat):.4f}")
    print(f"Variance ratio (Raw/NQCM): {np.var(raw_flat)/np.var(nqcm_flat):.2f}x")

    # ════════════════════════════════════════════════════════
        # ════════════════════════════════════════════════════════
    # PART 2 — HEATMAPS (k x fold accuracy grid)
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generating heatmaps...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(18/2.54, 7/2.54))

    fold_labels = [f"Fold {i+1}" for i in range(N_SPLITS)]
    k_labels    = [str(k) for k in K_VALUES]

    vmin = min(raw_matrix.min(), nqcm_matrix.min())
    vmax = max(raw_matrix.max(), nqcm_matrix.max())

    im0 = axes[0].imshow(raw_matrix, aspect='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    axes[0].set_title('Raw')
    axes[0].set_xticks(range(N_SPLITS))
    axes[0].set_xticklabels(fold_labels, rotation=45)
    axes[0].set_yticks(range(len(K_VALUES)))
    axes[0].set_yticklabels(k_labels)
    axes[0].set_ylabel('$k$')

    im1 = axes[1].imshow(nqcm_matrix, aspect='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    axes[1].set_title('NQCM Weighted')
    axes[1].set_xticks(range(N_SPLITS))
    axes[1].set_xticklabels(fold_labels, rotation=45)
    axes[1].set_yticks(range(len(K_VALUES)))
    axes[1].set_yticklabels([])

    # White annotations on the dark end of viridis, black on the light end
    threshold = vmin + 0.45 * (vmax - vmin)
    for ax, matrix in zip(axes, [raw_matrix, nqcm_matrix]):
        for i in range(len(K_VALUES)):
            for j in range(N_SPLITS):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.1f}",
                        ha='center', va='center', fontsize=7,
                        color='white' if val < threshold else 'black')

    fig.colorbar(im1, ax=axes, label='Accuracy (%)', shrink=0.8)

    out_path = OUT_DIR / 'fold_k_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
       main()