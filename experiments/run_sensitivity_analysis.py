# experiments/run_sensitivity_analysis.py
#
# Sensitivity Analysis — Weighted Manhattan KNN
# Conditions : Raw (uniform weights) vs NQCM Weighted (quality weights)
# k values   : [1, 3, 5, 7, 9, 11, 13, 15]
# CV         : 5-fold Stratified, shuffle=False
#
# Built directly from the verified isolated test that reproduced
# the correct Raw baseline (97.20% ~ 97.22%) at k=3.
#
# Run     : python experiments/run_sensitivity_analysis.py
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, FIGURES_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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

# ── NQCM PATH — update if different from settings.py DATA_PATH ───
NQCM_PATH = DATA_PATH

# ── SETTINGS ─────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
N_SPLITS = 5

# ── COLORBREWER COLORS ────────────────────────────────────────
COLOR_RAW  = '#F4A582'
COLOR_NQCM = '#4393C3'

# ── OUTPUT ────────────────────────────────────────────────────
OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# RUN ONE CONDITION ACROSS ALL k VALUES
# (identical structure to the verified isolated test)
# ══════════════════════════════════════════════════════════════

def run_condition(X_knn, X_raw, X_gqcm, y, k_values, use_weights, label):
    """
    X_knn       : feature matrix fed into the classifier (Raw or NQCM)
    X_raw       : raw distances — used for weight matrix (zero counting)
    X_gqcm      : GQCM distances — used for weight matrix (NaN counting)
    use_weights : False -> uniform weights (Raw condition)
                  True  -> quality weight matrix (NQCM Weighted condition)
    """
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    folds = list(skf.split(X_knn, y))

    means, stds = [], []

    for k in k_values:
        fold_accs = []

        for train_idx, test_idx in folds:
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
            fold_accs.append(acc)

        m = np.mean(fold_accs) * 100
        s = np.std(fold_accs)  * 100
        means.append(m)
        stds.append(s)
        print(f"  [{label}] k={k:2d} -> {m:.2f}% +/- {s:.4f}%")

    return means, stds


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

    assert np.array_equal(y, y_nqcm), \
        "ERROR: Zone labels do not match between Raw and NQCM files!"
    print(f"Labels aligned. Total samples: {len(y)}\n")

    # ── Raw condition: uniform weights ─────────────────────────
    print("-- Raw Condition (uniform weights) --")
    raw_means, raw_stds = run_condition(
        X_raw, X_raw, X_gqcm, y, K_VALUES,
        use_weights=False, label="Raw"
    )

    # ── NQCM Weighted condition: quality weight matrix ─────────
    print("\n-- NQCM Weighted Condition (quality weights) --")
    nqcm_means, nqcm_stds = run_condition(
        X_nqcm, X_raw, X_gqcm, y, K_VALUES,
        use_weights=True, label="NQCM"
    )

    # ── Print summary table ─────────────────────────────────────
    print("\n-- Summary Table --")
    print(f"{'k':>4} {'Raw(%)':>10} {'Raw Std':>10} "
          f"{'NQCM(%)':>10} {'NQCM Std':>10}")
    print("-" * 48)
    for i, k in enumerate(K_VALUES):
        print(f"{k:>4} {raw_means[i]:>10.2f} {raw_stds[i]:>10.4f} "
              f"{nqcm_means[i]:>10.2f} {nqcm_stds[i]:>10.4f}")

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16/2.54, 7/2.54))

    ax.errorbar(K_VALUES, raw_means,  yerr=raw_stds,
                fmt='o-',  color=COLOR_RAW,  linewidth=1.5,
                markersize=5, capsize=3, label='Raw')

    ax.errorbar(K_VALUES, nqcm_means, yerr=nqcm_stds,
                fmt='s--', color=COLOR_NQCM, linewidth=1.5,
                markersize=5, capsize=3, label='NQCM Weighted')

    ax.axvline(x=3, color='gray', linestyle=':', linewidth=1.0,
               label='Selected k = 3')

    ax.set_xlabel('Number of Nearest Neighbors (k)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_xticks(K_VALUES)
    ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0))
    ax.grid(True, linestyle='--', alpha=0.4)

    raw_lower  = [m - s for m, s in zip(raw_means, raw_stds)]
    raw_upper  = [m + s for m, s in zip(raw_means, raw_stds)]
    nqcm_lower = [m - s for m, s in zip(nqcm_means, nqcm_stds)]
    nqcm_upper = [m + s for m, s in zip(nqcm_means, nqcm_stds)]

    lowest_point  = min(raw_lower + nqcm_lower)
    highest_point = max(raw_upper + nqcm_upper)

    ax.set_ylim(lowest_point - 1, min(highest_point + 1, 102)) 

    out_path = OUT_DIR / 'fig_4_4_sensitivity_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
