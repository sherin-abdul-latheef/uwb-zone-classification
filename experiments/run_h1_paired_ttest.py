# experiments/run_h1_paired_ttest.py
#
# Hypothesis 1 (exploratory): Does NQCM (weighted) preprocessing significantly
# change performance compared to Raw (unweighted)?
#
# Tests THREE metrics separately: Accuracy, macro Recall, macro F1
# — to check whether any metric escapes the ceiling effect seen in accuracy.
#
# H0: mean difference in fold metric = 0
# H1: mean difference in fold metric != 0
# alpha = 0.05, two-tailed, df = 4
#
# Run: python experiments/run_h1_paired_ttest.py
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FOLD_INDICES_PATH

import numpy as np
from scipy.stats import ttest_rel

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, GQCM_Hermite_path, DATA_PATH
from src.data_loader import load_data
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities
from src.evaluator import evaluate

BEST_K = 3  # established best k from your k-sensitivity analysis
METRICS = ["accuracy", "recall", "f1"]  # pulled from evaluate()'s returned dict


def run_condition_all_folds(X_knn, X_raw, X_gqcm, y, folds, use_weights, label):
    """Run one preprocessing condition across ALL folds.
    Returns a dict: {"accuracy": [5 values], "recall": [5 values], "f1": [5 values]}
    """
    fold_metrics = {m: [] for m in METRICS}

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_knn[train_idx], X_knn[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_raw_train  = X_raw[train_idx]
        X_gqcm_train = X_gqcm[train_idx]

        weight_matrix = compute_weight_matrix(X_raw_train, X_gqcm_train, y_train)
        train_weights = get_train_weights(weight_matrix, y_train)

        if not use_weights:
            train_weights = np.ones_like(train_weights)

        weighted_md = compute_weighted_MD(X_test, X_train, train_weights)
        sorted_idx  = np.argsort(weighted_md, axis=1)
        top_k_labels, top_k_dist = get_top_k(sorted_idx, weighted_md, y_train, k=BEST_K)
        P, zones    = compute_probabilities(top_k_labels, top_k_dist)
        predictions = zones[np.argmax(P, axis=1)]
        metrics     = evaluate(y_test, predictions)

        for m in METRICS:
            fold_metrics[m].append(metrics[m])

        print(f"  [{label}] Fold {fold_num+1}: "
              f"Acc={metrics['accuracy']:.4f}  Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    return {m: np.array(v) for m, v in fold_metrics.items()}


def run_paired_ttest(metric_name, raw_vals, nqcm_vals, n_folds):
    """Run and print a paired t-test for one metric."""
    print(f"\n{'-'*65}")
    print(f"METRIC: {metric_name.upper()}")
    print(f"{'-'*65}")
    print(f"Raw  : {np.round(raw_vals, 4)}")
    print(f"NQCM : {np.round(nqcm_vals, 4)}")

    differences = nqcm_vals - raw_vals
    d_bar = differences.mean()
    s_d   = differences.std(ddof=1)

    print(f"Differences (NQCM - Raw): {np.round(differences, 4)}")
    print(f"Mean difference (d̄)     : {d_bar:.4f}")
    print(f"Std dev of differences  : {s_d:.4f}")

    t_stat, p_value = ttest_rel(nqcm_vals, raw_vals)
    df = n_folds - 1

    print(f"t-statistic : {t_stat:.4f}")
    print(f"p-value     : {p_value:.6f}")
    print(f"df          : {df}")

    alpha = 0.05
    if p_value < alpha:
        print(f"=> REJECT H0 (p = {p_value:.6f} < {alpha}) — significant difference detected")
    else:
        print(f"=> FAIL TO REJECT H0 (p = {p_value:.6f} >= {alpha}) — no significant difference")

    return t_stat, p_value


def main():
    # ── Load everything ──────────────────────────────────────────
    X_raw, X_zqcm, X_gqcm, X, y, X_gqcm_hermite = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )

    fold_path = FOLD_INDICES_PATH
    with open(fold_path, "rb") as f:
        folds = pickle.load(f)

    print(f"Loaded {len(folds)} folds from {fold_path}\n")
    print("=" * 65)
    print("H1 (exploratory): Raw (unweighted) vs NQCM (weighted)")
    print("Testing across THREE metrics: Accuracy, Recall, F1")
    print("=" * 65)

    # ── Condition A: Raw, unweighted ────────────────────────────
    print("\nRunning Raw (unweighted)...")
    raw_results = run_condition_all_folds(
        X_raw, X_raw, X_gqcm, y, folds, use_weights=False, label="Raw"
    )

    # ── Condition B: NQCM, weighted (full pipeline) ─────────────
    print("\nRunning NQCM (weighted)...")
    nqcm_results = run_condition_all_folds(
        X, X_raw, X_gqcm, y, folds, use_weights=True, label="NQCM"
    )

    # ── Paired t-test for EACH metric ───────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS — ALL METRICS")
    print("=" * 65)

    summary = {}
    for metric in METRICS:
        t_stat, p_value = run_paired_ttest(
            metric, raw_results[metric], nqcm_results[metric], len(folds)
        )
        summary[metric] = (t_stat, p_value)

    # ── Final summary table ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Metric':<12}{'t-statistic':<15}{'p-value':<12}{'Significant?'}")
    for metric, (t_stat, p_value) in summary.items():
        sig = "YES" if p_value < 0.05 else "no"
        print(f"{metric:<12}{t_stat:<15.4f}{p_value:<12.6f}{sig}")


if __name__ == "__main__":
    main()