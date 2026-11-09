# experiments/run_custom_knn_cv.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FOLD_INDICES_PATH
import numpy as np
import pandas as pd
import pickle

from config.settings import RAW_PATH, GQCM_PATH, DATA_PATH, ZQCM_PATH, GQCM_Hermite_path, k_values
from src.data_loader import load_data
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities
from src.evaluator import evaluate


def main():
    # Load data
    import pickle

    X_raw, X_zqcm, X_gqcm, X, y, X_gqcm_hermite = load_data(
    RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
)

    with open(FOLD_INDICES_PATH, "rb") as f:
         folds = pickle.load(f)
    
    n_rows = len(y)
    max_test_idx = max(test_idx.max() for _, test_idx in folds)
    assert max_test_idx < n_rows, (
        f"Fold indices go up to row {max_test_idx}, but this file only has {n_rows} rows."
    )
    # Check alignment
    print("── Data Alignment Check ──────────────────────────────")
    print(f"Raw rows  : {X_raw.shape[0]}")
    print(f"GQCM rows : {X_gqcm.shape[0]}")
    print(f"NQCM rows : {X.shape[0]}")

    assert X_raw.shape[0] == X_gqcm.shape[0] == X.shape[0], \
        "ERROR: Row count mismatch!"

    raw_df = pd.read_excel(RAW_PATH)
    y_raw = raw_df["zone_label"].values
    assert np.array_equal(y_raw, y), "ERROR: Zone labels don't match!"

    print("✅ All files aligned correctly!\n")

    # Store results
    all_results = {k: [] for k in k_values}

    # Fold loop
    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_raw_train = X_raw[train_idx]
        X_gqcm_train = X_gqcm[train_idx]

        # Compute weights
        weight_matrix = compute_weight_matrix(X_raw_train, X_gqcm_train, y_train)
        train_weights = get_train_weights(weight_matrix, y_train)

        # Compute distances once per fold
        weighted_md = compute_weighted_MD(X_test, X_train, train_weights)
        sorted_indices = np.argsort(weighted_md, axis=1)

        # k loop
        for k in k_values:
            top_k_labels, top_k_dist = get_top_k(
                sorted_indices,
                weighted_md,
                y_train,
                k=k
            )

            P, zones = compute_probabilities(top_k_labels, top_k_dist)
            predictions = zones[np.argmax(P, axis=1)]
            metrics = evaluate(y_test, predictions)

            all_results[k].append(metrics)

            print(
                f"Fold {fold} | k={k:3d} → "
                f"Accuracy: {metrics['accuracy']:.4f}  "
                f"F1: {metrics['f1']:.4f}"
            )

    # Compute averages
    print("\n── Summary ───────────────────────────────────────────")
    print(f"{'k':>6}  {'Avg Acc':>10}  {'Avg F1':>10}  {'Avg Pre':>10}  {'Avg Rec':>10}")
    print("-" * 58)

    avg_results = {}
    for k in k_values:
        avg_acc = np.mean([r["accuracy"] for r in all_results[k]])
        avg_f1 = np.mean([r["f1"] for r in all_results[k]])
        avg_pre = np.mean([r["precision"] for r in all_results[k]])
        avg_rec = np.mean([r["recall"] for r in all_results[k]])

        avg_results[k] = {
            "avg_accuracy": avg_acc,
            "avg_f1": avg_f1,
            "avg_precision": avg_pre,
            "avg_recall": avg_rec,
        }

        print(
            f"{k:>6}  {avg_acc:>10.4f}  {avg_f1:>10.4f}  "
            f"{avg_pre:>10.4f}  {avg_rec:>10.4f}"
        )

    # Print best k
    best_k = max(avg_results, key=lambda k: avg_results[k]["avg_accuracy"])
    print(f"\nBest k        = {best_k}")
    print(f"Best Accuracy = {avg_results[best_k]['avg_accuracy']:.4f}")
    print(f"Best F1       = {avg_results[best_k]['avg_f1']:.4f}")
    print(f"Best Precision= {avg_results[best_k]['avg_precision']:.4f}")
    print(f"Best Recall   = {avg_results[best_k]['avg_recall']:.4f}")


if __name__ == "__main__":
    main()