import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
from sklearn.metrics import classification_report

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
from src.data_loader import load_data, split_folds
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities


BEST_K = 3  # best k from run_custom_knn_cv.py (thesis result)


def main():
    X_raw, X_zqcm, X_gqcm, X, y, X_gqcm_hermite = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )
    folds = split_folds(X, y)

    train_idx, test_idx = folds[0]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_raw_train = X_raw[train_idx]
    X_gqcm_train = X_gqcm[train_idx]

    weight_matrix = compute_weight_matrix(X_raw_train, X_gqcm_train, y_train)
    train_weights = get_train_weights(weight_matrix, y_train)

    weighted_md = compute_weighted_MD(X_test, X_train, train_weights)
    sorted_idx = np.argsort(weighted_md, axis=1)

    top_k_labels, top_k_dist = get_top_k(sorted_idx, weighted_md, y_train, k=BEST_K)
    P, zones = compute_probabilities(top_k_labels, top_k_dist)
    predictions = zones[np.argmax(P, axis=1)]

    print(f"\nDetailed report for best k = {BEST_K}\n")
    print(classification_report(y_test, predictions))

    for zone in np.unique(y_test):
        mask = y_test == zone
        zone_acc = np.mean(predictions[mask] == y_test[mask])
        print(f"Zone {zone} accuracy: {zone_acc:.4f} ({mask.sum()} samples)")


if __name__ == "__main__":
    main()