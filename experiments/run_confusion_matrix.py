import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
from src.data_loader import load_data
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities

BEST_K = 3

X_raw, X_zqcm, X_gqcm,  X, y, X_gqcm_hermite = load_data(
    RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
)

for shuffle_setting in [True, False]:
    print(f"\n{'='*50}")
    print(f"shuffle = {shuffle_setting}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=5, shuffle=shuffle_setting,
                          random_state=42 if shuffle_setting else None)
    folds = list(skf.split(X, y))

    cm_total = np.zeros((4, 4), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train      = X[train_idx]
        X_test       = X[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]
        X_raw_train  = X_raw[train_idx]
        X_gqcm_train = X_gqcm[train_idx]

        weight_matrix = compute_weight_matrix(
            X_raw_train, X_gqcm_train, y_train
        )
        train_weights = get_train_weights(weight_matrix, y_train)

        weighted_md    = compute_weighted_MD(X_test, X_train, train_weights)
        sorted_indices = np.argsort(weighted_md, axis=1)

        top_k_labels, top_k_dist = get_top_k(
            sorted_indices, weighted_md, y_train, k=BEST_K
        )
        P, zones    = compute_probabilities(top_k_labels, top_k_dist)
        predictions = zones[np.argmax(P, axis=1)]

        cm = confusion_matrix(y_test, predictions, labels=[1, 2, 3, 4])
        cm_total += cm

        print(f"\nFold {fold}:")
        print(cm)

    print(f"\nTotal confusion matrix (shuffle={shuffle_setting}):")
    print(cm_total)
    print(f"Zone-wise accuracy:")
    for i, zone in enumerate([1, 2, 3, 4]):
        zone_acc = cm_total[i, i] / cm_total[i].sum()
        print(f"  Zone {zone}: {zone_acc:.4f} ({cm_total[i].sum()} samples)")