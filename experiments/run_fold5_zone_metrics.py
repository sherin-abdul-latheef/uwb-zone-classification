# experiments/run_fold5_zone_metrics.py
#
# For Fold 5 only, at k=3: compute standard per-zone classification metrics
# (Accuracy, Precision, Recall, F1, Support) for every preprocessing condition.
#
# This is the clean "headline numbers" script — use zone2_destination_diagnostic.py
# (or its per-zone variants) separately when you need the FN/FP source breakdown.
#
# Run: python experiments/run_fold5_zone_metrics.py
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FOLD_INDICES_PATH, RESULTS_DIR

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, GQCM_Hermite_path, DATA_PATH
from src.data_loader import load_data
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities

BEST_K = 3
ALL_ZONES = [1, 2, 3, 4]


def get_fold5_predictions(X_knn, X_raw, X_gqcm, y, train_idx, test_idx, use_weights):
    """Run KNN for Fold 5 only, for one preprocessing condition."""
    X_train = X_knn[train_idx]
    X_test  = X_knn[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]
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

    return y_test, predictions


def per_zone_metrics(y_test, predictions, label):
    """Compute Accuracy, Precision, Recall, F1, Support for every zone."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=ALL_ZONES, zero_division=0
    )

    rows = []
    for i, zone in enumerate(ALL_ZONES):
        # one-vs-rest accuracy for this zone (matches your original table's convention)
        zone_binary_actual    = (y_test == zone)
        zone_binary_predicted = (predictions == zone)
        zone_accuracy = accuracy_score(zone_binary_actual, zone_binary_predicted)

        rows.append({
            "preprocessing": label,
            "zone": zone,
            "accuracy": round(zone_accuracy, 4),
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i]),
        })
    return rows


def main():
    # ── Load everything ──────────────────────────────────────────────
    X_raw, X_zqcm, X_gqcm, X, y, X_gqcm_hermite = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )

    fold_path = FOLD_INDICES_PATH
    if not fold_path.exists():
        raise FileNotFoundError(
            f"{fold_path} not found. Run generate_fold_indices.py first."
        )
    with open(fold_path, "rb") as f:
        folds = pickle.load(f)

    train_idx, test_idx = folds[4]  # Fold 5

    # ── Same 5 conditions as run_ablation.py ─────────────────────────
    conditions = [
        ("1 — Raw (no weights)",                    X_raw,          False),
        ("2 — ZQCM (no weights)",                   X_zqcm,         False),
        ("3 — GQCM+Hermite (no weights)",           X_gqcm_hermite, False),
        ("4 — NQCM (no weights)",                    X,             False),
        ("5 — NQCM (with weights) — full pipeline", X,              True),
    ]

    print(f"Per-Zone Metrics — Fold 5, k={BEST_K}")
    print("=" * 75)

    all_rows = []
    for label, X_knn, use_weights in conditions:
        y_test, predictions = get_fold5_predictions(
            X_knn, X_raw, X_gqcm, y, train_idx, test_idx, use_weights
        )
        rows = per_zone_metrics(y_test, predictions, label)
        all_rows.extend(rows)

        print(f"\n{label}")
        for r in rows:
            print(f"  Zone {r['zone']}: Acc={r['accuracy']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"F1={r['f1']:.4f}  Support={r['support']}")

    # ── Save combined table ──────────────────────────────────────────
    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "fold5_zone_metrics_summary.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Zone_Metrics")
        worksheet = writer.sheets["Zone_Metrics"]
        for col_cells in worksheet.columns:
            max_len = max(len(str(cell.value)) for cell in col_cells) + 2
            worksheet.column_dimensions[col_cells[0].column_letter].width = max_len

    print(f"\n✅ Saved → {out_path}\n")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()