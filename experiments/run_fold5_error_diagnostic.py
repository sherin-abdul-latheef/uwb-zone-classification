# experiments/run_fold5_error_diagnostic.py
#
# For Fold 5 only, at k=3: for each preprocessing condition —
#   (a) FN direction: find every real TARGET_ZONE sample that got predicted
#       as something else, and tally WHICH zone it went to.
#   (b) FP direction: find every sample predicted as TARGET_ZONE that was
#       NOT actually TARGET_ZONE, and tally WHICH zone it came from.
#
# Together these fully explain both Recall (FN side) and Precision (FP side)
# for TARGET_ZONE, using the confusion matrix directly — no estimation needed.
#
# Run: python experiments/run_fold5_error_diagnostic.py
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import FOLD_INDICES_PATH, RESULTS_DIR

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, GQCM_Hermite_path, DATA_PATH
from src.data_loader import load_data
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities

BEST_K = 3
TARGET_ZONE = 2   # the zone whose FN/FP we're tracing

ALL_ZONES = [1, 2, 3, 4]
OTHER_ZONES = [z for z in ALL_ZONES if z != TARGET_ZONE]


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


def zone_destination_tally(y_test, predictions, label):
    """FN direction: where do real TARGET_ZONE samples get predicted instead?"""
    zone_mask  = (y_test == TARGET_ZONE)
    zone_wrong = zone_mask & (predictions != TARGET_ZONE)

    total_zone  = zone_mask.sum()
    total_wrong = zone_wrong.sum()
    recall      = 1 - (total_wrong / total_zone) if total_zone > 0 else np.nan

    destinations = pd.Series(predictions[zone_wrong]).value_counts().sort_index()

    row = {
        "preprocessing": label,
        "zone_total": total_zone,
        "zone_recall": round(recall, 4),
        "zone_false_negatives": total_wrong,
    }
    for zone in OTHER_ZONES:
        row[f"predicted_as_zone_{zone}"] = int(destinations.get(zone, 0))

    return row


def zone_fp_breakdown(y_test, predictions, label):
    """FP direction: where do TARGET_ZONE's false claims come from?"""
    cm = confusion_matrix(y_test, predictions, labels=ALL_ZONES)
    # cm[i][j] = actual ALL_ZONES[i], predicted ALL_ZONES[j]

    target_col_idx = ALL_ZONES.index(TARGET_ZONE)
    fp_total = int(cm[:, target_col_idx].sum() - cm[target_col_idx, target_col_idx])

    tp = int(cm[target_col_idx, target_col_idx])
    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else np.nan

    row = {
        "preprocessing": label,
        "zone_tp": tp,
        "zone_fp_total": fp_total,
        "zone_precision": round(precision, 4),
    }
    for zone in OTHER_ZONES:
        source_idx = ALL_ZONES.index(zone)
        row[f"fp_from_zone_{zone}"] = int(cm[source_idx, target_col_idx])

    return row


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

    print(f"Zone {TARGET_ZONE} FN/FP tally — Fold 5, k={BEST_K}")
    print("=" * 75)

    rows = []
    fp_rows = []
    for label, X_knn, use_weights in conditions:
        y_test, predictions = get_fold5_predictions(
            X_knn, X_raw, X_gqcm, y, train_idx, test_idx, use_weights
        )

        row = zone_destination_tally(y_test, predictions, label)
        rows.append(row)

        fp_row = zone_fp_breakdown(y_test, predictions, label)
        fp_rows.append(fp_row)

        print(f"\n{label}")
        print(f"  Zone {TARGET_ZONE} total samples : {row['zone_total']}")
        print(f"  Recall                  : {row['zone_recall']:.4f}")
        print(f"  False negatives         : {row['zone_false_negatives']}")
        for zone in OTHER_ZONES:
            print(f"  → predicted as Zone {zone}   : {row[f'predicted_as_zone_{zone}']}")

        print(f"  Precision               : {fp_row['zone_precision']:.4f}")
        print(f"  False positives (total) : {fp_row['zone_fp_total']}")
        for zone in OTHER_ZONES:
            print(f"  ← from Zone {zone}        : {fp_row[f'fp_from_zone_{zone}']}")

    # ── Save FN table ─────────────────────────────────────────────────
    result_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"zone{TARGET_ZONE}_destination_diagnostic.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name=f"Zone{TARGET_ZONE}_FN")
        worksheet = writer.sheets[f"Zone{TARGET_ZONE}_FN"]
        for col_cells in worksheet.columns:
            max_len = max(len(str(cell.value)) for cell in col_cells) + 2
            worksheet.column_dimensions[col_cells[0].column_letter].width = max_len

    # ── Save FP table ─────────────────────────────────────────────────
    fp_df = pd.DataFrame(fp_rows)
    fp_out_path = RESULTS_DIR / f"zone{TARGET_ZONE}_fp_breakdown.xlsx"
    with pd.ExcelWriter(fp_out_path, engine="openpyxl") as writer:
        fp_df.to_excel(writer, index=False, sheet_name=f"Zone{TARGET_ZONE}_FP")
        worksheet = writer.sheets[f"Zone{TARGET_ZONE}_FP"]
        for col_cells in worksheet.columns:
            max_len = max(len(str(cell.value)) for cell in col_cells) + 2
            worksheet.column_dimensions[col_cells[0].column_letter].width = max_len

    print(f"\n✅ Saved FN breakdown → {out_path}")
    print(f"✅ Saved FP breakdown → {fp_out_path}\n")
    print("FN table:")
    print(result_df.to_string(index=False))
    print("\nFP table:")
    print(fp_df.to_string(index=False))


if __name__ == "__main__":
    main()