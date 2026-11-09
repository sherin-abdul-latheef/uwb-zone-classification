# experiments/run_ablation.py
# Ablation study — isolates the contribution of each preprocessing stage.
# Run AFTER run_custom_knn_cv.py. Set BEST_K below to the best k found there.
#
# Distance metric is fixed to Manhattan (weighted MD) across ALL conditions —
# only the input features and the weight matrix change between conditions.
# This keeps the comparison fair: we are isolating preprocessing/weighting
# contribution, not metric choice.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import RESULTS_DIR

import numpy as np
import pandas as pd
from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, GQCM_Hermite_path, DATA_PATH
from src.data_loader import load_data, split_folds
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities
from src.evaluator import evaluate

# ── SET THIS to the best k from run_custom_knn_cv.py ──────────────────────
BEST_K = 3


def run_condition(label, folds, X_knn, y, X_raw, X_gqcm, use_weights):
    """Run 5-fold CV for one ablation condition.

    Parameters
    ----------
    label       : human-readable condition name
    folds       : list of (train_idx, test_idx) from split_folds
    X_knn       : the feature matrix that goes INTO the KNN distance calc
    y           : zone labels
    X_raw       : raw distances (needed for weight matrix — zero counting)
    X_gqcm      : GQCM distances (needed for weight matrix — NaN counting)
    use_weights : if False, replace quality weights with uniform ones
    """
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train      = X_knn[train_idx]
        X_test       = X_knn[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]
        X_raw_train  = X_raw[train_idx]
        X_gqcm_train = X_gqcm[train_idx]

        # Weight matrix is always computed from raw + gqcm (it counts
        # zeros and outliers — independent of which stage we feed to KNN)
        weight_matrix = compute_weight_matrix(X_raw_train, X_gqcm_train, y_train)
        train_weights = get_train_weights(weight_matrix, y_train)

        if not use_weights:
            # Uniform weights — every base station treated equally
            train_weights = np.ones_like(train_weights)
        
        # Manhattan distance (weighted), same metric in every condition
        weighted_md    = compute_weighted_MD(X_test, X_train, train_weights)
        sorted_indices = np.argsort(weighted_md, axis=1)

        top_k_labels, top_k_dist = get_top_k(
            sorted_indices, weighted_md, y_train, k=BEST_K
        )
        P, zones    = compute_probabilities(top_k_labels, top_k_dist)
        predictions = zones[np.argmax(P, axis=1)]
        metrics     = evaluate(y_test, predictions)
        fold_metrics.append(metrics)
        # NEW — capture Fold 5's per-zone breakdown only
        if fold == 4:
            fold5_zone_table = pd.DataFrame({
                "preprocessing": label,
                "zone": metrics["zone_labels"],
                "accuracy": metrics["accuracy_per_zone"],
                "precision": metrics["precision_per_zone"],
                "recall": metrics["recall_per_zone"],
                "f1": metrics["f1_per_zone"],
                "support": [np.sum(y_test == z) for z in metrics["zone_labels"]],
            })
    avg_acc = np.mean([r["accuracy"]  for r in fold_metrics])
    avg_f1  = np.mean([r["f1"]        for r in fold_metrics])
    avg_pre = np.mean([r["precision"] for r in fold_metrics])
    avg_rec = np.mean([r["recall"]    for r in fold_metrics])

    return dict(label=label, avg_accuracy=avg_acc, avg_f1=avg_f1,
                avg_precision=avg_pre, avg_recall=avg_rec, fold5_zone_table=fold5_zone_table)


def main():
    # ── Load all four data files ───────────────────────────────────────────
    X_raw, X_zqcm, X_gqcm,  X, y, X_gqcm_hermite = load_data(
    RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
)
    raw_check = pd.read_excel(RAW_PATH)["zone_label"].values
    assert np.array_equal(raw_check, y), "raw_df and knn_df (NQCM) rows are NOT aligned — this is likely the bug."
    # Folds built from NQCM shape (all files have identical row count)
    folds = split_folds(X, y)

    print(f"Ablation study  |  k = {BEST_K}  |  Manhattan distance  |  5-fold CV (shuffled)")
    print("=" * 75)

    # ── Four conditions ────────────────────────────────────────────────────
    #
    # A — Full pipeline:  NQCM features  + quality weights
    # B — No NQCM:        ZQCM+Hermite features (zeros filled, no NaN gaps)
    #                      + quality weights   → tests value of NQCM smoothing
    # C — No weights:     NQCM features  + uniform weights
    #                      → tests value of the quality weight matrix
    # D — Raw baseline:   Raw features   + uniform weights
    #                      → tests value of preprocessing entirely

    conditions = [
    ("1 — Raw (no weights)",                    X_raw,          False),
    ("2 — ZQCM (no weights)",                   X_zqcm,         False),
    ("3 — GQCM+Hermite (no weights)",           X_gqcm_hermite, False),
    ("4 — NQCM (no weights)",                   X,              False),
    ("5 — NQCM (with weights) — full pipeline", X,              True),
]

    results = []
    fold5_tables = []
    for label, X_knn, use_weights in conditions:
        r = run_condition(label, folds, X_knn, y, X_raw, X_gqcm, use_weights)
        results.append(r)
        fold5_tables.append(r["fold5_zone_table"])
        print(f"{label}")
        print(f"   Accuracy={r['avg_accuracy']:.4f}  F1={r['avg_f1']:.4f}  "
              f"Pre={r['avg_precision']:.4f}  Rec={r['avg_recall']:.4f}")
        print()
    
    # NEW — combine and save
    # ── Save Fold 5 per-zone table as a genuine Excel file ──────────────
    fold5_combined = pd.concat(fold5_tables, ignore_index=True)
    fold5_combined = fold5_combined.round({"accuracy": 4, "precision": 4, "recall": 4, "f1": 4})

    fold5_out_path = RESULTS_DIR / "fold5_zone_metrics_all_conditions.xlsx"
    with pd.ExcelWriter(fold5_out_path, engine="openpyxl") as writer:
        fold5_combined.to_excel(writer, index=False, sheet_name="Fold5_Per_Zone")
        worksheet = writer.sheets["Fold5_Per_Zone"]
        for col_cells in worksheet.columns:
            max_len = max(len(str(cell.value)) for cell in col_cells) + 2
            worksheet.column_dimensions[col_cells[0].column_letter].width = max_len

    print(f"\n✅ Saved Fold 5 per-zone metrics (all conditions) → {fold5_out_path}\n")
    print(fold5_combined.to_string(index=False))


if __name__ == "__main__":
    main()