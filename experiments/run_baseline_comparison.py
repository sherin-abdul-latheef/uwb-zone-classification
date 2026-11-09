import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config.settings import RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
from src.data_loader import load_data


def main():
    _, _, _, X, y, _ = load_data(
        RAW_PATH, ZQCM_PATH, GQCM_PATH, DATA_PATH, GQCM_Hermite_path
    )  # X = NQCM features

    print("\n" + "=" * 60)
    print("SKLEARN KNN CROSS-VALIDATION COMPARISON")
    print("=" * 60)
    print(f"{'k':>6}  {'Metric':>12}  {'Weights':>10}  {'Avg Acc':>10}  {'Std':>8}")
    print("-" * 60)

    k_test = [3, 5, 7, 11, 21]
    metrics = ["euclidean", "manhattan"]
    weights = ["uniform", "distance"]

    skf = StratifiedKFold(n_splits=5, shuffle=False)

    for k in k_test:
        for metric in metrics:
            for weight in weights:
                clf = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=metric,
                    weights=weight
                )
                scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

                print(
                    f"{k:>6}  {metric:>12}  {weight:>10}  "
                    f"{scores.mean():>10.4f}  {scores.std():>8.4f}"
                )

    print("=" * 60)
    print("Paper claims: euclidean → 97%")
    print("Check above: does manhattan beat euclidean?")
    print("=" * 60)


if __name__ == "__main__":
    main()