# figures/fig_lda_feature_space.py
"""
Visualizes class separability of the four-zone fingerprint database using
Linear Discriminant Analysis (LDA), a supervised dimensionality reduction
technique that maximizes between-class separation using the true zone
labels -- unlike PCA, which is blind to class structure and only maximizes
variance.

Companion figure to the existing pairwise anchor-distance scatter plots
(Figure 4.3): where Figure 4.3 shows two raw anchor distances at a time,
this figure compresses all four anchor distances into a single 2D view,
using the projection that best preserves class separability.

Input : NQCM fingerprint database (dist1-dist4 + zone_label), via
        config.settings.DATA_PATH — same source as every other figure.
Output: fig_4_4_lda_feature_space.png, via figures.style.save_figure
        (300 DPI, Times New Roman, thesis conventions applied centrally).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from figures.style import save_figure
from config.settings import DATA_PATH

# ColorBrewer Dark2 — consistent with other zone-colored figures
ZONE_COLORS = {1: '#1b9e77', 2: '#d95f02', 3: '#7570b3', 4: '#e7298a'}


def load_fingerprint_database(path):
    """Load the four-zone fingerprint database (final NQCM-smoothed features)."""
    df = pd.read_excel(path)
    X = df[['dist1', 'dist2', 'dist3', 'dist4']].values
    y = df['zone_label'].values
    return X, y


def fit_lda(X, y, n_components=2):
    """
    Fit LDA on the full 4D anchor-distance feature space.
    Returns the fitted model and the 2D projection of the training data.
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_2d = lda.fit_transform(X, y)
    return lda, X_2d


def plot_lda_projection(X_2d, y, lda):
    fig, ax = plt.subplots(figsize=(16/2.54, 12/2.54))

    for zone, color in ZONE_COLORS.items():
        mask = y == zone
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, alpha=0.35, s=10, label=f'Zone {zone}')

    var_ratio = lda.explained_variance_ratio_
    ax.set_xlabel(f'LD1 ({var_ratio[0]*100:.1f}% separation)')
    ax.set_ylabel(f'LD2 ({var_ratio[1]*100:.1f}% separation)')
    ax.legend(loc='best', frameon=True, fontsize=8)

    plt.tight_layout()
    save_figure("fig_4_4_lda_feature_space.png")

    print(f"Total between-class separation captured in 2D: "
          f"{var_ratio.sum()*100:.1f}%")


def main():
    X, y = load_fingerprint_database(DATA_PATH)
    lda, X_2d = fit_lda(X, y)
    plot_lda_projection(X_2d, y, lda)


if __name__ == "__main__":
    main()