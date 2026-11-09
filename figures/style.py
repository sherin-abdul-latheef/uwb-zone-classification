"""
figures/style.py
================
Shared academic style settings for all thesis figures.
Import this at the top of every figure script.
"""

import os
import matplotlib.pyplot as plt

# ── Output folder ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")  # same as config.settings.FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Academic style ─────────────────────────────────────────────
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

def save_figure(filename):
    """Save current figure to output folder at thesis quality."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved → {path}")