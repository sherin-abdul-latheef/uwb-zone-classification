# ══════════════════════════════════════════════════
# FIGURE 4.3 — Data Collection Sequence
# ══════════════════════════════════════════════════
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import save_figure

# ── Data ───────────────────────────────────────────────────────
ZONE_COL  = 'zone_label'

df = pd.read_excel(DATA_PATH)
y  = df[ZONE_COL].values

# ── Zone colors ────────────────────────────────────────────────
ZONE_COLORS = {1: '#1B9E77', 2: '#D95F02', 3: '#7570B3', 4: '#E7298A'}



# ── Compute temporal zone boundaries ──────────────────────────
changes    = np.where(np.diff(y) != 0)[0]
boundaries = np.concatenate([[0], changes + 1, [len(y)]])
zone_order = y[np.concatenate([[0], changes + 1])]

print(f"Collection order : {list(zone_order)}")
print(f"Boundaries at    : {list(changes)}")

# ── Build figure ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16/2.54, 5/2.54))

for i, zone in enumerate(zone_order):
    start = boundaries[i]
    end   = boundaries[i + 1]
    count = end - start

    # Filled bar segment
    ax.axvspan(
        start, end,
        ymin      = 0.0,
        ymax      = 1.0,
        facecolor = ZONE_COLORS[zone],
        alpha     = 1.0,
        edgecolor = 'none',
    )

    # Zone label above bar
    ax.text(
        (start + end) / 2,
        1.05,
        f"Zone {zone}\n({count} samples)",
        ha        = 'center',
        va        = 'bottom',
        fontweight = 'bold',
        color     = 'black',
        transform = ax.get_xaxis_transform()
    )

    # Dashed boundary line
    if i > 0:
        ax.axvline(
            x         = start,
            color     = '#444444',
            linewidth = 1.2,
            linestyle = '--',
            alpha     = 1.0
        )

# ── Axes formatting ────────────────────────────────────────────
ax.set_xlabel('Sample Index')
ax.set_xlim(0, len(y))
ax.set_ylim(0, 1)
ax.set_yticks([0.5])
ax.set_yticklabels(['Zone\n(class)'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.tight_layout()
save_figure("fig_4_2_cross_validation.png")