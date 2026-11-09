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
ZONE_COLORS = {
    1: '#4393C3',   # blue
    2: '#92C964',   # green
    3: '#F4A582',   # orange
    4: '#B2182B',   # red
}

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

    # ── Colored block ──────────────────────────────────────────
    ax.axvspan(
        start, end,
        ymin      = 0.0,
        ymax      = 1.0,
        facecolor = ZONE_COLORS[zone],
        alpha     = 1.0,
        edgecolor = 'none',
    )

    # ── Label above the box ────────────────────────────────────
    ax.text(
        (start + end) / 2,
        1.08,
        f"Zone {zone}\n({count} samples)",
        ha         = 'center',
        va         = 'bottom',
        fontweight = 'bold',
        color      = 'black',
        transform  = ax.get_xaxis_transform()
    )

    # ── Dashed boundary lines — full height inside box ─────────
    if i > 0:
        ax.axvline(
            x         = start,
            color     = '#444444',
            linewidth = 1.2,
            linestyle = '--',
            alpha     = 1.0,
            zorder    = 5        # draw on top of colored blocks
        )

# ── Black border box around entire colored area ────────────────
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('black')

# ── Axes formatting ────────────────────────────────────────────
ax.set_xlabel('Sample Index')
ax.set_xlim(0, len(y))
ax.set_ylim(0, 1)
ax.set_yticks([0.5])
ax.set_yticklabels(['Zone\n(class)'])
ax.tick_params(axis='y', length=0)   # hide y tick marks, keep label

plt.tight_layout()
save_figure("fig_4_3_cross_validation.png")