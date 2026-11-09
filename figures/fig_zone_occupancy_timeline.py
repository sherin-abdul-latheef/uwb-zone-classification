# ══════════════════════════════════════════════════
# FIGURE — Zone Occupancy Timeline
# Raw classifier output vs. debounced stable zone
# Matches thesis_figures/settings.py style exactly
# No title — written in Word document caption instead
# ══════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import PREDICTIONS_PATH

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.zone_state_machine import ZoneStateMachine

from figures.style import save_figure

# ── Load predictions ───────────────────────────────────────────

predictions = np.load(PREDICTIONS_PATH)
n           = len(predictions)
sample_rate = n / 80.0
t_full      = np.arange(n) / sample_rate

# ── Rebuild stable-zone stream ─────────────────────────────────
sm = ZoneStateMachine(
    window_seconds=1.5,
    dwell_seconds=0.4,
    sample_rate=sample_rate,
    alarm_from=3,
    alarm_to=1
)
stable_stream = np.zeros(n, dtype=int)
for i, pred in enumerate(predictions):
    sm.update(round(i / sample_rate, 1), int(pred))
    stable_stream[i] = sm.stable_zone

# ── Zone palette — matches thesis ColorBrewer palette ─────────
ZONE_COLORS = {
    1: '#1b9e77',   # Zone 1 — teal
    2: '#d95f02',   # Zone 2 — orange
    3: '#7570b3',   # Zone 3 — purple
    4: '#e7298a',   # Zone 4 — pink
}

# ── Helper — convert per-sample stream to (start, end, zone) ──
def zone_segments(stream, t):
    """Convert per-sample zone array into list of (t_start, t_end, zone)."""
    changes = np.where(np.diff(stream) != 0)[0] + 1
    starts  = np.concatenate([[0], changes])
    ends    = np.concatenate([changes, [len(stream)]])
    return [(t[s], t[e - 1], stream[s]) for s, e in zip(starts, ends)]

raw_segs    = zone_segments(predictions,   t_full)
stable_segs = zone_segments(stable_stream, t_full)

# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1,
    figsize=(16/2.54, 7/2.54),
    sharex=True
)

panel_data = [
    (axes[0], raw_segs,    'Raw prediction'),
    (axes[1], stable_segs, 'Debounced\n(stable zone)'),
]

for ax, segs, ylabel in panel_data:
    for t_start, t_end, zone in segs:
        ax.axvspan(
            t_start, t_end,
            ymin=0.0, ymax=1.0,
            facecolor=ZONE_COLORS.get(zone, '#CCCCCC'),
            alpha=1.0,
            edgecolor='none'
        )
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.set_yticks([])
    ax.set_xlim(0, 80)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)

axes[1].set_xlabel('Time (s)')

# ── Legend ─────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=ZONE_COLORS[z], label=f'Zone {z}')
    for z in [1, 2, 3, 4]
]
fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=4,
    fontsize=8,
    frameon=True,
    facecolor='white',
    edgecolor='#CCCCCC',
    bbox_to_anchor=(0.5, -0.08)
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.15, bottom=0.22)
save_figure("fig_zone_occupancy_timeline.png")