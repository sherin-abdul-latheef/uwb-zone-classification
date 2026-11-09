# ══════════════════════════════════════════════════
# FIGURE — Zone 3→1 Transition Detection
# Matches thesis_figures/settings.py style exactly
# No title — written in Word document caption instead
# ══════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, LIVE_TRAJECTORY_PATH, PREDICTIONS_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.zone_state_machine import ZoneStateMachine

from figures.style import save_figure

# ── Paths ──────────────────────────────────────────────────────

# ── Load predictions ───────────────────────────────────────────
predictions = np.load(PREDICTIONS_PATH)
n           = len(predictions)
sample_rate = n / 80.0
t_full      = np.arange(n) / sample_rate

# ── Raw Zone 3→1 crossing timestamps ──────────────────────────
raw_z31_idx  = np.where((predictions[:-1] == 3) & (predictions[1:] == 1))[0] + 1
raw_times    = t_full[raw_z31_idx]
raw_cumcount = np.arange(1, len(raw_times) + 1)

# ── Confirmed alarms via state machine ────────────────────────
sm = ZoneStateMachine(
    window_seconds=1.5,
    dwell_seconds=0.4,
    sample_rate=sample_rate,
    alarm_from=3,
    alarm_to=1
)
confirmed_times = []
for i, pred in enumerate(predictions):
    t = round(i / sample_rate, 1)
    if sm.update(t, int(pred)):
        confirmed_times.append(t)

confirmed_cumcount = np.arange(1, len(confirmed_times) + 1)

print(f"Raw Zone 3→1 crossings : {len(raw_times)}")
print(f"Confirmed alarms       : {len(confirmed_times)} at {confirmed_times}")

# ── Figure ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))

# Raw crossings — grey
ax.step(
    raw_times, raw_cumcount,
    where='post',
    color='#888888',
    linewidth=1.5,
    label=f'Raw Zone 3\u21921 crossings (n={len(raw_times)})'
)
ax.scatter(
    raw_times, raw_cumcount,
    color='#888888',
    s=20,
    zorder=5
)

# Confirmed alarms — dark red
ax.step(
    confirmed_times, confirmed_cumcount,
    where='post',
    color='#C0392B',
    linewidth=2.0,
    label=f'Confirmed alarms after debouncing (n={len(confirmed_times)})'
)
ax.scatter(
    confirmed_times, confirmed_cumcount,
    color='#C0392B',
    s=45,
    zorder=6
)

# ── Axes ───────────────────────────────────────────────────────
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cumulative Zone 3\u21921 transition count')

ax.legend(
    loc='upper left',
    fontsize=8,
    framealpha=0.9,
    edgecolor='#CCCCCC'
)

ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.set_xlim(0, 80)
ax.set_ylim(0, max(len(raw_times), len(confirmed_times)) + 1)

plt.tight_layout()
save_figure("fig_zone31_transition_count.png")