import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, FIGURES_DIR, LIVE_TRAJECTORY_PATH, PREDICTIONS_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.zone_state_machine import ZoneStateMachine

# --- Paths (adjust if needed) ---

# --- Load static DB (for LDA background) and real predictions ---
data_df = pd.read_excel(DATA_PATH)
X_static = data_df[['dist1', 'dist2', 'dist3', 'dist4']].values
y_static = data_df['zone_label'].values

traj_df = pd.read_excel(LIVE_TRAJECTORY_PATH)
predictions = np.load(PREDICTIONS_PATH)
n = len(predictions)
sample_rate = n / 80.0
t_full = np.arange(n) / sample_rate

# --- Identify EVERY raw Zone3->1 crossing (all 15) and mark which become confirmed alarms ---
raw_z31_idx = np.where((predictions[:-1] == 3) & (predictions[1:] == 1))[0] + 1

sm = ZoneStateMachine(window_seconds=1.5, dwell_seconds=0.4,
                       sample_rate=sample_rate, alarm_from=3, alarm_to=1)
confirmed_times = []
for i, pred in enumerate(predictions):
    t = round(i / sample_rate, 1)
    fired = sm.update(t, int(pred))
    if fired:
        confirmed_times.append(t)

print(f"Raw Zone3->1 crossings: {len(raw_z31_idx)}")
print(f"Confirmed alarms: {len(confirmed_times)} at {confirmed_times}")

# =========================================================
# STATIC FIGURE FOR THESIS: cumulative count over time
# =========================================================
raw_times = t_full[raw_z31_idx]
raw_cumcount = np.arange(1, len(raw_times) + 1)
confirmed_cumcount = np.arange(1, len(confirmed_times) + 1)

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.step(raw_times, raw_cumcount, where='post', color='#999999', linewidth=1.8,
        label=f'Raw Zone 3\u21921 crossings (n={len(raw_times)})')
ax.step(confirmed_times, confirmed_cumcount, where='post', color='#D4453A', linewidth=2.4,
        label=f'Confirmed alarms after debouncing (n={len(confirmed_times)})')
ax.scatter(raw_times, raw_cumcount, color='#999999', s=25, zorder=5)
ax.scatter(confirmed_times, confirmed_cumcount, color='#D4453A', s=60, zorder=6)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Cumulative Zone 3\u21921 transition count')
ax.set_title('Effect of debouncing on Zone 3\u21921 transition detections\n(15 raw crossings \u2192 5 confirmed alarms)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'zone31_transition_counter.png', dpi=300)
print("Saved static figure: zone31_transition_counter.png")

# =========================================================
# ANIMATION (portfolio/demo use only — NOT for thesis document)
# =========================================================
lda = LinearDiscriminantAnalysis(n_components=2)
X_2d_static = lda.fit_transform(X_static, y_static)
NQCM_COLS_TRAJ = traj_df.columns[:4]  # adjust to your real trajectory column names
X_2d_traj = lda.transform(traj_df[NQCM_COLS_TRAJ].values)

zone_colors = {1: '#1b9e77', 2: '#d95f02', 3: '#7570b3', 4: '#e7298a'}
FRAME_STEP = 15
idx = np.arange(0, n, FRAME_STEP)

fig2, ax2 = plt.subplots(figsize=(6.5, 6.5), dpi=110)
for zone, color in zone_colors.items():
    mask = y_static == zone
    ax2.scatter(X_2d_static[mask, 0], X_2d_static[mask, 1], c=color, alpha=0.1, s=6)

dot = ax2.scatter([], [], s=100, edgecolor='white', linewidth=1.2, zorder=6)
counter_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=12,
                        va='top', ha='left', fontweight='bold')
ax2.set_xticks([]); ax2.set_yticks([])
plt.tight_layout()

raw_set = set(raw_z31_idx.tolist())

def update(frame_i):
    i = idx[frame_i]
    dot.set_offsets([X_2d_traj[i]])
    dot.set_color(zone_colors[predictions[i]])
    count_so_far = int(np.sum(raw_z31_idx <= i))
    counter_text.set_text(f'Zone 3\u21921 crossings: {count_so_far} / 15')
    return dot, counter_text

ani = animation.FuncAnimation(fig2, update, frames=len(idx), interval=55, blit=False)
ani.save('zone31_counting_animation.gif', writer='pillow', fps=18)
print("Saved animation (demo use only): zone31_counting_animation.gif")