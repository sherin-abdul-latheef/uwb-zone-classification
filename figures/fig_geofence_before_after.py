import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, FIGURES_DIR, LIVE_TRAJECTORY_PATH, PREDICTIONS_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.zone_state_machine import ZoneStateMachine

# ---- EDIT THESE TO YOUR REAL FULL PATHS ----
# ---------------------------------------------

data_df = pd.read_excel(DATA_PATH)
X_static = data_df[['dist1', 'dist2', 'dist3', 'dist4']].values
y_static = data_df['zone_label'].values

traj_df = pd.read_excel(LIVE_TRAJECTORY_PATH)
NQCM_COLS_TRAJ = traj_df.columns[:4]
X_traj = traj_df[NQCM_COLS_TRAJ].values

predictions = np.load(PREDICTIONS_PATH)
n = len(predictions)
sample_rate = n / 80.0

lda = LinearDiscriminantAnalysis(n_components=2)
X_2d_static = lda.fit_transform(X_static, y_static)
X_2d_traj = lda.transform(X_traj)

sm = ZoneStateMachine(window_seconds=1.5, dwell_seconds=0.4,
                       sample_rate=sample_rate, alarm_from=3, alarm_to=1)
stable_stream = np.zeros(n, dtype=int)
alarm_indices = []
for i, pred in enumerate(predictions):
    fired = sm.update(round(i / sample_rate, 1), int(pred))
    stable_stream[i] = sm.stable_zone
    if fired:
        alarm_indices.append(i)
alarm_indices = np.array(alarm_indices)

raw_z31_idx = np.where((predictions[:-1] == 3) & (predictions[1:] == 1))[0] + 1
raw_transitions = int(np.sum(predictions[1:] != predictions[:-1]))
stable_transitions = len(sm.history)

zone_fill  = {1: '#e35b4f', 2: '#4f8de3', 3: '#3fb37f', 4: '#e3a13f'}
zone_edge  = {1: '#8f2e26', 2: '#28497d', 3: '#1f5f42', 4: '#8a611f'}
zone_names = {1: 'zone 1', 2: 'zone 2', 3: 'zone 3', 4: 'zone 4'}

def padded_hull(points, pad=0.12):
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    return centroid + (hull_pts - centroid) * (1 + pad)

def draw_zones(ax):
    ax.set_facecolor('#eef1ea')
    for zone in [1, 2, 3, 4]:
        pts = X_2d_static[y_static == zone]
        poly_pts = padded_hull(pts)
        poly = Polygon(poly_pts, closed=True, facecolor=zone_fill[zone], edgecolor=zone_edge[zone],
                        linewidth=1.6, alpha=0.45, zorder=1)
        ax.add_patch(poly)
        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, zone_names[zone], fontsize=10, fontweight='bold', color=zone_edge[zone],
                ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)

# ---------------- LEFT: RAW (no debouncing) ----------------
draw_zones(axL)
for zone in [1, 2, 3, 4]:
    seg = predictions == zone
    axL.plot(X_2d_traj[seg, 0], X_2d_traj[seg, 1], '.', color=zone_edge[zone], markersize=2, alpha=0.55, zorder=2)
axL.scatter(X_2d_traj[raw_z31_idx, 0], X_2d_traj[raw_z31_idx, 1],
            facecolors='none', edgecolors='gray', s=60, linewidth=1.3, zorder=5,
            label=f'Raw Zone 3\u21921 crossings (n={len(raw_z31_idx)})')
axL.scatter(*X_2d_traj[0], marker='o', s=140, c='black', edgecolor='white', linewidth=1.8, zorder=6, label='Start')
axL.scatter(*X_2d_traj[-1], marker='s', s=110, c='black', edgecolor='white', linewidth=1.8, zorder=6, label='End')
axL.set_title(f'WITHOUT debouncing (raw)\n{raw_transitions} total transitions, {len(raw_z31_idx)} Zone 3\u21921 crossings', fontsize=11)
axL.legend(loc='lower left', fontsize=7.5, frameon=True, facecolor='white', framealpha=0.9)

# ---------------- RIGHT: DEBOUNCED (state machine) ----------------
draw_zones(axR)
for zone in [1, 2, 3, 4]:
    seg = stable_stream == zone
    axR.plot(X_2d_traj[seg, 0], X_2d_traj[seg, 1], '.', color=zone_edge[zone], markersize=2, alpha=0.55, zorder=2)
axR.scatter(X_2d_traj[alarm_indices, 0], X_2d_traj[alarm_indices, 1],
            facecolors='none', edgecolors='red', s=220, linewidth=2.6, zorder=6,
            label=f'Confirmed alarms (n={len(alarm_indices)})')
axR.scatter(*X_2d_traj[0], marker='o', s=140, c='black', edgecolor='white', linewidth=1.8, zorder=6, label='Start')
axR.scatter(*X_2d_traj[-1], marker='s', s=110, c='black', edgecolor='white', linewidth=1.8, zorder=6, label='End')
axR.set_title(f'WITH debouncing (state machine)\n{stable_transitions} stable changes, {len(alarm_indices)} confirmed Zone 3\u21921 alarms', fontsize=11)
axR.legend(loc='lower left', fontsize=7.5, frameon=True, facecolor='white', framealpha=0.9)

fig.suptitle('Geofenced zone map: raw classifier output vs. debounced state-machine output', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'geofence_before_after.png', dpi=300, bbox_inches='tight')
print(f"Raw: {raw_transitions} transitions, {len(raw_z31_idx)} Zone3->1")
print(f"Debounced: {stable_transitions} transitions, {len(alarm_indices)} confirmed alarms")