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


data_df = pd.read_excel(DATA_PATH)
X_static = data_df[['dist1', 'dist2', 'dist3', 'dist4']].values
y_static = data_df['zone_label'].values

traj_df = pd.read_excel(LIVE_TRAJECTORY_PATH)
NQCM_COLS_TRAJ = traj_df.columns[:4]   # confirm against your real trajectory columns
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
for i, pred in enumerate(predictions):
    sm.update(round(i / sample_rate, 1), int(pred))
    stable_stream[i] = sm.stable_zone

flicker_mask = predictions != stable_stream

# --- Geofencing map palette, matching the scatter matrix ---
zone_fill = {1: '#3B7EB8', 2: '#7DC242', 3: '#F4A582', 4: '#B2182B'}
zone_edge = {1: '#1F4E75', 2: '#4A7A22', 3: '#C46A45', 4: '#6E0F1B'}
zone_names = {1: 'zone 1', 2: 'zone 2', 3: 'zone 3', 4: 'zone 4'}

fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
ax.set_facecolor('#eef1ea')   # soft map-like background, like the reference image

def padded_hull(points, pad=0.12):
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    return centroid + (hull_pts - centroid) * (1 + pad)

for zone in [1, 2, 3, 4]:
    pts = X_2d_static[y_static == zone]
    poly_pts = padded_hull(pts)
    poly = Polygon(poly_pts, closed=True, facecolor=zone_fill[zone], edgecolor=zone_edge[zone],
                    linewidth=2, alpha=0.55, zorder=1)
    ax.add_patch(poly)
    cx, cy = pts.mean(axis=0)
    ax.text(cx, cy, zone_names[zone], fontsize=13, fontweight='bold',
            color=zone_edge[zone], ha='center', va='center', zorder=4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

# --- Debounced path (clean line, like a tracked device path) ---
ax.plot(X_2d_traj[:, 0], X_2d_traj[:, 1], color='#2b2b2b', linewidth=1.1, alpha=0.55, zorder=2)

# --- Raw flicker points (noise the state machine filtered) ---
ax.scatter(X_2d_traj[flicker_mask, 0], X_2d_traj[flicker_mask, 1],
           facecolors='none', edgecolors='red', s=55, linewidth=1.4, zorder=5,
           label=f'Raw flicker (filtered), n={flicker_mask.sum()}')

# --- Start / end markers, like a tracking-device icon ---
ax.scatter(*X_2d_traj[0], marker='o', s=180, c='black', edgecolor='white', linewidth=2, zorder=6, label='Trajectory start')
ax.scatter(*X_2d_traj[-1], marker='s', s=140, c='black', edgecolor='white', linewidth=2, zorder=6, label='Trajectory end')

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('Geofenced zone map (LDA-projected feature space)\nDebounced trajectory with raw classifier flicker highlighted',
              fontsize=12, pad=14)
ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'geofence_style_map.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: geofence_style_map.png")
print(f"Flicker points shown: {flicker_mask.sum()} / {n}")