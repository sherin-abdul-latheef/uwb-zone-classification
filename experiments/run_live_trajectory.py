import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, GQCM_PATH, LIVE_TRAJECTORY_PATH, PREDICTIONS_PATH, RAW_PATH

import numpy as np
import pandas as pd
from src.weights import compute_weight_matrix, get_train_weights
from src.knn_custom import compute_weighted_MD, get_top_k, compute_probabilities

# --- your existing paths ---

# --- ONLY THING YOU NEED TO FILL IN: path to your live_zone1 trajectory file ---

# --- Load full training data (no split, no folds) ---
raw_df  = pd.read_excel(RAW_PATH)
gqcm_df = pd.read_excel(GQCM_PATH)
data_df = pd.read_excel(DATA_PATH)
data_df = pd.read_excel(DATA_PATH)
print(data_df.columns.tolist())   # ADD THIS LINE
X_raw   = raw_df[['dist1','dist2','dist3','dist4']].values
X_gqcm  = gqcm_df[['gqcm_dist1','gqcm_dist2','gqcm_dist3','gqcm_dist4']].values
X       = data_df[['dist1','dist2','dist3','dist4']].values
y       = data_df['zone_label'].values

# --- Compute weight matrix on the FULL training set (no fold splitting) ---
weight_matrix = compute_weight_matrix(X_raw, X_gqcm, y)
train_weights = get_train_weights(weight_matrix, y)

# --- Load trajectory, predict with k fixed at 3 ---
traj_df = pd.read_excel(LIVE_TRAJECTORY_PATH)
X_traj  = traj_df[['zqcm_GQCM_0_dist1_nqcm','zqcm_GQCM_0_dist2_nqcm','zqcm_GQCM_0_dist3_nqcm','zqcm_GQCM_0_dist4_nqcm']].values  # adjust column names if different

K = 3
weighted_md   = compute_weighted_MD(X_traj, X, train_weights)
sorted_idx    = np.argsort(weighted_md, axis=1)
top_k_labels, top_k_dist = get_top_k(sorted_idx, weighted_md, y, k=K)
P, zones      = compute_probabilities(top_k_labels, top_k_dist)
predictions   = zones[np.argmax(P, axis=1)]

np.save(PREDICTIONS_PATH, predictions)
print("Done. Total transitions:", (predictions[1:] != predictions[:-1]).sum())
print("Zone3->1 transitions:", ((predictions[:-1]==3)&(predictions[1:]==1)).sum())

from src.zone_state_machine import ZoneStateMachine

n = len(predictions)
sample_rate = n / 80.0

sm = ZoneStateMachine(window_seconds=1.5, dwell_seconds=0.4,
                       sample_rate=sample_rate, alarm_from=3, alarm_to=1)
for i, pred in enumerate(predictions):
    sm.update(round(i / sample_rate, 1), int(pred))

print("Stable changes:", len(sm.history))
print("Alarms fired:", sm.alarms)