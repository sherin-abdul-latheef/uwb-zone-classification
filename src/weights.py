import pandas as pd
import numpy as np

import numpy as np

def compute_weight_matrix(X_raw_train, X_gqcm_train, y_train):
    zones      = np.unique(y_train)
    n_stations = X_raw_train.shape[1]  # 4 base stations
    counts     = np.zeros((len(zones), n_stations))

    for i, zone in enumerate(zones):
        zone_mask = y_train == zone

        # Count zeros from RAW train data
        zero_count = np.sum(
            X_raw_train[zone_mask] == 0, axis=0
        )

        # Count outliers (NaN) from GQCM train data
        nan_count = np.sum(
            np.isnan(X_gqcm_train[zone_mask]), axis=0
        )

        # Paper equation 20 → add them
        counts[i] = zero_count + nan_count

    # Normalize per zone → paper equation 20
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    weight_matrix = counts / row_sums

    return weight_matrix  # shape (4 zones, 4 stations)

def get_train_weights(weight_matrix, y_train):
    zones        = np.unique(y_train)
    zone_to_idx  = {z: i for i, z in enumerate(zones)}
    train_idx    = np.array([zone_to_idx[z] for z in y_train])
    train_weights = weight_matrix[train_idx]
    return train_weights  # shape (4320, 4)