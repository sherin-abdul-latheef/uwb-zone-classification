import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_data(raw_path, ZQCM_PATH,gqcm_path, data_path, GQCM_Hermite_path, ):

    raw_df  = pd.read_excel(raw_path)
    zqcm_df = pd.read_excel(ZQCM_PATH)
    gqcm_df = pd.read_excel(gqcm_path)
    knn_df  = pd.read_excel(data_path)          # total_nqcm.xlsx
    gqcmhermite_df = pd.read_excel(GQCM_Hermite_path)
    

    # ── Raw features ───────────────────────────────────────────────
    X_raw  = raw_df[['dist1', 'dist2',
                      'dist3', 'dist4']].values
    X_zqcm = zqcm_df[['dist1', 'dist2',
                      'dist3', 'dist4']].values
    X_gqcm = gqcm_df[['gqcm_dist1', 'gqcm_dist2',
                        'gqcm_dist3', 'gqcm_dist4']].values
    # ── NQCM features — zqcm columns are the processed values ──────
    X      = knn_df[['dist1', 'dist2',
                      'dist3', 'dist4']].values

    # ── Zone labels — capital Z and L in total_nqcm.xlsx ───────────
    y      = knn_df['zone_label'].values
    X_gqcm_hermite = gqcmhermite_df[['zqcm_GQCM_0_dist1_nqcm', 'zqcm_GQCM_0_dist2_nqcm', 'zqcm_GQCM_0_dist3_nqcm', 'zqcm_GQCM_0_dist4_nqcm']].values
    # ── GQCM features ──────────────────────────────────────────────
   
    
    

    return X_raw, X_zqcm, X_gqcm, X, y, X_gqcm_hermite


def split_folds(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    return list(skf.split(X, y))