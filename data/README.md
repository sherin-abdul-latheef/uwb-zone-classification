# Dataset

The dataset is **not included** in this repository. It was collected by the author with four UWB
anchors and one mobile tag in an indoor test area. It is available on request.

To run the experiments, place these files in this `data/` folder
(or point the environment variable `UWB_DATA_DIR` to another folder):

| File | Contents | Columns used |
|---|---|---|
| `raw_data.xlsx` | Raw anchor distances | `dist1`–`dist4`, `zone_label` |
| `zqcm_data.xlsx` | After zero-value correction | `dist1`–`dist4` |
| `gqcm_data.xlsx` | Gross errors flagged as empty (NaN) | `gqcm_dist1`–`gqcm_dist4` |
| `gqcm_hermite.xlsx` | GQCM gaps refilled by Hermite interpolation | `zqcm_GQCM_0_dist1_nqcm`–`zqcm_GQCM_0_dist4_nqcm` |
| `total_nqcm.xlsx` | Final NQCM features + labels (the fingerprint database) | `dist1`–`dist4`, `zone_label` |
| `live_zone1_nqcm.xlsx` | Separate 80 s live walk used for real-time validation | `zqcm_GQCM_0_dist1_nqcm`–`zqcm_GQCM_0_dist4_nqcm` |

**Fingerprint database:** 5,400 samples (about 67 Hz), recorded zone by zone in the order
Zone 3 → Zone 1 → Zone 2 → Zone 4 (1,128 / 1,420 / 1,375 / 1,477 samples).

**Live trajectory:** 4,983 samples over 80 seconds (about 62 Hz).

All files must keep the same row order, because the frozen folds in `config/fold_indices.pkl` refer to row numbers.
