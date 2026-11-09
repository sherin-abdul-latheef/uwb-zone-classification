import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from figures.style import save_figure


# ── Data ───────────────────────────────────────────────────────
cm_true = np.array([
    [1419,    1,    0,    0],
    [   0, 1375,    0,    0],
    [   0,    0, 1128,    0],
    [   0,    0,    0, 1477]
])

cm_false = np.array([
    [1420,    0,    0,    0],
    [  10, 1365,    0,    0],
    [  19,    0, 1109,    0],
    [   0,    0,    0, 1477]
])

ZONE_LABELS = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']


# ── Main ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16/2.54, 7/2.54),  # 16 cm x 7 cm
    constrained_layout=True
)

# Left: shuffle=True
disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true,
                                   display_labels=ZONE_LABELS)
disp_true.plot(ax=ax1, cmap=plt.cm.viridis, colorbar=False)
ax1.set_title('shuffle=True (shuffled CV)', fontsize=11)
ax1.set_xlabel('Predicted zone', fontsize=11)
ax1.set_ylabel('True zone', fontsize=11)

# Right: shuffle=False
disp_false = ConfusionMatrixDisplay(confusion_matrix=cm_false,
                                    display_labels=ZONE_LABELS)
disp_false.plot(ax=ax2, cmap=plt.cm.viridis, colorbar=False)
ax2.set_title('shuffle=False (temporal CV)', fontsize=11)
ax2.set_xlabel('Predicted zone', fontsize=11)
ax2.set_ylabel('True zone', fontsize=11)

# Optional: rotate x tick labels a bit if they collide
for ax in (ax1, ax2):
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

save_figure("fig_4_1_confusion_matrix.png")
