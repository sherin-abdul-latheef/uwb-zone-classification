# demo_zone3_trigger.py — Zone 3 Boundary Trigger Demonstrator
# 
# What this does
# --------------
# 1. Trains KNN on your fingerprint database (total_nqcm.xlsx)
# 2. Projects training data into LDA space — draws zone blobs as background map
# 3. Loops through live_zone1_nqcm.xlsx row by row at 15ms intervals
# 4. Projects each reading onto same LDA axes → moves dot to that position
# 5. When prediction crosses INTO Zone 3 → TRIGGER fires (flash + sound)
#
# Run from project root:
#   python demos/zone3_trigger_demo.py
#
# Requirements:
#   pip install matplotlib scikit-learn pandas openpyxl scipy
# ══════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import DATA_PATH, LIVE_TRAJECTORY_PATH

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths
# ══════════════════════════════════════════════════════════════════════

TRAINING_DATA_PATH = DATA_PATH

LIVE_DATA_PATH = LIVE_TRAJECTORY_PATH

# KNN settings — your best configuration
K            = 37
METRIC       = 'manhattan'
DELAY_SEC    = 0.015          # 15ms between readings

# Feature columns
TRAIN_COLS   = ['zqcm_GQCM_0_dist1', 'zqcm_GQCM_0_dist2',
                'zqcm_GQCM_0_dist3', 'zqcm_GQCM_0_dist4']
LABEL_COL    = 'zone_label'
LIVE_COLS    = ['zqcm_GQCM_0_dist1_nqcm', 'zqcm_GQCM_0_dist2_nqcm',
                'zqcm_GQCM_0_dist3_nqcm', 'zqcm_GQCM_0_dist4_nqcm']

# Zone visual identity
ZONE_COLORS  = {1: '#185FA5', 2: '#3B6D11', 3: '#993C1D', 4: '#534AB7'}
ZONE_FILLS   = {1: '#D0E8FA', 2: '#D4EDBA', 3: '#F5C8BC', 4: '#D9D6F7'}
ZONE_NAMES   = {1: 'Zone 1', 2: 'Zone 2', 3: 'Zone 3', 4: 'Zone 4'}
TRIGGER_ZONE = 3              # Zone that fires the trigger

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Load and train
# ══════════════════════════════════════════════════════════════════════

def load_and_train():
    print("Loading fingerprint database...")
    df    = pd.read_excel(TRAINING_DATA_PATH)
    X     = df[TRAIN_COLS].values
    y     = df[LABEL_COL].values

    print(f"Training KNN — k={K}, metric={METRIC}...")
    knn   = KNeighborsClassifier(n_neighbors=K,
                                  metric=METRIC,
                                  algorithm='ball_tree')
    knn.fit(X, y)

    print("Fitting LDA projection axes on training data...")
    lda   = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    print(f"  LD1={lda.explained_variance_ratio_[0]*100:.1f}%  "
          f"LD2={lda.explained_variance_ratio_[1]*100:.1f}%")

    return knn, lda, X_lda, y


def load_live_data():
    print("Loading live walk data...")
    df    = pd.read_excel(LIVE_DATA_PATH)
    X     = df[LIVE_COLS].values
    print(f"  {len(X)} readings × 15ms = {len(X)*0.015:.1f} seconds")
    return X


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Build the background LDA map
# ══════════════════════════════════════════════════════════════════════

def build_lda_map(ax, X_lda, y_train):
    """
    Draw the zone fingerprint blobs in LDA space.
    This IS the map — not a room sketch.
    Each blob shows where that zone lives in classifier space.
    """
    zones = sorted(np.unique(y_train))

    x_min = X_lda[:, 0].min() - 1.2
    x_max = X_lda[:, 0].max() + 1.2
    y_min = X_lda[:, 1].min() - 1.2
    y_max = X_lda[:, 1].max() + 1.2

    xx, yy = np.mgrid[x_min:x_max:180j, y_min:y_max:180j]
    grid   = np.vstack([xx.ravel(), yy.ravel()])

    zone_centroids = {}

    for zone in zones:
        pts = X_lda[y_train == zone]
        zone_centroids[zone] = pts.mean(axis=0)

        if len(pts) < 5:
            continue

        kde = gaussian_kde(pts.T, bw_method='scott')
        zz  = kde(grid).reshape(xx.shape)
        zz /= zz.max()

        # Filled territory
        ax.contourf(xx, yy, zz,
                    levels=[0.25, 1.0],
                    colors=[ZONE_FILLS[zone]],
                    alpha=0.35, zorder=1)

        # Boundary contours
        ax.contour(xx, yy, zz,
                   levels=[0.25, 0.65, 0.90],
                   colors=[ZONE_COLORS[zone]],
                   linewidths=[0.6, 1.0, 1.5],
                   linestyles=['dotted', 'dashed', 'solid'],
                   zorder=2)

        # Zone label at centroid
        cx, cy = zone_centroids[zone]
        ax.text(cx, cy,
                ZONE_NAMES[zone],
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color=ZONE_COLORS[zone],
                alpha=0.6, zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return zone_centroids


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Real-time demo loop
# ══════════════════════════════════════════════════════════════════════

def run_demo(knn, lda, X_lda, y_train, X_live):

    zones = sorted(np.unique(y_train))

    # ── Build figure ──────────────────────────────────────────────────
    fig   = plt.figure(figsize=(14, 8), facecolor='#0D0D0D')
    fig.canvas.manager.set_window_title(
        'UWB Zone Classifier — Live Boundary Trigger Demo')

    # Left: LDA map
    ax_map = fig.add_axes([0.04, 0.12, 0.56, 0.80])
    ax_map.set_facecolor('#1A1A2E')
    ax_map.tick_params(colors='#888888', labelsize=8)
    for spine in ax_map.spines.values():
        spine.set_edgecolor('#333355')

    # Right: status panel
    ax_status = fig.add_axes([0.63, 0.12, 0.34, 0.80])
    ax_status.set_facecolor('#0D0D0D')
    ax_status.axis('off')

    # Title
    fig.text(0.5, 0.97,
             'UWB Indoor Zone Classification — Real-Time Demo',
             ha='center', va='top',
             fontsize=13, fontweight='bold', color='white')
    fig.text(0.5, 0.93,
             f'KNN  k={K}  |  Manhattan distance  |  NQCM features  |  '
             f'15ms update rate',
             ha='center', va='top',
             fontsize=9, color='#888888')

    # Draw static background map
    zone_centroids = build_lda_map(ax_map, X_lda, y_train)

    ax_map.set_xlabel('LD1  (82.0% between-class variance)',
                      color='#888888', fontsize=9)
    ax_map.set_ylabel('LD2  (14.2% between-class variance)',
                      color='#888888', fontsize=9)
    ax_map.set_title('Fingerprint Space — LDA Projection',
                     color='white', fontsize=10, pad=8)
    ax_map.grid(True, alpha=0.08, color='white')

    # ── Dynamic elements ──────────────────────────────────────────────
    # Current position dot on map
    dot, = ax_map.plot([], [], 'o',
                       markersize=14,
                       markeredgecolor='white',
                       markeredgewidth=2,
                       zorder=10)

    # Trail — last N positions
    TRAIL_LEN = 25
    trail_x, trail_y = [], []
    trail_line, = ax_map.plot([], [], '-',
                               color='white', alpha=0.3,
                               linewidth=1.5, zorder=9)

    # ── Status panel elements ─────────────────────────────────────────
    # Current zone box
    zone_box = ax_status.add_patch(
        FancyBboxPatch((0.05, 0.55), 0.90, 0.38,
                       boxstyle='round,pad=0.02',
                       facecolor='#1A1A2E',
                       edgecolor='#333355',
                       linewidth=2,
                       transform=ax_status.transAxes,
                       zorder=1))

    zone_label_text = ax_status.text(
        0.50, 0.77, '—',
        transform=ax_status.transAxes,
        ha='center', va='center',
        fontsize=28, fontweight='bold',
        color='white', zorder=2)

    zone_sub_text = ax_status.text(
        0.50, 0.62, 'Initialising...',
        transform=ax_status.transAxes,
        ha='center', va='center',
        fontsize=10, color='#888888', zorder=2)

    # Trigger panel
    trigger_box = ax_status.add_patch(
        FancyBboxPatch((0.05, 0.10), 0.90, 0.38,
                       boxstyle='round,pad=0.02',
                       facecolor='#0D0D0D',
                       edgecolor='#333333',
                       linewidth=2,
                       transform=ax_status.transAxes,
                       zorder=1))

    trigger_title = ax_status.text(
        0.50, 0.44, '⬤  ZONE 3 TRIGGER',
        transform=ax_status.transAxes,
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        color='#444444', zorder=2)

    trigger_count_text = ax_status.text(
        0.50, 0.32, 'Crossings: 0',
        transform=ax_status.transAxes,
        ha='center', va='center',
        fontsize=22, fontweight='bold',
        color='#555555', zorder=2)

    trigger_time_text = ax_status.text(
        0.50, 0.18, '',
        transform=ax_status.transAxes,
        ha='center', va='center',
        fontsize=9, color='#555555', zorder=2)

    # Progress bar
    prog_bg = ax_status.add_patch(
        FancyBboxPatch((0.05, 0.03), 0.90, 0.04,
                       boxstyle='round,pad=0.01',
                       facecolor='#1A1A1A',
                       edgecolor='#333333',
                       linewidth=1,
                       transform=ax_status.transAxes))

    prog_bar = ax_status.add_patch(
        FancyBboxPatch((0.05, 0.03), 0.001, 0.04,
                       boxstyle='round,pad=0.01',
                       facecolor='#534AB7',
                       edgecolor='none',
                       linewidth=0,
                       transform=ax_status.transAxes))

    plt.ion()
    plt.show()

    # ── State ─────────────────────────────────────────────────────────
    prev_zone      = None
    trigger_count  = 0
    trigger_times  = []
    flash_frames   = 0       # how many frames left in the trigger flash

    n_total = len(X_live)

    print()
    print("=" * 55)
    print("DEMO RUNNING — close window to stop")
    print("=" * 55)
    print(f"{'Row':>5} | {'Time':>6} | {'Zone':>6} | {'Event'}")
    print("-" * 55)

    for i, row in enumerate(X_live):

        t_sec = i * DELAY_SEC

        # ── Classify ──────────────────────────────────────────────────
        pred = knn.predict(row.reshape(1, -1))[0]

        # ── Project onto LDA axes ─────────────────────────────────────
        proj = lda.transform(row.reshape(1, -1))[0]

        # ── Check trigger ─────────────────────────────────────────────
        trigger_fired = (prev_zone != TRIGGER_ZONE and
                         pred       == TRIGGER_ZONE)

        if trigger_fired:
            trigger_count += 1
            trigger_times.append(t_sec)
            flash_frames = 12    # flash for 12 frames (~180ms)
            print(f"{i:5d} | {t_sec:5.1f}s | Zone {pred} | "
                  f"*** ZONE 3 TRIGGER #{trigger_count} ***")
        elif pred != prev_zone:
            print(f"{i:5d} | {t_sec:5.1f}s | Zone {pred} | "
                  f"→ Zone {prev_zone} to Zone {pred}")

        # ── Update trail ──────────────────────────────────────────────
        trail_x.append(proj[0])
        trail_y.append(proj[1])
        if len(trail_x) > TRAIL_LEN:
            trail_x.pop(0)
            trail_y.pop(0)

        # ── Update map dot ────────────────────────────────────────────
        dot.set_data([proj[0]], [proj[1]])
        dot.set_color(ZONE_COLORS[pred])
        dot.set_markersize(16 if flash_frames > 0 else 12)
        trail_line.set_data(trail_x, trail_y)

        # ── Update status panel ───────────────────────────────────────
        if flash_frames > 0:
            # Trigger active — full flash
            zone_box.set_facecolor('#3D0000')
            zone_box.set_edgecolor('#FF2200')
            zone_box.set_linewidth(3)
            zone_label_text.set_text('⚡ ZONE 3')
            zone_label_text.set_color('#FF4400')
            zone_label_text.set_fontsize(32)
            zone_sub_text.set_text('BOUNDARY CROSSED')
            zone_sub_text.set_color('#FF6644')
            trigger_box.set_facecolor('#1A0800')
            trigger_box.set_edgecolor('#FF4400')
            trigger_title.set_color('#FF4400')
            trigger_count_text.set_text(f'Crossings: {trigger_count}')
            trigger_count_text.set_color('#FF6644')
            trigger_time_text.set_text(
                '\n'.join([f't = {tt:.1f}s'
                           for tt in trigger_times[-3:]]))
            trigger_time_text.set_color('#FF8866')
            flash_frames -= 1

        else:
            # Normal display
            col = ZONE_COLORS[pred]
            fill = ZONE_FILLS[pred]
            zone_box.set_facecolor(fill + '33')
            zone_box.set_edgecolor(col)
            zone_box.set_linewidth(2)
            zone_label_text.set_text(ZONE_NAMES[pred])
            zone_label_text.set_color(col)
            zone_label_text.set_fontsize(28)
            zone_sub_text.set_text(f't = {t_sec:.1f}s')
            zone_sub_text.set_color('#888888')
            if trigger_count > 0:
                trigger_box.set_facecolor('#0D0D0D')
                trigger_box.set_edgecolor('#444444')
                trigger_title.set_color('#888888')
                trigger_count_text.set_text(
                    f'Crossings: {trigger_count}')
                trigger_count_text.set_color('#AAAAAA')

        # Progress bar
        prog = (i + 1) / n_total * 0.90
        prog_bar.set_width(prog)
        prog_bar.set_facecolor(ZONE_COLORS[pred])

        prev_zone = pred

        # ── Render ────────────────────────────────────────────────────
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(DELAY_SEC)

        if not plt.fignum_exists(fig.number):
            break

    # ── Final summary ─────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("DEMO COMPLETE")
    print("=" * 55)
    print(f"Total readings    : {n_total}")
    print(f"Duration          : {n_total * DELAY_SEC:.1f} seconds")
    print(f"Zone 3 crossings  : {trigger_count}")
    if trigger_times:
        print(f"Trigger times     : "
              f"{', '.join([f'{t:.1f}s' for t in trigger_times])}")
    print()

    plt.ioff()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("UWB ZONE 3 BOUNDARY TRIGGER — DEMO")
    print(f"KNN k={K} | {METRIC} | NQCM features")
    print("=" * 55)
    print()

    knn, lda, X_lda, y_train = load_and_train()
    X_live                    = load_live_data()

    print()
    print("Starting demo in 2 seconds...")
    time.sleep(2)

    run_demo(knn, lda, X_lda, y_train, X_live)


if __name__ == '__main__':
    main()
