# experiments/run_state_machine_demo.py
#
# Runs the zone state machine on the classifier's saved predictions for the
# 80-second live validation trajectory (results/predictions_live_zone1.npy).
#
# This script needs NO private data — it runs straight after cloning:
#     python experiments/run_state_machine_demo.py
#
# Expected output (thesis Section 4.6):
#     raw zone changes 45  -> stable changes 22
#     raw Zone 3->1 crossings 15 -> confirmed alarms 5

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from config.settings import PREDICTIONS_PATH

import numpy as np
from src.zone_state_machine import ZoneStateMachine

DURATION_S     = 80.0   # length of the live recording
WINDOW_SECONDS = 1.5    # majority-vote window
DWELL_SECONDS  = 0.4    # how long a new zone must persist before it is accepted


def main():
    predictions = np.load(PREDICTIONS_PATH)
    n = len(predictions)
    sample_rate = n / DURATION_S

    raw_changes = int((predictions[1:] != predictions[:-1]).sum())
    raw_31 = int(((predictions[:-1] == 3) & (predictions[1:] == 1)).sum())

    sm = ZoneStateMachine(window_seconds=WINDOW_SECONDS, dwell_seconds=DWELL_SECONDS,
                          sample_rate=sample_rate, alarm_from=3, alarm_to=1)
    for i, pred in enumerate(predictions):
        sm.update(round(i / sample_rate, 1), int(pred))

    print(f"Samples: {n}  (~{sample_rate:.1f} Hz over {DURATION_S:.0f} s)")
    print(f"Zone changes      : raw {raw_changes:3d}  ->  stable {len(sm.history):3d}")
    print(f"Zone 3 -> 1 events: raw {raw_31:3d}  ->  alarms {len(sm.alarms):3d}")
    print(f"Alarm times (s)   : {sm.alarms}")


if __name__ == "__main__":
    main()
