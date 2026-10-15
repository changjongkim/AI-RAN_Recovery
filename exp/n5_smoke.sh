#!/usr/bin/env bash
# Check of the receivers added to N5 before the full run: on the same slots (one GPU, Eb/No 10 dB, 64 slots),
# IDD with zero rounds must decode about as often as the conventional receiver (it is MMSE-PIC without prior = LMMSE),
# IDD with three rounds and the conventional receiver with 50 BP iterations must not decode fewer blocks.
# usage: GPUS=1 bash exp/n5_smoke.sh
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
rm -rf $RESULTS/n5_smoke $RESULTS/n5_smoke.json
GPUS=${GPUS:-1} bash $REC/exp/classical.sh n5_smoke --systems baseline_lslin_lmmse,lslin_idd0,lslin_idd3,baseline_lslin_lmmse@50 \
    --ebno 8,10 --slots 32 --batch 8 > /dev/null || { echo "n5 smoke run failed"; tail -20 $RESULTS/n5_smoke/g*.log; exit 1; }
python3 - $RESULTS/n5_smoke.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
fail = r["fail_pct"]
print("block failure %:", {k: round(v, 1) for k, v in fail.items()})
print("seconds per slot:", {k: round(v, 3) for k, v in r["seconds_per_slot"].items()})
conv = fail["baseline_lslin_lmmse"]
ok = abs(fail["lslin_idd0"] - conv) <= max(3.0, 0.3 * conv) and fail["lslin_idd3"] <= fail["lslin_idd0"] + 1.0 \
     and fail["baseline_lslin_lmmse@50"] <= conv + 0.5
print("N5_SMOKE", "pass" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
