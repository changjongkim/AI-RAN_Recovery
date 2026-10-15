#!/usr/bin/env bash
# Smoke test of the HARQ pipeline: RV 0 equals Sionna's TBEncoder bit for bit; a tiny chunk generates, builds and
# decodes; the separable receiver decodes about what cuPHY decodes on the first transmission; combining never loses.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
t0=$(date +%s)
tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --check-rv0 --ebno 8 --output $DATA/smoke_rv0 2>&1 | grep -E '^\{|rror|Traceback|  File' || exit 1
echo "rv0 check $(($(date +%s) - t0)) s"
t0=$(date +%s)
KEEP=1 GPUS="0" bash $REC/exp/harq_ir.sh smoke 1 --channel UMi --speed 0.83 --ebno 8,12 --slots 8 --batch 4 || exit 1
echo "chunk of 16 slots, 4 transmissions: $(($(date +%s) - t0)) s"
grep -E '^point' $DATA/smoke/c0/gen.log
GPU=0 rx python3 - <<'PY'
import json, sys
r = json.load(open("/pscratch/sd/s/sgkim/kcj/airan_recovery/results/smoke.json"))
d, tbs = r["decoded"], r["slots"] * r["users"]
sep, conv = r["check"]["sep_vs_conv_first"]
print("decoded:", {k: d[k] for k in sorted(d)})
ok = abs(sep - conv) <= 0.1 * tbs + 1
for rx in ("sep", "nrx", "small"):
    seq = [d[f"{rx}_c{t}"] for t in range(r["transmissions"])]
    print(rx, "after 1..4 transmissions:", seq)
    ok &= all(b >= a for a, b in zip(seq, seq[1:]))
print("SMOKE_OK" if ok else "SMOKE_FAIL")
sys.exit(0 if ok else 1)
PY
