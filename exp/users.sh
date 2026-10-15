#!/usr/bin/env bash
# N8a: failures of the conventional receiver with 1-4 users on the same resources (UMi), and how they depend on how
# alike the users' channels are (largest pair similarity per slot).  One user count per GPU.
# usage: bash exp/users.sh NAME [EBNO] [SLOTS]      results/<NAME>_u<n>.json/.npz
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
name=$1; ebno=${2:-0,2,4,6,8,10,12}; slots=${3:-256}
mkdir -p $DATA/$name
pids=()
for n in 1 2 3 4; do
  (
    g=$((n - 1)); f=$DATA/$name/u$n.npz
    [ -s $RESULTS/${name}_u$n.npz ] && exit 0
    cd $NRX_REPO/scripts && tfgen python3 $REC/src/vendor/nv_dataset_mu.py --num-tx $n --ebno $ebno --slots $slots \
        --channel UMi --gpu $g --output $f > $DATA/$name/gen_u$n.log 2>&1 || { echo "u$n generation failed"; exit 1; }
    GPU=$g rx python3 /rec/vendor/nv_conventional.py --input $f --output $RESULTS/${name}_u$n.json > $DATA/$name/dec_u$n.log 2>&1 \
        || { echo "u$n decode failed"; tail -5 $DATA/$name/dec_u$n.log; exit 1; }
    rm -f $f
  ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
GPU=0 rx python3 - $RESULTS/$name <<'PY' | tee $RESULTS/$name.txt
import json, sys
import numpy as np
base = sys.argv[1]
print("| users | blocks | conventional fail % | slots with max pair similarity >= 0.6 | fail % there | fail % below 0.4 |")
print("|---|---|---|---|---|---|")
out = {}
for n in (1, 2, 3, 4):
    try:
        d = np.load(f"{base}_u{n}.npz")
    except FileNotFoundError:
        continue
    ok, cx = d["conv_ok"], d["corr_max"]
    fail = ~ok
    hi, lo = cx >= 0.6, cx < 0.4
    r = {"blocks": int(ok.size), "fail_pct": 100 * float(fail.mean()), "share_hi": float(hi.mean()),
         "fail_pct_hi": 100 * float(fail[hi].mean()) if hi.any() else None,
         "fail_pct_lo": 100 * float(fail[lo].mean()) if lo.any() else None,
         "by_ebno": {f"{e:g}": 100 * float(fail[d["ebno"] == e].mean()) for e in sorted(set(d["ebno"].tolist()))}}
    out[n] = r
    f = lambda v: "-" if v is None else f"{v:.1f}"
    print(f"| {n} | {r['blocks']} | {r['fail_pct']:.1f} | {100 * r['share_hi']:.0f}% | {f(r['fail_pct_hi'])} | {f(r['fail_pct_lo'])} |")
json.dump(out, open(f"{base}.json", "w"), indent=2)
PY
echo USERS_DONE $name rc=$rc
exit $rc
