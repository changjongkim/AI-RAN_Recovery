#!/usr/bin/env bash
# N5 / P1 (antennas): classical_rx.py on the four GPUs in parallel (one seed per GPU), then one merged summary.
# usage: bash exp/classical.sh NAME [classical_rx options]   e.g.  bash exp/classical.sh n5_umi_v3 --ebno 6,8,10,12 --slots 64
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
name=$1; shift
out=$RESULTS/$name; mkdir -p $out
pids=()
for g in ${GPUS:-0 1 2 3}; do
  [ -s $out/g$g.npz ] && continue
  tfgen python3 $REC/src/classical_rx.py --nrx-repo $NRX_REPO --gpu $g --seed $((${SEED0:-300} + g)) "$@" --output $out/g$g.json \
      > $out/g$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
GPU=0 rx python3 - "$out" <<'PY' | tee $RESULTS/$name.txt
import glob, json, sys
import numpy as np
out = sys.argv[1]
files = sorted(glob.glob(f"{out}/g*.npz"))
d = [np.load(f) for f in files]
keys = [k for k in d[0].files if k.startswith("ok_")]
ok = {k[3:]: np.concatenate([x[k] for x in d]) for k in keys}
corr = np.concatenate([x["corr"] for x in d]); point = np.concatenate([x["point"] for x in d])
rep = [json.load(open(f.replace(".npz", ".json"))) for f in files]
systems = rep[0]["systems"]; conv = ok[systems[0]]
rho = np.repeat(corr, conv.shape[1])
res = {"files": files, "systems": systems, "slots": int(conv.shape[0]), "num_rx": rep[0]["num_rx"],
       "seconds_per_slot": {s: float(np.mean([r["seconds_per_slot"][s] for r in rep])) for s in systems},
       "fail_pct": {s: 100.0 * float((~ok[s]).mean()) for s in systems},
       "recovers_pct_of_conventional_failures": {s: 100.0 * float((ok[s] & ~conv).sum()) / max(1, int((~conv).sum())) for s in systems[1:]},
       "by_similarity": {}}
for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01)):
    m = (rho >= lo) & (rho < hi)
    f = ~conv.reshape(-1) & m
    res["by_similarity"][f"{lo:.1f}-{hi:.1f}"] = {
        "share_of_blocks": float(m.mean()), "conventional_fail_pct": 100.0 * float(f.sum()) / max(1, int(m.sum())),
        "share_of_failures": float(f.sum()) / max(1, int((~conv).sum())),
        "recovers_pct": {s: 100.0 * float((ok[s].reshape(-1) & f).sum()) / max(1, int(f.sum())) for s in systems[1:]}}
json.dump(res, open(f"{out}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PY
echo CLASSICAL_DONE $name rc=$rc
exit $rc
