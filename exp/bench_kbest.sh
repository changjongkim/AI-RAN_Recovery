#!/usr/bin/env bash
# GPU time of the classical recovery receiver (src/bench_kbest.py) for the configurations of Figure 12 (d, e):
# 4 RX with 2-4 users (nrx_large_mu4.cfg) and 8 RX with 1-4 users (nrx_large_8rx_mu4.cfg), one after another on GPU 0
# of an otherwise idle node, then one summary.   usage: bash exp/bench_kbest.sh
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
out=$RESULTS/kbest_gpu_time; mkdir -p $out
rc=0
for spec in "4 2 nrx_large_mu4.cfg" "4 3 nrx_large_mu4.cfg" "4 4 nrx_large_mu4.cfg" \
            "8 1 nrx_large_8rx_mu4.cfg" "8 2 nrx_large_8rx_mu4.cfg" "8 3 nrx_large_8rx_mu4.cfg" "8 4 nrx_large_8rx_mu4.cfg"; do
  read -r r u cfg <<<"$spec"
  [ -s $out/r${r}_u${u}.json ] && continue
  tfgen python3 $REC/src/bench_kbest.py --nrx-repo $NRX_REPO --gpu 0 --base-config $cfg --num-rx $r --users $u \
      --output $out/r${r}_u${u}.json > $out/r${r}_u${u}.log 2>&1 || { rc=1; tail -5 $out/r${r}_u${u}.log; }
  grep -E "^r[0-9]" $out/r${r}_u${u}.log
done
python3 - $out <<'PY' | tee $RESULTS/kbest_gpu_time.txt
import glob, json, sys
rows = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/r*_u*.json")):
    d = json.load(open(f))
    c = d["config"]
    rows[f"r{c['num_rx']}_u{c['users']}"] = {"num_rx": c["num_rx"], "users": c["users"], "device": d["device"],
        **{k: {"median_ms": s["median_ms"], "p99_ms": s["p99_ms"], "mode": s["mode"]} for k, s in d["stages"].items()}}
json.dump(rows, open(f"{sys.argv[1]}.json", "w"), indent=1)
for k, r in rows.items():
    print(k, r["device"], "  ".join(f"{s} {r[s]['median_ms']:.2f}/{r[s]['p99_ms']:.2f} ms ({r[s]['mode']})"
                                   for s in ("lslin_est", "lmmse_det", "lmmse_est", "kbest_det", "nrx")))
PY
exit $rc
