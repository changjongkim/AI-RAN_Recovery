#!/usr/bin/env bash
# N25 slots for closed-loop link adaptation in the real-time runs: first transmissions of MCS 10-16 on the same channel
# realizations (gen_harq.py --mcs LIST) on UMa (3 km/h, 3.5 GHz, two users, 273 PRBs, 4 RX) at the SNR points of the
# trace-driven link adaptation (N6: 6-16 dB per resource element), 64 slots per point on each of four GPUs (seeds
# 2500-2503).  One runtime dataset per SNR point, data/n25_la_s<SNR>, with one ring per MCS (profiles nv_mu2_m10..16).
# env: POINTS (default 6,8,10,12,14,16), MCS_LIST (default 10,11,12,13,14,15,16)
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
points=${POINTS:-6,8,10,12,14,16}; list=${MCS_LIST:-10,11,12,13,14,15,16}; last=${list##*,}
d=$DATA/n25_la_gen; mkdir -p $d
missing=0; for s in ${points//,/ }; do [ -s $DATA/n25_la_s$s/nv_mu2_m${last}_meta.json ] || missing=1; done
[ $missing = 0 ] && { echo "datasets exist"; exit 0; }
pids=()
for g in 0 1 2 3; do
  [ -s $d/p${g}_m${last}_t0.npz ] && continue
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((2500 + g)) --mcs $list --rvs 0 --channel UMa --carrier-ghz 3.5 \
      --speed 0.83 --snr-db $points --slots 64 --batch 8 --output $d/p$g > $d/gen_p$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "generation failed"; tail -5 $d/gen_p*.log; exit 1; }
build() {   # one SNR point: one ring per MCS
  local s=$1 g=$2 m
  for m in ${list//,/ }; do
    GPU=$g rx python3 /rec/vendor/nv_build_dataset.py --input $d/p0_m${m}_t0.npz $d/p1_m${m}_t0.npz $d/p2_m${m}_t0.npz $d/p3_m${m}_t0.npz \
        --ebno $s --profile nv_mu2_m$m --output $DATA/n25_la_s$s > $d/build_s${s}_m$m.log 2>&1 || { echo "build $s $m failed"; tail -5 $d/build_s${s}_m$m.log; return 1; }
  done
}
pids=(); i=0
for s in ${points//,/ }; do build $s $((i % 4)) & pids+=($!); i=$((i + 1)); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || exit 1
for s in ${points//,/ }; do echo "SNR $s: $(ls $DATA/n25_la_s$s/*_meta.json | wc -l) rings, $(tail -1 $d/build_s${s}_m$last.log)"; done
rm -f $d/p*_m*_t0.npz
