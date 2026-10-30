#!/usr/bin/env bash
# N25: closed-loop link adaptation in the real-time runs.  Every cell has two users and an outer loop over MCS 10-16
# (target 10% of the TBs needing a retransmission, step down 0.25 per such TB, feedback 6 uplink periods = 15 ms, after
# the recovery deadline; start MCS 13), on the slots of one SNR point (exp/n25_la_data.sh).  With POLICY all, a TB that
# a recovery decodes before the recovery deadline counts as received (Reverb: large NRx, one recovery per GPU, rejection
# after the latest start, deadline 11.0 ms, bound 6.8 ms); with off, only the TBs the L1 decodes.  Settings otherwise as
# exp/n16_runtime.sh; raw results in airan_cloudlab/results/backstop_slot/raw under rec25*, src/analysis/n25_la_summary.py.
# usage: bash exp/n25_runtime_la.sh POLICY "CELLS" SEED "SNRS"
#        env RT_GPUS (default 4), PERIODS (default 4800 = 12 s), RING (slots per MCS ring, default 128)
set -uo pipefail
pol=$1; cells=$2; seed=${3:-1}; snrs=${4:-6 8 10 12 14 16}
REC=/pscratch/sd/s/sgkim/kcj/airan_recovery
case $pol in off) nrx=off ;; all) nrx=rescue ;; *) echo "policy?"; exit 2 ;; esac
gpus=${RT_GPUS:-4}; periods=${PERIODS:-4800}; ring=${RING:-128}
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab || exit 1
rc=0
for snr in $snrs; do
  data=$REC/data/n25_la_s$snr
  [ -s $data/nv_mu2_m16_meta.json ] || { echo "no dataset $data (exp/n25_la_data.sh)"; rc=1; continue; }
  matrix=""; for c in $cells; do matrix="$matrix c$c:$nrx:none"; done
  out=$(SLOT_TAG=rec25${pol}g${gpus}n${snr}s$seed SLOT_PERIODS=$periods SLOT_AI_RATE=4 \
    SLOT_EXTRA="--gpus $gpus --weak-fraction 1.0 --weak-profile nv_mu2_m16 --dataset $data --la 10,11,12,13,14,15,16:0.1:0.25:6:3 --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --lanes-per-gpu 1 --nrx-max-cb-fail 10 --rescue-deadline-ms 11.0 --nrx-bound-ms 6.8 --nrx-bound-corun-ms 6.8 --ring $ring --seed $((83000 + 1000 * seed)) --ring-offset $((8 * (seed - 1)))" \
    SLOT_MATRIX="$matrix" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1)
  echo "$out" | grep -E "^===|FAILED"
  echo "$out" | grep -q "^FAILED" && rc=1
done
exit $rc
