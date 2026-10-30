#!/usr/bin/env bash
# N22: the real-time runs of N16/S1 with AI inference on the same GPUs.  The L1 of C cells (cuPHY PUSCH, LDPC 20
# iterations) and the recoveries of Reverb (the large NRx, one recovery per GPU, rejection after the latest start,
# recovery deadline 11.0 ms, bound 6.8 ms, the 3.5 GHz slots) share the GPUs with AI inference served as in an AI-RAN
# server: the AI process runs with a fixed MPS share of each GPU (ai-policy static of the slot runtime, the mechanism
# of the SoftBank AI-RAN and NVIDIA proof of concept, where the AI receives the GPU share the RAN does not use),
# Qwen2.5-1.5B prefill requests with BurstGPT lengths (chunks 128/512/1024), Poisson arrivals, SLO 200 ms, admission
# on.  The runtime is the slot runtime of airan_cloudlab (scripts_for_node/backstop_slot, read only); raw results stay
# in airan_cloudlab/results/backstop_slot/raw under the tag rec22*.
# usage: bash exp/n22_runtime_ai.sh POLICY "CELLS" SEED     POLICY: off (L1 and AI only) | all (Reverb)
#        env: RT_GPUS (default 1), AI_RATE (requests/s per GPU, default 4), AI_PCT (MPS share of the AI process in %,
#        default 60: NVIDIA's proof of concept gives 60% of the GPU to AI), PERIODS (default 2400 = 6 s)
set -uo pipefail
pol=$1; cells=$2; seed=${3:-1}
REC=/pscratch/sd/s/sgkim/kcj/airan_recovery
case $pol in off) nrx=off ;; all) nrx=rescue ;; *) echo "policy?"; exit 2 ;; esac
gpus=${RT_GPUS:-1}; rate=${AI_RATE:-4}; pct=${AI_PCT:-60}; data=n16_uma_f35
tagpol=${pol}b68; [ "$gpus" = 1 ] || tagpol=${tagpol}g$gpus
tagpol=${tagpol}d110_f35r${rate}m${pct}
[ "${PERIODS:-2400}" = 2400 ] || tagpol=${tagpol}p${PERIODS}
[ -s $REC/data/$data/nv_mu2_meta.json ] || { echo "no dataset (exp/n16_data.sh)"; exit 1; }
matrix=""; for c in $cells; do matrix="$matrix c$c:$nrx:static"; done
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab || exit 1
out=$(SLOT_TAG=rec22${tagpol}s$seed SLOT_PERIODS=${PERIODS:-2400} SLOT_AI_RATE=$rate \
  SLOT_EXTRA="--gpus $gpus --weak-fraction 1.0 --weak-profile nv_mu2 --dataset $REC/data/$data --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --lanes-per-gpu 1 --nrx-max-cb-fail 10 --rescue-deadline-ms 11.0 --nrx-bound-ms 6.8 --nrx-bound-corun-ms 6.8 --ring 512 --seed $((83000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --static-mps-pct $pct --ai-admission 1 --ai-chunks 128,512,1024" \
  SLOT_MATRIX="$matrix" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1)
echo "$out"
echo "$out" | grep -q "^FAILED" && exit 1
exit 0
