#!/usr/bin/env bash
# N16: one A100 runs the Layer 1 (cuPHY PUSCH, LDPC 20 iterations) of C cells every uplink slot (DDDSU, 2.5 ms, L1
# deadline 4.0 ms) and, for blocks that fail CRC, the large neural receiver on one recovery lane (recovery deadline
# 11.5 ms, a candidate is dropped when it cannot start by 11.5 - 7.6 ms); no AI.  The runtime is the slot runtime of
# airan_cloudlab (scripts_for_node/backstop_slot, read only) with our UMa slots (exp/n16_data.sh) and the large engine
# every HARQ run of this project decodes with (ENGINE_LARGE in env.sh, /softwall_runtime in the slot container); its
# raw results stay in airan_cloudlab/results/backstop_slot/raw under the tag rec16*, src/analysis/n16_summary.py
# collects them into results/n16/.
# usage: bash exp/n16_runtime.sh POLICY "CELLS" SEED     POLICY: off (L1 only) | all (every failed block) | val5 (slots
#        whose blocks have at most 5 failed code blocks in all, the runtime's value rule) | cb5 (as all: the runtime
#        applies the code-block limit only with the value rule; kept for the runs of 2026-10-29)
#        env: PERIODS (default 2400 = 6 s), BOUND (path time the controller assumes, default 7.6 ms as in the
#        Antiphase runs with AI; the path alone takes 5.3 ms; 6.8 is the reference since 2026-10-29): a candidate must
#        start by 11.5 - BOUND after release.  RT_GPUS (default 1): GPUs of the server, cell c on GPU c mod RT_GPUS, one
#        recovery lane per GPU, a failure goes to any free lane (S1, a four-GPU server)
set -uo pipefail
pol=$1; cells=$2; seed=${3:-1}
REC=/pscratch/sd/s/sgkim/kcj/airan_recovery
case $pol in off) nrx=off; cb=10 ;; all) nrx=rescue; cb=10 ;; cb5) nrx=rescue; cb=5 ;; val5) nrx=rescue_value; cb=5 ;;
  *) echo "policy?"; exit 2 ;; esac
#        DEADLINE (recovery deadline in ms, default 11.5; 11.0 since 2026-10-29: the retransmission slot 15.0 ms minus the
#        grant lead of 4.0 ms) and N16_DATA (dataset, default n16_uma; n16_uma_f35 is the 3.5 GHz one) add to the tag.
bound=${BOUND:-7.6}; gpus=${RT_GPUS:-1}; tagpol=$pol; [ "$bound" = 7.6 ] || tagpol=${pol}b${bound/./}
deadline=${DEADLINE:-11.5}; data=${N16_DATA:-n16_uma}
[ "$gpus" = 1 ] || tagpol=${tagpol}g$gpus
[ "${PERIODS:-2400}" = 2400 ] || tagpol=${tagpol}p${PERIODS}
[ "$deadline" = 11.5 ] || tagpol=${tagpol}d${deadline/./}
[ "$data" = n16_uma ] || tagpol=${tagpol}${data#n16_uma}
[ -s $REC/data/$data/nv_mu2_meta.json ] || { echo "no dataset (exp/n16_data.sh)"; exit 1; }
matrix=""; for c in $cells; do matrix="$matrix c$c:$nrx:none"; done
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab || exit 1
out=$(SLOT_TAG=rec16${tagpol}s$seed SLOT_PERIODS=${PERIODS:-2400} SLOT_AI_RATE=4 \
  SLOT_EXTRA="--gpus $gpus --weak-fraction 1.0 --weak-profile nv_mu2 --dataset $REC/data/$data --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --lanes-per-gpu 1 --nrx-max-cb-fail $cb --rescue-deadline-ms $deadline --nrx-bound-ms $bound --nrx-bound-corun-ms $bound --ring 512 --seed $((83000 + 1000 * seed)) --ring-offset $((50 * (seed - 1)))" \
  SLOT_MATRIX="$matrix" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1)
echo "$out"
echo "$out" | grep -q "^FAILED" && exit 1
exit 0
