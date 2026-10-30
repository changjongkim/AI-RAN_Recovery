#!/usr/bin/env bash
# N29: N16 with partial cell load, up to 48 cells on the four GPUs of a node.  Every (cell, uplink period) carries the
# PUSCH of its two users with probability ACT (--activity-prob of the slot runtime, activity.py), independently
# (ACT_MODE=bernoulli) or in busy and idle runs with a mean busy run of 8 periods (ACT_MODE=bursty); an idle period has
# no transmission and no recovery.  A real server hosts more cells at partial load than at full load, so the recovered
# share is measured at the number of cells and the load of the cells together.  Settings otherwise as exp/n16_runtime.sh
# (the large NRx, one recovery per GPU, rejection after the latest start, recovery deadline DEADLINE, bound BOUND).
# usage: ACT=0.5 [ACT_MODE=bernoulli] bash exp/n29_runtime_load.sh POLICY "CELLS" SEED      (tag: ...a<ACT*100><b|u>)
set -uo pipefail
pol=$1; cells=$2; seed=${3:-1}
REC=/pscratch/sd/s/sgkim/kcj/airan_recovery
case $pol in off) nrx=off; cb=10 ;; all) nrx=rescue; cb=10 ;; cb5) nrx=rescue; cb=5 ;; val5) nrx=rescue_value; cb=5 ;;
  par) nrx=parallel; cb=10 ;; para) nrx=parallel_admit; cb=10 ;; naive) nrx=rescue_naive; cb=10 ;;
  *) echo "policy?"; exit 2 ;; esac
#        DEADLINE (recovery deadline in ms, default 11.5; 11.0 since 2026-10-29: the retransmission slot 15.0 ms minus the
#        grant lead of 4.0 ms) and N16_DATA (dataset, default n16_uma; n16_uma_f35 is the 3.5 GHz one) add to the tag.
#        NRX_FLAGS overrides the NeuralRx mechanisms of the runtime (make_config.py --nrx-flags) and adds its letters and
#        digits to the tag: admit=0 starts a recovery even after its latest start (no rejection), rank=value serves
#        TBs with fewer failed code blocks first (the TBs whose grant waits before those granted at once),
#        lanes=partner keeps the recovery of a cell on the GPU of that cell (no placement on another free GPU).
#        LANES (default 1): recoveries that may run at once on one GPU (2 or 4: concurrent runs, each slower; the
#        runtime starts the second one without a deadline check), adds l<LANES> to the tag.
#        More POLICY values (2026-10-29, the comparisons of Section 5): par (the large NRx for every TB at its
#        arrival, before the CRC, no rejection, on the GPU of the cell), para (every TB at its arrival, rejection after
#        the latest start, any free GPU), naive (after the CRC, no rejection, on the GPU of the cell).
bound=${BOUND:-7.6}; gpus=${RT_GPUS:-1}; tagpol=$pol; [ "$bound" = 7.6 ] || tagpol=${pol}b${bound/./}
deadline=${DEADLINE:-11.5}; data=${N16_DATA:-n16_uma}
[ "$gpus" = 1 ] || tagpol=${tagpol}g$gpus
[ "${PERIODS:-2400}" = 2400 ] || tagpol=${tagpol}p${PERIODS}
[ "$deadline" = 11.5 ] || tagpol=${tagpol}d${deadline/./}
[ "$data" = n16_uma ] || tagpol=${tagpol}${data#n16_uma}
[ -z "${NRX_FLAGS:-}" ] || tagpol=${tagpol}$(echo "$NRX_FLAGS" | tr -cd 'a-z0-9')
lanes=${LANES:-1}; [ "$lanes" = 1 ] || tagpol=${tagpol}l$lanes
act=${ACT:?ACT}; mode=${ACT_MODE:-bernoulli}
tagpol=${tagpol}a$(python3 -c "print(round(100*$act))")${mode:0:1}
[ -s $REC/data/$data/nv_mu2_meta.json ] || { echo "no dataset (exp/n16_data.sh)"; exit 1; }
matrix=""; for c in $cells; do matrix="$matrix c$c:$nrx:none"; done
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab || exit 1
out=$(SLOT_TAG=rec16${tagpol}s$seed SLOT_PERIODS=${PERIODS:-2400} SLOT_AI_RATE=4 \
  SLOT_EXTRA="--gpus $gpus --weak-fraction 1.0 --weak-profile nv_mu2 --dataset $REC/data/$data --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --lanes-per-gpu $lanes --nrx-max-cb-fail $cb --rescue-deadline-ms $deadline --nrx-bound-ms $bound --nrx-bound-corun-ms $bound --activity-prob $act --activity-mode $mode --ring 512 --seed $((83000 + 1000 * seed)) --ring-offset $((50 * (seed - 1)))${NRX_FLAGS:+ --nrx-flags $NRX_FLAGS}" \
  SLOT_MATRIX="$matrix" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1)
echo "$out"
echo "$out" | grep -q "^FAILED" && exit 1
exit 0
