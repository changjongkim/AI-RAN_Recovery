#!/usr/bin/env bash
# Side pool: while the N4 training uses GPU 0, GPUs 1-3 work through queue_b/side_items (exp/side_items.py) until
# MINUTES have passed; every item is one classical_rx.py run with its own output, so a run cut by the deadline is
# simply done again by the next pool.  An item that fails twice (not by the deadline) is left out (<item>.fails, .err).  The first GPU first runs the N5 smoke test and the LMMSE covariance matrices;
# N5 items wait for both.  Merge with exp/side_merge.sh.
# usage: bash exp/side_pool.sh MINUTES ["GPUS"]        results: results/side/<item>.json/.npz/.log, results/side/pool.log
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
minutes=$1; gpus=(${2:-1 2 3})
deadline=$(( $(date +%s) + minutes * 60 ))
S=$RESULTS/side; mkdir -p $S
items=$REC/queue_b/side_items
[ -s $items ] || python3 $REC/exp/side_items.py > $items
find $S -maxdepth 1 -name '*.claim' -mmin +60 -exec rm -rf {} + 2>/dev/null   # claims of a killed pool (an item takes < 25 min); pools on two nodes can share the list
cov=$NRX_REPO/weights/nrx_large_freq_cov_mat.npy
smoke_ok() { grep -q "N5_SMOKE pass" $S/n5_smoke.log 2>/dev/null; }
log() { echo "$(date '+%F %T') $*" >> $S/pool.log; }
tf() { local s=$1; shift; timeout -k 30 $s bash -c 'source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh; tfgen "$@"' _ "$@"; }   # timeout cannot run a shell function

prep() {
  local g=$1
  if ! smoke_ok; then
    GPUS=$g timeout -k 30 $(( deadline - $(date +%s) )) bash $REC/exp/n5_smoke.sh > $S/n5_smoke.log 2>&1
    log "GPU $g n5 smoke: $(grep N5_SMOKE $S/n5_smoke.log || echo 'no verdict')"
  fi
  if smoke_ok && [ ! -f $cov ]; then
    (cd $NRX_REPO/scripts && tf $(( deadline - $(date +%s) )) python3 compute_cov_mat.py -config_name nrx_large.cfg -num_samples 100000 -gpu $g) > $S/cov.log 2>&1
    log "GPU $g covariance matrices: $([ -f $cov ] && echo done || echo 'not done')"
  fi
}

worker() {
  local g=$1 name needs opts pick left rc
  [ "$g" = "${gpus[0]}" ] && prep $g
  while true; do
    left=$(( deadline - $(date +%s) )); [ $left -gt 180 ] || return 0
    pick=""
    while IFS='|' read -r name needs opts; do
      grep -q '"prbs": 273' $S/$name.json 2>/dev/null && continue       # done (runs at 132 PRBs are redone)
      [ "$(cat $S/$name.fails 2>/dev/null || echo 0)" -ge 2 ] && continue     # failed twice: leave it, see <item>.err
      if [ "$needs" = n5 ]; then smoke_ok && [ -f $cov ] || continue; fi
      mkdir $S/$name.claim 2>/dev/null || continue
      pick=$name; break
    done < $items
    [ -z "$pick" ] && { log "GPU $g: nothing left that can run"; return 0; }
    tf $left python3 $REC/src/classical_rx.py --nrx-repo $NRX_REPO --gpu $g $opts \
        --output $S/$pick.json > $S/$pick.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      rm -f $S/$pick.json $S/$pick.npz
      # a run cut by the deadline (timeout: 124, or killed: 137) is not a failure of the item
      if [ $rc -ne 124 ] && [ $rc -ne 137 ]; then
        echo $(( $(cat $S/$pick.fails 2>/dev/null || echo 0) + 1 )) > $S/$pick.fails
        cp $S/$pick.log $S/$pick.err
      fi
    fi
    rmdir $S/$pick.claim
    log "GPU $g $pick rc=$rc"
  done
}

log "pool start: $minutes min on GPUs ${gpus[*]}"
pids=()
for g in "${gpus[@]}"; do worker $g & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
log "pool end: $(ls $S/*.json 2>/dev/null | grep -c . ) items done of $(grep -c . $items)"
