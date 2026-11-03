#!/usr/bin/env bash
# U3n: neural receivers for more users on the configurations of the user sweep U3b (exp/u3_configs.sh), trained in
# parallel on one node until a wall-clock end, so the receiver of each sweep point is a trained NRx and not the bound.
#   GPU 0  nrx_large_8rx_mu8    8 RX, 1-8 users (8 DMRS ports, type 1, two symbols), from the 8-RX weights of N4b
#   GPU 1  nrx_large_16rx_mu12  16 RX, 1-12 users (12 ports, type 2, two symbols), from the 8-RX weights doubled to
#                               16 RX by src/nrx_widen.py
#   GPU 2  nrx_large_mu4        4 RX, 1-4 users, continues N4 (exp/n4_train.sh) at a lower learning rate
# NVlabs training loop and data (UMi at 2.14 GHz and 0-56 m/s as the published NRx and N4/N4b, so the sweep on UMa
# at 3.5 GHz is outside the training data for every NRx alike; 4 PRBs per sample, MCS 14), batch 64, XLA off (cuDNN 8.9, see
# n4_train.sh).  Eb/No per number of active users follows the sweep ranges of each antenna count.  The NVlabs script
# saves the weights every 1000 iterations and loads them when it starts, so a run cut by the end of an allocation
# continues in the next one.  Before each start the weights are checked to load and are copied to a backup; a file
# cut while it was written is restored from the backup.  After the end time the script only snapshots and exits 0.
# usage: bash exp/u3n_train.sh END_TIME          e.g. "2026-11-03 14:00"
#   weights: third_party/neural_rx/weights/<label>_weights, snapshots results/u3n/<label>_<time>, logs results/u3n/
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
end=$(date -d "$1" +%s) || exit 2
O=$RESULTS/u3n; mkdir -p $O
C=$NRX_REPO/config; W=$NRX_REPO/weights
log() { echo "$(date '+%F %T') $*" | tee -a $O/train.log; }

config() {   # label lr min max  (min/max: comma lists, one Eb/No per number of active users)
  python3 - "$C/$1.cfg" "$2" "$3" "$4" <<'PY'
import os, re, sys
cfg, lr, lo, hi = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
t = open(cfg).read()
m = re.search(r"(?m)^max_num_tx\s*=\s*(\d+)", t); n = int(m.group(1))
lo = [float(x) for x in lo.split(",")]; hi = [float(x) for x in hi.split(",")]
assert len(lo) == len(hi) == n, (n, lo, hi)
f = lambda v: "[[" + ", ".join(f"{x:.1f}" for x in v) + "]]"
schedule = ('training_schedule = {\n'
            '    "num_iter": [10000000],\n'
            f'    "learning_rate": [{lr}],\n'
            '    "batch_size": [64],\n'
            '    "train_tx": [False],\n'
            f'    "min_training_snr_db": {f(lo)},\n'
            f'    "max_training_snr_db": {f(hi)},\n'
            '    "double_readout": [True],\n'
            '    "apply_multiloss": [True],\n'
            '    "weighting_double_readout": [0.01]}\n')
t, k = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S); assert k == 1
t, k = re.subn(r"(?m)^xla\s*=.*", "xla = False", t); assert k == 1
open(cfg + ".tmp", "w").write(t); os.replace(cfg + ".tmp", cfg)
PY
}

loads() { python3 -c "import pickle,sys; w=pickle.load(open(sys.argv[1],'rb')); assert len(w) > 10" "$1" 2>/dev/null; }

train() {    # gpu label lr min max
  local g=$1 label=$2 w=$W/$2_weights left
  config $label $3 "$4" "$5" || { log "$label: configuration failed"; return 1; }
  if [ -f $w ] && ! loads $w; then
    log "$label: weights do not load (cut while saved); restored from the backup"; cp $w.bak $w
  fi
  [ -f $w ] && cp $w $w.bak
  left=$(( end - $(date +%s) ))
  [ $left -gt 300 ] || { log "$label: end time reached"; return 0; }
  log "$label: training on GPU $g for at most $(( left / 60 )) min"
  ( cd $NRX_REPO/scripts && timeout -k 60 $left bash -c 'source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh; tfgen "$@"' _ \
      python3 train_neural_rx.py -config_name $label.cfg -gpu $g ) >> $O/${label}.log 2>&1
  local rc=$?
  loads $w || { log "$label: weights cut at the end; restored from the backup"; cp $w.bak $w; }
  log "$label: stopped with exit code $rc"
}

# starting points (once)
if [ ! -f $W/nrx_large_8rx_mu8_weights ]; then
  cp $W/nrx_large_8rx_weights $W/nrx_large_8rx_mu8_weights; log "nrx_large_8rx_mu8: starts from nrx_large_8rx_weights"
fi
if [ ! -f $W/nrx_large_16rx_mu12_weights ]; then
  tfgen python3 $REC/src/nrx_widen.py --nrx-repo $NRX_REPO --src-config nrx_large_8rx.cfg --dst-config nrx_large_16rx_mu12.cfg \
      --src-weights $W/nrx_large_8rx_weights --out $W/nrx_large_16rx_mu12_weights --gpu 1 >> $O/widen.log 2>&1 \
      || { log "widening to 16 RX failed (results/u3n/widen.log)"; }
  [ -f $W/nrx_large_16rx_mu12_weights ] && cp $W/nrx_large_16rx_mu12_weights $O/nrx_large_16rx_mu12_widened
  log "nrx_large_16rx_mu12: starts from the 8-RX weights doubled to 16 RX"
fi

#     GPU label                 lr       Eb/No min per users                    Eb/No max per users
train 0 nrx_large_8rx_mu8   0.0001 "-4,-2,0,2,3,4,5,6"                  "8,10,11,12,13,14,15,16" &
p1=$!
train 1 nrx_large_16rx_mu12 0.0001 "-8,-7,-6,-5,-4,-3,-2,-1,0,1,1.5,2" "4,5,6,7,7.5,8,8.5,9,9.5,10,11,12" &
p2=$!
train 2 nrx_large_mu4       0.00005 "0,2,4,6"                           "10,12,15,18" &
p3=$!
wait $p1 $p2 $p3
stamp=$(date +%m%d_%H%M)
for l in nrx_large_8rx_mu8 nrx_large_16rx_mu12 nrx_large_mu4; do [ -f $W/${l}_weights ] && cp $W/${l}_weights $O/${l}_$stamp; done
if [ $(date +%s) -lt $(( end - 300 )) ]; then
  log "U3N_TRAIN round ended before the end time"; exit 3      # cut early (error): the side runner records it; see logs
fi
log "U3N_TRAIN_DONE"; exit 0
