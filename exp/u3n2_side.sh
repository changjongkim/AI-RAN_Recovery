#!/usr/bin/env bash
# U3n2, second interactive allocation (queue_b): three more training variants next to A, B, C of exp/u3n2_train.sh,
# so that a working recipe for the two-symbol DMRS shows up sooner.
#   GPU 0  nrx_large_8rx_mu8d    U3n weights of A, learning rate 1e-3 and batch 128 as the NVlabs schedule  (D)
#   GPU 1  nrx_large_16rx_mu12e  U3n weights of C, learning rate 1e-4 (the control of C at 1e-3)           (E)
#   GPU 2  nrx_large_8rx_mu8f    the one-symbol 8-RX weights of N4b (before U3n), learning rate 1e-3      (F)
#   GPU 3  snapshots of D, E, F on 8 RX with 2 and 8 users and on 16 RX with 4 and 8 users (seed 900, UMa 3.5 GHz,
#          16 slots per Eb/No, LS + LMMSE and NRx), results/u3n2/curve.tsv
# One call runs for MINUTES or until END, whichever comes first, and exits 0; the next call continues from the saved
# weights (the NVlabs script saves every 1000 iterations and loads the weights when it starts).
# usage: bash exp/u3n2_side.sh MINUTES END_TIME
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
R=/pscratch/sd/s/sgkim/kcj/airan_recovery; cd $R || exit 1
stop=$(( $(date +%s) + $1 * 60 )); end=$(date -d "$2" +%s) || exit 2; [ $end -lt $stop ] && stop=$end
O=$RESULTS/u3n2; mkdir -p $O
C=$NRX_REPO/config; W=$NRX_REPO/weights
log() { echo "$(date '+%F %T') $*" | tee -a $O/train.log; }
[ $(( stop - $(date +%s) )) -gt 600 ] || { log "end time reached"; exit 0; }

config() {   # label lr min max batch
  python3 - "$C/$1.cfg" "$2" "$3" "$4" "${5:-64}" <<'PY'
import os, re, sys
cfg, lr, lo, hi, bs = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
t = open(cfg).read()
n = int(re.search(r"(?m)^max_num_tx\s*=\s*(\d+)", t).group(1))
lo = [float(x) for x in lo.split(",")]; hi = [float(x) for x in hi.split(",")]
assert len(lo) == len(hi) == n, (n, lo, hi)
f = lambda v: "[[" + ", ".join(f"{x:.1f}" for x in v) + "]]"
schedule = ('training_schedule = {\n    "num_iter": [10000000],\n' f'    "learning_rate": [{lr}],\n'
            f'    "batch_size": [{bs}],\n    "train_tx": [False],\n'
            f'    "min_training_snr_db": {f(lo)},\n    "max_training_snr_db": {f(hi)},\n'
            '    "double_readout": [True],\n    "apply_multiloss": [True],\n    "weighting_double_readout": [0.01]}\n')
t, k = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S); assert k == 1
t, k = re.subn(r"(?m)^xla\s*=.*", "xla = False", t); assert k == 1
open(cfg + ".tmp", "w").write(t); os.replace(cfg + ".tmp", cfg)
PY
}
loads() { python3 -c "import pickle,sys; w=pickle.load(open(sys.argv[1],'rb')); assert len(w) > 10" "$1" 2>/dev/null; }

variant() {  # new-label base-label start-weights
  if [ ! -f $C/$1.cfg ]; then
    sed "s/^label = '$2'/label = '$1'/" $C/$2.cfg > $C/$1.cfg
    grep -q "label = '$1'" $C/$1.cfg || { log "$1: configuration failed"; return 1; }
    for k in freq time space; do ln -sf $2_${k}_cov_mat.npy $W/$1_${k}_cov_mat.npy; done
  fi
  [ -f $W/$1_weights ] || cp $3 $W/$1_weights
}
variant nrx_large_8rx_mu8d nrx_large_8rx_mu8 $RESULTS/u3n/nrx_large_8rx_mu8_final || exit 1
variant nrx_large_16rx_mu12e nrx_large_16rx_mu12 $RESULTS/u3n/nrx_large_16rx_mu12_final || exit 1
variant nrx_large_8rx_mu8f nrx_large_8rx_mu8 $W/nrx_large_8rx_weights || exit 1

train() {    # gpu label lr min max [batch]
  local g=$1 label=$2 w=$W/$2_weights left
  config $label $3 "$4" "$5" "${6:-64}" || { log "$label: configuration failed"; return 1; }
  if [ -f $w ] && ! loads $w; then log "$label: weights do not load; restored from the backup"; cp $w.bak $w; fi
  cp $w $w.bak
  left=$(( stop - $(date +%s) )); log "$label: training on GPU $g at $3 for $(( left / 60 )) min"
  ( cd $NRX_REPO/scripts && timeout -k 60 $left bash -c 'source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh; tfgen "$@"' _ \
      python3 train_neural_rx.py -config_name $label.cfg -gpu $g ) >> $O/${label}.log 2>&1
  loads $w || { log "$label: weights cut at the end; restored from the backup"; cp $w.bak $w; }
  log "$label: stopped"
}

curve() {    # GPU 3: snapshots every two hours until 40 minutes before the stop
  local t0 stamp l w point
  while [ $(( stop - $(date +%s) )) -gt 2400 ]; do
    t0=$(date +%s); stamp=$(date +%m%d_%H%M)
    for l in nrx_large_8rx_mu8d nrx_large_8rx_mu8f nrx_large_16rx_mu12e; do
      w=$O/w_${l}_$stamp; cp $W/${l}_weights $w; loads $w || { sleep 30; cp $W/${l}_weights $w; }
      if [[ $l == *16rx* ]]; then pts="16:4:2,4 16:8:4,6"; cfg=nrx_large_16rx_mu12.cfg; else pts="8:2:2,4 8:8:10,12"; cfg=nrx_large_8rx_mu8.cfg; fi
      for point in $pts; do
        IFS=: read rx u eb <<<"$point"
        name=u3n2/c_${l}_${stamp}_r${rx}u${u}
        GPUS=3 SEED0=900 CARRIER_GHZ=3.5 bash exp/classical.sh $name --base-config $cfg --num-rx $rx --users $u --channel UMa \
            --systems baseline_lslin_lmmse,nrx --nrx-weights $w --ebno=$eb --slots 16 --batch 1 > /dev/null 2>&1
        python3 - "$RESULTS/$name.json" "$l" "$stamp" "$rx" "$u" >> $O/curve.tsv <<'PY'
import json, sys
f, l, stamp, rx, u = sys.argv[1:]
try:
    d = json.load(open(f)); r = d["recovers_pct_of_conventional_failures"]["nrx"]; c = d["fail_pct"]["baseline_lslin_lmmse"]
    print(f"{stamp}\t{l}\t{rx}\t{u}\t{c:.1f}\t{r:.1f}")
except Exception as e:
    print(f"{stamp}\t{l}\t{rx}\t{u}\tfailed\t{e}")
PY
        [ $(( stop - $(date +%s) )) -gt 900 ] || return 0
      done
    done
    while [ $(( $(date +%s) - t0 )) -lt 7200 ] && [ $(( stop - $(date +%s) )) -gt 2400 ]; do sleep 60; done
  done
}

#     GPU label                  lr      Eb/No min per users                    Eb/No max per users              batch
train 0 nrx_large_8rx_mu8d   0.001  "-4,-2,0,2,3,4,5,6"                  "8,10,11,12,13,14,15,16"            128 &
train 1 nrx_large_16rx_mu12e 0.0001 "-8,-7,-6,-5,-4,-3,-2,-1,0,1,1.5,2" "4,5,6,7,7.5,8,8.5,9,9.5,10,11,12" 64 &
train 2 nrx_large_8rx_mu8f   0.001  "-4,-2,0,2,3,4,5,6"                  "8,10,11,12,13,14,15,16"            64 &
curve &
wait
for l in nrx_large_8rx_mu8d nrx_large_8rx_mu8f nrx_large_16rx_mu12e; do cp $W/${l}_weights $O/${l}_$(date +%m%d_%H%M); done
log "U3N2 side chunk done"
exit 0
