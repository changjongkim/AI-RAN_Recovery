#!/usr/bin/env bash
# U3s: neural receiver for 8 RX and 1-4 users on the one-symbol DMRS with the port order of the 4-RX NRx that works
# (nrx_large_mu4: ports 0, 2, 1, 3, so the first two users are on different CDM groups).  The two-symbol 8-port
# configuration of U3n/U3n2 (ports 0-7 in order, the first two users on one CDM group) gave an NRx that recovered none
# of the conventional failures after about 16 hours of training, also on its own training channel (UMi), so the 8-RX
# point of the user sweep uses this configuration instead: nrx_large_8rx_mu4.cfg is nrx_large_mu4.cfg with 8 RX
# (exp/u1_configs.sh; covariances of nrx_large_8rx).  Two sets of three variants, one set per allocation; GPU 3 of each
# set evaluates snapshots of its variants every two hours (seed 900, 16 slots per Eb/No, LS + LMMSE and NRx): 8 RX
# with 2 users at Eb/No 0, 2 dB and 4 users at 6, 8 dB on UMa at 3.5 GHz (the sweep), and 2 users at 0, 2 dB on UMi at
# the configuration carrier (the training channel, rows marked umi), into results/u3s/curve.tsv.
#   set a  GPU 0  nrx_large_8rx_mu4a  from nrx_large_8rx_weights (N4b: 8 RX, 2 users)   lr 1e-4, batch 64 (N4 recipe)
#          GPU 1  nrx_large_8rx_mu4b  from the 4-RX 1-4-user weights of U3n widened     lr 1e-4, batch 64
#                                     to 8 RX by src/nrx_widen.py
#          GPU 2  nrx_large_8rx_mu4c  from nrx_large_8rx_weights                       lr 5e-4, batch 64 (N4b rate)
#   set b  GPU 0  nrx_large_8rx_mu4d  widened 4-RX weights                               lr 5e-4, batch 64
#          GPU 1  nrx_large_8rx_mu4e  from nrx_large_8rx_weights                       lr 1e-3, batch 128 (NVlabs)
#          GPU 2  nrx_large_8rx_mu4f  widened 4-RX weights                               lr 1e-3, batch 128
# NVlabs training loop and data (UMi at 2.14 GHz, 4 PRBs per sample, MCS 14, XLA off as n4_train.sh); Eb/No per number
# of active users 1/2/3/4: -3..9, -1..11, 1..13, 3..15 dB (the 4-RX ranges of N4 shifted by -3 dB, as N4b did for 8 RX).
# One call runs for MINUTES or until END, whichever comes first, and exits 0; the next call continues from the saved
# weights (the NVlabs script saves every 1000 iterations and loads the weights when it starts).
# usage: bash exp/u3s_train.sh SET MINUTES END_TIME        e.g. bash exp/u3s_train.sh a 200 "2026-11-05 07:00"
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
R=/pscratch/sd/s/sgkim/kcj/airan_recovery; cd $R || exit 1
set_=$1
stop=$(( $(date +%s) + $2 * 60 )); end=$(date -d "$3" +%s) || exit 2; [ $end -lt $stop ] && stop=$end
O=$RESULTS/u3s; mkdir -p $O
C=$NRX_REPO/config; W=$NRX_REPO/weights
BASE=nrx_large_8rx_mu4
log() { echo "$(date '+%F %T') [$set_] $*" | tee -a $O/train.log; }
[ $(( stop - $(date +%s) )) -gt 600 ] || { log "end time reached"; exit 0; }

config() {   # label lr batch
  python3 - "$C/$1.cfg" "$2" "$3" <<'PY'
import os, re, sys
cfg, lr, bs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
t = open(cfg).read()
n = int(re.search(r"(?m)^max_num_tx\s*=\s*(\d+)", t).group(1)); assert n == 4, n
assert re.search(r"(?m)^num_rx_antennas\s*=\s*8\b", t) and re.search(r"(?m)^dmrs_length\s*=\s*1\b", t)
assert re.search(r"(?m)^dmrs_port_sets\s*=\s*\[\[0\], \[2\], \[1\], \[3\]\]", t)
schedule = ('training_schedule = {\n    "num_iter": [10000000],\n' f'    "learning_rate": [{lr}],\n'
            f'    "batch_size": [{bs}],\n    "train_tx": [False],\n'
            '    "min_training_snr_db": [[-3.0, -1.0, 1.0, 3.0]],\n    "max_training_snr_db": [[9.0, 11.0, 13.0, 15.0]],\n'
            '    "double_readout": [True],\n    "apply_multiloss": [True],\n    "weighting_double_readout": [0.01]}\n')
t, k = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S); assert k == 1
t, k = re.subn(r"(?m)^xla\s*=.*", "xla = False", t); assert k == 1
open(cfg + ".tmp", "w").write(t); os.replace(cfg + ".tmp", cfg)
PY
}
loads() { python3 -c "import pickle,sys; w=pickle.load(open(sys.argv[1],'rb')); assert len(w) > 10" "$1" 2>/dev/null; }

# starting point from the 4-RX weights (once; made on the login node at 2026-11-04 01:34, made again here if missing)
widened=$O/nrx_large_8rx_mu4_widened
if ! loads $widened; then
  tfgen python3 $REC/src/nrx_widen.py --nrx-repo $NRX_REPO --src-config nrx_large_mu4.cfg --dst-config $BASE.cfg \
      --src-weights $R/models/nrx_large_mu4_u3n_weights --out $widened.$set_.tmp --gpu 0 >> $O/widen.log 2>&1 \
    && mv $widened.$set_.tmp $widened
  loads $widened || { log "widening the 4-RX weights failed (results/u3s/widen.log)"; exit 1; }
fi

variant() {  # label start-weights
  if [ ! -f $C/$1.cfg ]; then
    sed "s/^label = '$BASE'/label = '$1'/" $C/$BASE.cfg > $C/$1.cfg
    grep -q "^label = '$1'" $C/$1.cfg || { log "$1: configuration failed"; return 1; }
    for k in freq time space; do ln -sf nrx_large_8rx_${k}_cov_mat.npy $W/$1_${k}_cov_mat.npy; done
  fi
  [ -f $W/$1_weights ] || { cp $2 $W/$1_weights; log "$1: starts from $2"; }
}

train() {    # gpu label lr batch
  local g=$1 label=$2 w=$W/$2_weights left
  config $label $3 $4 || { log "$label: configuration failed"; return 1; }
  if ! loads $w; then log "$label: weights do not load; restored from the backup"; cp $w.bak $w; fi
  cp $w $w.bak
  left=$(( stop - $(date +%s) )); log "$label: training on GPU $g at $3, batch $4, for $(( left / 60 )) min"
  ( cd $NRX_REPO/scripts && timeout -k 60 $left bash -c 'source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh; tfgen "$@"' _ \
      python3 train_neural_rx.py -config_name $label.cfg -gpu $g ) >> $O/${label}.log 2>&1
  loads $w || { log "$label: weights cut at the end; restored from the backup"; cp $w.bak $w; }
  log "$label: stopped"
}

curve() {    # GPU 3: snapshots every two hours until 40 minutes before the stop
  local t0 stamp l w point rx u eb ch tag ghz name
  while [ $(( stop - $(date +%s) )) -gt 2400 ]; do
    t0=$(date +%s); stamp=$(date +%m%d_%H%M)
    for l in "${labels[@]}"; do
      w=$O/w_${l}_$stamp; cp $W/${l}_weights $w; loads $w || { sleep 30; cp $W/${l}_weights $w; }
      for point in 8:2:0,2:UMa 8:4:6,8:UMa 8:2:0,2:UMi; do
        IFS=: read rx u eb ch <<<"$point"
        if [ $ch = UMi ]; then tag=umi; ghz=; else tag=; ghz=3.5; fi
        name=u3s/c_${l}_${stamp}_r${rx}u${u}${tag}
        GPUS=3 SEED0=900 CARRIER_GHZ=$ghz bash exp/classical.sh $name --base-config $BASE.cfg --num-rx $rx --users $u \
            --channel $ch --systems baseline_lslin_lmmse,nrx --nrx-weights $w --ebno=$eb --slots 16 --batch 1 > /dev/null 2>&1
        python3 - "$RESULTS/$name.json" "$l" "$stamp" "$rx" "$u$tag" >> $O/curve.tsv <<'PY'
import json, sys
f, l, stamp, rx, u = sys.argv[1:]
try:
    d = json.load(open(f)); r = d["recovers_pct_of_conventional_failures"]["nrx"]; c = d["fail_pct"]["baseline_lslin_lmmse"]
    n = d["fail_pct"]["nrx"]
    print(f"{stamp}\t{l}\t{rx}\t{u}\t{c:.1f}\t{n:.1f}\t{r:.1f}")
except Exception as e:
    print(f"{stamp}\t{l}\t{rx}\t{u}\tfailed\t{e}")
PY
        [ $(( stop - $(date +%s) )) -gt 900 ] || return 0
      done
    done
    while [ $(( $(date +%s) - t0 )) -lt 7200 ] && [ $(( stop - $(date +%s) )) -gt 2400 ]; do sleep 60; done
  done
}

[ -s $O/curve.tsv ] || printf 'stamp\tlabel\trx\tusers\tconventional_fail_pct\tnrx_fail_pct\tnrx_recovers_pct\n' > $O/curve.tsv
case $set_ in
  a) variant ${BASE}a $W/nrx_large_8rx_weights || exit 1
     variant ${BASE}b $widened || exit 1
     variant ${BASE}c $W/nrx_large_8rx_weights || exit 1
     labels=(${BASE}a ${BASE}b ${BASE}c)
     train 0 ${BASE}a 0.0001 64 &
     train 1 ${BASE}b 0.0001 64 &
     train 2 ${BASE}c 0.0005 64 & ;;
  b) variant ${BASE}d $widened || exit 1
     variant ${BASE}e $W/nrx_large_8rx_weights || exit 1
     variant ${BASE}f $widened || exit 1
     labels=(${BASE}d ${BASE}e ${BASE}f)
     train 0 ${BASE}d 0.0005 64 &
     train 1 ${BASE}e 0.001 128 &
     train 2 ${BASE}f 0.001 128 & ;;
  *) echo "set must be a or b"; exit 2 ;;
esac
curve &
wait
for l in "${labels[@]}"; do cp $W/${l}_weights $O/${l}_$(date +%m%d_%H%M); done
log "U3S chunk done"
exit 0
