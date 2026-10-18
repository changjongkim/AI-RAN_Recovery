#!/usr/bin/env bash
# N4b: fine-tune an 8-RX large neural receiver (1-2 users, UMi as NVlabs) from the published 4-RX weights widened by
# src/nrx_widen.py.  NVlabs training loop and data (4 PRBs per sample, MCS 14), learning rate 1e-4, batch 64, XLA off
# (cuDNN 8.9 fails under XLA, see n4_train.sh); Eb/No for 1/2 active users -3..9 / -1..11 dB (the 4-RX ranges moved
# down by the 3 dB array gain).  One call trains ITER iterations and saves every 1000; calls continue from the weights.
# usage: bash exp/n4b_train.sh STEP [ITER]     (STEP=config only writes the configuration)
#   weights: third_party/neural_rx/weights/nrx_large_8rx_weights, snapshots results/n4b/weights_step<STEP>
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
step=$1; iters=${2:-35000}
mkdir -p $RESULTS/n4b
cfg=$NRX_REPO/config/nrx_large_8rx.cfg
python3 - "$NRX_REPO/config/nrx_large.cfg" "$cfg" "$iters" "${LR:-0.0001}" <<'PY'
import os, re, sys
src, dst, iters, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
t = open(src).read()
t = re.sub(r"label\s*=\s*'[^']+'", "label = 'nrx_large_8rx'", t, count=1)
t, n = re.subn(r"(?m)^num_rx_antennas\s*=.*", "num_rx_antennas = 8", t); assert n == 1
schedule = ('training_schedule = {\n'
            f'    "num_iter": [{iters}],\n'
            f'    "learning_rate": [{lr}],\n'
            '    "batch_size": [64],\n'
            '    "train_tx": [False],\n'
            '    "min_training_snr_db": [[-3., -1.]],\n'
            '    "max_training_snr_db": [[9., 11.]],\n'
            '    "double_readout": [True],\n'
            '    "apply_multiloss": [True],\n'
            '    "weighting_double_readout": [0.01]}\n')
t, n = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S); assert n == 1
t, n = re.subn(r"(?m)^xla\s*=.*", "xla = False", t); assert n == 1
open(dst + ".tmp", "w").write(t); os.replace(dst + ".tmp", dst)
PY
[ "$step" = config ] && { echo "wrote $cfg"; exit 0; }
w=$NRX_REPO/weights/nrx_large_8rx_weights
if [ ! -f $w ]; then
  tfgen python3 $REC/src/nrx_widen.py --nrx-repo $NRX_REPO --dst-config nrx_large_8rx.cfg \
      --src-weights $NRX_REPO/weights/nrx_large_weights --out $w --gpu 0 || { echo "widening failed"; exit 1; }
  cp $w $RESULTS/n4b/weights_widened
fi
echo "step $step: $iters iterations, learning rate ${LR:-0.0001}, from $(stat -c %y $w) $(date +%T)"
( cd $NRX_REPO/scripts && tfgen python3 train_neural_rx.py -config_name nrx_large_8rx.cfg -gpu 0 ) \
    > $RESULTS/n4b/train_step$step.log 2>&1
rc=$?
cp $w $RESULTS/n4b/weights_step$step
grep -iE "error|Traceback" $RESULTS/n4b/train_step$step.log | tail -5
echo N4B_TRAIN_DONE step $step rc=$rc $(date +%T)
exit $rc
