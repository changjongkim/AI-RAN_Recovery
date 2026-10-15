#!/usr/bin/env bash
# N4: fine-tune the large NVlabs neural receiver for 1-4 users on the same resources, starting from the published
# two-user weights (the network keeps one state per user, so they load into the four-user receiver).  NVlabs training
# loop and data (UMi, 4 PRBs per sample, MCS 14), learning rate 1e-4, batch 64; Eb/No per number of active users
# 1/2/3/4: 0-10, 2-12, 4-15, 6-18 dB.  One call trains ITER iterations and saves the weights every 1000 (the training
# script loads the saved weights when it starts, so calls continue where the last one stopped).
# usage: bash exp/n4_train.sh STEP [ITER]        weights: third_party/neural_rx/weights/nrx_large_mu4_weights
#        snapshots: results/n4/weights_step<STEP>, log: results/n4/train_step<STEP>.log
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
step=$1; iters=${2:-50000}
mkdir -p $RESULTS/n4
cfg=$NRX_REPO/config/nrx_large_mu4.cfg
python3 - "$NRX_REPO/config/mu_nrx_large_273_UMi_4.cfg" "$cfg" "$iters" <<'PY'
import re, sys
src, dst, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(src).read()
t = re.sub(r"label\s*=\s*'[^']+'", "label = 'nrx_large_mu4'", t, count=1)
schedule = ('training_schedule = {\n'
            f'    "num_iter": [{iters}],\n'
            '    "learning_rate": [0.0001],\n'
            '    "batch_size": [64],\n'
            '    "train_tx": [False],\n'
            '    "min_training_snr_db": [[0., 2., 4., 6.]],\n'
            '    "max_training_snr_db": [[10., 12., 15., 18.]],\n'
            '    "double_readout": [True],\n'
            '    "apply_multiloss": [True],\n'
            '    "weighting_double_readout": [0.01]}\n')
t, n = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S)
assert n == 1, "no training schedule"
open(dst + ".tmp", "w").write(t)
import os; os.replace(dst + ".tmp", dst)
PY
w=$NRX_REPO/weights/nrx_large_mu4_weights
[ -f $w ] || cp $NRX_REPO/weights/nrx_large_weights $w
echo "step $step: $iters iterations from $(stat -c %y $w) $(date +%T)"
( cd $NRX_REPO/scripts && tfgen python3 train_neural_rx.py -config_name nrx_large_mu4.cfg -gpu 0 ) \
    > $RESULTS/n4/train_step$step.log 2>&1
rc=$?
cp $w $RESULTS/n4/weights_step$step
grep -iE "loss|iter|error|Traceback" $RESULTS/n4/train_step$step.log | tail -15
echo N4_TRAIN_DONE step $step rc=$rc $(date +%T)
exit $rc
