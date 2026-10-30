#!/usr/bin/env bash
# N27: the large neural receiver fine-tuned for the small cells of the evaluation (UMi at 3.5 GHz; the published weights
# are trained on UMi at 2.14 GHz).  Starts from the published 2-user 4-RX weights and trains without interference
# (src/train_nrx_ici.py with --p-none 1.0), NVlabs data (UMi, 4 PRBs per sample) at a carrier of 3.5 GHz, learning
# rate LR (default 1e-4), batch 64, XLA off, the published Eb/No ranges 0..10 / 2..12 dB for 1 / 2 users.
# usage: bash exp/n27_train.sh STEP [ITER]   (STEP=config writes the configuration only)
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
step=$1; iters=${2:-40000}
mkdir -p $RESULTS/n27
cfg=$NRX_REPO/config/nrx_large_f35.cfg
python3 - "$NRX_REPO/config/nrx_large.cfg" "$cfg" "$iters" "${LR:-0.0001}" <<'PY'
import os, re, sys
src, dst, iters, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
t = open(src).read()
t = re.sub(r"label\s*=\s*'[^']+'", "label = 'nrx_large_f35'", t, count=1)
t, n = re.subn(r"(?m)^carrier_frequency\s*=.*", "carrier_frequency = 3500000000.0", t); assert n == 1
schedule = ('training_schedule = {\n'
            f'    "num_iter": [{iters}],\n'
            f'    "learning_rate": [{lr}],\n'
            '    "batch_size": [64],\n'
            '    "train_tx": [False],\n'
            '    "min_training_snr_db": [[0., 2.]],\n'
            '    "max_training_snr_db": [[10., 12.]],\n'
            '    "double_readout": [True],\n'
            '    "apply_multiloss": [True],\n'
            '    "weighting_double_readout": [0.01]}\n')
t, n = re.subn(r"training_schedule = \{.*?\}[^\n]*\n", schedule, t, count=1, flags=re.S); assert n == 1
t, n = re.subn(r"(?m)^xla\s*=.*", "xla = False", t); assert n == 1
open(dst + ".tmp", "w").write(t); os.replace(dst + ".tmp", dst)
PY
[ "$step" = config ] && { echo "wrote $cfg"; exit 0; }
w=$NRX_REPO/weights/nrx_large_f35_weights
[ -f $w ] || cp $NRX_REPO/weights/nrx_large_weights $w
echo "step $step: $iters iterations, learning rate ${LR:-0.0001}, from $(stat -c %y $w) $(date +%T)"
( cd $NRX_REPO/scripts && tfgen python3 $REC/src/train_nrx_ici.py -config_name nrx_large_f35.cfg -gpu 0 --p-none 1.0 ) \
    > $RESULTS/n27/train_step$step.log 2>&1
rc=$?
cp $w $RESULTS/n27/weights_step$step
grep -iE "error|Traceback" $RESULTS/n27/train_step$step.log | tail -5
echo N27_TRAIN_DONE step $step rc=$rc $(date +%T)
exit $rc
