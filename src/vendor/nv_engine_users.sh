#!/usr/bin/env bash
# v38 S3: GPU time of the small (nrx_rt) and the large (nrx_large) NeuralRx engine with 1-4 users on the same
# resources.  The run time does not depend on the values of the weights, so no training is needed (nv_export_mu.py).
# Engines in runtime/softwall_same_gpu/engines/nv_users; GPU compute time of trtexec (FP16, 273 PRB, one GPU, alone,
# 10 s) in results/backstop_slot/v38_engine_users.txt.  The 2-user engines check the method against README 4.4
# (1.35 ms and 4.29 ms).
# usage: bash nv_engine_users.sh          (on a GPU node)
set -uo pipefail
ROOT=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
source $ROOT/scripts_for_node/backstop_slot/slot_env.sh
D=$SOFTWALL_RUNTIME/engines/nv_users; E=/softwall_runtime/engines/nv_users; mkdir -p $D
OUT=$ROOT/results/backstop_slot/v38_engine_users.txt
echo "# label users weights | GPU compute time (ms): mean median p99 | $(date '+%F %T') job ${SLURM_JOB_ID:-}" > $OUT
for label in ${LABELS:-nrx_rt nrx_large}; do
  for n in ${USERS:-1 2 3 4}; do
    stem=${label}_273prb_${n}ue
    echo "=== $stem $(date +%T)"
    ( cd $ROOT/third_party/neural_rx/scripts && shifter --module=gpu --image="$AERIAL_IMAGE" \
        --env=PYTHONPATH=$ROOT/runtime/softwall_same_gpu/nrx_export_deps --env=PYTHONNOUSERSITE=1 \
        python3 $ROOT/scripts_for_node/backstop_slot/nv/nv_export_mu.py --base-config $label.cfg --num-tx $n \
        --output-dir $D ) > $D/$stem.export.log 2>&1
    [ -s $D/$stem.onnx ] || { echo "export failed: $(tail -2 $D/$stem.export.log)"; echo "$label $n FAILED export" >> $OUT; continue; }
    shapes=$(python3 -c "
import json; m = json.load(open('$D/$stem.json'))['input_shapes']
print(','.join(k + ':' + 'x'.join(map(str, m[k])) for k in ('rx_slot_real', 'rx_slot_imag', 'h_hat_real', 'h_hat_imag')))")
    weights=$(python3 -c "import json; print(json.load(open('$D/$stem.json'))['weights'].split()[0])")
    shifter_slot trtexec --onnx=$E/$stem.onnx --saveEngine=$E/$stem.trt --skipInference --fp16 \
      --memPoolSize=workspace:4096 --shapes=$shapes > $D/$stem.build.log 2>&1
    [ -s $D/$stem.trt ] || { echo "build failed: $(tail -2 $D/$stem.build.log)"; echo "$label $n FAILED build" >> $OUT; continue; }
    shifter_slot trtexec --loadEngine=$E/$stem.trt --shapes=$shapes --warmUp=2000 --duration=10 > $D/$stem.time.log 2>&1
    line=$(grep "GPU Compute Time: min" $D/$stem.time.log | tail -1)
    echo "$line"
    mean=$(echo "$line" | sed -n 's/.*mean = \([0-9.]*\) ms.*/\1/p')
    median=$(echo "$line" | sed -n 's/.*median = \([0-9.]*\) ms.*/\1/p')
    p99=$(echo "$line" | sed -n 's/.*percentile(99%) = \([0-9.]*\) ms.*/\1/p')
    echo "$label $n $weights | $mean $median $p99" >> $OUT
  done
done
cat $OUT
echo V38_ENGINE_USERS_DONE
