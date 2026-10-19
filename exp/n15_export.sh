#!/usr/bin/env bash
# N15: export the published large receiver (control) and the interference-aware one to TensorRT engines in
# src/engines (seen as /rec/engines in the runtime container).  The control must decode like the engine of the runtime.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
D=$REC/src/engines; mkdir -p $D
for label in ${LABELS:-nrx_large nrx_large_ici}; do
  stem=${label}_273prb_2ue
  ( cd $NRX_REPO/scripts && tfgen python3 $REC/src/export_nrx.py --base-config $label.cfg --num-tx 2 --output-dir $D ) > $D/$stem.export.log 2>&1 \
    || { echo "export of $label failed"; tail -3 $D/$stem.export.log; exit 1; }
  shapes=$(python3 -c "
import json; m = json.load(open('$D/$stem.json'))['input_shapes']
print(','.join(k + ':' + 'x'.join(map(str, m[k])) for k in ('rx_slot_real', 'rx_slot_imag', 'h_hat_real', 'h_hat_imag')))")
  GPU=0 rx trtexec --onnx=/rec/engines/$stem.onnx --saveEngine=/rec/engines/$stem.trt --skipInference --fp16 \
      --memPoolSize=workspace:4096 --shapes=$shapes > $D/$stem.build.log 2>&1
  [ -s $D/$stem.trt ] || { echo "build of $label failed"; tail -3 $D/$stem.build.log; exit 1; }
  echo "$label: $(python3 -c "import json; print(json.load(open('$D/$stem.json'))['weights'])") weights, engine $(du -h $D/$stem.trt | cut -f1)"
done
