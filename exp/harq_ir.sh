#!/usr/bin/env bash
# One condition of the HARQ experiments, in chunks: generate slots with their retransmissions (gen_harq.py), turn every
# transmission into a runtime dataset, decode with IR combining (decode_harq.py), delete the received samples, and at
# the end merge the chunks (harq_stats.py).  Chunks run on the four GPUs of the node in parallel.
# usage: bash exp/harq_ir.sh NAME CHUNKS [generator options]      e.g.
#        bash exp/harq_ir.sh n1_umi_v3 8 --channel UMi --speed 0.83 --ebno 6,8,10,12 --slots 64
#   environment: MCS (default 14; one MCS per call), GPUS (default "0 1 2 3"), KEEP=1 keeps the received samples
#                SEED0 (default 1000; chunk k uses seed SEED0 + k)
# results: results/<NAME>/c<k>.json/.npz per chunk, results/<NAME>.json and .txt merged
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
name=$1; chunks=$2; shift 2
gen_opts="$*"
mcs=${MCS:-14}
gpus=(${GPUS:-0 1 2 3})
out=$RESULTS/$name; mkdir -p $out $DATA/$name

chunk() {   # K GPU
  local k=$1 g=$2 d=$DATA/$name/c$1 t n
  [ -s $out/c$k.npz ] && { echo "chunk $k done before"; return 0; }
  mkdir -p $d
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((${SEED0:-1000} + k)) --mcs $mcs $gen_opts \
      --output $d/s > $d/gen.log 2>&1 || { echo "chunk $k: generation failed"; tail -5 $d/gen.log; return 1; }
  n=$(GPU=$g rx python3 -c "import json,numpy as np; print(len(json.loads(str(np.load('$d/s_info.npz')['meta']))['rvs']))")
  for ((t = 0; t < n; t++)); do
    GPU=$g rx python3 /rec/vendor/nv_build_dataset.py --input $d/s_m${mcs}_t$t.npz --ebno all --profile nv_mu2_m$mcs \
        --output $d/rt_s_m${mcs}_t$t > $d/build_t$t.log 2>&1 || { echo "chunk $k: build failed"; return 1; }
    [ "${KEEP:-0}" = 1 ] || rm -f $d/s_m${mcs}_t$t.npz
  done
  GPU=$g rx python3 /rec/decode_harq.py --root $d --set s --mcs $mcs --engine $ENGINE_LARGE --small-engine $ENGINE_SMALL \
      --output $out/c$k.json > $d/decode.log 2>&1 || { echo "chunk $k: decode failed"; tail -8 $d/decode.log; return 1; }
  [ "${KEEP:-0}" = 1 ] || rm -rf $d/rt_s_m${mcs}_t*
  echo "chunk $k (GPU $g) $(date +%T)"; tail -6 $d/decode.log
}

k=0; rc=0
while [ $k -lt $chunks ]; do
  pids=()
  for g in "${gpus[@]}"; do
    [ $k -lt $chunks ] || break
    chunk $k $g & pids+=($!); k=$((k + 1))
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
done
ls $out/c*.npz > /dev/null 2>&1 || { echo "no chunk finished"; exit 1; }
GPU=0 rx python3 /rec/harq_stats.py --inputs $out/c*.npz --output $RESULTS/$name.json | tee $RESULTS/$name.txt
echo HARQ_IR_DONE $name rc=$rc
exit $rc
