#!/usr/bin/env bash
# N6 data: every MCS of the ladder generated on the same channel realizations and noise (gen_harq.py --mcs LIST
# --snr-db ...), each MCS decoded with IR combining; then the outer-loop link adaptation (src/sim/olla.py).
# usage: bash exp/harq_mcs.sh NAME CHUNKS --channel UMi --speed 0.83 --snr-db 4,6,8,10,12,14 --slots 16 --batch 8
#   environment: MCS_LIST (default 10,11,12,13,14,15,16), GPUS (default "0 1 2 3"), as in harq_ir.sh: IRC=1 adds the
#                MMSE-IRC conventional receiver, NRX_ENGINE / SMALL_ENGINE replace the large / small neural receiver,
#                OLLA_OPTS extra options of olla.py (e.g. --sir-bins -5,0,5,10,15,25 for link adaptation per SIR group)
# results: results/<NAME>/m<mcs>_c<k>.npz/.json, results/<NAME>_olla.json and .txt
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
name=$1; chunks=$2; shift 2
gen_opts="$*"
list=${MCS_LIST:-10,11,12,13,14,15,16}
gpus=(${GPUS:-0 1 2 3})
out=$RESULTS/$name; mkdir -p $out $DATA/$name

chunk() {   # K GPU
  local k=$1 g=$2 d=$DATA/$name/c$1 t n m
  local last=${list##*,}
  [ -s $out/m${last}_c$k.npz ] && { echo "chunk $k done before"; return 0; }
  mkdir -p $d
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((2000 + k)) --mcs $list $gen_opts \
      --output $d/s > $d/gen.log 2>&1 || { echo "chunk $k: generation failed"; tail -5 $d/gen.log; return 1; }
  n=$(GPU=$g rx python3 -c "import json,numpy as np; print(len(json.loads(str(np.load('$d/s_info.npz')['meta']))['rvs']))")
  for m in ${list//,/ }; do
    for ((t = 0; t < n; t++)); do
      GPU=$g rx python3 /rec/vendor/nv_build_dataset.py --input $d/s_m${m}_t$t.npz --ebno all --profile nv_mu2_m$m \
          --output $d/rt_s_m${m}_t$t > $d/build_m${m}_t$t.log 2>&1 || { echo "chunk $k: build failed"; return 1; }
      rm -f $d/s_m${m}_t$t.npz
    done
    GPU=$g rx python3 /rec/decode_harq.py --root $d --set s --mcs $m --engine ${NRX_ENGINE:-$ENGINE_LARGE} \
        --small-engine ${SMALL_ENGINE:-$ENGINE_SMALL} ${IRC:+--irc} --output $out/m${m}_c$k.json > $d/decode_m$m.log 2>&1 || { echo "chunk $k MCS $m: decode failed"; tail -8 $d/decode_m$m.log; return 1; }
    rm -rf $d/rt_s_m${m}_t*
  done
  echo "chunk $k (GPU $g) $(date +%T)"
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
GPU=0 rx python3 /rec/sim/olla.py --root $RESULTS/$name --output $RESULTS/${name}_olla.json ${OLLA_OPTS:-} 2>&1 | tee $RESULTS/${name}_olla.txt
echo HARQ_MCS_DONE $name rc=$rc
exit $rc
