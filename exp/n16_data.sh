#!/usr/bin/env bash
# N16 slots: 1,024 first transmissions on UMa (3 km/h, Eb/No 6/8/10/12 dB, two users, MCS 14, 273 PRBs, 4 RX), generated
# on the four GPUs (seeds 1600-1603) and built into one runtime dataset of the nv_mu2 profile, the format the slot
# runtime of airan_cloudlab replays (data/n16_uma/nv_mu2_{rx,tb}.npy, nv_mu2_meta.json).
# env: N16_DATA (default n16_uma) names the dataset, CARRIER_GHZ (e.g. 3.5) the carrier of gen_harq.py.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
name=${N16_DATA:-n16_uma}; d=$DATA/${name/n16_uma/n16_gen}; out=$DATA/$name; mkdir -p $d
[ -s $out/nv_mu2_meta.json ] && { echo "dataset exists: $out"; exit 0; }
pids=()
for g in 0 1 2 3; do
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((1600 + g)) --mcs 14 --rvs 0 --channel UMa ${CARRIER_GHZ:+--carrier-ghz $CARRIER_GHZ} \
      --speed 0.83 --ebno 6,8,10,12 --slots 64 --batch 8 --output $d/p$g > $d/gen_p$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "generation failed"; tail -5 $d/gen_p*.log; exit 1; }
GPU=0 rx python3 /rec/vendor/nv_build_dataset.py --input $d/p0_m14_t0.npz $d/p1_m14_t0.npz $d/p2_m14_t0.npz $d/p3_m14_t0.npz \
    --ebno all --profile nv_mu2 --output $out > $d/build.log 2>&1 || { echo "build failed"; tail -8 $d/build.log; exit 1; }
ls -la $out; tail -3 $d/build.log
