#!/usr/bin/env bash
# N26 slots: the N23 slots (MCS 16, UMa, 3 km/h, Eb/No 6/8/10/12 dB, two users, 273 PRBs, 4 RX, 3.5 GHz, seeds
# 1600-1603) with two users of neighbor cells at an interference over thermal of INR dB, built into a runtime dataset of
# the nv_mu2_m16 profile.  The L1 then fails more TBs with the same slot size and receivers, so the real-time runs on
# these slots measure the on-time share of the recoveries at the recovery load of the timeline (1.5x the runs at MCS 14)
# and of link adaptation (1.8x), where the MCS 16 slots of N23 reach 1.3x (check of the load matching, Section 5.6).
# env: INR (dB, required), N16_DATA (default n16_uma_f35_m16_i<INR>), CARRIER_GHZ (default 3.5)
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
inr=${INR:?INR}; name=${N16_DATA:-n16_uma_f35_m16_i$inr}; d=$DATA/${name/n16_uma/n16_gen}; out=$DATA/$name; mkdir -p $d
[ -s $out/nv_mu2_m16_meta.json ] && { echo "dataset exists: $out"; exit 0; }
pids=()
for g in 0 1 2 3; do
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((1600 + g)) --mcs 16 --rvs 0 --channel UMa --carrier-ghz ${CARRIER_GHZ:-3.5} \
      --speed 0.83 --ebno 6,8,10,12 --slots 64 --batch 8 --interferers 2 --inr-db $inr --output $d/p$g > $d/gen_p$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "generation failed"; tail -5 $d/gen_p*.log; exit 1; }
GPU=0 rx python3 /rec/vendor/nv_build_dataset.py --input $d/p0_m16_t0.npz $d/p1_m16_t0.npz $d/p2_m16_t0.npz $d/p3_m16_t0.npz \
    --ebno all --profile nv_mu2_m16 --output $out > $d/build.log 2>&1 || { echo "build failed"; tail -8 $d/build.log; exit 1; }
rm -f $d/p*_m16_t0.npz
ls -la $out; tail -3 $d/build.log
