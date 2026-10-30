#!/usr/bin/env bash
# N23 slots: the N16 slots at another MCS (default 16), 1,024 first transmissions on UMa (3 km/h, Eb/No 6/8/10/12 dB, two
# users, 273 PRBs, 4 RX, 3.5 GHz), seeds 1600-1603 as N16, built into a runtime dataset of the nv_mu2 profile.  At a
# higher MCS the L1 fails more TBs, so the real-time runs on these slots measure the on-time share of the recoveries at
# a higher recovery load (the load of link adaptation and of the recoveries of retransmissions, Section 5.6).
# env: MCS (default 16), N16_DATA (default n16_uma_f35_m16), CARRIER_GHZ (default 3.5)
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
mcs=${MCS:-16}; name=${N16_DATA:-n16_uma_f35_m16}; d=$DATA/${name/n16_uma/n16_gen}; out=$DATA/$name; mkdir -p $d
[ -s $out/nv_mu2_meta.json ] && { echo "dataset exists: $out"; exit 0; }
pids=()
for g in 0 1 2 3; do
  tfgen python3 $REC/src/gen_harq.py --nrx-repo $NRX_REPO --gpu $g --seed $((1600 + g)) --mcs $mcs --rvs 0 --channel UMa --carrier-ghz ${CARRIER_GHZ:-3.5} \
      --speed 0.83 --ebno 6,8,10,12 --slots 64 --batch 8 --output $d/p$g > $d/gen_p$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "generation failed"; tail -5 $d/gen_p*.log; exit 1; }
GPU=0 rx python3 /rec/vendor/nv_build_dataset.py --input $d/p0_m${mcs}_t0.npz $d/p1_m${mcs}_t0.npz $d/p2_m${mcs}_t0.npz $d/p3_m${mcs}_t0.npz \
    --ebno all --profile nv_mu2 --output $out > $d/build.log 2>&1 || { echo "build failed"; tail -8 $d/build.log; exit 1; }
ls -la $out; tail -3 $d/build.log
