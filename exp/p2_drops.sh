#!/usr/bin/env bash
# P2 step 1: pair similarities of user drops in UMi and UMa (pair_drops.py), on two GPUs.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
tfgen python3 $REC/src/pair_drops.py --nrx-repo $NRX_REPO --channel UMi --gpu 0 --output $RESULTS/p2_drops_umi.npz > $RESULTS/p2_drops_umi.log 2>&1 &
a=$!
tfgen python3 $REC/src/pair_drops.py --nrx-repo $NRX_REPO --channel UMa --gpu 1 --output $RESULTS/p2_drops_uma.npz > $RESULTS/p2_drops_uma.log 2>&1 &
b=$!
wait $a; ra=$?; wait $b; rb=$?
grep -h "U=\|saved\|rror" $RESULTS/p2_drops_umi.log $RESULTS/p2_drops_uma.log
exit $((ra | rb))
