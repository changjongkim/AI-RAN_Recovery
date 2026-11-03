#!/usr/bin/env bash
# U3n: train the NRx for more users until END (exp/u3n_train.sh), keep the weights at the end as <label>_final, and,
# when the evaluation arguments are given, evaluate one trained NRx on one point of the user sweep U3b with the same
# seed, Eb/No, and slots, next to the conventional receiver of the same run (LS + LMMSE).  K-best and the
# true-channel bound come from U3b.  Before END every call trains (so an evaluation step that starts first trains
# instead of failing); after END the training part returns at once.  Every part can be cut by the end of an
# allocation and run again: training continues from its weights, and exp/classical.sh skips finished GPUs.
# usage: bash exp/u3n_side.sh END_TIME [NAME SEED0 CONFIG RX USERS EBNO SLOTS BATCH WEIGHTS]
set -uo pipefail
R=/pscratch/sd/s/sgkim/kcj/airan_recovery; cd $R || exit 1
source $R/env.sh
bash exp/u3n_train.sh "$1"; rc=$?
[ $rc -eq 0 ] || { echo "training ended with exit code $rc"; exit $rc; }
O=$RESULTS/u3n
for l in nrx_large_8rx_mu8 nrx_large_16rx_mu12 nrx_large_mu4; do
  [ -f $O/${l}_final ] || cp $NRX_REPO/weights/${l}_weights $O/${l}_final
  [ -f $R/models/${l}_u3n_weights ] || cp $O/${l}_final $R/models/${l}_u3n_weights
done
[ $# -ge 10 ] || { echo U3N_TRAINED; exit 0; }
shift
name=$1 seed=$2 cfg=$3 rx=$4 users=$5 ebno=$6 slots=$7 batch=$8 w=$9
[ -s $RESULTS/$name.json ] && { echo "$name done before"; exit 0; }
SEED0=$seed CARRIER_GHZ=3.5 bash exp/classical.sh $name --base-config $cfg --num-rx $rx --users $users --channel UMa \
    --systems baseline_lslin_lmmse,nrx --nrx-weights $O/$w --ebno=$ebno --slots $slots --batch $batch
