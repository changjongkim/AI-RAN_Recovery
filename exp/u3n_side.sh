#!/usr/bin/env bash
# U3n on the second node of the queue (queue_b/side_todo): train the NRx for more users until END (exp/u3n_train.sh),
# then evaluate each trained NRx on the points of the user sweep U3b with the same seeds, Eb/No, and slots, next to
# the conventional receiver of the same run (LS + LMMSE).  K-best and the true-channel bound come from U3b.
# Every part can be cut by the end of an allocation and run again: training continues from its weights, and
# exp/classical.sh skips the GPUs that already wrote their results.
# usage: bash exp/u3n_side.sh END_TIME
set -uo pipefail
R=/pscratch/sd/s/sgkim/kcj/airan_recovery; cd $R || exit 1
source $R/env.sh
bash exp/u3n_train.sh "$1"; rc=$?
[ $rc -eq 0 ] || { echo "training ended with exit code $rc"; exit $rc; }
O=$RESULTS/u3n
for l in nrx_large_8rx_mu8 nrx_large_16rx_mu12 nrx_large_mu4; do
  [ -f $O/${l}_final ] || cp $NRX_REPO/weights/${l}_weights $O/${l}_final
done
cp -n $O/nrx_large_8rx_mu8_final $R/models/nrx_large_8rx_mu8_u3n_weights
cp -n $O/nrx_large_16rx_mu12_final $R/models/nrx_large_16rx_mu12_u3n_weights
cp -n $O/nrx_large_mu4_final $R/models/nrx_large_mu4_u3n_weights
S=baseline_lslin_lmmse,nrx
ev() {   # name seed0 cfg rx users ebno slots batch weights
  [ -s $RESULTS/$1.json ] && return 0
  SEED0=$2 CARRIER_GHZ=3.5 bash exp/classical.sh $1 --base-config $3 --num-rx $4 --users $5 --channel UMa --systems $S \
      --nrx-weights $O/$9 --ebno=$6 --slots $7 --batch $8 || echo "evaluation $1 failed"
}
#  name                    seed cfg                      rx users Eb/No              slots batch weights
ev u3n_uma_r4_u2_t_f35     300 nrx_large_mu4.cfg          4  2  6,8,10,12           48 4 nrx_large_mu4_final
ev u3n_uma_r4_u3_t_f35     300 nrx_large_mu4.cfg          4  3  8,10,12,14,16       48 2 nrx_large_mu4_final
ev u3n_uma_r4_u4_t_f35     300 nrx_large_mu4.cfg          4  4  10,12,14,16,18      48 2 nrx_large_mu4_final
ev u3n_uma_r8_u2_f35       300 nrx_large_8rx_mu8.cfg      8  2  -2,0,2,4,6          48 2 nrx_large_8rx_mu8_final
ev u3n_uma_r8_u4_f35       300 nrx_large_8rx_mu8.cfg      8  4  2,4,6,8,10          48 1 nrx_large_8rx_mu8_final
ev u3n_uma_r8_u6_f35       300 nrx_large_8rx_mu8.cfg      8  6  4,6,8,10,12         32 1 nrx_large_8rx_mu8_final
ev u3n_uma_r8_u8a_f35      300 nrx_large_8rx_mu8.cfg      8  8  6,8,10              32 1 nrx_large_8rx_mu8_final
ev u3n_uma_r8_u8b_f35      400 nrx_large_8rx_mu8.cfg      8  8  12,14               32 1 nrx_large_8rx_mu8_final
ev u3n_uma_r16_u2_f35      300 nrx_large_16rx_mu12.cfg   16  2  -6,-4,-2,0,2        32 1 nrx_large_16rx_mu12_final
ev u3n_uma_r16_u4_f35      300 nrx_large_16rx_mu12.cfg   16  4  -2,0,2,4,6          32 1 nrx_large_16rx_mu12_final
ev u3n_uma_r16_u8a_f35     300 nrx_large_16rx_mu12.cfg   16  8  0,2,4               32 1 nrx_large_16rx_mu12_final
ev u3n_uma_r16_u8b_f35     400 nrx_large_16rx_mu12.cfg   16  8  6,8                 32 1 nrx_large_16rx_mu12_final
ev u3n_uma_r16_u12a_f35    300 nrx_large_16rx_mu12.cfg   16 12  2,4,6               32 1 nrx_large_16rx_mu12_final
ev u3n_uma_r16_u12b_f35    400 nrx_large_16rx_mu12.cfg   16 12  8,10                32 1 nrx_large_16rx_mu12_final
echo U3N_SIDE_DONE
