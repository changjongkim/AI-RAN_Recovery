#!/usr/bin/env bash
# Check of --retx-alone (retransmissions of user 0 sent without user 1): one chunk with the seed and options of
# n1_umi_v3 chunk 2 (generated on the GPU in n1_umi_v3_fill).  The first transmission does not change, so user 0's
# first-transmission results must equal those of results/n1_umi_v3/c2.npz; the retransmissions are new.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
rm -rf $RESULTS/n1b_smoke
SEED0=1002 GPUS=0 bash $REC/exp/harq_ir.sh n1b_smoke 1 --channel UMi --speed 0.83 --ebno 6,8,10,12 --slots 64 --batch 8 \
    --retx-alone || exit 1
GPU=0 rx python3 - $RESULTS/n1b_smoke/c0.npz $RESULTS/n1_umi_v3/c2.npz <<'PY'
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print("users in the alone run:", a["conv_t0"].shape[1])
same = {k: bool(np.array_equal(a[k][:, 0], b[k][:, 0])) for k in ("conv_t0", "sep_c0", "nrx_c0", "small_c0")}
print("first transmission of user 0 equal to n1_umi_v3 c2:", same)
for k in ("sep_s1", "sep_s2", "nrx_s1", "sep_c1", "sep_c3", "nrx_c1", "nrx_c3"):
    print(f"{k}: alone {a[k][:, 0].mean():.3f}  same pair {b[k][:, 0].mean():.3f}")
sys.exit(0 if same["conv_t0"] and same["sep_c0"] else 1)
PY
