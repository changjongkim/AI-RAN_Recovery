#!/usr/bin/env bash
# N15 control: the published large receiver exported by src/export_nrx.py must decode the slots of n1_umi_v3 chunk 2
# (same seed, generated on the GPU) like the runtime engine did.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
rm -rf $RESULTS/n15_control
SEED0=1002 GPUS=0 NRX_ENGINE=/rec/engines/nrx_large_273prb_2ue.trt bash $REC/exp/harq_ir.sh n15_control 1 --channel UMi \
    --speed 0.83 --ebno 6,8,10,12 --slots 64 --batch 8 > /dev/null || exit 1
/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3 - $RESULTS/n15_control/c0.npz $RESULTS/n1_umi_v3/c2.npz <<'PY'
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
same = {k: float((a[k] == b[k]).mean()) for k in ("conv_t0", "nrx_c0", "nrx_c1", "nrx_c3", "sep_c3")}
print("share of blocks with the same result as the runtime engine:", same)
print("N15_CONTROL", "pass" if min(same.values()) >= 0.99 else "FAIL")
sys.exit(0 if min(same.values()) >= 0.99 else 1)
PY
