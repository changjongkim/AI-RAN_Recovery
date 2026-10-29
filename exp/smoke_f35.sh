#!/usr/bin/env bash
# Smoke test of the 3.5 GHz runs (2026-10-29): one small chunk of the HARQ pipeline and one of the classical receivers
# with CARRIER_GHZ=3.5, then a check that both record a carrier of 3.5 GHz.  The runs of the paper move from the 2.14 GHz
# of the NVlabs configuration to 3.5 GHz (the band of the TDD patterns in the paper); their steps are named *_f35.
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
export CARRIER_GHZ=3.5
KEEP=1 GPUS="0" bash $REC/exp/harq_ir.sh smoke_f35 1 --channel UMa --speed 0.83 --ebno 8,12 --slots 8 --batch 4 || exit 1
GPUS="0" bash $REC/exp/classical.sh smoke_cls_f35 --channel UMa --speed 0.83 --ebno 8 --slots 4 --batch 2 \
    --systems baseline_lslin_lmmse,baseline_lmmse_kbest || exit 1
GPU=0 rx python3 - <<'PY'
import json, sys
import numpy as np
res = "/pscratch/sd/s/sgkim/kcj/airan_recovery/results"
meta = json.loads(str(np.load(f"{res}/smoke_f35/c0.npz")["meta"]))
harq = meta["generator"].get("carrier_ghz")
cls = json.load(open(f"{res}/smoke_cls_f35/g0.json")).get("carrier_ghz")
d = json.load(open(f"{res}/smoke_f35.json"))["decoded"]
print("carrier (HARQ, classical):", harq, cls, " decoded:", {k: d[k] for k in sorted(d)})
ok = abs(harq - 3.5) < 1e-6 and abs(cls - 3.5) < 1e-6
print("SMOKE_F35_OK" if ok else "SMOKE_F35_FAIL")
sys.exit(0 if ok else 1)
PY
