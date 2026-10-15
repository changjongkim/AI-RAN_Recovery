#!/usr/bin/env python3
"""N9: GPU time that recovery needs, from the measured failure rates and NeuralRx run times.

A slot needs a recovery run when any of its users' blocks failed on the conventional receiver (the NeuralRx decodes
the whole slot).  Per cell, the recovery work is   slots per second x P(slot needs recovery) x run time,   in GPU
seconds per second (1.0 = one GPU busy all the time).  The table is computed for uplink patterns (one uplink slot every
2.5 ms; two every 5 ms -- the same 400 uplink slots per second), several cells per server, and the run time of the
large model per number of users per slot (measured: README of airan_cloudlab, S3).
usage: demand.py --traces results/<name>/c*.npz --output OUT.json [--cells 4,8,16,32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harq_stats import load   # noqa: E402

UL_SLOTS_PER_S = 400.0                     # 20% of 2000 slots per second at 30 kHz
RUN_MS = {1: 3.23, 2: 6.3, 3: 6.68, 4: 8.64}   # large model; 2 users: full runtime path, others: engine (trtexec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--cells", default="4,8,16,32")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    good, point, corr, change, gain, total, meta = load(a.traces)
    fail = ~good["conv_t0"]
    slot_needs = fail.any(axis=1)
    out = {"traces": [str(p) for p in a.traces], "run_ms": RUN_MS, "by_point": {}}
    for q in sorted(set(point.tolist())) + ["all"]:
        sel = np.ones(len(point), bool) if q == "all" else point == q
        p_slot = float(slot_needs[sel].mean())
        row = {"p_slot_needs_recovery": p_slot, "p_block_fails": float(fail[sel].mean()), "cells": {}}
        for c in [int(x) for x in a.cells.split(",")]:
            gpu_s = c * UL_SLOTS_PER_S * p_slot * RUN_MS[2] / 1000.0
            row["cells"][c] = {"recovery_gpu_seconds_per_second": gpu_s, "neuralrx_lanes_busy_pct_per_gpu_of_4": 100.0 * gpu_s / 4}
        out["by_point"][str(q)] = row
    a.output.write_text(json.dumps(out, indent=2))
    print("| Eb/No | P(block fails) | P(slot needs recovery) | " + " | ".join(f"{c} cells: GPU-s/s" for c in a.cells.split(",")) + " |")
    print("|---|---|---|" + "---|" * len(a.cells.split(",")))
    for q, r in out["by_point"].items():
        print(f"| {q} | {100 * r['p_block_fails']:.1f}% | {100 * r['p_slot_needs_recovery']:.1f}% | " +
              " | ".join(f"{v['recovery_gpu_seconds_per_second']:.2f}" for v in r["cells"].values()) + " |")


if __name__ == "__main__":
    main()
