#!/usr/bin/env python3
"""Time of the recovery path, measured in the slot runtime (N16): from the start of a recovery on its GPU to its result,
which covers the copy of the received samples, the large NRx, rate recovery, LDPC decoding and the CRC check.  The
runs with one cell on one GPU (policy allb68<N16_SUFFIX>, every seed) have no other recovery on the GPU, only the L1
of the cell, which is the condition the trace-driven timeline assumes.  The timeline uses the 99th percentile,
rounded up to 0.1 ms.  Read by src/analysis/paper_numbers.py and figures/paper/make_paper_figures.py.
usage: recovery_path.py            (prints the statistics; N16_SUFFIX selects the runs, as in paper_numbers.py)"""
import glob
import json
import math
import os

import numpy as np

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"


def path_ms(n16=None, cells=1, gpus=1):
    """Run times (ms) of every recovery of the runs allb68[g4]<n16>s<seed>_c<cells>."""
    n16 = os.environ.get("N16_SUFFIX", "") if n16 is None else n16
    tag = f"rec16allb68{'g4' if gpus == 4 else ''}{n16}s*_c{cells}_rescue_none_j*_work"
    times = []
    for d in sorted(glob.glob(os.path.join(RAW, tag))):
        for f in sorted(glob.glob(os.path.join(d, "lane*.json"))):
            r = np.array(json.load(open(f))["records"], dtype=float)
            if len(r):
                times.append((r[:, 3] - r[:, 2]) / 1e6)
    if not times:
        raise FileNotFoundError(f"no runtime records for {tag}")
    return np.concatenate(times)


def stats(n16=None):
    t = path_ms(n16)
    p99 = float(np.percentile(t, 99))
    return {"runs": int(t.size), "p50_ms": float(np.median(t)), "p99_ms": p99,
            "timeline_run_ms": math.ceil(p99 * 10 - 1e-9) / 10}


if __name__ == "__main__":
    print(json.dumps(stats(), indent=1))
