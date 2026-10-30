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


def path_ms(n16=None, cells=1, gpus=1, pol="allb68"):
    """Run times (ms) of every recovery of the runs <pol>[g4]<n16>s<seed>_c<cells>."""
    n16 = os.environ.get("N16_SUFFIX", "") if n16 is None else n16
    tag = f"rec16{pol}{'g4' if gpus == 4 else ''}{n16}s*_c{cells}_rescue_none_j*_work"
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


def bound_ms(pol, n16=None, margin_ms=0.9):
    """Latest-start bound of a receiver: the 99th percentile of its path with one cell on one GPU plus the margin of the
    large NRx (6.8 ms against a 99th percentile of 5.9 ms), rounded up to 0.1 ms."""
    p99 = float(np.percentile(path_ms(n16, pol=pol), 99))
    return math.ceil((p99 + margin_ms) * 10 - 1e-9) / 10


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--bound":     # recovery_path.py --bound allsmb68 [N16 suffix]
        print(f"{bound_ms(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None):.1f}")
    else:
        print(json.dumps(stats(), indent=1))
