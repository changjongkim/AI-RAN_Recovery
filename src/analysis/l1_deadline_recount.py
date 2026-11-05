#!/usr/bin/env python3
"""E4 (docs/plan_setup_and_sensitivity.md): the share of L1 results later than a threshold of 3.0-5.0 ms after the
slot, recounted from the per-slot records of the real-time runs (conv<k>.json of each run's _work directory: release_ns
and done_ns of every slot of cell k, the warmup periods left out as the runtime does), for the four-GPU server with
Reverb (one recovery per GPU, rejection at the L1 result, deadline 11.0 ms) and without recoveries (L1 only), and for
two recoveries at a time per GPU (l2), the runs of Section 5.6.  Each slot holds two TBs, so the share of slots is the
share of TBs.
usage: python3 src/analysis/l1_deadline_recount.py        output: results/e4_l1_deadline_recount.json
"""

from __future__ import annotations

import collections
import glob
import json
import re

import numpy as np

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw/"
TAGS = {"reverb": "rec16allb68g4d110_f35arrivalpred60arrivalmargin015", "none": "rec16offb68g4d110_f35",
        "two": "rec16allb68g4d110_f35l2"}
THRESH = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

out = {}
for name, tag in TAGS.items():
    lat = collections.defaultdict(list)
    for w in sorted(glob.glob(RAW + tag + "s[0-9]_c*_*_work")):
        c = int(re.search(r"s\d_c(\d+)_", w).group(1))
        for f in glob.glob(w + "/conv*.json"):
            d = json.load(open(f)); cols = d["columns"]; r = np.array(d["records"], dtype=np.int64)
            if not len(r):
                continue
            keep = r[:, cols.index("period")] >= d.get("warmup", 0)
            ms = (r[keep, cols.index("done_ns")] - r[keep, cols.index("release_ns")]) / 1e6
            lat[c].append(ms)
    out[name] = {}
    for c in sorted(lat):
        ms = np.concatenate(lat[c])
        out[name][c] = {"slots": int(len(ms)), **{f"late_pct_{t:g}": float(100.0 * (ms > t).mean()) for t in THRESH},
                        "p99_ms": float(np.percentile(ms, 99)), "p999_ms": float(np.percentile(ms, 99.9))}
    if out[name]:
        print(name, tag)
        for c, v in out[name].items():
            print(f"  cells {c:3d} slots {v['slots']:7d}  " + "  ".join(f">{t:g}ms {v[f'late_pct_{t:g}']:.3f}%" for t in THRESH)
                  + f"  p99.9 {v['p999_ms']:.2f} ms")
    else:
        print(name, tag, "no runs found")
json.dump(out, open("results/e4_l1_deadline_recount.json", "w"), indent=1)
print("wrote results/e4_l1_deadline_recount.json")
