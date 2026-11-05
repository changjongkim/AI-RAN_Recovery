#!/usr/bin/env python3
"""E1a: distance between the radio units and the GPU server (fronthaul), in the replay of Sections 5.2-5.5.

One-way fronthaul delay t = 5 us per km of fiber plus a fixed switch delay (--switch-us, default 0).  The samples of a
slot reach the server t later, so the L1 result is at 1.4 + t ms after the slot.  The grant must reach the radio unit
before its air time, so the server sends it t earlier: the lead becomes 4.0 + t ms, and the first recovery must end by
11.0 - t ms after the slot (11.0 - 2t ms after the samples arrive).  The recovery path itself is unchanged, since the
samples are already in the GPU memory of the same server.  Everything else is paper_numbers.latency_policies: the
macro traces, DDDSU, the measured 99th-percentile recovery path, Full Wait (wait), Latency-Neutral (select_cb5), and
standard HARQ (harq_only).
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 python3 src/analysis/fronthaul_replay.py [--km 0,5,10,20,40,80]
output: results/e1_fronthaul_replay.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as pn  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--km", default="0,5,10,20,40,80")
parser.add_argument("--us-per-km", type=float, default=5.0)
parser.add_argument("--switch-us", type=float, default=0.0)
parser.add_argument("--output", default="results/e1_fronthaul_replay.json")
args = parser.parse_args()

base = {k: getattr(pn.Timing, k) for k in ("l1", "run", "deadline", "lead")}
rows = {}
for km in [float(x) for x in args.km.split(",")]:
    t = (args.us_per_km * km + args.switch_us) / 1000.0
    pn.Timing.l1, pn.Timing.lead, pn.Timing.deadline = base["l1"] + t, base["lead"] + t, base["deadline"] - t
    out = pn.latency_policies()
    keep = {p: out[p] for p in ("harq_only", "wait", "select_cb5")}
    keep.update({k: out[k] for k in ("parallel", "first_only", "first_retx_ms", "wait_delay_ms", "delivered_at_recovery_pct",
                                     "delivered_final_pct", "neutral_waits_pct_of_failed")})
    keep["one_way_ms"] = t
    keep["timing"] = {"l1": pn.Timing.l1, "run": pn.Timing.run, "lead": pn.Timing.lead, "deadline": pn.Timing.deadline}
    conv = keep["harq_only"]["retx_per_100"]
    keep["retx_cut_wait"] = conv / keep["wait"]["retx_per_100"]
    keep["retx_cut_neutral"] = conv / keep["select_cb5"]["retx_per_100"]
    rows[f"{km:g}"] = keep
    print(f"{km:5g} km  t={t*1000:5.0f} us  conventional {conv:6.2f}  wait {keep['wait']['retx_per_100']:6.2f} "
          f"({keep['retx_cut_wait']:.2f}x)  neutral {keep['select_cb5']['retx_per_100']:6.2f} ({keep['retx_cut_neutral']:.2f}x)  "
          f"first retx {keep['first_retx_ms']:.1f} ms  wait delay {keep['wait_delay_ms']} ms", flush=True)
pn.Timing.l1, pn.Timing.lead, pn.Timing.deadline = base["l1"], base["lead"], base["deadline"]
json.dump({"params": vars(args), "base_timing": base, "rows": rows}, open(args.output, "w"), indent=1)
print("wrote", args.output)
