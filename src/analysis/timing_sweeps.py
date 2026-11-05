#!/usr/bin/env python3
"""E2, E3, E6 (docs/plan_setup_and_sensitivity.md): the replay of Sections 5.2-5.5 under other TDD patterns and grant
leads, and per Eb/No point.

E2  TDD pattern: DDDSU (the reference), DDDSUUDDDD (the structure of DDDDDDDSUU, the GSMA pattern next to LTE), and
    two uplink-heavier patterns, DDSUU and DSUUU.
E3  grant lead L from 1.5 ms (three slots ahead, as Aerial schedules) to 6 ms.
In every case the recovery deadline is D = 15.0 - L ms after the slot, as the reference 11.0 ms = 15.0 - 4.0 (the
retransmission slot 15.0 ms minus the lead), and the L1 result and the recovery path are the measured 1.4 ms and the
99th-percentile path of paper_numbers.  Policies: standard HARQ (harq_only), Full Wait (wait), Latency-Neutral
(select_cb5); the same macro traces at MCS 14 (paper_numbers.MACRO_SETS).
E6  per Eb/No point of the macro traces at the reference timing: first-transmission failure of the conventional
    receiver, retransmissions per 100 TBs of each policy, and the reduction.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 python3 src/analysis/timing_sweeps.py
output: results/e2e3e6_timing_sweeps.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as pn  # noqa: E402

T = pn.Timing
base = {k: getattr(T, k) for k in ("l1", "run", "deadline", "lead")}
conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
fail = ~conv


def policies(pattern, mask=None):
    m = np.ones(len(conv), bool) if mask is None else mask
    sel = lambda a: a[m]
    st = [None] + [s[m] for s in steps[1:]]
    cst = [None] + [s[m] for s in conv_steps[1:]]
    out, dones = {}, {}
    for pol in ("harq_only", "wait", "select_cb5"):
        done, tx = pn.simulate(sel(conv), sel(rec), cst if pol == "harq_only" else st, sel(cand), pol, pattern, T, l1_steps=cst)
        dones[pol] = done
        out[pol] = {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)), "lost_pct": float(100.0 * (~np.isfinite(done)).mean())}
    f = sel(fail); d0 = dones["harq_only"]
    for pol in ("wait", "select_cb5"):
        d = dones[pol]
        at = np.unique(np.concatenate([[0.0], d0[f & np.isfinite(d0)], d[f & np.isfinite(d)]]))
        out[pol]["share_min_diff_pp"] = float((pn.delivered_share(d, f, at) - pn.delivered_share(d0, f, at)).min())
        both = f & np.isfinite(d0) & np.isfinite(d)
        out[pol]["p99_failed_ms"] = float(np.percentile(d[both], 99))
        out[pol]["cut"] = out["harq_only"]["retx_per_100"] / out[pol]["retx_per_100"]
    out["harq_only"]["p99_failed_ms"] = float(np.percentile(d0[f & np.isfinite(d0)], 99))
    out["first_fail_pct"] = float(100.0 * f.mean())
    return out


res = {"base_timing": base, "e2_pattern": {}, "e3_lead": {}, "e6_point": {}}
for pattern in ("DDDSU", "DDDSUUDDDD", "DDSUU", "DSUUU"):
    for lead in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        T.lead, T.deadline = lead, 15.0 - lead
        r = policies(pattern)
        r["timing"] = {"lead": lead, "deadline": T.deadline}
        key = "e3_lead" if pattern == "DDDSU" else "e2_pattern"
        res[key][f"{pattern}/L{lead:g}"] = r
        if pattern == "DDDSU" or lead == 4.0:
            res["e2_pattern"][f"{pattern}/L{lead:g}"] = r
        print(f"{pattern:11s} L={lead:3.1f} D={T.deadline:4.1f}  conv {r['harq_only']['retx_per_100']:6.2f}  "
              f"wait {r['wait']['retx_per_100']:6.2f} ({r['wait']['cut']:.2f}x)  neutral {r['select_cb5']['retx_per_100']:6.2f} "
              f"({r['select_cb5']['cut']:.2f}x, min diff {r['select_cb5']['share_min_diff_pp']:+.2f}pp)  "
              f"p99 conv {r['harq_only']['p99_failed_ms']:.1f} wait {r['wait']['p99_failed_ms']:.1f} neutral {r['select_cb5']['p99_failed_ms']:.1f}",
              flush=True)
T.lead, T.deadline = base["lead"], base["deadline"]

# E6: Eb/No point of each TB (point is stored per slot; every slot holds the same number of users)
pts = []
for s in pn.MACRO_SETS:
    for f in sorted(glob.glob(f"results/{s}{pn.SUFFIX}/c*.npz")):
        d = np.load(f)
        users = d["conv_t0"].size // d["point"].size
        pts.append(np.repeat(d["point"].reshape(-1), users))
point = np.concatenate(pts)
assert len(point) == len(conv), (len(point), len(conv))
for p in np.unique(point):
    r = policies("DDDSU", point == p)
    res["e6_point"][f"{p:g}"] = r
    print(f"Eb/No {p:4g} dB  first fail {r['first_fail_pct']:5.1f}%  conv {r['harq_only']['retx_per_100']:6.2f}  "
          f"wait {r['wait']['retx_per_100']:6.2f} ({r['wait']['cut']:.2f}x)  neutral ({r['select_cb5']['cut']:.2f}x)", flush=True)
json.dump(res, open("results/e2e3e6_timing_sweeps.json", "w"), indent=1)
print("wrote results/e2e3e6_timing_sweeps.json")
