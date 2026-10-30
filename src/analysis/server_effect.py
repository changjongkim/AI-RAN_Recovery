#!/usr/bin/env python3
"""Server-level effect of Reverb: the trace-driven results with only the recoveries that finish on time on shared GPUs.

1. Availability, from the real-time runs (N16, S1; policy allb68[g4]<N16_SUFFIX>, every seed): the share of the
   recoverable failed TBs that a run recovers by the recovery deadline.  Recoverable TBs are estimated from the slots the
   run decoded: in those slots, the TBs that the L1 failed (conv<cell>.json, ue_crc_mask) and the NRx decoded
   (lane<gpu>.json, ue_good_mask) over the TBs the L1 failed; the rejected slots have the same channels, so the run's
   failed TBs times this share is the number of recoverable TBs.  Availability = on-time recoveries / recoverable TBs.
2. Effect: each recovery of the macro traces (Full Wait) is kept with probability equal to the availability (one random
   draw per TB and transmission, seed 7), the first recovery and the NRx decodes of the retransmissions alike, and the timeline simulation (src/sim/harq_latency.py, the timing of paper_numbers.py) and the
   outer-loop link adaptation of src/sim/olla.py (target 10%, SNR points of N6) give retransmissions per 100 TBs, lost
   TBs, the 99th-percentile delivery time of the delivered failed TBs that recovery does not decode, and goodput.
   The runtime slots and the traces differ, so the availability transfers as a rate, not slot by slot.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 server_effect.py [OUT.json]       (from the repository root)"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/sim")
sys.path.insert(0, "src/analysis")
import harq_stats  # noqa: E402
import paper_numbers as pn  # noqa: E402
from harq_latency import simulate  # noqa: E402
import olla  # noqa: E402

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"
SUFFIX, N16 = pn.SUFFIX, pn.N16


def popcount(x):
    return np.array([bin(int(v)).count("1") for v in np.atleast_1d(x)])


def availability():
    rows = {}
    for f in glob.glob(f"{RAW}/rec16*_c*_*_none_j*.json"):
        m = re.match(r"rec16(\w+?)s(\d+)_c(\d+)_\w+_none_j(\d+)\.json", os.path.basename(f))
        if not m or m.group(1) not in (f"allb68{N16}", f"allb68g4{N16}"):
            continue
        a = json.load(open(f))["all"]
        work = f[:-5] + "_work"
        conv = {}
        for cf in glob.glob(f"{work}/conv*.json"):
            cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
            c = json.load(open(cf))
            col = {k: i for i, k in enumerate(c["columns"])}
            for r in c["records"]:
                conv[(cell, int(r[col["period"]]))] = int(r[col["ue_crc_mask"]])
        failed_run = recovered_run = 0
        for lf in glob.glob(f"{work}/lane*.json"):
            l = json.load(open(lf))
            col = {k: i for i, k in enumerate(l["columns"])}
            for r in l["records"]:
                key = (int(r[col["cell"]]), int(r[col["period"]]))
                if key not in conv:
                    continue
                failed = ~conv[key] & 0b11
                failed_run += bin(failed).count("1")
                recovered_run += bin(failed & int(r[col["ue_good_mask"]])).count("1")
        key = ("four" if "g4" in m.group(1) else "one", int(m.group(3)))
        failed_all = a["tbs"] * (1 - a["conv_crc_pass"])
        rows.setdefault(key, []).append((failed_all, a.get("nrx_rescues_on_time", 0), failed_run, recovered_run))
    out = {}
    for key, v in sorted(rows.items()):
        failed_all, on_time, failed_run, recovered_run = (sum(x[i] for x in v) for i in range(4))
        share = recovered_run / max(1, failed_run)
        out[key] = {"gpus": key[0], "cells": key[1], "recoverable_share": share,
                    "availability": min(1.0, on_time / max(1e-9, failed_all * share))}
    return out


def effect(avail):
    conv, rec, steps, conv_steps, cand = pn.load_traces(pn.MACRO_SETS)
    rng = np.random.default_rng(7)
    draw = rng.random(len(conv))
    T = pn.Timing
    d0, tx0 = simulate(conv, rec, conv_steps, cand, "harq_only", "DDDSU", T)
    root = Path(f"results/n6_uma_v3{SUFFIX}")
    tab = olla.per_mcs(root)
    ladder = sorted(tab)
    # per-MCS plans of the ours structure, to thin its first-transmission recoveries
    plans_m = {}
    for mcs in ladder:
        fs = sorted(glob.glob(str(root / f"m{mcs}_c*.npz")), key=lambda f: int(re.search(r"_c(\d+)\.npz$", f).group(1)))
        good, point, corr, _, _, total, _ = harq_stats.load(fs)
        flat = {k: v.reshape(-1) for k, v in good.items()}
        pl = harq_stats.plans(flat, total)
        plans_m[mcs] = (pl["ours"], pl["ours_conv_harq"][2], total)
    points = sorted(set(tab[ladder[0]]["point"].tolist()))
    draw_la = np.random.default_rng(11).random(len(tab[ladder[0]]["point"]))

    n_steps = len(steps)
    draws = [np.random.default_rng(100 + t).random(len(conv)) for t in range(n_steps)]

    def run(a):
        keep = draw < a
        r = rec & keep
        st = [None] + [conv_steps[t] | (steps[t] & ~conv_steps[t] & (draws[t] < a)) for t in range(1, n_steps)]
        d, tx = simulate(conv, r, st, cand, "wait", "DDDSU", T)
        unrec = ~conv & ~(r & (T.l1 + T.run <= T.deadline)) & np.isfinite(d)
        lost = 100.0 * (~np.isfinite(d)).mean()
        for mcs in ladder:
            (f, rr, stp), st_conv, total_m = plans_m[mcs]
            r2 = rr & (draw_la < a)
            st2 = [None] + [st_conv[t] | (stp[t] & ~st_conv[t] & (np.random.default_rng(200 + t).random(len(f)) < a)) for t in range(1, total_m)]
            tab[mcs]["first"]["ours_thin"] = f | r2
            tab[mcs]["n_tx"]["ours_thin"] = harq_stats.transmissions(f, r2, st2, total_m)
        gains = []
        for q in points:
            sel = tab[ladder[0]]["point"] == q
            base = olla.simulate(tab, ladder, sel, "ai_and_ran", 0.1, 20000, 0.25, np.random.default_rng(5))
            ours = olla.simulate(tab, ladder, sel, "ours_thin", 0.1, 20000, 0.25, np.random.default_rng(5))
            gains.append(ours["goodput_bits_per_tx"] / base["goodput_bits_per_tx"])
        return {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)), "lost_pct": lost,
                "p99_unrecovered_ms": float(np.percentile(d[unrec], 99)), "goodput_gain_la": float(np.mean(gains))}

    base = {"retx_per_100": float(100.0 * (tx0 - 1).sum() / len(tx0)), "lost_pct": float(100.0 * (~np.isfinite(d0)).mean()),
            "p99_unrecovered_ms": float(np.percentile(d0[~conv & np.isfinite(d0)], 99))}
    rows = {"every_recovery": run(1.0)}
    for (g, c), v in avail.items():
        rows[f"{g}_{c}"] = {**v, **run(v["availability"])}
    return base, rows


if __name__ == "__main__":
    av = availability()
    base, rows = effect(av)
    out = sys.argv[1] if len(sys.argv) > 1 else f"results/server_effect{SUFFIX}.json"
    json.dump({"conventional": base, "rows": rows}, open(out, "w"), indent=1, sort_keys=True)
    print(f"conventional: retx {base['retx_per_100']:.1f}, lost {base['lost_pct']:.2f}%, p99 {base['p99_unrecovered_ms']:.1f} ms")
    print("| GPUs | cells | recoverable share | availability | retx /100 | cut | lost % | p99 unrecovered ms | goodput (LA) |")
    for k, r in rows.items():
        a = r.get("availability", 1.0)
        print(f"| {r.get('gpus', '-')} | {r.get('cells', '-')} | {r.get('recoverable_share', float('nan')):.2f} | {a:.2f} | "
              f"{r['retx_per_100']:.1f} | {base['retx_per_100'] / r['retx_per_100']:.2f}x | {r['lost_pct']:.2f} | "
              f"{r['p99_unrecovered_ms']:.1f} | {r['goodput_gain_la']:.2f}x |")
