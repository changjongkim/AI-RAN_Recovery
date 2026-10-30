#!/usr/bin/env python3
"""Analyses for the review of 2026-10-30, from stored results only (no new runs).
  small_nrx       the real-time (small) NRx as the recovery receiver of the failed TBs, against the large NRx, with
                  every recovery completed (trace-driven; the small NRx outcomes are in every trace as small_c<t>)
  no_delay_window the time between the L1 result and D_0 = U_s - L, the last grant that keeps the first retransmission
                  slot, for grant leads L of 1-6 ms on DDDSU and DDDDDDDSUU: a recovery adds no delay only if its run
                  time is below this window (the measured recovery path is src/analysis/recovery_path.py)
  capacity        retransmissions against transmissions and goodput: transmissions per delivered TB at a fixed MCS
                  (traces) and the mean MCS and goodput under outer-loop link adaptation (N6)
  shared_gpu      retransmissions per 100 TBs when only the recoveries that finished on time in the real-time runs
                  count: the failed TBs per TB and the on-time recoveries per TB of each run (N16) times the
                  retransmissions that one failure costs and one recovery saves in the traces (a linear model; the
                  runtime and the traces use different slots)
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 review_analyses.py [OUT.json]     (from the repository root)"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/sim")
import harq_stats  # noqa: E402
from harq_latency import uplink_times  # noqa: E402

SUFFIX = os.environ.get("RESULT_SUFFIX", "")
N16 = os.environ.get("N16_SUFFIX", "")
RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"
MACRO = ["n3_uma_v3_cb", "n3_uma_v3_more"]
SMALL = ["n1_umi_v3_cb"]


def load(names):
    paths = [p for n in names for p in sorted(glob.glob(f"results/{n}{SUFFIX}/c*.npz"))]
    return harq_stats.load(paths)


def small_nrx():
    out = {}
    for cell, names in (("macro", MACRO), ("small", SMALL)):
        good, point, corr, _, _, total, _ = load(names)
        large = harq_stats.report(good, point, corr, total)["all"]
        g = dict(good)
        for t in range(total):
            g[f"nrx_c{t}"] = good[f"small_c{t}"]
        small = harq_stats.report(g, point, corr, total)["all"]
        conv = large["ai_and_ran"]["resends_per_100"]
        out[cell] = {"conventional_retx": conv,
                     "large_retx": large["ours"]["resends_per_100"], "small_retx": small["ours"]["resends_per_100"],
                     "large_recovered_pct": large["ours"]["recovered_pct_of_failures"],
                     "small_recovered_pct": small["ours"]["recovered_pct_of_failures"],
                     "large_cut": conv / large["ours"]["resends_per_100"], "small_cut": conv / small["ours"]["resends_per_100"]}
    return out


def no_delay_window(l1=1.4):
    out = {}
    for pattern in ("DDDSU", "DDDDDDDSUU"):
        times = uplink_times(pattern, 200.0)
        rows = {}
        for lead in np.arange(1.0, 6.01, 0.5):
            us = times[np.searchsorted(times, l1 + lead)]
            rows[f"{lead:.1f}"] = {"U_s_ms": float(us), "D0_ms": float(us - lead), "window_ms": float(us - lead - l1)}
        out[pattern] = rows
    return out


def capacity():
    a = json.load(open(f"results/n3_uma_v3_all{SUFFIX}.json"))["all"]
    fixed = {}
    for s in ("ai_and_ran", "ours"):
        tx = 1 + a[s]["resends_per_100"] / 100
        delivered = 1 - a[s]["residual_after_all_pct"] / 100
        fixed[s] = {"transmissions_per_tb": tx, "delivered": delivered, "delivered_per_transmission": delivered / tx}
    fixed["gain"] = fixed["ours"]["delivered_per_transmission"] / fixed["ai_and_ran"]["delivered_per_transmission"]
    olla = json.load(open(f"results/n6_uma_v3{SUFFIX}_olla.json"))["results"]
    la = {}
    for k, r in sorted(olla.items()):
        if not k.endswith("_target0.1"):
            continue
        la[k.split("_")[0][3:]] = {"mcs_conventional": r["ai_and_ran"]["mean_mcs"], "mcs_reverb": r["ours"]["mean_mcs"],
                                   "retx_conventional": r["ai_and_ran"]["resends_per_100"], "retx_reverb": r["ours"]["resends_per_100"],
                                   "goodput_gain": r["ours"]["goodput_vs_ai_and_ran"]}
    return {"fixed_mcs": fixed, "link_adaptation_by_snr_db": la}


def shared_gpu():
    a = json.load(open(f"results/n3_uma_v3_all{SUFFIX}.json"))["all"]
    conv, ours = a["ai_and_ran"], a["ours"]
    fails = conv["first_fail_pct"]
    recovered = ours["done_at_bs_pct"]                       # recovered TBs per 100 TBs
    per_failure = conv["resends_per_100"] / fails             # retransmissions one failed TB costs
    per_recovery = (conv["resends_per_100"] - ours["resends_per_100"]) / recovered   # retransmissions one recovery saves
    runs = {}
    for f in glob.glob(f"{RAW}/rec16*_c*_*_none_j*.json"):
        m = re.match(r"rec16(\w+?)s(\d+)_c(\d+)_\w+_none_j\d+\.json", os.path.basename(f))
        if not m or m.group(1) not in (f"allb68{N16}", f"allb68g4{N16}"):
            continue
        r = json.load(open(f))["all"]
        key = ("four" if "g4" in m.group(1) else "one", int(m.group(3)))
        runs.setdefault(key, []).append((r["tbs"], r["tbs"] * (1 - r["conv_crc_pass"]), r.get("nrx_rescues_on_time", 0)))
    out = {"retx_per_failure": per_failure, "retx_saved_per_recovery": per_recovery, "rows": {}}
    for (gpus, cells), v in sorted(runs.items()):
        tbs, failed, rescued = (sum(x[i] for x in v) for i in range(3))
        c = 100 * failed / tbs * per_failure
        r = c - 100 * rescued / tbs * per_recovery
        out["rows"][f"{gpus}_{cells}"] = {"gpus": gpus, "cells": cells, "conventional_retx": c, "reverb_retx": r,
                                         "cut": c / r, "recovered_pct_of_failed": 100 * rescued / failed}
    return out


if __name__ == "__main__":
    res = {"small_nrx": small_nrx(), "no_delay_window": no_delay_window(), "capacity": capacity(), "shared_gpu": shared_gpu()}
    out = sys.argv[1] if len(sys.argv) > 1 else f"results/review_analyses{SUFFIX}.json"
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    s = res["small_nrx"]
    for cell in s:
        print(f"{cell}: conventional {s[cell]['conventional_retx']:.1f} | large NRx {s[cell]['large_retx']:.1f} "
              f"({s[cell]['large_cut']:.2f}x, recovers {s[cell]['large_recovered_pct']:.1f}%) | small NRx {s[cell]['small_retx']:.1f} "
              f"({s[cell]['small_cut']:.2f}x, recovers {s[cell]['small_recovered_pct']:.1f}%)")
    for p, rows in res["no_delay_window"].items():
        print(p, " ".join(f"L={k}:{v['window_ms']:.1f}" for k, v in rows.items()))
    c = res["capacity"]
    print("fixed MCS: delivered per transmission gain %.3fx (transmissions per TB %.3f -> %.3f)" % (
        c["fixed_mcs"]["gain"], c["fixed_mcs"]["ai_and_ran"]["transmissions_per_tb"], c["fixed_mcs"]["ours"]["transmissions_per_tb"]))
    for snr, r in c["link_adaptation_by_snr_db"].items():
        print(f"  LA {snr} dB: MCS {r['mcs_conventional']:.1f} -> {r['mcs_reverb']:.1f}, retx {r['retx_conventional']:.1f} -> "
              f"{r['retx_reverb']:.1f}, goodput {r['goodput_gain']:.2f}x")
    g = res["shared_gpu"]
    print("shared GPU: retx per failure %.2f, saved per recovery %.2f" % (g["retx_per_failure"], g["retx_saved_per_recovery"]))
    for k, r in g["rows"].items():
        print(f"  {r['gpus']} GPU(s), {r['cells']} cells: retx {r['conventional_retx']:.1f} -> {r['reverb_retx']:.1f} "
              f"({r['cut']:.2f}x), recovered on time {r['recovered_pct_of_failed']:.1f}%")
