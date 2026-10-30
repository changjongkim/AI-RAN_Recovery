#!/usr/bin/env python3
"""N30: rejection at the L1 result in the slot runtime (exp/n16_runtime.sh with CONTROLLER=src/runtime/controller_arrival.py
and NRX_FLAGS arrival_pred=6.0, raw results rec16allb68[g4]<N16>arrivalpred60*) against the runtime's rule (rejection at
the latest start, the s1_server and n16_all runs of the same settings).

Per server and number of cells: the on-time recoveries per failed slot of both rules, the rejections at the L1 result and
at the latest start per failed slot, the error of the predicted end (actual - predicted, ms) of the recoveries that ran,
and wait_split(): the on-time recoveries whose predicted end is at most the decode time of the first retransmission (the
slots that wait under that rule) with their end times, in the form of admission_replay.wait_split for delay_under_load,
and share_scale, the on-time recoveries of the rejection at the L1 result over those of the runtime's rule.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 n30_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path[:0] = ["src", "src/sim", "src/analysis"]
import paper_numbers as pn  # noqa: E402
import server_effect as se  # noqa: E402

ARRIVAL = os.environ.get("N30_TAG", "arrivalpred60")   # arrivalpred60arrivalmargin015: the rerun with a margin


def run_slots(f):
    """Per failed slot of one run: (rejected at L1, rejected later, ran, end ms after the slot, predicted end ms)."""
    d = json.load(open(f))
    cfg, work = d["config"], f[:-5] + "_work"
    skip, deadline = int(cfg["skip_periods"]), float(cfg["rescue_deadline_ms"])
    ctl = json.load(open(f"{work}/controller.json"))
    pred = ctl.get("predicted_end_ms", {})
    lanes = {}
    for lf in glob.glob(f"{work}/lane*.json"):
        lane = json.load(open(lf))
        col = {k: i for i, k in enumerate(lane["columns"])}
        for r in lane["records"]:
            lanes[(int(r[col["cell"]]), int(r[col["period"]]))] = int(r[col["done_ns"]])
    out = []
    for cf in glob.glob(f"{work}/conv*.json"):
        cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
        c = json.load(open(cf))
        col = {k: i for i, k in enumerate(c["columns"])}
        for r in c["records"]:
            k = int(r[col["period"]])
            if k < skip or not (~int(r[col["ue_crc_mask"]]) & 0b11):
                continue
            end = (lanes[(cell, k)] - int(r[col["release_ns"]])) / 1e6 if (cell, k) in lanes else np.nan
            p = pred.get(f"{cell}:{k}", np.nan)
            out.append((cell, k, end, p, deadline))
    return out, ctl["counters"]


def summarize(tag, cells):
    slots, counters = [], {}
    for f in sorted(glob.glob(f"{se.RAW}/rec16{tag}s*_c{cells}_*_none_j*.json")):
        s, c = run_slots(f)
        slots += s
        for k, v in c.items():
            counters[k] = counters.get(k, 0) + v
    if not slots:
        return None
    end = np.array([x[2] for x in slots]); pred = np.array([x[3] for x in slots]); dl = slots[0][4]
    ran = np.isfinite(end)
    on = ran & (end <= dl)
    admitted = np.isfinite(pred)
    return {"failed_slots": len(slots), "on_time_per_failed": float(on.mean()),
            "rejected_at_l1_per_failed": float((~admitted).mean()) if tag.endswith(ARRIVAL) else 0.0,
            "rejected_later_per_failed": float((admitted & ~ran).mean()) if tag.endswith(ARRIVAL) else float((~ran).mean()),
            "pred_error_ms": (np.percentile((end - pred)[ran & admitted], [10, 50, 90]).tolist()
                              if (ran & admitted).any() else None),
            "end_ms": end[on], "pred_ms": pred[on]}


def wait_split(gpus, cells, cut_ms):
    base = f"allb68{'g4' if gpus == 'four' else ''}{pn.N16}"
    s, b = summarize(base + ARRIVAL, cells), summarize(base, cells)
    if s is None or b is None:
        return None
    wait = s["pred_ms"] <= cut_ms
    return {"wait_share_of_on_time": float(wait.mean()) if len(wait) else 0.0, "wait_end_ms": s["end_ms"][wait],
            "late_per_failed": s["rejected_later_per_failed"], "on_time_per_failed": s["on_time_per_failed"],
            "share_scale": s["on_time_per_failed"] / b["on_time_per_failed"]}   # on-time runs against the runtime's rule


def main():
    cut_ms = 9.4
    out = {}
    for gpus, base in (("four", f"allb68g4{pn.N16}"), ("one", f"allb68{pn.N16}")):
        for c in ((4, 8, 12, 16, 20, 24) if gpus == "four" else (1, 2, 3, 4, 6, 8)):
            a, b = summarize(base + ARRIVAL, c), summarize(base, c)
            if a is None or b is None:
                continue
            w = a["pred_ms"] <= cut_ms
            row = {"gpus": gpus, "cells": c,
                   "latest_start": {k: v for k, v in b.items() if not k.endswith("_ms")},
                   "at_l1": {k: v for k, v in a.items() if not k.endswith("_ms") or k == "pred_error_ms"},
                   "wait_share_of_on_time": float(w.mean()) if len(w) else 0.0,
                   "wait_end_p50_ms": float(np.median(a["end_ms"][w])) if w.any() else None}
            out[f"{gpus}_{c}"] = row
            e = a["pred_error_ms"]
            print(f"{gpus:4s} {c:2d} cells: on time per failed slot {b['on_time_per_failed']:.3f} (latest start) -> "
                  f"{a['on_time_per_failed']:.3f} (L1 result); rejected at L1 {a['rejected_at_l1_per_failed']:.3f}, "
                  f"later {a['rejected_later_per_failed']:.3f}; end - predicted p10/p50/p90 "
                  + ("/".join(f"{x:+.2f}" for x in e) if e else "-")
                  + f" ms; wait (predicted by {cut_ms} ms) {row['wait_share_of_on_time']:.2f} of on-time")
    path = sys.argv[1] if len(sys.argv) > 1 else f"results/n30_summary{pn.SUFFIX}{'' if ARRIVAL == 'arrivalpred60' else '_' + ARRIVAL}.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
