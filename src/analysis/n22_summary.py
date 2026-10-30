#!/usr/bin/env python3
"""N22: the real-time runs with AI inference on the same GPUs (exp/n22_runtime_ai.sh, raw results rec22* of the slot
runtime).  Per (policy, GPUs, AI rate, cells), over the runs: the failed TBs recovered before the deadline, the TBs the
L1 decodes after the L1 deadline, and the AI requests (within the SLO, rejected by admission, the 99th percentile of the
time to first token).  The 60 s runs (tag p24000) are used when every configuration has one, else the 6 s runs.
usage: n22_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import os
import re
import sys

import numpy as np

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"
PAT = re.compile(r"rec22(all|off)b68(g4)?d110_f35r(\d+)m60(p\d+)?s(\d)_c(\d+)_\w+_static_j(\d+)\.json")


def rows(long):
    out = {}
    for f in glob.glob(f"{RAW}/rec22*_static_j*.json"):
        m = PAT.match(os.path.basename(f))
        if not m or bool(m.group(4)) != long:
            continue
        d = json.load(open(f))
        a, ai = d["all"], d.get("ai", {})
        failed = a["tbs"] * (1 - a["conv_crc_pass"])
        key = (m.group(1), 4 if m.group(2) else 1, int(m.group(3)), int(m.group(6)))
        r = out.setdefault(key, {"failed": 0.0, "on_time": 0, "tbs": 0, "late": 0, "arrived": 0, "within": 0,
                                 "rejected": 0, "ttft_p99": []})
        r["failed"] += failed
        r["on_time"] += a.get("nrx_rescues_on_time", 0)
        r["tbs"] += a["tbs"]
        r["late"] += a.get("late_tbs", 0)
        for w in ai.values():
            r["arrived"] += w["arrived"]
            r["within"] += w["within_slo"]
            r["rejected"] += w["rejected"]
            if w.get("ttft_ms", {}).get("n"):
                r["ttft_p99"].append(w["ttft_ms"]["p99"])
    return out


def main():
    long, short = rows(True), rows(False)
    use = long if long and set(long) >= set(short) else short
    res = {}
    for (pol, g, rate, c), r in sorted(use.items()):
        res[f"{pol}_g{g}_r{rate}_c{c}"] = {
            "policy": pol, "gpus": g, "rate_per_gpu": rate, "cells": c,
            "recovered_pct": 100.0 * r["on_time"] / max(1.0, r["failed"]),
            "late_l1_pct": 100.0 * r["late"] / max(1, r["tbs"]),
            "ai_within_slo_pct": 100.0 * r["within"] / max(1, r["arrived"]),
            "ai_rejected_pct": 100.0 * r["rejected"] / max(1, r["arrived"]),
            "ai_ttft_p99_ms": float(np.max(r["ttft_p99"])) if r["ttft_p99"] else None,
            "ai_requests": r["arrived"]}
    out = sys.argv[1] if len(sys.argv) > 1 else "results/n22_summary.json"
    json.dump({"runs": "60 s" if use is long else "6 s", "rows": res}, open(out, "w"), indent=1, sort_keys=True)
    print("runs:", "60 s" if use is long else "6 s")
    for k, v in res.items():
        print(f"{k:18s} recovered {v['recovered_pct']:5.1f}%  late L1 {v['late_l1_pct']:.3f}%  AI within SLO {v['ai_within_slo_pct']:5.1f}%"
              f"  rejected {v['ai_rejected_pct']:4.1f}%  TTFT p99 {v['ai_ttft_p99_ms']}  requests {v['ai_requests']}")


if __name__ == "__main__":
    main()
