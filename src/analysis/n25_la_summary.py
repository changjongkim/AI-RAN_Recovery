#!/usr/bin/env python3
"""N25: closed-loop link adaptation in the real-time runs (exp/n25_runtime_la.sh, raw results rec25* of the slot runtime).

Per run (one SNR point, one number of cells), after the first WARM uplink periods in which the outer loop settles, over
every cell: the mean MCS, the TBs the L1 decodes, the TBs a recovery decodes by the recovery deadline (lane records,
done - release <= deadline), the TBs that need a retransmission (neither), and the goodput in bits per user slot,
sum of TB bits / (TBs + retransmissions / RETX_OK) as in the Antiphase closed-loop analysis (a retransmission takes a
slot of the user and succeeds with probability RETX_OK).  Reverb (all) against the L1 alone (off) of the same SNR,
cells, GPUs and seed gives the throughput gain; the mean over the SNR points is the number of the trace-driven link
adaptation (src/sim/olla.py), with every recovery finishing, and of the replay of Section 5.6.
usage: n25_la_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import os
import re
import sys

import numpy as np

RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw"
PAT = re.compile(r"rec25(all|off)g(\d+)n(\d+)s(\d+)_c(\d+)_\w+_none_j\d+\.json")
WARM = 800
RETX_OK = 0.95


def run_stats(path):
    d = json.load(open(path))
    cfg, work = d["config"], path[:-5] + "_work"
    deadline_ns = float(cfg["rescue_deadline_ms"]) * 1e6
    mcs_of = cfg["la"]["mcs"]
    lanes = {}
    for lf in glob.glob(f"{work}/lane*.json"):
        lane = json.load(open(lf))
        col = {k: i for i, k in enumerate(lane["columns"])}
        for r in lane["records"]:
            lanes[(int(r[col["cell"]]), int(r[col["period"]]))] = (int(r[col["done_ns"]]), int(r[col["ue_good_mask"]]))
    tbs = bits = l1_ok = recovered = retx = mcs_sum = slots = 0
    for cf in glob.glob(f"{work}/conv*.json"):
        cell = int(re.search(r"conv(\d+)\.json$", cf).group(1))
        c = json.load(open(cf))
        col = {k: i for i, k in enumerate(c["columns"])}
        size = [8 * b for b in c["la"]["tb_bytes"]]
        for r in c["records"]:
            period = int(r[col["period"]])
            if period < WARM:
                continue
            level, release, good = int(r[col["mcs_level"]]), int(r[col["release_ns"]]), int(r[col["ue_good_mask"]])
            saved = 0
            ran = lanes.get((cell, period))
            if ran is not None and ran[0] - release <= deadline_ns:
                saved = ran[1] & ~good & 0b11
            final = good | saved
            tbs += 2
            bits += 2 * size[level]
            l1_ok += bin(good & 0b11).count("1")
            recovered += bin(saved).count("1")
            retx += 2 - bin(final & 0b11).count("1")
            mcs_sum += mcs_of[level]
            slots += 1
    return {"tbs": tbs, "mean_mcs": mcs_sum / max(1, slots), "first_error_pct": 100.0 * (tbs - l1_ok) / max(1, tbs),
            "recovered_pct_of_failed": 100.0 * recovered / max(1, tbs - l1_ok), "retx_pct": 100.0 * retx / max(1, tbs),
            "goodput_bits": bits / max(1.0, tbs + retx / RETX_OK)}


def main():
    runs = {}
    for f in glob.glob(f"{RAW}/rec25*_c*_*_none_j*.json"):
        m = PAT.match(os.path.basename(f))
        if m:
            pol, g, snr, seed, cells = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            runs[(pol, g, snr, seed, cells)] = run_stats(f)
    rows = {}
    for (pol, g, snr, seed, cells), r in sorted(runs.items()):
        if pol != "all" or ("off", g, snr, seed, cells) not in runs:
            continue
        o = runs[("off", g, snr, seed, cells)]
        rows[f"g{g}_c{cells}_snr{snr}_s{seed}"] = {"gpus": g, "cells": cells, "snr": snr, "seed": seed,
                                                   "gain": r["goodput_bits"] / o["goodput_bits"], "reverb": r, "off": o}
    summary = {}
    for g in sorted({v["gpus"] for v in rows.values()}):
        for c in sorted({v["cells"] for v in rows.values() if v["gpus"] == g}):
            sel = [v for v in rows.values() if v["gpus"] == g and v["cells"] == c]
            snrs = sorted({v["snr"] for v in sel})
            by_snr = {s: float(np.mean([v["gain"] for v in sel if v["snr"] == s])) for s in snrs}
            summary[f"g{g}_c{c}"] = {"gpus": g, "cells": c, "snrs": snrs, "gain_by_snr": by_snr,
                                     "gain_mean": float(np.mean(list(by_snr.values()))),
                                     "recovered_pct_of_failed": float(np.mean([v["reverb"]["recovered_pct_of_failed"] for v in sel])),
                                     "mcs_off": float(np.mean([v["off"]["mean_mcs"] for v in sel])),
                                     "mcs_reverb": float(np.mean([v["reverb"]["mean_mcs"] for v in sel]))}
    out = sys.argv[1] if len(sys.argv) > 1 else "results/n25_la_summary.json"
    json.dump({"rows": rows, "summary": summary}, open(out, "w"), indent=1, sort_keys=True)
    for k, v in summary.items():
        print(f"{k:8s} SNR {v['snrs']} gain {v['gain_mean']:.3f}x  by SNR "
              + " ".join(f"{s}:{g:.2f}" for s, g in v["gain_by_snr"].items())
              + f"  recovered {v['recovered_pct_of_failed']:.1f}% of failed, MCS {v['mcs_off']:.1f} -> {v['mcs_reverb']:.1f}")


if __name__ == "__main__":
    main()
