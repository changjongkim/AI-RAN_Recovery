#!/usr/bin/env python3
"""N29: recoveries on a server of partially loaded cells (exp/n29_runtime_load.sh, raw results rec16*a<load><b|u>).

For each cell load (share of the uplink periods in which a cell transmits; b: independent periods, u: busy and idle
runs) and server (one or four GPUs): the on-time share of the recoverable failed TBs at each number of cells
(server_effect.availability), the late L1 blocks, and the share that the full-load runs give at the same number of
transmitting cells (cells x load, interpolated; the check that a partially loaded server behaves like a fully loaded
server with fewer cells).  Then the effect on the server as in Section 5.6 (server_effect.effect with the measured shares
of that load): retransmission reduction and throughput under link adaptation at every number of cells.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 n29_load_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import re
import sys

sys.path[:0] = ["src", "src/sim", "src/analysis"]
import paper_numbers as pn  # noqa: E402
import server_effect as se  # noqa: E402

N16 = pn.N16


def late_l1(tag, cells):
    late = tbs = 0
    for f in glob.glob(f"{se.RAW}/rec16{tag}s*_c{cells}_*_j*.json"):
        a = json.load(open(f))["all"]
        late += a.get("late_tbs", 0)
        tbs += a["tbs"]
    return 100.0 * late / max(1, tbs)


def main():
    full = se.availability()
    loads = sorted({m.group(1) for f in glob.glob(f"{se.RAW}/rec16allb68*{N16}a*s*_c*_*_none_j*.json")
                    if (m := re.search(rf"{N16}(a\d+[bu])s\d", f))})
    out = {}
    for load in loads:
        act = int(load[1:-1]) / 100.0
        tags = {f"allb68{N16}{load}": "one", f"allb68g4{N16}{load}": "four"}
        avail = se.availability(tags)
        if not avail:
            continue
        rows = {}
        for (g, c), v in sorted(avail.items(), key=lambda x: (x[0][0], x[0][1])):
            tag = f"allb68g4{N16}{load}" if g == "four" else f"allb68{N16}{load}"
            rows[f"{g}_{c}"] = {"gpus": g, "cells": c, "share": v["availability"], "failed_per_tb": v["failed_per_tb"],
                                "full_load_share_same_transmitting_cells": se.matched(full, g, c, act),
                                "late_l1_pct": late_l1(tag, c)}
        base, eff, _ = se.effect(avail)
        for k, r in eff.items():
            if k in rows:
                rows[k].update({"retx_cut": base["retx_per_100"] / r["retx_per_100"],
                                "goodput_gain_la": r["goodput_gain_la"],
                                "share_load_matched": r["availability_fixed"]})
        out[load] = rows
        for k, r in rows.items():
            print(f"{load} {k:8s} share {100 * r['share']:5.1f}% (full load at {act:.2f}x cells: "
                  f"{100 * r['full_load_share_same_transmitting_cells']:5.1f}%), late L1 {r['late_l1_pct']:.3f}%"
                  + (f", retx {r['retx_cut']:.2f}x, goodput {r['goodput_gain_la']:.2f}x" if "retx_cut" in r else ""))
    path = sys.argv[1] if len(sys.argv) > 1 else f"results/n29_load_summary{pn.SUFFIX}.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
