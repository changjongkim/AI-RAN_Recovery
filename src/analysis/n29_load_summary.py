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


END_PERIODS = 100   # the end of a 48-cell run stalls the L1 of every cell in its last 12-100 periods (exp n29_end_*:
                    # at 4800 periods the stall moved to periods 4700-4799 and none came near period 2400)


def late_l1(tag, cells, steady=True):
    """Slots whose L1 result comes later than pn.L1_DEADLINE_MS after the slot, in percent.  steady: over the periods
    after the first skip_periods and before the last END_PERIODS; else the late TBs that the runtime counts."""
    late = tbs = 0
    for f in glob.glob(f"{se.RAW}/rec16{tag}s*_c{cells}_*_none_j*.json"):
        d = json.load(open(f))
        if not steady:
            late += d["all"].get("late_tbs", 0)
            tbs += d["all"]["tbs"]
            continue
        skip, periods = int(d["config"]["skip_periods"]), int(d["config"]["periods"])
        for cf in glob.glob(f"{f[:-5]}_work/conv*.json"):
            c = json.load(open(cf))
            col = {k: i for i, k in enumerate(c["columns"])}
            for r in c["records"]:
                if skip <= int(r[col["period"]]) < periods - END_PERIODS:
                    tbs += 1
                    late += int(r[col["done_ns"]]) - int(r[col["release_ns"]]) > pn.L1_DEADLINE_MS * 1e6
    return 100.0 * late / max(1, tbs)


def main():
    # the runs with the rejection at the L1 result of Section 4.3 (N29 rerun, tag ARRIVAL before the load) when every load
    # has them, compared with the full-load runs of the same rule; else the first runs (rejection at the latest start)
    x = pn.ARRIVAL
    found = lambda pre: sorted({m.group(1) for f in glob.glob(f"{se.RAW}/rec16allb68*{N16}{pre}a*s*_c*_*_none_j*.json")
                                if (m := re.search(rf"{N16}{pre}(a\d+[bu])s\d", f))})
    loads_x, loads_l = found(x), found("")
    if loads_x and set(loads_x) >= set(loads_l):
        pre, full, loads = x, se.availability(), loads_x
    else:
        pre, full, loads = "", se.availability(se.LATEST), loads_l
    out = {"rejection": "L1 result" if pre else "latest start"}
    for load in loads:
        act = int(load[1:-1]) / 100.0
        tags = {f"allb68{N16}{pre}{load}": "one", f"allb68g4{N16}{pre}{load}": "four"}
        avail = se.availability(tags)
        if not avail:
            continue
        rows = {}
        for (g, c), v in sorted(avail.items(), key=lambda x: (x[0][0], x[0][1])):
            tag = f"allb68g4{N16}{pre}{load}" if g == "four" else f"allb68{N16}{pre}{load}"
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
