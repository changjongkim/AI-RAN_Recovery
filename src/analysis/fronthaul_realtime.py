#!/usr/bin/env python3
"""E1b from existing runs (docs/plan_setup_and_sensitivity.md): the real-time runs of the four-GPU server at recovery
deadlines of 9.0-13.0 ms after the samples arrive (rec16allb68g4d{90,...,130}_f35arrivalpred60arrivalmargin015, the
settings of Figures 14-15), read as fronthaul distances: the deadline after arrival is 11.0 - 2t with t = 5 us per km,
so 10.0 ms is 100 km and 9.0 ms is 200 km.  Per deadline and number of cells: failed TBs recovered on time (%) and L1
results later than 4.0 ms (%), summed over the seeds.
usage: python3 src/analysis/fronthaul_realtime.py        output: results/e1_fronthaul_realtime.json
"""
import collections, glob, json, re
RAW = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw/"
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(RAW + "rec16allb68g4d*_f35arrivalpred60arrivalmargin015s[0-9]_c*_rescue_none_*.json"):
    m = re.search(r"g4d(\d+)_f35arrivalpred60arrivalmargin015s(\d)_c(\d+)_", f)
    a = json.load(open(f))["all"]
    rows[int(m.group(1)) / 10][int(m.group(3))].append(a)
out = {}
for D in sorted(rows):
    km = (11.0 - D) / 2 / 0.005
    out[f"{D:g}"] = {"km": km, "cells": {}}
    for c in sorted(rows[D]):
        r = rows[D][c]
        rec, fail = sum(a["nrx_rescues_on_time"] for a in r), sum(a["conv_fail_tbs"] for a in r)
        late, tbs = sum(a["late_tbs"] for a in r), sum(a["tbs"] for a in r)
        out[f"{D:g}"]["cells"][c] = {"recovered_on_time_pct": 100 * rec / fail, "l1_late_pct": 100 * late / tbs, "seeds": len(r)}
    print(f"D {D:4.1f} ms ({km:+5.0f} km): " + "  ".join(f"c{c} {v['recovered_on_time_pct']:.1f}%" for c, v in out[f'{D:g}']['cells'].items()))
json.dump(out, open("results/e1_fronthaul_realtime.json", "w"), indent=1)
