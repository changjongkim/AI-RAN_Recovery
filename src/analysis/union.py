#!/usr/bin/env python3
"""N12: which failures each recovery receiver decodes, on the same slots; unions and cascades.
For a set of classical_rx outputs (npz with ok_<system>), the failures of the first (conventional) receiver are split
by which receivers decode them.  A cascade tries receivers in order and stops at the first success; its cost is the
mean number of receiver runs per failure.  usage: union.py NAME:pattern [...]  -> results/n12_union.json"""
import glob, itertools, json, sys
import numpy as np

out = {}
for arg in sys.argv[1:]:
    name, pat = arg.split(":", 1)
    files = sorted(glob.glob(pat))
    keep = [f for f in files if json.load(open(f[:-4] + ".json")).get("prbs", 273) == 273]
    d = [np.load(f) for f in keep]
    systems = [k[3:] for k in d[0].files if k.startswith("ok_")]
    ok = {s: np.concatenate([x["ok_" + s].reshape(-1) for x in d]) for s in systems}
    conv = ok[systems[0]]; fail = ~conv; n = int(fail.sum())
    pct = lambda m: round(100.0 * float((m & fail).sum()) / n, 1)
    r = {"failures": n, "single": {s: pct(ok[s]) for s in systems[1:]}}
    rec = [s for s in systems[1:] if not s.startswith("baseline_perf")]
    r["pairs"] = {f"{a}|{b}": pct(ok[a] | ok[b]) for a, b in itertools.combinations(rec, 2)}
    r["all_practical"] = pct(np.logical_or.reduce([ok[s] for s in rec]))
    if "nrx" in ok:
        r["only_nrx"] = pct(ok["nrx"] & ~np.logical_or.reduce([ok[s] for s in rec if s != "nrx"]))
        r["missed_by_nrx_got_by_others"] = pct(~ok["nrx"] & np.logical_or.reduce([ok[s] for s in rec if s != "nrx"]))
    if "nrx_rt" in ok and "nrx" in ok:   # cascade: small model first, large only if small fails
        r["cascade_small_then_large"] = {"recovered": pct(ok["nrx_rt"] | ok["nrx"]),
                                         "large_runs_per_failure": round(float((~ok["nrx_rt"] & fail).sum()) / n, 3)}
    out[name] = r
    print(name, json.dumps(r))
json.dump(out, open("results/n12_union.json", "w"), indent=1)
