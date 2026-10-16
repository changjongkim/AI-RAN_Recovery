#!/usr/bin/env python3
"""95% intervals of the headline numbers by resampling chunks (each chunk has its own seed, so chunks are independent).
For every results/<name>/c*.npz set: recovered share of failures, resends per 100 blocks (L1 and ours) and their ratio,
residual after all transmissions.  usage: ci.py NAME [NAME ...]  -> results/ci.json and a table."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "src")
from harq_stats import plans, transmissions

rng = np.random.default_rng(0)
out = {}
for name in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(f"results/{name}/c*.npz")):
        d = np.load(f); total = int(d["total"])
        flat = {k: d[k].reshape(-1) for k in d.files if k.startswith(("conv_", "sep_", "nrx_", "small_"))}
        p = plans(flat, total)
        r = {}
        for s in ("ai_and_ran", "ours"):
            first, rec, steps = p[s]
            n = transmissions(first, rec, steps, total)
            r[s] = (np.clip(n - 1, 0, total - 1).sum(), (n > total).sum())
        fail = (~flat["conv_t0"]).sum(); rec = (flat["nrx_c0"] & ~flat["conv_t0"]).sum()
        rows.append((len(flat["conv_t0"]), fail, rec, r["ai_and_ran"][0], r["ours"][0], r["ai_and_ran"][1], r["ours"][1]))
    a = np.array(rows, dtype=float)
    def stat(x):
        s = x.sum(0)
        return [100 * s[2] / s[1], 100 * s[3] / s[0], 100 * s[4] / s[0], s[4] / s[3], 100 * s[5] / s[0], 100 * s[6] / s[0]]
    point = stat(a)
    boot = np.array([stat(a[rng.integers(0, len(a), len(a))]) for _ in range(2000)])
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    keys = ["recovered_pct", "resends_l1", "resends_ours", "resends_ratio", "residual_l1", "residual_ours"]
    out[name] = {k: [round(float(p), 4), round(float(l), 4), round(float(h), 4)] for k, p, l, h in zip(keys, point, lo, hi)}
    print(name, len(a), "chunks", {k: f"{v[0]:.3g} [{v[1]:.3g}, {v[2]:.3g}]" for k, v in out[name].items()})
json.dump(out, open("results/ci.json", "w"), indent=1)
