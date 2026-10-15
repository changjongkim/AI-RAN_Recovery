#!/usr/bin/env python3
"""Merge classical_rx.py outputs (npz + json per run) into one summary: failure rate per receiver, the share of the
first receiver's failures each other receiver decodes, both by similarity bin of the two users' channels, and the
GPU time per slot.  Runs with plain numpy (login node or container).
usage: merge_classical.py --inputs A.npz [B.npz ...] --output SUMMARY.json
"""

import argparse
import json

import numpy as np

BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))


def merge(files, prbs=273):
    """Runs of another bandwidth (before 2026-10-15 04:00 classical_rx ran at the config's 132 PRBs) are left out."""
    keep = [f for f in files if json.load(open(f[:-4] + ".json")).get("prbs") == prbs]
    if len(keep) < len(files):
        print("left out %d runs that are not at %d PRBs" % (len(files) - len(keep), prbs))
    files = keep
    d = [np.load(f) for f in files]
    rep = [json.load(open(f[:-4] + ".json")) for f in files]
    systems = rep[0]["systems"]
    assert all(r["systems"] == systems for r in rep), "runs with different receiver lists"
    ok = {s: np.concatenate([x["ok_" + s] for x in d]) for s in systems}
    corr = np.concatenate([x["corr"] for x in d])
    point = np.concatenate([x["point"] for x in d])
    conv = ok[systems[0]]
    rho = np.repeat(corr, conv.shape[1])
    res = {"files": list(files), "systems": systems, "slots": int(conv.shape[0]), "blocks": int(conv.size),
           "num_rx": rep[0]["num_rx"], "users": rep[0].get("users", 2), "channel": rep[0].get("channel"),
           "speed": rep[0].get("speed"),
           "points": {"%g" % p: int((point == p).sum()) for p in sorted(set(point.tolist()))},
           "seconds_per_slot": {s: float(np.mean([r["seconds_per_slot"][s] for r in rep])) for s in systems},
           "fail_pct": {s: 100.0 * float((~ok[s]).mean()) for s in systems},
           "first_failures": int((~conv).sum()),
           "recovers_pct_of_conventional_failures": {
               s: 100.0 * float((ok[s] & ~conv).sum()) / max(1, int((~conv).sum())) for s in systems[1:]},
           "by_point": {}, "by_similarity": {}}
    pts = np.repeat(point, conv.shape[1])
    for p in sorted(set(point.tolist())):
        m = pts == p
        f = ~conv.reshape(-1) & m
        res["by_point"]["%g" % p] = {
            "fail_pct": {s: 100.0 * float((~ok[s].reshape(-1) & m).sum()) / max(1, int(m.sum())) for s in systems},
            "recovers_pct": {s: 100.0 * float((ok[s].reshape(-1) & f).sum()) / max(1, int(f.sum())) for s in systems[1:]}}
    for lo, hi in BINS:
        m = (rho >= lo) & (rho < hi)
        f = ~conv.reshape(-1) & m
        res["by_similarity"]["%.1f-%.1f" % (lo, hi)] = {
            "share_of_blocks": float(m.mean()),
            "conventional_fail_pct": 100.0 * float(f.sum()) / max(1, int(m.sum())),
            "share_of_failures": float(f.sum()) / max(1, int((~conv).sum())),
            "recovers_pct": {s: 100.0 * float((ok[s].reshape(-1) & f).sum()) / max(1, int(f.sum())) for s in systems[1:]}}
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    res = merge(sorted(args.inputs))
    json.dump(res, open(args.output, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("slots", "blocks", "first_failures", "fail_pct",
                                          "recovers_pct_of_conventional_failures", "seconds_per_slot")}, indent=2))


if __name__ == "__main__":
    main()
