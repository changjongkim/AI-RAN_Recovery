#!/usr/bin/env python3
"""P2, step 2: how much the base station depends on finding well-separated pairs, with and without recovery.

Link level (from the HARQ traces of decode_harq.py): for every block the similarity of its pair, whether the
conventional receiver decoded it, whether recovery decoded it, and how many transmissions each structure needed.
These are binned by pair similarity (bin width 0.05).  System level (pair_drops.py): for U users per drop, the
similarity of every pair.  A scheduler that co-schedules two users per slot picks either a random pair or the most
separable pair of the drop; the binned link outcomes give, per structure, the first-transmission failures, the
retransmissions per 100 blocks and the blocks delivered per transmission.  The question: with recovery, how many users
does the scheduler need to choose from to match the conventional receiver with a well-separated pair?
usage: pairing.py --traces A.npz [...] --drops DROPS.npz --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harq_stats import STRUCTURES, load, plans, transmissions   # noqa: E402

EDGES = np.round(np.arange(0.0, 1.0001, 0.05), 3)


def tables(good, corr, total):
    users = good["conv_t0"].shape[1]
    flat = {k: v.reshape(-1) for k, v in good.items()}
    p = plans(flat, total)
    rho = np.repeat(corr[:, 0], users)
    b = np.clip(np.digitize(rho, EDGES) - 1, 0, len(EDGES) - 2)
    out = {}
    for s, (first, rec, steps) in p.items():
        n_tx = transmissions(first, rec, steps, total)
        rows = []
        for k in range(len(EDGES) - 1):
            m = b == k
            c = int(m.sum())
            rows.append({"count": c,
                         "first_fail": float((~first & m).sum() / c) if c else np.nan,
                         "resends": float((np.clip(n_tx - 1, 0, total - 1) * m).sum() / c) if c else np.nan,
                         "residual": float(((n_tx > total) & m).sum() / c) if c else np.nan})
        out[s] = rows
    return out


def fill(rows, key):
    """Bin values with empty bins taken from the nearest filled bin."""
    v = np.array([r[key] for r in rows], dtype=float)
    idx = np.where(~np.isnan(v))[0]
    return np.interp(np.arange(len(v)), idx, v[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--drops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    good, point, corr, change, gain, total, meta = load(a.traces)
    tab = tables(good, corr, total)
    drops = np.load(a.drops)
    out = {"traces": [str(p) for p in a.traces], "drops": str(a.drops), "bin_edges": EDGES.tolist(), "tables": tab,
           "policies": {}}
    for key in sorted(k for k in drops.files if k.startswith("rho_u")):
        users = int(key[5:])
        rho = drops[key]
        iu = np.triu_indices(users, 1)
        pairs = rho[:, iu[0], iu[1]]
        rng = np.random.default_rng(1)
        choice = {"random": pairs[np.arange(len(pairs)), rng.integers(0, pairs.shape[1], len(pairs))],
                  "best": pairs.min(axis=1)}
        for policy, r in choice.items():
            b = np.clip(np.digitize(r, EDGES) - 1, 0, len(EDGES) - 2)
            row = {"users": users, "policy": policy, "similarity_median": float(np.median(r))}
            for s in STRUCTURES:
                ff, rs, res = (fill(tab[s], k)[b].mean() for k in ("first_fail", "resends", "residual"))
                row[s] = {"first_fail_pct": 100 * ff, "resends_per_100": 100 * rs, "residual_pct": 100 * res,
                          "blocks_per_transmission": (1 - res) / (1 + rs)}
            out["policies"][f"u{users}_{policy}"] = row
    a.output.write_text(json.dumps(out, indent=2, default=float))
    print("| users | pairing | similarity | ai_and_ran resends/100 | ours resends/100 | ai_and_ran blocks/tx | ours blocks/tx |")
    print("|---|---|---|---|---|---|---|")
    for k, r in out["policies"].items():
        print(f"| {r['users']} | {r['policy']} | {r['similarity_median']:.2f} | {r['ai_and_ran']['resends_per_100']:.1f} | "
              f"{r['ours']['resends_per_100']:.1f} | {r['ai_and_ran']['blocks_per_transmission']:.3f} | "
              f"{r['ours']['blocks_per_transmission']:.3f} |")


if __name__ == "__main__":
    main()
