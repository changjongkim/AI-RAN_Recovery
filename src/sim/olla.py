#!/usr/bin/env python3
"""N6: outer-loop link adaptation with IR HARQ, with and without recovery (trace-driven over MCS 10-16).

Input: decode_harq.py outputs of one experiment in which every MCS was generated on the same channel realizations and
noise (gen_harq.py --mcs 10,...,16 --snr-db ...): results/<name>/m<mcs>_c<k>.npz.  For each SNR point a link draws
blocks from that point's slots; the outer loop keeps a pointer x on the MCS ladder, sends at MCS round(x), and moves
the pointer down by delta after a first-transmission failure and up by delta * target / (1 - target) after a success
(the usual ACK/NACK step ratio that settles at the target first-transmission error rate).  "Success" is what the
scheduler sees before asking for a retransmission: the conventional receiver decoded it, or, with recovery, recovery
did.  A block that fails is retransmitted with IR HARQ up to the traced number of transmissions.
Goodput = delivered bits / transmissions used (bits per uplink allocation).
usage: olla.py --root results/<name> --output OUT.json [--targets 0.1,0.03,0.01 --steps 20000 --delta 0.25]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harq_stats import STRUCTURES, load, plans, transmissions   # noqa: E402


def per_mcs(root: Path):
    files = sorted(glob.glob(str(root / "m*_c*.npz")))
    groups = {}
    for f in files:
        m = int(re.search(r"/m(\d+)_c\d+\.npz$", f).group(1))
        groups.setdefault(m, []).append(f)
    out = {}
    for m, fs in sorted(groups.items()):
        fs.sort(key=lambda f: int(re.search(r"_c(\d+)\.npz$", f).group(1)))
        good, point, corr, change, gain, total, meta = load(fs)
        tb_bits = 8 * int(json.loads(str(np.load(fs[0])["meta"]))["tb_bytes"])
        flat = {k: v.reshape(-1) for k, v in good.items()}
        p = plans(flat, total)
        out[m] = {"point": np.repeat(point, good["conv_t0"].shape[1]), "tb_bits": tb_bits, "total": total,
                  "first": {s: p[s][0] | p[s][1] for s in STRUCTURES},
                  "n_tx": {s: transmissions(*p[s], total) for s in STRUCTURES}}
    return out


def simulate(tab, ladder, sel, structure, target, steps, delta, rng):
    x = float(ladder[len(ladder) // 2])
    up = delta * target / (1 - target)
    bits = used = fails = resends = residual = 0
    mcs_sum = 0.0
    idx = np.flatnonzero(sel)
    warm = steps // 10
    for k in range(steps):
        m = int(np.clip(round(x), ladder[0], ladder[-1]))
        i = idx[rng.integers(0, len(idx))]
        t = tab[m]
        ok = bool(t["first"][structure][i])
        n = int(t["n_tx"][structure][i])
        x = x + up if ok else x - delta
        x = float(np.clip(x, ladder[0] - 0.49, ladder[-1] + 0.49))
        if k < warm:
            continue
        mcs_sum += m
        fails += not ok
        total = t["total"]
        used += min(n, total)
        resends += min(n, total) - 1
        if n <= total:
            bits += t["tb_bits"]
        else:
            residual += 1
    count = steps - warm
    return {"goodput_bits_per_tx": bits / max(1, used), "mean_mcs": mcs_sum / count,
            "first_fail_pct": 100.0 * fails / count, "resends_per_100": 100.0 * resends / count,
            "residual_pct": 100.0 * residual / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default="0.1,0.03,0.01")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--delta", type=float, default=0.25)
    a = parser.parse_args()
    tab = per_mcs(a.root)
    ladder = sorted(tab)
    points = sorted(set(tab[ladder[0]]["point"].tolist()))
    out = {"root": str(a.root), "ladder": ladder, "points": points, "results": {}}
    rng = np.random.default_rng(5)
    for q in points:
        sel = tab[ladder[0]]["point"] == q
        for target in [float(t) for t in a.targets.split(",")]:
            row = {s: simulate(tab, ladder, sel, s, target, a.steps, a.delta, rng) for s in STRUCTURES}
            base = row["ai_and_ran"]["goodput_bits_per_tx"]
            for s in STRUCTURES:
                row[s]["goodput_vs_ai_and_ran"] = row[s]["goodput_bits_per_tx"] / max(1e-9, base)
            out["results"][f"snr{q:g}_target{target:g}"] = row
    a.output.write_text(json.dumps(out, indent=2))
    print("| SNR | target | AI-and-RAN MCS / goodput | ours MCS / goodput (x) | AI-for-RAN goodput (x) |")
    print("|---|---|---|---|---|")
    for k, r in out["results"].items():
        q, t = re.match(r"snr(.+)_target(.+)", k).groups()
        print(f"| {q} | {t} | {r['ai_and_ran']['mean_mcs']:.1f} / {r['ai_and_ran']['goodput_bits_per_tx']:.0f} | "
              f"{r['ours']['mean_mcs']:.1f} / {r['ours']['goodput_bits_per_tx']:.0f} ({r['ours']['goodput_vs_ai_and_ran']:.2f}) | "
              f"{r['ai_for_ran']['goodput_bits_per_tx']:.0f} ({r['ai_for_ran']['goodput_vs_ai_and_ran']:.2f}) |")


if __name__ == "__main__":
    main()
