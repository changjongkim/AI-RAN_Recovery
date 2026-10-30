#!/usr/bin/env python3
"""Recovery receivers on top of the MMSE-IRC L1 (decode_harq.py --irc --whiten --cb-mix): for each set, the
retransmissions per 100 TBs of the conventional structures and of each recovery receiver, the share of the L1's failed
TBs it recovers, and the multiple of the retransmissions of the same L1 without recovery.  With --check REF, the
outcomes that the new decode shares with an earlier run of the same seeds (results/REF/c<k>.npz) must be equal.
usage: irc_recovery.py NAME [NAME ...] [--check NAME=REF ...] [--json OUT]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harq_stats  # noqa: E402

R = Path(__file__).resolve().parents[2]
ROWS = (  # structure, L1 it is compared with, what the recovery receiver is
    ("ai_and_ran", None, "MMSE L1, no recovery"),
    ("ours", "ai_and_ran", "MMSE L1 + neural receiver"),
    ("ours_mix", "ai_and_ran", "MMSE L1 + neural receiver, code blocks of both"),
    ("ai_and_ran_irc", None, "IRC L1, no recovery"),
    ("ours_irc", "ai_and_ran_irc", "IRC L1 + neural receiver"),
    ("ours_irc_mix", "ai_and_ran_irc", "IRC L1 + neural receiver, code blocks of both"),
    ("ours_irc_w", "ai_and_ran_irc", "IRC L1 + neural receiver on the IRC-whitened slot (per PRB)"),
    ("ours_irc_wb", "ai_and_ran_irc", "IRC L1 + neural receiver on the IRC-whitened slot (band)"),
    ("ours_irc_wmix", "ai_and_ran_irc", "IRC L1 + whitened neural receiver, code blocks of both"),
)


def check(name, ref):
    """Outcomes present in both runs, chunk by chunk (same seeds), must be equal."""
    bad, n = [], 0
    for p in sorted((R / "results" / name).glob("c*.npz")):
        q = R / "results" / ref / p.name
        if not q.exists():
            continue
        a, b = np.load(p), np.load(q)
        for k in a.files:
            if k in b.files and k.startswith(("conv_", "sep_", "nrx_", "small_", "irc_")):
                n += 1
                if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                    bad.append(f"{p.name}:{k}")
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    for pair in args.check:
        name, ref = pair.split("=")
        n, bad = check(name, ref)
        print(f"check {name} against {ref}: {n} outcome arrays compared, {len(bad)} differ"
              + (f" ({', '.join(bad[:6])})" if bad else ""))
    out = {}
    for name in args.names:
        paths = sorted((R / "results" / name).glob("c*.npz"))
        good, point, corr, _, _, total, _ = harq_stats.load(paths)
        a = harq_stats.report(good, point, corr, total)["all"]
        out[name] = {"chunks": len(paths), "tbs": a["ai_and_ran"]["tbs"], "rows": {}}
        print(f"\n{name}: {len(paths)} chunks, {a['ai_and_ran']['tbs']} TBs")
        print("| structure | retx /100 | recovered % of L1 failures | cut vs same L1 | lost after all % |")
        print("|---|---|---|---|---|")
        for s, base, label in ROWS:
            if s not in a:
                continue
            r = a[s]
            cut = a[base]["resends_per_100"] / max(1e-9, r["resends_per_100"]) if base else None
            out[name]["rows"][s] = {"label": label, "resends_per_100": r["resends_per_100"],
                                    "recovered_pct_of_failures": r["recovered_pct_of_failures"],
                                    "cut_vs_same_l1": cut, "residual_after_all_pct": r["residual_after_all_pct"]}
            print(f"| {label} | {r['resends_per_100']:.1f} | {r['recovered_pct_of_failures']:.1f} | "
                  f"{'-' if cut is None else f'{cut:.2f}x'} | {r['residual_after_all_pct']:.2f} |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
