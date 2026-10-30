#!/usr/bin/env python3
"""Transmissions per block for each structure, from the per-block outcomes of decode_harq.py; merges chunks.

Structures (first transmission, what may end a failed block without a retransmission, what decodes after
retransmission t):
  ai_and_ran       conventional receiver; nothing; HARQ with the conventional receiver (sep_c<t>)
  ai_for_ran       small neural receiver inside L1 (small_c0); nothing; HARQ with it (small_c<t>)
  ours             conventional receiver; recovery by the large neural receiver (nrx_c0); sep_c<t> or nrx_c<t>
  ours_conv_harq   conventional receiver; recovery; HARQ with the conventional receiver only
With decode_harq.py --irc the same with the MMSE-IRC receiver as L1 (ai_and_ran_irc, ours_irc), and recovery receivers
on top of it when their outcomes are present: ours_irc_w / ours_irc_wb (the large neural receiver on the IRC-whitened
slot, per PRB / over the band; --whiten), ours_mix / ours_irc_mix / ours_irc_wmix (code blocks taken from either
receiver: conventional and neural, IRC and neural, IRC and whitened neural; --cb-mix).
usage: harq_stats.py --inputs A.npz [B.npz ...] --output REPORT.json   (npz files written by decode_harq.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STRUCTURES = ("ai_and_ran", "ai_for_ran", "ours", "ours_conv_harq")
BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))


def transmissions(first_ok, recovered, step_ok, total):
    """Transmissions per block: 1..total, total + 1 when it still failed after all of them."""
    n = np.where(first_ok | recovered, 1, total + 1)
    for t in range(1, total):
        n = np.where((n == total + 1) & step_ok[t], t + 1, n)
    return n


def plans(flat, total):
    step = lambda r: [None] + [flat[f"{r}_c{t}"] for t in range(1, total)]
    nothing = np.zeros_like(flat["conv_t0"])
    return {
        "ai_and_ran": (flat["conv_t0"], nothing, step("sep")),
        "ai_for_ran": (flat["small_c0"], nothing, step("small")),
        "ours": (flat["conv_t0"], flat["nrx_c0"] & ~flat["conv_t0"],
                 [None] + [flat[f"sep_c{t}"] | flat[f"nrx_c{t}"] for t in range(1, total)]),
        "ours_conv_harq": (flat["conv_t0"], flat["nrx_c0"] & ~flat["conv_t0"], step("sep")),
    } | ({    # with the MMSE-IRC conventional receiver as L1 (decode_harq.py --irc)
        "ai_and_ran_irc": (flat["irc_c0"], nothing, step("irc")),
        "ours_irc": (flat["irc_c0"], flat["nrx_c0"] & ~flat["irc_c0"],
                     [None] + [flat[f"irc_c{t}"] | flat[f"nrx_c{t}"] for t in range(1, total)]),
    } if "irc_c0" in flat else {}) | {
        name: (flat[l1], flat[f"{rec}_c0"] & ~flat[l1],
               [None] + [flat[f"{a}_c{t}"] | flat[f"{b}_c{t}"] | flat[f"{rec}_c{t}"] for t in range(1, total)])
        for name, l1, rec, a, b in (    # structure, L1, recovery receiver, the receivers whose HARQ buffers decode
            ("ours_irc_w", "irc_c0", "nrxw", "irc", "nrxw"), ("ours_irc_wb", "irc_c0", "nrxwb", "irc", "nrxwb"),
            ("ours_mix", "conv_t0", "mixsep", "sep", "nrx"), ("ours_irc_mix", "irc_c0", "mixirc", "irc", "nrx"),
            ("ours_irc_wmix", "irc_c0", "mixw", "irc", "nrxw"))
        if f"{rec}_c0" in flat
    }


def stats(n_tx, first_ok, recovered, sel, total):
    m = max(1, int(sel.sum()))
    fail = ~first_ok & sel
    nf = max(1, int(fail.sum()))
    resends = np.clip(n_tx - 1, 0, total - 1)
    return {
        "tbs": int(sel.sum()), "first_failures": int(fail.sum()),
        "first_fail_pct": 100.0 * fail.sum() / m,
        "done_at_bs_pct": 100.0 * (recovered & fail).sum() / m,
        "recovered_pct_of_failures": 100.0 * (recovered & fail).sum() / nf,
        "resends_per_100": 100.0 * (resends * sel).sum() / m,
        "still_failed_after_one_resend_pct_of_failures": 100.0 * ((n_tx > 2) & fail).sum() / nf,
        "residual_after_all_pct": 100.0 * ((n_tx > total) & sel).sum() / m,
        "transmissions_hist": [int(((n_tx == k) & sel).sum()) for k in range(1, total + 2)],
        "mean_transmissions": float((np.minimum(n_tx, total) * sel).sum() / m),
    }


def report(good: dict, point, corr, total: int, extra: dict | None = None) -> dict:
    """good: name -> bool [slots, users]; point: [slots]; corr: [slots, transmissions]."""
    users = good["conv_t0"].shape[1]
    flat = {k: v.reshape(-1) for k, v in good.items()}
    p = plans(flat, total)
    n_tx = {s: transmissions(f, r, st, total) for s, (f, r, st) in p.items()}
    slot_corr = np.repeat(corr[:, 0], users)
    slot_point = np.repeat(point, users)

    def summary(sel):
        return {s: stats(n_tx[s], p[s][0], p[s][1], sel, total) for s in p}

    out = dict(extra or {})
    out.update({
        "slots": int(len(point)), "users": int(users), "transmissions": total,
        "decoded": {k: int(v.sum()) for k, v in good.items()},
        "corr_median": float(np.median(corr[:, 0])),
        "all": summary(np.ones_like(flat["conv_t0"])),
        "by_point": {f"{q:g}": summary(slot_point == q) for q in sorted(set(np.asarray(point).tolist()))},
        "by_similarity": {f"{lo:.1f}-{hi:.1f}": summary((slot_corr >= lo) & (slot_corr < hi)) for lo, hi in BINS},
        "similarity_share": {f"{lo:.1f}-{hi:.1f}": float(((corr[:, 0] >= lo) & (corr[:, 0] < hi)).mean()) for lo, hi in BINS},
        "check": {"sep_vs_conv_first": [int(flat["sep_c0"].sum()), int(flat["conv_t0"].sum())]},
    })
    return out


def table(rep: dict) -> str:
    lines = ["| structure | first fail % | done at BS % | resends /100 | failed after one resend % | residual % | mean tx |",
             "|---|---|---|---|---|---|---|"]
    for s in rep["all"]:
        r = rep["all"][s]
        lines.append(f"| {s} | {r['first_fail_pct']:.1f} | {r['done_at_bs_pct']:.1f} | {r['resends_per_100']:.1f} | "
                     f"{r['still_failed_after_one_resend_pct_of_failures']:.0f} | {r['residual_after_all_pct']:.2f} | "
                     f"{r['mean_transmissions']:.3f} |")
    return "\n".join(lines)


def load(paths):
    good, point, corr, change, gain, total, meta = {}, [], [], [], [], None, None
    for path in paths:
        d = np.load(path, allow_pickle=False)
        total = int(d["total"]) if total is None else total
        meta = json.loads(str(d["meta"])) if meta is None else meta
        for k in d.files:
            if k.startswith(("conv_", "sep_", "nrx_", "small_", "irc_", "nrxw_", "nrxwb_", "mixsep_", "mixirc_", "mixw_")):
                good.setdefault(k, []).append(d[k])
        point.append(d["point"]); corr.append(d["corr"]); change.append(d["change"]); gain.append(d["gain"])
    good = {k: np.concatenate(v) for k, v in good.items()}
    return good, np.concatenate(point), np.concatenate(corr), np.concatenate(change), np.concatenate(gain), total, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    good, point, corr, change, gain, total, meta = load(args.inputs)
    rep = report(good, point, corr, total, {"chunks": [str(p) for p in args.inputs], "generator": meta,
                                            "change_median": [float(np.median(change[:, t])) for t in range(total)]})
    args.output.write_text(json.dumps(rep, indent=2))
    print(f"{args.output.name}: {rep['slots']} slots, {rep['slots'] * rep['users']} TBs, similarity median "
          f"{rep['corr_median']:.2f}, separable vs cuPHY first {rep['check']['sep_vs_conv_first']}")
    print(table(rep))


if __name__ == "__main__":
    main()
