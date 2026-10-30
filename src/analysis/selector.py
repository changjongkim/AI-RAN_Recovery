#!/usr/bin/env python3
"""Receiver Selector (Section 4.2) on the traces of mixed interference (N19).  Every batch of slots draws its own
interference over thermal (none with probability 0.25, else uniform in 0-10 dB; gen_harq.py --inr-mix), and the same
slots are decoded twice: with the large NRx that NVIDIA publishes (PUB) and with the NRx trained with interference
(TRAINED, same seeds).  For a threshold theta, a cell uses the trained NRx when its interference is at least theta and
the published one otherwise; "published" and "trained" use one NRx for every slot, and "oracle" takes for each TB and
transmission whichever NRx decodes it (a bound that no measurement reaches).  For each choice: the "ours" structure of
harq_stats.py (retransmissions per 100 TBs, recovered share of the failed TBs, TBs lost after all transmissions) and
the multiple of the conventional retransmissions.
usage: selector.py PUB TRAINED OUT.json      (results/<name>/c<k>.npz, data/<name>/c<k>/s_info.npz)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harq_stats  # noqa: E402

R = Path(__file__).resolve().parents[2]
pub, trained, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ks = sorted(int(p.stem[1:]) for p in (R / "results" / pub).glob("c*.npz")
            if (R / "results" / trained / p.name).exists())

def rows_of(c, g):
    """A key per slot: decode_harq.py writes the slots in another order than gen_harq.py, so slots are matched by their
    channel features (similarity and gains, unique per slot)."""
    return [c[i].tobytes() + g[i].tobytes() for i in range(len(c))]


parts = {"gp": [], "gt": [], "point": [], "corr": [], "inr": []}
for k in ks:
    gp_k, point_k, corr_k, _, gain_k, total, _ = harq_stats.load([R / "results" / pub / f"c{k}.npz"])
    gt_k, _, corr_t, _, gain_t, _, _ = harq_stats.load([R / "results" / trained / f"c{k}.npz"])
    info = np.load(R / "data" / pub / f"c{k}" / "s_info.npz")
    gen = {key: i for i, key in enumerate(rows_of(info["corr"], info["gain"]))}
    to_gen = np.array([gen[key] for key in rows_of(corr_k, gain_k)])
    assert np.array_equal(info["point"][to_gen], point_k), f"chunk {k}: slots do not match the generator"
    tr = {key: i for i, key in enumerate(rows_of(corr_t, gain_t))}
    to_t = np.array([tr[key] for key in rows_of(corr_k, gain_k)])
    parts["gp"].append(gp_k); parts["gt"].append({n: v[to_t] for n, v in gt_k.items()})
    parts["point"].append(point_k); parts["corr"].append(corr_k); parts["inr"].append(info["inr"][to_gen])
gp = {n: np.concatenate([p[n] for p in parts["gp"]]) for n in parts["gp"][0]}
gt = {n: np.concatenate([p[n] for p in parts["gt"]]) for n in parts["gt"][0]}
point, corr, inr = (np.concatenate(parts[n]) for n in ("point", "corr", "inr"))
same_l1 = float((gp["conv_t0"] == gt["conv_t0"]).mean())
inr_db = np.where(np.isnan(inr), -np.inf, inr)


def run(use_trained=None, oracle=False):
    g = dict(gp)
    for k in gp:
        if k.startswith("nrx_"):
            g[k] = (gp[k] | gt[k]) if oracle else np.where(use_trained[:, None], gt[k], gp[k])
    a = harq_stats.report(g, point, corr, total)["all"]
    o, c = a["ours"], a["ai_and_ran"]
    return {"resends_per_100": o["resends_per_100"], "recovered_pct_of_failures": o["recovered_pct_of_failures"],
            "residual_after_all_pct": o["residual_after_all_pct"], "conventional_resends_per_100": c["resends_per_100"],
            "retx_cut_x": c["resends_per_100"] / max(1e-9, o["resends_per_100"]),
            "trained_share_of_slots": None if use_trained is None else float(use_trained.mean())}


rows = {"published": run(np.zeros(len(point), bool)), "trained": run(np.ones(len(point), bool))}
for th in range(0, 11):
    rows[f"theta_{th}dB"] = run(inr_db >= th)
rows["oracle"] = run(oracle=True)
best = max((k for k in rows if k.startswith("theta")), key=lambda k: rows[k]["recovered_pct_of_failures"])
res = {"pub": pub, "trained": trained, "chunks": ks, "slots": int(len(point)), "same_l1_outcome": same_l1,
       "slots_without_interference": float(np.isnan(inr).mean()), "best_theta": best, "choices": rows}
out.write_text(json.dumps(res, indent=2))
print(f"{pub} / {trained}: {len(point)} slots in {len(ks)} chunks, L1 outcome equal in {100 * same_l1:.2f}% of TBs, "
      f"{100 * np.isnan(inr).mean():.0f}% of slots without interference")
print("| choice | trained share | recovered % of failed | retx /100 | cut vs conventional | lost % |")
print("|---|---|---|---|---|---|")
for k, r in rows.items():
    ts = "-" if r["trained_share_of_slots"] is None else f"{100 * r['trained_share_of_slots']:.0f}%"
    print(f"| {k} | {ts} | {r['recovered_pct_of_failures']:.1f} | {r['resends_per_100']:.1f} | {r['retx_cut_x']:.2f}x | "
          f"{r['residual_after_all_pct']:.2f} |")
