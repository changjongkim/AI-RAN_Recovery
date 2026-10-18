#!/usr/bin/env python3
"""Collect every result of the project into tables (results/summary/*.csv and *.md): HARQ conditions with 95% intervals
(chunks resampled), classical/neural receivers on the same slots, link adaptation, latency policies, GPU demand.
usage: summarize.py   (run from the repository root; reads results/*.json and results/<set>/c*.npz)"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, "src")
from harq_stats import plans, transmissions

os.makedirs("results/summary", exist_ok=True)
rng = np.random.default_rng(0)


def write(name, rows, cols):
    with open(f"results/summary/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    with open(f"results/summary/{name}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(name, len(rows), "rows")


def chunk_ci(name):
    rows = []
    for f in sorted(glob.glob(f"results/{name}/c*.npz")):
        d = np.load(f); total = int(d["total"])
        flat = {k: d[k].reshape(-1) for k in d.files if k.startswith(("conv_", "sep_", "nrx_", "small_"))}
        p = plans(flat, total)
        r = []
        for s in ("ai_and_ran", "ours"):
            first, rec, steps = p[s]; n = transmissions(first, rec, steps, total)
            r += [np.clip(n - 1, 0, total - 1).sum(), (n > total).sum()]
        rows.append([len(flat["conv_t0"]), (~flat["conv_t0"]).sum(), (flat["nrx_c0"] & ~flat["conv_t0"]).sum()] + r)
    a = np.array(rows, float)
    f = lambda s: [100 * s[2] / s[1], s[5] / s[3], 100 * (1 - s[6] / max(s[4], 1))]
    b = np.array([f(a[rng.integers(0, len(a), len(a))].sum(0)) for _ in range(1000)])
    lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
    return lo, hi


harq = []
for path in sorted(glob.glob("results/*.json")):
    name = os.path.basename(path)[:-5]
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if not isinstance(d, dict) or "all" not in d or "ai_and_ran" not in d.get("all", {}) or not os.path.isdir(f"results/{name}"):
        continue
    gen = d.get("generator", {}); gen = gen.get("generator", gen) if isinstance(gen, dict) else {}
    a, f, o, c = (d["all"][s] for s in ("ai_and_ran", "ai_for_ran", "ours", "ours_conv_harq"))
    lo, hi = chunk_ci(name)
    harq.append({"set": name, "channel": gen.get("channel"), "speed_kmh": round(3.6 * gen.get("speed", 0), 0), "los": gen.get("los"),
                 "mcs": (gen.get("mcs") or [None])[0], "retx_alone": gen.get("retx_alone", False), "drop_users": gen.get("drop_users", 0),
                 "pairing": gen.get("pairing", "") if gen.get("drop_users") else "", "ebno": "/".join(sorted(d["by_point"], key=float)),
                 "blocks": d["slots"] * d["users"], "pair_similarity_median": round(d["corr_median"], 3),
                 "first_fail_pct": round(a["first_fail_pct"], 2), "first_fail_small_nrx_pct": round(f["first_fail_pct"], 2),
                 "recovered_pct": round(o["recovered_pct_of_failures"], 1), "recovered_ci": f"{lo[0]:.1f}-{hi[0]:.1f}",
                 "resends_l1": round(a["resends_per_100"], 2), "resends_small_nrx": round(f["resends_per_100"], 2),
                 "resends_ours": round(o["resends_per_100"], 2), "resends_recovery_only": round(c["resends_per_100"], 2),
                 "resend_ratio": round(o["resends_per_100"] / a["resends_per_100"], 3), "resend_ratio_ci": f"{lo[1]:.3f}-{hi[1]:.3f}",
                 "residual_l1_pct": round(a["residual_after_all_pct"], 2), "residual_ours_pct": round(o["residual_after_all_pct"], 2),
                 "residual_cut_pct_ci": f"{lo[2]:.0f}-{hi[2]:.0f}"})
write("harq", harq, list(harq[0].keys()))

cls = []
for path in sorted(glob.glob("results/*.json")):
    name = os.path.basename(path)[:-5]
    d = json.load(open(path))
    if not isinstance(d, dict) or "recovers_pct_of_conventional_failures" not in d or "fail_pct" not in d:
        continue
    first = json.load(open(d["files"][0][:-4] + ".json")) if d.get("files") else {}
    row = {"set": name, "channel": d.get("channel") or first.get("channel"), "users": d.get("users") or first.get("users"),
           "num_rx": d.get("num_rx"), "base_config": first.get("base_config"), "prbs": first.get("prbs"),
           "slots": d.get("slots"), "conventional_failures": d.get("first_failures")}
    for s, v in d["fail_pct"].items():
        row[f"fail%_{s}"] = round(v, 2)
    for s, v in d["recovers_pct_of_conventional_failures"].items():
        row[f"recovers%_{s}"] = round(v, 1)
    cls.append(row)
cols = sorted({k for r in cls for k in r}, key=lambda k: (not k in ("set", "channel", "users", "num_rx", "base_config", "prbs", "slots", "conventional_failures"), k))
write("receivers", cls, cols)

olla = []
for path in sorted(glob.glob("results/n6_*_olla.json")):
    d = json.load(open(path))
    for k, r in d["results"].items():
        snr, tgt = k.split("_target")
        olla.append({"set": os.path.basename(path)[:-10], "snr_db": snr[3:], "target": tgt,
                     "mcs_l1": round(r["ai_and_ran"]["mean_mcs"], 2), "mcs_ours": round(r["ours"]["mean_mcs"], 2),
                     "goodput_l1_kbit": round(r["ai_and_ran"]["goodput_bits_per_tx"] / 1e3, 1),
                     "goodput_ours_x": round(r["ours"]["goodput_vs_ai_and_ran"], 3), "goodput_small_nrx_x": round(r["ai_for_ran"]["goodput_vs_ai_and_ran"], 3),
                     "residual_l1": round(r["ai_and_ran"]["residual_pct"], 2), "residual_ours": round(r["ours"]["residual_pct"], 2)})
write("link_adaptation", olla, list(olla[0].keys()))

lat = []
for path in sorted(glob.glob("results/n7_*.json")):
    d = json.load(open(path))
    for k, r in d.get("policies", {}).items():
        lat.append({"set": os.path.basename(path)[:-5], "policy": k, "lead_ms": d["params"]["lead"],
                    "unrecovered_p50": r.get("unrecovered_p50_ms"), "unrecovered_p99": r.get("unrecovered_p99_ms"),
                    "recovered_p50": r.get("recovered_p50_ms"), "uplink_tx_per_100": round(r["uplink_slots_per_100"], 2), "never_pct": round(r["never_pct"], 2)})
write("latency", lat, list(lat[0].keys()))
