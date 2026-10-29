#!/usr/bin/env python3
"""Numbers of the Reverb paper: every number the manuscript states is a LaTeX macro written here from the stored
results, so the text never carries a typed number (paper repo rule 2.1).  The paper repo copies the output with
scripts/update_numbers.sh; a rerun must reproduce the file byte for byte.

  python3 src/analysis/paper_numbers.py        (from the repository root; numpy only)
  outputs: results/paper/numbers.tex (macros), results/paper/numbers.json (values, sources, checks)

Macros (value, source):
  \\RetxCutFullWait     retransmissions per 100 TBs, conventional receiver only / Reverb waiting for every recovery,
                       macro cell, the 20,480 TBs of n3_uma_v3_cb + n3_uma_v3_more (results/n3_uma_v3_all.json)
  \\RetxCutNoDelay      the same with the latency-neutral rule (wait only when at most 5 of 10 code blocks failed),
                       from the latency simulation (src/sim/harq_latency.py, DDDSU, L1 1.4 ms, recovery 6.3 ms,
                       recovery deadline 11.5 ms, lead 4.0 ms) on the same traces; checked: no added p99 delay for
                       the TBs recovery does not decode
  \\LostCutFullWait     TBs never received after four transmissions, conventional / Reverb waiting for every recovery
  \\ThroughputGainLA    goodput under outer-loop link adaptation (target 10%), Reverb / conventional, mean over the
                       SNR points 6-16 dB, macro cell (results/n6_uma_v3_olla.json)
  \\PairingCutBestSixteen  retransmissions per 100 TBs, conventional receiver with the best partner among 16
                       candidates by MMSE sum rate (results/p2b_uma_u16_pf.json) / Reverb with a random partner
                       (results/p2b_uma_u8_random.json), macro cell, the same seeds
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src/sim")
from harq_latency import simulate  # noqa: E402

OUT = "results/paper"
MACRO_SETS = ["n3_uma_v3_cb", "n3_uma_v3_more"]


class Timing:
    l1, run, deadline, lead = 1.4, 6.3, 11.5, 4.0


def load_traces(sets):
    parts = []
    for s in sets:
        good, total = {}, None
        for f in sorted(glob.glob(f"results/{s}/c*.npz")):
            d = np.load(f)
            total = int(d["total"])
            for k in d.files:
                if k.startswith(("conv_", "sep_", "nrx_")):
                    good.setdefault(k, []).append(d[k].reshape(-1))
            good.setdefault("cbfail", []).append(d["cbfail_t0"].reshape(-1))
        parts.append(({k: np.concatenate(v) for k, v in good.items()}, total))
    total = parts[0][1]

    def cat(k):
        return np.concatenate([g[k] for g, _ in parts])
    conv, rec, cbfail = cat("conv_t0"), cat("nrx_c0"), cat("cbfail")
    steps = [None] + [cat(f"sep_c{t}") | cat(f"nrx_c{t}") for t in range(1, total)]
    conv_steps = [None] + [cat(f"sep_c{t}") for t in range(1, total)]
    return conv, rec, steps, conv_steps, (cbfail >= 0) & (cbfail <= 5)


def latency_policies():
    conv, rec, steps, conv_steps, cand = load_traces(MACRO_SETS)
    out = {}
    for pol in ("harq_only", "wait", "select_cb5"):
        done, tx = simulate(conv, rec, conv_steps if pol == "harq_only" else steps, cand, pol, "DDDSU", Timing)
        unrec = ~conv & ~(rec & (Timing.l1 + Timing.run <= Timing.deadline)) & np.isfinite(done)
        out[pol] = {"retx_per_100": float(100.0 * (tx - 1).sum() / len(tx)),
                    "p99_unrecovered_ms": float(np.percentile(done[unrec], 99)),
                    "tbs": int(len(tx))}
    return out


def main():
    allstats = json.load(open("results/n3_uma_v3_all.json"))["all"]
    lat = latency_policies()
    olla = json.load(open("results/n6_uma_v3_olla.json"))["results"]
    snr = sorted({float(k.split("_")[0][3:]) for k in olla})
    goodput = [olla[f"snr{x:g}_target0.1"]["ours"]["goodput_vs_ai_and_ran"] for x in snr]
    pf16 = json.load(open("results/p2b_uma_u16_pf.json"))["all"]
    rand = json.load(open("results/p2b_uma_u8_random.json"))["all"]

    conv_retx, ours_retx = allstats["ai_and_ran"]["resends_per_100"], allstats["ours"]["resends_per_100"]
    added = lat["select_cb5"]["p99_unrecovered_ms"] - lat["harq_only"]["p99_unrecovered_ms"]
    checks = {
        "latency_sim_full_wait_equals_harq_stats": abs(lat["wait"]["retx_per_100"] - ours_retx) < 1e-9,
        "latency_sim_conventional_equals_harq_stats": abs(lat["harq_only"]["retx_per_100"] - conv_retx) < 1e-9,
        "no_added_delay_latency_neutral": added <= 0.05,
    }
    values = {
        "RetxCutFullWait": (conv_retx / ours_retx, 1),
        "RetxCutNoDelay": (lat["harq_only"]["retx_per_100"] / lat["select_cb5"]["retx_per_100"], 1),
        "LostCutFullWait": (allstats["ai_and_ran"]["residual_after_all_pct"] / allstats["ours"]["residual_after_all_pct"], 1),
        "ThroughputGainLA": (float(np.mean(goodput)), 2),
        "PairingCutBestSixteen": (pf16["ai_and_ran"]["resends_per_100"] / rand["ours"]["resends_per_100"], 1),
    }
    os.makedirs(OUT, exist_ok=True)
    report = {"values": {k: v for k, (v, _) in values.items()},
              "inputs": {"retx_per_100": {"conventional": conv_retx, "reverb_full_wait": ours_retx,
                                          "latency_policies": lat},
                         "lost_pct": {"conventional": allstats["ai_and_ran"]["residual_after_all_pct"],
                                      "reverb_full_wait": allstats["ours"]["residual_after_all_pct"]},
                         "goodput_vs_conventional_by_snr": dict(zip([f"{x:g}" for x in snr], goodput)),
                         "pairing_retx_per_100": {"conventional_best_of_16_sum_rate": pf16["ai_and_ran"]["resends_per_100"],
                                                  "reverb_random_partner": rand["ours"]["resends_per_100"]},
                         "added_p99_delay_latency_neutral_ms": added},
              "checks": checks}
    json.dump(report, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
    lines = ["% Generated by airan_recovery/src/analysis/paper_numbers.py from the stored results.  Do not edit.",
             "% Values, sources and checks: airan_recovery/results/paper/numbers.json"]
    for k, (v, digits) in values.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v:.{digits}f}}}")
    open(f"{OUT}/numbers.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        print("CHECKS FAILED:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
