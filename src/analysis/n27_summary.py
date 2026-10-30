#!/usr/bin/env python3
"""N27: the large NRx fine-tuned at the carrier of the evaluation (exp/n27_train.sh: UMi at 3.5 GHz, from the published
weights trained at 2.14 GHz) against the published weights, on the same slots (the *_f35t traces repeat the seeds of the
*_f35 traces with the fine-tuned engine).  Per cell type: first failures, failed TBs the recovery decodes (in total and
per Eb/No), and the retransmissions per 100 TBs of the conventional receiver and of Reverb.
usage: n27_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import sys

import numpy as np

PAIRS = {"macro": "n3_uma_v3_cb_f35", "small": "n1_umi_v3_cb_f35"}


def trace(name):
    z = [np.load(f) for f in sorted(glob.glob(f"results/{name}/c*.npz"))]
    conv = np.concatenate([x["conv_t0"].reshape(-1) for x in z])
    rec = np.concatenate([x["nrx_c0"].reshape(-1) for x in z])
    point = np.concatenate([np.repeat(x["point"], x["conv_t0"].shape[1]) for x in z])
    s = json.load(open(f"results/{name}.json"))["all"]
    f = ~conv
    return {"first_failure_pct": 100.0 * f.mean(), "recovered_pct_of_failed": 100.0 * (rec & f).sum() / f.sum(),
            "recovered_pct_by_ebno": {f"{q:g}": 100.0 * (rec & f)[point == q].sum() / max(1, f[point == q].sum())
                                      for q in sorted(set(point.tolist()))},
            "conventional_retx_per_100": s["ai_and_ran"]["resends_per_100"], "reverb_retx_per_100": s["ours"]["resends_per_100"],
            "retx_cut": s["ai_and_ran"]["resends_per_100"] / s["ours"]["resends_per_100"]}


def main():
    out = {}
    for cell, name in PAIRS.items():
        out[cell] = {"published": trace(name), "fine_tuned": trace(name + "t")}
        for k, r in out[cell].items():
            print(f"{cell:5s} {k:10s} first failures {r['first_failure_pct']:.1f}%, recovered {r['recovered_pct_of_failed']:.1f}% "
                  f"of failed, retransmissions {r['conventional_retx_per_100']:.2f} -> {r['reverb_retx_per_100']:.2f} per 100 TBs "
                  f"({r['retx_cut']:.2f}x)")
    path = sys.argv[1] if len(sys.argv) > 1 else "results/n27_summary_f35.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
