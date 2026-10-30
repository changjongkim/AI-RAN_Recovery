#!/usr/bin/env python3
"""N26: check of the load matching of Section 5.6 on real-time runs at a higher recovery load (exp/n26_load_data.sh,
raw results rec16allb68*_m16_<data>).  The MCS 16 slots with interference of neighbor cells fail more TBs with the same
slot size and receivers.  For each dataset and server: the load (failed TBs per TB over the MCS 14 runs), the measured
on-time share against the share that server_effect.matched predicts from the MCS 14 runs at that load, and the same
prediction at the load of the recovery runs (slots with a failed TB per TB), since two failed TBs of a slot share one
recovery run and the runs grow slower than the failed TBs.
usage: RESULT_SUFFIX=_f35 N16_SUFFIX=d110_f35 n26_load_summary.py [OUT.json]       (from the repository root)"""
import glob
import json
import sys

sys.path[:0] = ["src", "src/sim", "src/analysis"]
import paper_numbers as pn  # noqa: E402
import server_effect as se  # noqa: E402

DATA = {"_m16": None, "_m16_wm8": -8.0, "_m16_wm5": -5.0, "_m16_wm3": -3.0, "_m16_w05": 0.5, "_m16_i3": 3.0,
        "_m16_i6": 6.0, "_m16_i10": 10.0}


def runs_per_tb(tag):
    """Failed TBs and recovery runs (slots with a failed TB) per TB over the runs of a tag."""
    slots = failed = runs = 0
    for f in glob.glob(f"{se.RAW}/rec16{tag}s*_c*_*_none_j*.json"):
        d = json.load(open(f))
        skip = int(d["config"]["skip_periods"])
        for cf in glob.glob(f"{f[:-5]}_work/conv*.json"):
            c = json.load(open(cf))
            col = {k: i for i, k in enumerate(c["columns"])}
            for r in c["records"]:
                if int(r[col["period"]]) >= skip:
                    n = bin(~int(r[col["ue_crc_mask"]]) & 0b11).count("1")
                    slots += 1
                    failed += n
                    runs += n > 0
    return failed / max(1, 2 * slots), runs / max(1, 2 * slots)


def main():
    avail = se.availability()
    base = {g: runs_per_tb(t)[1] for g, t in (("one", f"allb68{pn.N16}"), ("four", f"allb68g4{pn.N16}"))}
    out = {}
    for data, inr in DATA.items():
        check = se.check_load_matching(avail, data)
        if not check:
            continue
        rpt = {g: runs_per_tb(t)[1] for g, t in (("one", f"allb68{pn.N16}{data}"), ("four", f"allb68g4{pn.N16}{data}"))}
        for k, v in check.items():
            g = v["gpus"]
            v["run_load"] = rpt[g] / base[g]
            v["predicted_by_runs"] = se.matched(avail, g, v["cells"], v["run_load"])
        out[data] = {"inr_db": inr, "points": check}
        for g in ("four", "one"):
            pts = [v for v in check.values() if v["gpus"] == g]
            d = [100 * (v["measured"] - v["predicted"]) for v in pts]
            dr = [100 * (v["measured"] - v["predicted_by_runs"]) for v in pts]
            print(f"{data:9s} {g:4s} load {min(v['load'] for v in pts):.2f}-{max(v['load'] for v in pts):.2f}x  "
                  f"measured - predicted {min(d):+5.1f}..{max(d):+5.1f} pp  "
                  f"(at the run load {pts[0]['run_load']:.2f}x: {min(dr):+5.1f}..{max(dr):+5.1f} pp)")
    path = sys.argv[1] if len(sys.argv) > 1 else f"results/n26_load_summary{pn.SUFFIX}.json"
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
