#!/usr/bin/env python3
"""v40b: v40 re-decoded with the small neural receiver as well (AI-for-RAN: the small model in place of the
conventional receiver, HARQ with its own combined LLRs), and per 100 blocks the resends of the three structures.
v40, realistic recovery against retransmission: decode the first reception and the HARQ resend 7.5 ms later on the
time-correlated UMi channel (nv_retx_dataset2.py) with the runtime receivers.  The helpers are those of v39b:

For one set of nv_retx_dataset.py (first reception, a resend on the same channel, their average = chase combining on
the same channel, a resend on a new channel; built into runtime datasets with the same slot order) this decodes every
reception with the receivers Antiphase runs, LDPC 20 iterations each:
  conv   the cuPHY PUSCH pipeline (fixed 20 iterations)
  sep    the conventional front end from the separable pyAerial components, whose LLRs can be added (soft combining)
  nrx    the large neural receiver (TensorRT engine + LDPC), whose LLRs can be added as well
and soft-combines the first reception with each resend by adding LLRs (chase combining of HARQ, RV 0):
  sep_comb_same / sep_comb_new    the conventional receiver with HARQ combining
  nrx_comb_same / nrx_comb_new    the neural receiver on the combined LLRs
Of the TBs the conventional receiver fails at the first reception, it reports per scenario the share decoded without a
resend, after one resend, and still failed after one resend:
  harq          today: conventional receiver, HARQ resend with combining
  recovery      recovery first (neural receiver on the first reception), then HARQ for the rest
  recovery_both recovery first, then HARQ, and the neural receiver also on the combined resend
usage: probe_retx.py --root DIR --set NAME --engine TRT [--count N] --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from aerial.phy5g.ldpc import LdpcDecoder
from aerial import pycuphy

from cell_ring import CellRing
from probe_ldpc import SeparableConv, fixed_iteration_conv
from slot_radio import nrx_path, result_masks

ITER = 20
KINDS = ("first", "retx")


def nrx_llrs(nrx, rx_slot, slot: int) -> cp.ndarray:
    """LLRs of the neural receiver for one reception: (bit, UE, subcarrier, data symbol), float32 copy."""
    with nrx.stream:
        estimate = nrx._ls_fast(rx_slot, slot)
        window = rx_slot[None, ...]
        inputs = nrx.engine.inputs
        cp.copyto(inputs["rx_slot_real"], window.real)
        cp.copyto(inputs["rx_slot_imag"], window.imag)
        cp.copyto(inputs["h_hat_real"], estimate.real)
        cp.copyto(inputs["h_hat_imag"], estimate.imag)
        outputs = nrx.engine.launch(use_graph=True)
        llrs = cp.take(outputs["output_1"][0, ...], nrx.data_symbols, axis=3).astype(cp.float32)
    nrx.stream.synchronize()
    return llrs


def nrx_decode(nrx, llrs: cp.ndarray) -> tuple[bool, list]:
    """The bit-level chain of MuNrxPath._run_fast on given LLRs."""
    with nrx.stream:
        for u in range(nrx.num_ue):
            nrx.padded[u][:nrx.bits, :] = cp.transpose(llrs[:, u], (0, 2, 1)).reshape(nrx.bits, -1).astype(cp.float16)
        c = nrx.chain
        coded = nrx.derate.pycuphy_ldpc_derate_match.derate_match(
            [pycuphy.CudaArrayHalf(a) for a in nrx.padded], c["tb"], c["rate"], c["length"], c["mod"],
            c["layers"], c["rv"], c["ndi"], c["cinit"], c["group"])
        decoded = nrx.decoder.pycuphy_ldpc_decoder.decode(coded, c["tb"], c["rate"], c["rv"], c["length"])
        blocks = [cp.asarray(d) for d in decoded]
        if nrx.crc_input is None:
            nrx.crc_input = cp.zeros((8448, sum(b.shape[1] for b in blocks)), dtype=cp.float16, order="F")
        column = 0
        for b in blocks:
            nrx.crc_input[:nrx.cb_size, column:column + b.shape[1]] = b[:nrx.cb_size, :]
            column += b.shape[1]
        payloads = nrx.crc.crc_checker.check_crc(pycuphy.CudaArrayHalf(nrx.crc_input), c["tb"], c["rate"])
        crcs = nrx.crc.crc_checker.get_tb_crcs()
    nrx.stream.synchronize()
    mask, out = 0, []
    for u in range(nrx.num_ue):
        out.append(cp.asnumpy(cp.asarray(payloads[u])))
        if int(cp.asnumpy(cp.asarray(crcs[u])).reshape(-1)[0]) == 0:
            mask |= 1 << u
    nrx.last_crc_mask = mask
    return mask == (1 << nrx.num_ue) - 1, out


def sep_llrs(sep, rx_slot, slot: int) -> list:
    """LLRs of the separable conventional receiver, copied on its own stream after it finished (the stream does not
    wait for the default stream, so every copy and sum stays on it)."""
    with sep.stream:
        out = [cp.array(x, copy=True) for x in sep.llrs(rx_slot, slot)]
    sep.stream.synchronize()
    return out


def on(stream, fn):
    """Run fn() on the stream of the receivers and wait for it."""
    with stream:
        out = fn()
    stream.synchronize()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--small-engine", required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads((args.root / f"rt_{args.set}_first" / "nv_mu2_meta.json").read_text())
    count = min(args.count or meta["count"], meta["count"])
    rings = {k: CellRing(args.root / f"rt_{args.set}_{k}", "nv_mu2", count, 0) for k in KINDS}
    first = rings["first"]
    profile, num_ue, tb = first.profile, first.profile.num_ue, first.tb_bytes
    # per-slot similarity of the two users and channel change, in the order of the runtime dataset
    raw = np.load(args.root / f"{args.set}_first.npz")
    order = np.random.default_rng(int(meta.get("shuffle_seed", 73001))).permutation(len(raw["ebno"]))
    corr = raw["corr_first"][order][:count]
    change = raw["channel_change"][order][:count]
    ebno = raw["ebno"][order][:count]
    assert np.allclose(ebno, np.asarray(meta["esno_db"][:count]), atol=1e-3), "slot order differs from the dataset"

    stream = cp.cuda.Stream(non_blocking=True)
    conv = fixed_iteration_conv(profile, tb, stream, ITER)
    nrx = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": ITER}, profile, tb, stream)
    nrx.decoder = LdpcDecoder(num_iterations=ITER, cuda_stream=int(stream.ptr))
    sep = SeparableConv(profile, tb, stream, ITER)
    small = nrx_path({"engine": args.small_engine, "nrx_ldpc_iterations": ITER}, profile, tb, stream)
    small.decoder = LdpcDecoder(num_iterations=ITER, cuda_stream=int(stream.ptr))
    names = ["conv_first", "conv_retx", "nrx_first", "nrx_retx", "sep_first", "sep_retx", "sep_comb", "nrx_comb",
             "small_first", "small_retx", "small_comb"]
    good = {n: np.zeros((count, num_ue), dtype=bool) for n in names}

    def put(name, path, ok, payload, i):
        _, mask = result_masks(path, ok, payload, first.blocks[i], tb)
        good[name][i] = [(mask >> u) & 1 for u in range(num_ue)]

    for i in range(count):
        slot = first.slots[i]
        for k in KINDS:
            ok, payload = conv.run(rings[k].views[i], slot)
            put(f"conv_{k}", conv, ok, payload, i)
            ok, payload = nrx.run(rings[k].views[i], slot)
            put(f"nrx_{k}", nrx, ok, payload, i)
        s1 = sep_llrs(sep, first.views[i], slot)
        s2 = sep_llrs(sep, rings["retx"].views[i], slot)
        ok, payload = sep.decode(on(stream, lambda: [x.copy() for x in s1])); put("sep_first", sep, ok, payload, i)
        ok, payload = sep.decode(on(stream, lambda: [x.copy() for x in s2])); put("sep_retx", sep, ok, payload, i)
        ok, payload = sep.decode(on(stream, lambda: [a + b for a, b in zip(s1, s2)])); put("sep_comb", sep, ok, payload, i)
        n1 = nrx_llrs(nrx, first.views[i], slot)
        n2 = nrx_llrs(nrx, rings["retx"].views[i], slot)
        ok, payload = nrx_decode(nrx, on(stream, lambda: n1 + n2)); put("nrx_comb", nrx, ok, payload, i)
        for k in KINDS:
            ok, payload = small.run(rings[k].views[i], slot)
            put(f"small_{k}", small, ok, payload, i)
        m1 = nrx_llrs(small, first.views[i], slot)
        m2 = nrx_llrs(small, rings["retx"].views[i], slot)
        ok, payload = nrx_decode(small, on(stream, lambda: m1 + m2)); put("small_comb", small, ok, payload, i)

    np.savez(args.output.with_suffix(".npz"), corr=corr, change=change, ebno=ebno, **good)
    slot_corr = np.repeat(corr[:, None], num_ue, axis=1)
    slot_ebno = np.repeat(ebno[:, None], num_ue, axis=1)

    def summary(sel):
        failed = ~good["conv_first"] & sel
        n_fail = int(failed.sum())
        out = {"tbs": int(sel.sum()), "conv_first_fail_pct": 100 * n_fail / max(1, int(sel.sum())), "conv_failed": n_fail,
               "of_conv_failures_pct": {n: 100 * float((good[n] & failed).sum()) / max(1, n_fail) for n in names}}
        rec = good["nrx_first"]
        for name, after, no_resend in (("harq", good["sep_comb"], np.zeros_like(rec)),
                                        ("recovery", rec | good["sep_comb"], rec),
                                        ("recovery_both", rec | good["sep_comb"] | good["nrx_comb"], rec)):
            out[name] = {"without_resend_pct": 100 * float((no_resend & failed).sum()) / max(1, n_fail),
                         "after_one_resend_pct": 100 * float((after & failed).sum()) / max(1, n_fail),
                         "still_failed_pct": 100 * float((~after & failed).sum()) / max(1, n_fail),
                         "resends_per_failed_block": float((~no_resend & failed).sum()) / max(1, n_fail)}
        return out

    def per100(sel):
        """Resends per 100 blocks of each structure (first resend, and the second one when the first fails too),
        and the blocks done at the base station without a resend."""
        n = max(1, int(sel.sum()))
        conv_fail = ~good["conv_first"] & sel
        harq_after = good["sep_comb"]
        small_fail = ~good["small_first"] & sel
        rec = good["nrx_first"] & conv_fail
        need = conv_fail & ~good["nrx_first"]
        both_after = good["sep_comb"] | good["nrx_comb"]
        return {
            "ai_and_ran": {"first_fail": 100 * conv_fail.sum() / n, "done_at_bs": 0.0,
                           "first_resends": 100 * conv_fail.sum() / n,
                           "second_resends": 100 * (conv_fail & ~harq_after).sum() / n},
            "ai_for_ran": {"first_fail": 100 * small_fail.sum() / n, "done_at_bs": 0.0,
                           "first_resends": 100 * small_fail.sum() / n,
                           "second_resends": 100 * (small_fail & ~good["small_comb"]).sum() / n},
            "ours": {"first_fail": 100 * conv_fail.sum() / n, "done_at_bs": 100 * rec.sum() / n,
                     "first_resends": 100 * need.sum() / n,
                     "second_resends": 100 * (need & ~both_after).sum() / n}}

    every = np.ones_like(good["conv_first"])
    report = {"set": args.set, "slots": count, "ldpc_iterations": ITER, "decoded": {n: int(good[n].sum()) for n in names}, "speed": json.loads(str(raw["meta"]))["speed"],
              "change_median": float(np.median(change)), "corr_median": float(np.median(corr)),
              "all": summary(every), "per100": per100(every),
              "per100_by_similarity": {f"{lo:.1f}-{hi:.1f}": per100((slot_corr >= lo) & (slot_corr < hi))
                                       for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))},
              "by_ebno": {f"{e:g}": summary(slot_ebno == e) for e in sorted(set(ebno.tolist()))},
              "by_similarity": {f"{lo:.1f}-{hi:.1f}": summary((slot_corr >= lo) & (slot_corr < hi))
                                for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))},
              "slots_by_similarity": {f"{lo:.1f}-{hi:.1f}": int(((corr >= lo) & (corr < hi)).sum())
                                      for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 1.01))}}
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print("| structure | first fail /100 | done at the base station /100 | first resends /100 | second resends /100 |")
    print("|---|---|---|---|---|")
    for k, v in report["per100"].items():
        print(f"| {k} | {v['first_fail']:.1f} | {v['done_at_bs']:.1f} | {v['first_resends']:.1f} | {v['second_resends']:.1f} |")
    print(f"set {args.set}: {count} slots, speed {report['speed']} m/s, channel change in 7.5 ms (median) "
          f"{report['change_median']:.3f}, user similarity (median) {report['corr_median']:.2f}")
    print("| part | TBs | first fail | conv resend alone | HARQ combined | recovery | NRx on combined | "
          "HARQ: still failed | recovery: no resend / still failed | recovery also on resend: still failed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    parts = [("all", report["all"])] + [(f"Eb/No {k}", v) for k, v in report["by_ebno"].items()] + \
            [(f"similarity {k}", v) for k, v in report["by_similarity"].items()]
    for name, r in parts:
        f = r["of_conv_failures_pct"]
        print(f"| {name} | {r['tbs']} | {r['conv_first_fail_pct']:.1f}% | {f['conv_retx']:.0f}% | {f['sep_comb']:.0f}% | "
              f"{f['nrx_first']:.0f}% | {f['nrx_comb']:.0f}% | {r['harq']['still_failed_pct']:.0f}% | "
              f"{r['recovery']['without_resend_pct']:.0f}% / {r['recovery']['still_failed_pct']:.0f}% | "
              f"{r['recovery_both']['still_failed_pct']:.0f}% |")


if __name__ == "__main__":
    main()
