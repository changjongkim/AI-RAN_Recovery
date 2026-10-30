#!/usr/bin/env python3
"""Decode the slots of gen_harq.py and their retransmissions with the runtime receivers, with IR HARQ combining.

Receivers (LDPC iterations fixed, 20 by default, the same for all):
  conv    the cuPHY PUSCH pipeline (first transmission only; this is what the base station runs on every block)
  sep     the conventional front end from the separable pyAerial components (its LLRs can be combined)
  nrx     the large neural receiver (TensorRT engine)
  small   the small real-time neural receiver (TensorRT engine)
IR combining: the LLRs of every reception are rate-recovered with that reception's redundancy version into the
mother-code buffer (cuPHY derate-match output, N_cb + 2 Zc per code block) and summed; the decoder then runs on the sum.
For a sum of more than one reception the decoder is given RV 3 so that it uses every parity node the sum may cover
(cuPHY sets the parity nodes from E + k0(RV)); a single reception is decoded with its own RV.

Per slot and user it stores (results <output>.npz, summary <output>.json):
  conv_t0                     conventional receiver, first transmission
  <rx>_c<t>  (rx = sep, nrx, small)   after combining transmissions 0..t
  <rx>_s<t>  (t >= 1)         that retransmission alone (is the RV self-decodable at this SNR)
the conventional receiver's failed code blocks per user (cbfail_t0), and the summary counts, per structure, how many transmissions each block needs:
  ai_and_ran   conventional receiver, HARQ with the conventional receiver
  ai_for_ran   small neural receiver inside L1 on every block, HARQ with it
  ours         conventional receiver, then recovery by the large neural receiver on the first reception (no
               retransmission if it succeeds), then HARQ where either receiver may decode the combined buffer
  ours_conv_harq   as ours, but HARQ with the conventional receiver only
With retransmissions sent alone (gen_harq.py --retx-alone, read from the generator's metadata) the retransmissions
are received as one-user slots and only user 0's blocks are counted.
Recovery receivers on top of the MMSE-IRC L1 (both off by default):
  --whiten   with --irc: the large neural receiver on the slot whitened with the IRC estimate of the noise-plus-
             interference covariance, per PRB as the IRC equalizer does (nrxw) and averaged over the band (nrxwb)
  --cb-mix   a TB decoded from code blocks of two receivers: every block whose CRC failed in the first and passed
             in the second is taken from the second; conventional and neural (mixsep), with --irc IRC and neural
             (mixirc), with --whiten IRC and whitened neural (mixw)
usage: decode_harq.py --root DIR --set NAME --mcs 14 --engine LARGE --small-engine SMALL [--count N] --output OUT
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cupy as cp
import numpy as np

from aerial import pycuphy
from aerial.phy5g.ldpc import LdpcDecoder

from cell_ring import CellRing
from probe_ldpc import SeparableConv, fixed_iteration_conv
from slot_radio import nrx_path, result_masks
from harq_stats import report, table



def on(stream, fn):
    with stream:
        out = fn()
    stream.synchronize()
    return out


def nrx_llrs(nrx, rx_slot, slot: int) -> cp.ndarray:
    """LLRs of a neural receiver for one reception: (bit, UE, subcarrier, data symbol), float32 copy."""
    with nrx.stream:
        estimate = nrx._ls_fast(rx_slot, slot)
        inputs = nrx.engine.inputs
        cp.copyto(inputs["rx_slot_real"], rx_slot[None, ...].real)
        cp.copyto(inputs["rx_slot_imag"], rx_slot[None, ...].imag)
        cp.copyto(inputs["h_hat_real"], estimate.real)
        cp.copyto(inputs["h_hat_imag"], estimate.imag)
        outputs = nrx.engine.launch(use_graph=True)
        llrs = cp.take(outputs["output_1"][0, ...], nrx.data_symbols, axis=3).astype(cp.float32)
    nrx.stream.synchronize()
    return llrs


def whitened(view: cp.ndarray, lw_inv: cp.ndarray, wideband: bool) -> cp.ndarray:
    """The slot whitened with the IRC estimate of the noise-plus-interference covariance R = L L^H (cuPHY returns L^-1
    per PRB, (antenna, antenna, PRB), and its MMSE-IRC equalizer applies it as L^-1 y), scaled so that the whitened
    noise plus interference keeps the average per-antenna power of the original: interference becomes white noise."""
    w = cp.transpose(lw_inv, (2, 0, 1))                                   # (PRB, antenna, antenna)
    prbs, ant = w.shape[0], w.shape[1]
    assert view.shape[0] == 12 * prbs, "the whitening needs the allocation over the whole band"
    inv = cp.linalg.inv(w)
    cov = inv @ cp.conj(cp.transpose(inv, (0, 2, 1)))                     # R per PRB
    scale = cp.sqrt(cp.trace(cov, axis1=1, axis2=2).real.mean() / ant)
    if wideband:
        out = cp.einsum("ij,ktj->kti", cp.linalg.inv(cp.linalg.cholesky(cov.mean(axis=0))), view)
    else:
        out = cp.einsum("pij,pktj->pkti", w, view.reshape(prbs, 12, view.shape[1], ant)).reshape(view.shape)
    return cp.asfortranarray((out * scale).astype(cp.complex64))


class Combiner:
    """Rate recovery with a chosen RV, buffer sums and the LDPC + CRC chain, on the bit-level objects of a
    neural-receiver path (the chain constants depend only on the TB size and the MCS, not on the receiver)."""

    def __init__(self, nrx, sep, iterations: int) -> None:
        self.nrx, self.sep = nrx, sep
        self.users = nrx.num_ue
        self.decoder = LdpcDecoder(num_iterations=iterations, cuda_stream=int(nrx.stream.ptr))
        self.sep_cfg = {}

    def _sep_configs(self, rv: int):
        if rv not in self.sep_cfg:
            cfgs = copy.deepcopy(self.sep.per_ue)
            for cfg in cfgs:
                for ue in cfg.ue_configs:
                    ue.rv, ue.ndi = rv, 1
            self.sep_cfg[rv] = cfgs
        return self.sep_cfg[rv]

    def derate_nrx(self, llrs: cp.ndarray, rv: int) -> list:
        n = self.nrx
        c = n.chain
        with n.stream:
            for u in range(self.users):
                n.padded[u][:n.bits, :] = cp.transpose(llrs[:, u], (0, 2, 1)).reshape(n.bits, -1).astype(cp.float16)
            out = n.derate.pycuphy_ldpc_derate_match.derate_match(
                [pycuphy.CudaArrayHalf(a) for a in n.padded], c["tb"], c["rate"], c["length"], c["mod"],
                c["layers"], [np.uint32(rv)] * self.users, [np.uint32(1)] * self.users, c["cinit"], c["group"])
            out = [cp.array(a, dtype=cp.float32) for a in out]
        n.stream.synchronize()
        return out

    def derate_sep(self, llrs: list, rv: int, alone: bool = False) -> list:
        s = self.sep
        with s.stream:
            if alone:      # equalizer output of the one-user configuration; the other user's buffer stays zero
                u0 = cp.ascontiguousarray(llrs[0][:, 0:1])
                per_ue = [u0] + [cp.zeros_like(u0) for _ in range(1, self.users)]
            else:
                per_ue = [cp.ascontiguousarray(llrs[0][:, u:u + 1]) for u in range(self.users)]
            out = s.derate.derate_match(input_llrs=per_ue, pusch_configs=self._sep_configs(rv))
            out = [cp.array(a, dtype=cp.float32) for a in out]
        s.stream.synchronize()
        return out

    def decode(self, buffers: list, rv: int, reference, tb_bytes: int) -> int:
        """LDPC + CRC on summed buffers; returns the per-user mask of correct blocks."""
        n = self.nrx
        c = n.chain
        with n.stream:
            half = [cp.asfortranarray(cp.clip(b, -60000, 60000).astype(cp.float16)) for b in buffers]
            decoded = self.decoder.pycuphy_ldpc_decoder.decode(
                [pycuphy.CudaArrayHalf(a) for a in half], c["tb"], c["rate"], [np.uint32(rv)] * self.users, c["length"])
            blocks = [cp.asarray(d) for d in decoded]
            if n.crc_input is None:
                n.crc_input = cp.zeros((8448, sum(b.shape[1] for b in blocks)), dtype=cp.float16, order="F")
            column = 0
            for b in blocks:
                n.crc_input[:n.cb_size, column:column + b.shape[1]] = b[:n.cb_size, :]
                column += b.shape[1]
            payloads = n.crc.crc_checker.check_crc(pycuphy.CudaArrayHalf(n.crc_input), c["tb"], c["rate"])
            crcs = n.crc.crc_checker.get_tb_crcs()
        n.stream.synchronize()
        return self._masks(payloads, crcs, reference, tb_bytes)

    def snapshot(self):
        """The code blocks of the last decode and their block CRCs (0 = passed), for mixed()."""
        n = self.nrx
        with n.stream:
            blocks = n.crc_input.copy(order="F")
            cbs = cp.concatenate([cp.asarray(x).reshape(-1).copy() for x in n.crc.crc_checker.get_cb_crcs()])
        n.stream.synchronize()
        return blocks, cbs

    def mixed(self, a, b, reference, tb_bytes: int) -> int:
        """CRC of the blocks of snapshot a, with every block whose CRC failed in a and passed in b taken from b."""
        n = self.nrx
        c = n.chain
        with n.stream:
            n.crc_input[...] = a[0]
            take = cp.nonzero((a[1] != 0) & (b[1] == 0))[0]
            n.crc_input[:, take] = b[0][:, take]
            payloads = n.crc.crc_checker.check_crc(pycuphy.CudaArrayHalf(n.crc_input), c["tb"], c["rate"])
            crcs = n.crc.crc_checker.get_tb_crcs()
        n.stream.synchronize()
        return self._masks(payloads, crcs, reference, tb_bytes)

    def _masks(self, payloads, crcs, reference, tb_bytes: int) -> int:
        n = self.nrx
        mask, out = 0, []
        for u in range(self.users):
            out.append(cp.asnumpy(cp.asarray(payloads[u])))
            if int(cp.asnumpy(cp.asarray(crcs[u])).reshape(-1)[0]) == 0:
                mask |= 1 << u
        n.last_crc_mask = mask
        return result_masks(n, mask == (1 << self.users) - 1, out, reference, tb_bytes)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--mcs", type=int, default=14)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--small-engine", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--irc", action="store_true",
                        help="also receive with the conventional front end using MMSE-IRC (interference rejection combining)")
    parser.add_argument("--whiten", action="store_true",
                        help="with --irc: also the large neural receiver on the IRC-whitened slot (nrxw, nrxwb)")
    parser.add_argument("--cb-mix", action="store_true",
                        help="also TBs decoded from the code blocks of two receivers (mixsep, mixirc, mixw)")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = f"nv_mu2_m{args.mcs}"
    info = np.load(args.root / f"{args.set}_info.npz")
    gmeta = json.loads(str(info["meta"]))
    rvs = gmeta["rvs"]
    total = len(rvs)
    dirs = [args.root / f"rt_{args.set}_m{args.mcs}_t{t}" for t in range(total)]
    meta = json.loads((dirs[0] / f"{profile}_meta.json").read_text())
    count = min(args.count or meta["count"], meta["count"])
    rings = [CellRing(d, profile, count, 0) for d in dirs]
    first = rings[0]
    prof, users, tb = first.profile, first.profile.num_ue, first.tb_bytes

    # per-slot channel information in the order of the runtime datasets (same shuffle as nv_build_dataset.py)
    order = np.random.default_rng(int(meta.get("shuffle_seed", 73001))).permutation(len(info["point"]))[:count]
    point, corr, change = info["point"][order], info["corr"][order], info["change"][order]
    gain = info["gain"][order]
    extra = {"sir": info["sir"][order]} if "sir" in info.files else {}   # per slot and transmission, with interferers
    assert np.allclose(point, np.asarray(meta["esno_db"][:count]), atol=1e-3), "slot order differs from the dataset"

    stream = cp.cuda.Stream(non_blocking=True)
    conv = fixed_iteration_conv(prof, tb, stream, args.iterations)
    sep = SeparableConv(prof, tb, stream, args.iterations)
    big = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": args.iterations}, prof, tb, stream)
    small = nrx_path({"engine": args.small_engine, "nrx_ldpc_iterations": args.iterations}, prof, tb, stream)
    comb = {"sep": Combiner(big, sep, args.iterations), "nrx": Combiner(big, sep, args.iterations),
            "small": Combiner(small, sep, args.iterations)}
    # MMSE-IRC conventional receiver (equalizer algorithm 2: noise-plus-interference covariance from the DMRS), the usual
    # base-station answer to inter-cell interference; cuPHY's default (algorithm 1) uses the noise variance only.
    irc = None
    if args.irc and not gmeta.get("retx_alone", False):
        from aerial.phy5g.algorithms import ChannelEqualizer, NoiseIntfEstimator
        from ul_profiles import NUM_RX_ANT
        irc = SeparableConv(prof, tb, stream, args.iterations)
        irc.noise = NoiseIntfEstimator(num_rx_ant=NUM_RX_ANT, eq_coeff_algo=2, cuda_stream=int(stream.ptr))
        irc.equalizer = ChannelEqualizer(num_rx_ant=NUM_RX_ANT, enable_pusch_tdi=prof.enable_pusch_tdi, eq_coeff_algo=2,
                                         cuda_stream=int(stream.ptr))
        comb["irc"] = Combiner(big, sep, args.iterations)
        if args.whiten:
            comb["nrxw"] = Combiner(big, sep, args.iterations)
            comb["nrxwb"] = Combiner(big, sep, args.iterations)

    # Retransmissions sent alone (gen_harq.py --retx-alone): the conventional front end is configured for user 0 only
    # (same DMRS port, scrambling and CDM groups as in the two-user slot), the neural receivers get user 1 masked out
    # (the NVlabs models are trained with one or two active users), and only user 0's blocks are counted.
    alone = bool(gmeta.get("retx_alone", False))
    if alone:
        from dataclasses import replace
        from mu_radio import mu_pusch_configs
        sep1 = SeparableConv(prof, tb, stream, args.iterations)
        sep1.configs, sep1.per_ue = mu_pusch_configs(replace(prof, ue_dmrs_ports=prof.ue_dmrs_ports[:1]), tb)

    def active(ports):
        with stream:
            for n in (big, small):
                a = n.engine.inputs["active_dmrs_ports"]
                a[...] = cp.asarray([ports], dtype=a.dtype)
        stream.synchronize()

    # code-block mixes (first receiver, second receiver), decoded after combining transmissions 0..t
    mixes = {}
    if args.cb_mix and not alone:
        mixes["mixsep"] = ("sep", "nrx")
        if irc is not None:
            mixes["mixirc"] = ("irc", "nrx")
        if "nrxw" in comb:
            mixes["mixw"] = ("irc", "nrxw")
    parts = {r for pair in mixes.values() for r in pair}

    names = ["conv_t0"] + [f"{r}_c{t}" for r in comb for t in range(total)] + \
            [f"{r}_s{t}" for r in comb for t in range(1, total)] + [f"{m}_c{t}" for m in mixes for t in range(total)]
    good = {k: np.zeros((count, users), dtype=bool) for k in names}

    def mark(name, mask, i):
        good[name][i] = [(mask >> u) & 1 for u in range(users)]

    # failed code blocks per user in the conventional receiver's first decode (a predictor of whether recovery will
    # succeed, known when L1 finishes; -1 where the path does not report it)
    cbfail = np.full((count, users), -1, dtype=np.int16)

    for i in range(count):
        slot, ref = first.slots[i], first.blocks[i]
        ok, payload = conv.run(rings[0].views[i], slot)
        mark("conv_t0", result_masks(conv, ok, payload, ref, tb)[1], i)
        by_ue = getattr(conv, "last_cb_fail_by_ue", None)
        if by_ue is not None and len(by_ue) == users:
            cbfail[i] = by_ue
        sums = {}
        for t, rv in enumerate(rvs):
            view = rings[t].views[i]
            solo = alone and t >= 1
            if alone and t <= 1:
                active([1, 0] if solo else [1, 1])
            front = sep1 if solo else sep
            buffers = {
                "sep": comb["sep"].derate_sep(on(stream, lambda: [cp.array(x, copy=True) for x in front.llrs(view, slot)]),
                                              rv, solo),
                "nrx": comb["nrx"].derate_nrx(nrx_llrs(big, view, slot), rv),
                "small": comb["small"].derate_nrx(nrx_llrs(small, view, slot), rv),
            }
            if irc is not None:
                buffers["irc"] = comb["irc"].derate_sep(on(stream, lambda: [cp.array(x, copy=True) for x in irc.llrs(view, slot)]), rv)
                if "nrxw" in comb:      # irc.llrs left this slot's IRC estimate in irc.noise.lw_inv
                    for r, wide in (("nrxw", False), ("nrxwb", True)):
                        white = on(stream, lambda: whitened(view, irc.noise.lw_inv[0], wide))
                        buffers[r] = comb[r].derate_nrx(nrx_llrs(big, white, slot), rv)
            snaps = {}
            for r, buf in buffers.items():
                sums[r] = buf if t == 0 else [a + b for a, b in zip(sums[r], buf)]
                mark(f"{r}_c{t}", comb[r].decode(sums[r], rv if t == 0 else 3, ref, tb), i)
                if r in parts:
                    snaps[r] = comb[r].snapshot()
                if t >= 1:
                    mark(f"{r}_s{t}", comb[r].decode(buf, rv, ref, tb), i)
            for m, (a, b) in mixes.items():
                mark(f"{m}_c{t}", comb[b].mixed(snaps[a], snaps[b], ref, tb), i)

    if alone:
        good = {k: v[:, :1] for k, v in good.items()}
        cbfail = cbfail[:, :1]
    rep = report(good, point, corr, total, {"set": args.set, "mcs": args.mcs, "rvs": rvs, "generator": gmeta,
                                            "ldpc_iterations": args.iterations, "retx_alone": alone})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), point=point, corr=corr, change=change, gain=gain, total=total,
             meta=json.dumps({"set": args.set, "mcs": args.mcs, "rvs": rvs, "tb_bytes": int(tb), "generator": gmeta,
                              "retx_alone": alone}), cbfail_t0=cbfail, **extra, **good)
    args.output.write_text(json.dumps(rep, indent=2))
    print(f"set {args.set} MCS {args.mcs}: {count} slots, similarity median {rep['corr_median']:.2f}, "
          f"separable vs cuPHY first {rep['check']['sep_vs_conv_first']}")
    print(table(rep))


if __name__ == "__main__":
    main()
