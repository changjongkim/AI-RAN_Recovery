#!/usr/bin/env python3
"""Uplink slots with their HARQ retransmissions (incremental redundancy) on a time-correlated 3GPP channel.

For every slot the channel model is sampled once over (number of transmissions - 1) x gap + one slot, and the
frequency response at 0, gap, 2 gap, ... is used for the first transmission and each retransmission, so a
retransmission sees the channel as it has changed at the users' speed.  The same transport block is sent each time with
the redundancy version of the 3GPP order (RV 0, 2, 3, 1 by default; TS 38.214 Table 6.1.2.1-2).  RV r starts the bit
selection of every code block at k0(r) of TS 38.212 Table 5.4.2.1-2 (BG1: 0, 17, 33, 56 Zc; BG2: 0, 13, 25, 43 Zc),
skipping the filler bits, with N_cb = N (no limited buffer).  RV 0 is checked bit for bit against Sionna's TBEncoder.

Writes, per MCS m and transmission index t, <output>_m<m>_t<t>.npz in the format of the runtime dataset builder
(rx: slot, subcarrier, symbol, antenna; payload: packed TB bits per user; meta), and <output>_info.npz with per slot:
  ebno (or SNR), corr[t]   similarity of the two users' channels at transmission t: mean over subcarriers and symbols
                           of |h1^H h2| / (|h1||h2|) over the receive antennas (0 orthogonal, 1 same direction),
  change[t]                1 - |<h_0, h_t>| / (|h_0||h_t|): how much the channel changed by transmission t,
  gain[t]                  mean |h|^2 per user (to separate weak users from hard-to-separate pairs).
Several MCS share the same channel realizations and noise level (--snr-db), so a rate-control simulation can pick an
MCS per slot afterwards.  With --ebno instead, the noise follows the rate-adjusted Eb/No of the first MCS (as NVlabs).
Run with the NVlabs repository (third_party/neural_rx) as working directory's parent: python3 gen_harq.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--nrx-repo", type=Path, required=True)
parser.add_argument("--base-config", default="nrx_large.cfg")
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--channel", default="UMi", help="UMi, UMa, CDL-A..E, TDL-*, DoubleTDLlow/high")
parser.add_argument("--speed", type=float, default=0.83, help="user speed in m/s (min = max)")
parser.add_argument("--los", choices=["auto", "los", "nlos"], default="auto", help="force LoS state (UMi/UMa)")
parser.add_argument("--num-rx", type=int, default=0, help="override the number of receive antennas")
parser.add_argument("--mcs", default="14", help="comma list; all share the channel realizations")
parser.add_argument("--rvs", default="0,2,3,1", help="redundancy version of each transmission")
parser.add_argument("--gap-ms", type=float, default=7.5, help="time between transmissions")
parser.add_argument("--ebno", default="", help="comma list of Eb/No (rate adjusted to the first MCS)")
parser.add_argument("--snr-db", default="", help="comma list of SNR per resource element (same for every MCS)")
parser.add_argument("--slots", type=int, default=100, help="slots per Eb/No or SNR point")
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu", default="0")
parser.add_argument("--check-rv0", action="store_true", help="only verify RV 0 against Sionna's TBEncoder")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(args.nrx_repo))
os.chdir(args.nrx_repo / "scripts")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.get_logger().setLevel("ERROR")
for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)
tf.random.set_seed(args.seed)
np.random.seed(args.seed)

from sionna.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies   # noqa: E402
from utils import Parameters, DataGeneratorAerial                                          # noqa: E402

MCS = [int(m) for m in args.mcs.split(",")]
RVS = [int(r) for r in args.rvs.split(",")]
assert (args.ebno == "") != (args.snr_db == ""), "give exactly one of --ebno and --snr-db"
POINTS = [float(e) for e in (args.ebno or args.snr_db).split(",")]
K0 = {1: {0: 0, 1: 17, 2: 33, 3: 56}, 2: {0: 0, 1: 13, 2: 25, 3: 43}}   # x Zc, N_cb = N


def make_config(mcs: int) -> str:
    text = (Path("../config") / args.base_config).read_text()
    label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
    subs = [(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}"),
            (r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"'),
            (r"max_ut_velocity_eval\s*=.*", f"max_ut_velocity_eval = {args.speed}"),
            (r"min_ut_velocity_eval\s*=.*", f"min_ut_velocity_eval = {args.speed}"),
            (r"(?m)^mcs_index\s*=.*", f"mcs_index = [{mcs}]")]
    if args.num_rx:
        subs.append((r"(?m)^num_rx_antennas\s*=.*", f"num_rx_antennas = {args.num_rx}"))
    for pattern, value in subs:
        text, n = re.subn(pattern, value, text)
        assert n >= 1, pattern
    name = f"rec_{label}_{args.prbs}_{args.channel}_v{args.speed:g}_m{mcs}_r{args.num_rx or 4}_{args.los}.cfg"
    (Path("../config") / name).write_text(text)
    return name


class Link:
    """One MCS: the NVlabs generator, its transmitter and the RV bit selection of its TB encoder."""

    def __init__(self, mcs: int) -> None:
        self.mcs = mcs
        self.params = Parameters(make_config(mcs), training=False, system="nrx", num_tx_eval=2)
        assert not self.params.channel_norm, "the channel must not be normalized per slot"
        assert self.params.frequency_offset is None, "CFO would have to be applied to every retransmission"
        self.gen = DataGeneratorAerial(self.params)
        self.tx = self.params.transmitters[0]
        tbe = self.tx._tb_encoder
        enc = tbe._encoder
        z, k, k_ldpc, n_ldpc = enc._z, enc._k, enc._k_ldpc, enc._n_ldpc
        bg = 1 if enc._bg == "bg1" else 2
        e_max = int(np.max(tbe._cw_lengths))
        n_cb = n_ldpc - 2 * z
        filler = set(range(k - 2 * z, k_ldpc - 2 * z))      # filler positions in d (after the 2 Zc punctured bits)
        self.select = {}
        for rv in sorted(set(RVS) | {0}):
            idx, j = [], K0[bg][rv] * z
            while len(idx) < e_max:
                d = j % n_cb
                if d not in filler:
                    idx.append(d + 2 * z)                      # index into the full codeword c (length n_ldpc)
                j += 1
            self.select[rv] = tf.constant(idx, tf.int32)
        self.bg, self.z = bg, z
        captured = {}
        source = self.gen._source

        def keep_bits(shape):
            captured["b"] = source(shape)
            return captured["b"]
        self.gen._source = keep_bits
        self.captured = captured

    def encode(self, b: tf.Tensor, rv: int) -> tf.Tensor:
        """TBEncoder.call with the bit selection of RV rv."""
        tbe = self.tx._tb_encoder
        enc = tbe._encoder
        u = tf.cast(b, tf.float32)
        if tbe._k_padding > 0:
            s = tf.concat((tf.shape(u)[:-1], [tbe._k_padding]), axis=0)
            u = tf.concat((u, tf.zeros(s, u.dtype)), axis=-1)
        u_crc = tbe._tb_crc_encoder(u)
        u_cb = tf.reshape(u_crc, (-1, tbe._num_tx, tbe._num_cbs, tbe._cb_size - tbe._cb_crc_length))
        u_cb_crc = tbe._cb_crc_encoder(u_cb) if tbe._cb_crc_length == 24 else u_cb
        flat = tf.reshape(u_cb_crc, [-1, enc._k])
        u_fill = tf.concat([flat, tf.zeros([tf.shape(flat)[0], enc._k_ldpc - enc._k], flat.dtype)], 1)
        full = tf.reshape(enc._encode_fast(u_fill), [-1, enc._n_ldpc])
        e = tf.gather(full, self.select[rv], axis=-1)
        c = tf.reshape(e, (-1, tbe._num_tx, tbe._num_cbs * int(np.max(tbe._cw_lengths))))
        c = tf.gather(c, tbe._output_perm, axis=-1)[:, :, :int(np.sum(tbe._cw_lengths))]
        if tbe._use_scrambler:
            c = tbe._scrambler(c)
        return tf.cast(c, tbe.dtype)

    def modulate(self, c: tf.Tensor) -> tf.Tensor:
        """PUSCHTransmitter.call after TB encoding."""
        tx = self.tx
        x = tx._resource_grid_mapper(tx._layer_mapper(tx._mapper(c)))
        if tx._precoding == "codebook":
            x = tx._precoder(x)
        assert tx._output_domain == "freq"
        return x


links = [Link(m) for m in MCS]
first = links[0]

if args.check_rv0:
    b = first.gen._source([4, first.params.max_num_tx, first.tx._tb_size])
    ok = bool(tf.reduce_all(tf.equal(first.encode(b, 0), first.tx._tb_encoder(b))))
    print(json.dumps({"rv0_matches_tbencoder": ok, "bg": first.bg, "zc": first.z,
                      "k0": {rv: int(first.select[rv][0]) - 2 * first.z for rv in first.select}}))
    sys.exit(0 if ok else 1)

model = first.params.channel_model
rg = first.tx._resource_grid
symbols = int(rg.num_ofdm_symbols)
symbol_s = float(rg.ofdm_symbol_duration)
gap = int(round(args.gap_ms * 1e-3 / symbol_s))
span = gap * (len(RVS) - 1) + symbols
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
apply_channel = ApplyOFDMChannel(add_awgn=True)
kept = {}

if args.los != "auto":
    set_topology = model.set_topology

    def forced(*topology, **kw):
        kw["los"] = args.los == "los"
        return set_topology(*topology[:6], **kw)
    model.set_topology = forced


def time_correlated(channel_inputs):
    """Replaces the generator's channel: samples the model over all transmissions, keeps every response."""
    x, no = channel_inputs
    a, tau = model(tf.shape(x)[0], span, 1.0 / symbol_s)
    hs = [cir_to_ofdm_channel(frequencies, a[..., t * gap:t * gap + symbols], tau, False) for t in range(len(RVS))]
    kept["h"], kept["no"] = hs, no
    return apply_channel([x, hs[0], no]), hs[0]


first.gen._channel = time_correlated
first.gen(1, 10.0)          # build


def aerial(y):
    return tf.transpose(tf.squeeze(y, axis=1), (0, 3, 2, 1)).numpy().astype(np.complex64)


def similarity(h):
    """[batch, 1, rx_ant, tx, 1, symbol, subcarrier] -> per slot mean |h1^H h2| / (|h1||h2|)."""
    v = h.numpy()[:, 0, :, :, 0]
    u1, u2 = v[:, :, 0], v[:, :, 1]
    num = np.abs(np.sum(np.conj(u1) * u2, axis=1))
    den = np.linalg.norm(u1, axis=1) * np.linalg.norm(u2, axis=1) + 1e-12
    return (num / den).mean(axis=(1, 2))


def gains(h):
    v = h.numpy()[:, 0, :, :, 0]
    return np.mean(np.abs(v) ** 2, axis=(1, 3, 4))       # [batch, user]


out = {(m, t): [] for m in MCS for t in range(len(RVS))}
bits = {m: [] for m in MCS}
info = {"point": [], "corr": [], "change": [], "gain": []}
for point in POINTS:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        ebno = point if args.ebno else 10.0
        first.gen(n, ebno)                                   # draws topology, bits and channel for MCS[0]
        hs = kept["h"]
        no = kept["no"] if args.ebno else tf.cast(10 ** (-point / 10), tf.float32)
        for link in links:
            if link is first:
                b = first.captured["b"]
            else:
                b = link.gen._source([n, link.params.max_num_tx, link.tx._tb_size])
            bits[link.mcs].append(np.packbits(np.asarray(b).astype(np.uint8), axis=-1))
            for t, rv in enumerate(RVS):
                x = link.modulate(link.encode(b, rv))
                out[(link.mcs, t)].append(aerial(apply_channel([x, hs[t], no])))
        info["point"] += [point] * n
        info["corr"].append(np.stack([similarity(h) for h in hs], axis=1))
        flat0 = hs[0].numpy().reshape(n, -1)
        info["change"].append(np.stack([1 - np.abs(np.sum(np.conj(flat0) * h.numpy().reshape(n, -1), axis=1)) /
                                        (np.linalg.norm(flat0, axis=1) * np.linalg.norm(h.numpy().reshape(n, -1), axis=1) + 1e-12)
                                        for h in hs], axis=1))
        info["gain"].append(np.stack([gains(h) for h in hs], axis=1))
        done += n
    c = np.concatenate(info["corr"])[-args.slots:, 0]
    print(f"point {point}: user similarity median {np.median(c):.2f}, slots {args.slots}", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
point = np.asarray(info["point"], dtype=np.float32)
corr, change, gain = (np.concatenate(info[k]).astype(np.float32) for k in ("corr", "change", "gain"))
np.savez(args.output.parent / f"{args.output.name}_info.npz", point=point, corr=corr, change=change, gain=gain,
         meta=json.dumps({"channel": args.channel, "speed": args.speed, "los": args.los, "rvs": RVS, "mcs": MCS,
                          "gap_ms": args.gap_ms, "gap_symbols": gap, "axis": "ebno" if args.ebno else "snr_db",
                          "num_rx": int(first.params.num_rx_antennas), "seed": args.seed}))
for link in links:
    payload = np.concatenate(bits[link.mcs])
    for t, rv in enumerate(RVS):
        meta = {"base_config": args.base_config, "prbs": args.prbs, "num_tx": 2, "channel": args.channel,
                "speed": args.speed, "los": args.los, "mcs_index": link.mcs, "rv": rv, "transmission": t,
                "gap_ms": args.gap_ms, "carrier_hz": float(link.params.carrier_frequency),
                "num_rx_ant": int(link.params.num_rx_antennas), "tb_size_bits": int(link.tx._tb_size),
                "num_bits_per_symbol": int(link.tx._num_bits_per_symbol),
                "target_coderate": float(link.tx._target_coderate), "n_cell_id": int(link.params.n_cell_id),
                "slot_number": int(link.params.slot_number), "ebno_db": POINTS, "axis": "ebno" if args.ebno else "snr_db",
                "slots": args.slots, "neural_receiver": None}
        path = args.output.parent / f"{args.output.name}_m{link.mcs}_t{t}.npz"
        np.savez(path, rx=np.concatenate(out[(link.mcs, t)]), payload=payload, ebno=point,
                 nrx_ok=np.zeros(payload.shape[:2], dtype=bool), meta=json.dumps(meta))
        print("saved", path, flush=True)
