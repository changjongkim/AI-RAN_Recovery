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
With --drop-users U, U users are dropped per slot and user 0 is co-scheduled with one partner (--pairing random
or partner: best (1 - rho^2) * min gain), as a MU-MIMO scheduler picks a partner; the pair's channels go on as usual.
With --interferers K, K users of neighbor cells transmit on the same PRBs (own time-correlated channels, new data every
transmission, DMRS decorrelated by a random phase per RE) at a mean power set by --sir-db.
With --retx-alone the retransmissions carry user 0 only (user 1's data and DMRS are zero), so HARQ is not limited by
the same partner; the first transmission still carries both users.
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
parser.add_argument("--carrier-ghz", type=float, default=None,
                    help="carrier frequency in GHz (default: the NVlabs configuration, 2.14 GHz); sets the Doppler, the\n"
                         "                         UMi large-scale parameters (UMa uses 6 GHz values below 6 GHz) and the layout path loss")
parser.add_argument("--channel", default="UMi", help="UMi, UMa, CDL-A..E, TDL-*, DoubleTDLlow/high")
parser.add_argument("--speed", type=float, default=0.83, help="user speed in m/s (min = max)")
parser.add_argument("--los", choices=["auto", "los", "nlos"], default="auto", help="force LoS state (UMi/UMa)")
parser.add_argument("--num-rx", type=int, default=0, help="override the number of receive antennas")
parser.add_argument("--mcs", default="14", help="comma list; all share the channel realizations")
parser.add_argument("--rvs", default="0,2,3,1", help="redundancy version of each transmission")
parser.add_argument("--gap-ms", type=float, default=7.5, help="time between transmissions")
parser.add_argument("--drop-users", type=int, default=0,
                    help="drop this many users per slot and co-schedule user 0 with a partner chosen by --pairing")
parser.add_argument("--pairing", choices=["random", "partner", "minrho", "pf"], default="random",
                    help="random: any other user; partner: the user j that maximizes (1 - rho_0j^2) * min(g_0, g_j), the weaker user's channel gain left after zero-forcing (the scheduler knows both from SRS/previous slots); pf: the user j that maximizes the pair's sum rate after MMSE detection, sum_k log2(1 + SINR_k) per PRB at the slot's noise (proportional fair with equal past rates)")
parser.add_argument("--interferers", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="users of neighbor cells on the same PRBs, each with its own time-correlated channel to this cell")
parser.add_argument("--sir-db", type=float, default=10.0,
                    help="mean power of one served user over the total power of the interferers (with unit-mean channels)")
parser.add_argument("--inr-db", type=float, default=None,
                    help="instead of --sir-db: total interference power over the thermal noise power (interference over\n                         thermal, the operator measure; 3-10 dB in loaded networks)")
parser.add_argument("--inr-mix", default=None,
                    help="P_NONE:LO:HI with --interferers: every batch of slots draws its interference over thermal, none\n                         with probability P_NONE, else uniform in [LO, HI] dB, and its Eb/No is the point plus the rise\n                         10 log10(1 + 10^(INR/10)), so the signal to interference-plus-noise stays on the grid (per-slot INR\n                         in the info file, nan = none)")
parser.add_argument("--layout", default=None,
                    help="hex:LOAD[:ALPHA[:P0]] with --interferers K: neighbor-cell users from a multi-cell layout (our site's two other\n                         sectors and the six first-tier sites, a pair per cell with probability LOAD) with powers from\n                         3GPP path loss and power control; the K strongest get their own channels, the rest add white noise")
parser.add_argument("--retx-alone", action="store_true",
                    help="retransmissions of user 0 are sent alone (user 1 silent), as when the scheduler gives a "
                         "retransmission its own resources; only user 0's blocks are counted then")
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
from sionna.channel import gen_single_sector_topology                                      # noqa: E402
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
    if args.carrier_ghz:
        subs.append((r"(?m)^carrier_frequency\s*=.*", f"carrier_frequency = {args.carrier_ghz * 1e9:.1f}"))
    for pattern, value in subs:
        text, n = re.subn(pattern, value, text)
        assert n >= 1, pattern
    # one file per process: chunks run in parallel and must not read each other's half-written file
    name = f"rec_{label}_{args.prbs}_{args.channel}_v{args.speed:g}_m{mcs}_r{args.num_rx or 4}_{args.los}_p{os.getpid()}.cfg"
    tmp = Path("../config") / (name + ".tmp")
    tmp.write_text(text)
    tmp.rename(Path("../config") / name)
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
apply_noiseless = ApplyOFDMChannel(add_awgn=False)
ISCALE = float(np.sqrt(10 ** (-args.sir_db / 10) / max(1, args.interferers)))
kept = {}

assert not (args.drop_users and args.interferers), "--drop-users and --interferers are not combined"
LAYOUT = float(args.layout.split(":")[1]) if args.layout else None
ALPHA = float(args.layout.split(":")[2]) if args.layout and args.layout.count(":") >= 2 else 1.0
# hex:LOAD:ALPHA:P0 caps every user's power at 23 dBm (power class 3, TS 38.101-1) under the open-loop power control
# P = min(23, P0 + 10 log10(2^mu M_RB) + ALPHA PL) of TS 38.213 7.1.1, P0 in dBm per resource block (TR 38.858 Table C.2-1
# calibrates UMa with P0 = -80 dBm, ALPHA = 0.8); C below is the headroom Pmax - P0 - 10 log10(2^mu M_RB) in dB.
P0_DBM = float(args.layout.split(":")[3]) if args.layout and args.layout.count(":") >= 3 else None
HEADROOM = None if P0_DBM is None else 23.0 - P0_DBM - 10 * np.log10(2 * args.prbs)   # 30 kHz: mu = 1
assert LAYOUT is None or (args.interferers and not args.inr_mix and args.channel in ("UMa", "UMi")), \
    "--layout needs --interferers, UMa or UMi, and no --inr-mix"
layout_rng = np.random.default_rng(args.seed + 104729)
FC_GHZ = float(first.params.carrier_frequency) / 1e9


def pathloss_db(d2, h_bs, h_ut=1.5):
    """3GPP TR 38.901 path loss with a random LoS state and shadow fading (UMa or UMi), d2: 2D distances [m]."""
    d3 = np.sqrt(d2 ** 2 + (h_bs - h_ut) ** 2)
    d_bp = 4 * (h_bs - 1) * (h_ut - 1) * FC_GHZ * 1e9 / 3e8
    if args.channel == "UMa":
        p_los = np.where(d2 <= 18, 1.0, 18 / d2 + np.exp(-d2 / 63) * (1 - 18 / d2))
        los = np.where(d2 < d_bp, 28 + 22 * np.log10(d3), 28 + 40 * np.log10(d3) - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)) + 20 * np.log10(FC_GHZ)
        nlos = np.maximum(los, 13.54 + 39.08 * np.log10(d3) + 20 * np.log10(FC_GHZ) - 0.6 * (h_ut - 1.5))
        sf = (4.0, 6.0)
    else:
        p_los = np.where(d2 <= 18, 1.0, 18 / d2 + np.exp(-d2 / 36) * (1 - 18 / d2))
        los = np.where(d2 < d_bp, 32.4 + 21 * np.log10(d3), 32.4 + 40 * np.log10(d3) - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)) + 20 * np.log10(FC_GHZ)
        nlos = np.maximum(los, 22.4 + 35.3 * np.log10(d3) + 21.3 * np.log10(FC_GHZ) - 0.3 * (h_ut - 1.5))
        sf = (4.0, 7.82)
    is_los = layout_rng.random(d2.shape) < p_los
    return np.where(is_los, los + layout_rng.normal(0, sf[0], d2.shape), nlos + layout_rng.normal(0, sf[1], d2.shape))


_PL_REF = {}


def pl_ref(h_bs):
    """Median path loss of a user dropped uniformly in our sector (the fractional power control's reference)."""
    if h_bs not in _PL_REF:
        isd, r_min = (500.0, 35.0) if args.channel == "UMa" else (200.0, 10.0)
        r = np.sqrt(np.random.default_rng(7).uniform(r_min ** 2, (isd / np.sqrt(3)) ** 2, 20000))
        _PL_REF[h_bs] = float(np.median(pathloss_db(r, h_bs)))
    return _PL_REF[h_bs]


def layout_interferers(b, theta0, h_bs):
    """--layout hex:LOAD.  Our cell is one sector (boresight theta0) of the site at the origin; the two other sectors of
    this site and the three sectors of each of the six first-tier sites (inter-site distance 500 m UMa, 200 m UMi) each
    carry one pair of users on our PRBs with probability LOAD, dropped uniformly in their sector and kept only if their
    own cell serves them at least as well as ours (best-server association; else drawn again).  Every user is power
    controlled so that it arrives at its own base station like our users arrive at ours (full compensation, no power
    cap: the conservative side, cell-edge neighbors transmit loud), so its power at our base station relative to one of
    our users is  delta = (PL_own - G_own) - PL_ours  [dB]  (G: 3GPP horizontal sector pattern -min(12 (phi/65)^2, 30)
    dB; our own antenna gain is in the channel, which is not normalized; a co-site user has one path loss to both).
    With hex:LOAD:ALPHA the power control compensates only ALPHA of the loss (fractional, 3GPP TS 38.213 7.1.1):
    delta = ALPHA (PL_own - G_own) + (1 - ALPHA) PL_ref - PL_ours, PL_ref the median loss of a user of our cell (a user
    there arrives at the target).  With hex:LOAD:ALPHA:P0 every user also transmits at most 23 dBm: with the headroom C of
    P0 (above), delta = min(C, ALPHA (PL_own - G_own)) - PL_ours - min(C, ALPHA PL_ref) + PL_ref, our reference user at
    PL_ref capped the same way (without P0 this is the formula above; PL is the path loss without the antenna gain).
    Returns, per slot, the positions and delta of the K strongest at our antenna (explicit
    interferers with their own channels) and the summed power of the rest after our sector pattern (white noise, scaled
    by the served channel's mean gain where it is added)."""
    isd, r_min = (500.0, 35.0) if args.channel == "UMa" else (200.0, 10.0)
    r_max = isd / np.sqrt(3)
    gain = lambda phi: -np.minimum(12 * (np.degrees(np.angle(np.exp(1j * phi))) / 65) ** 2, 30.0)
    pos = np.zeros((b, args.interferers, 2)); delta = np.full((b, args.interferers), -200.0); rest = np.zeros(b)
    for i in range(b):
        cells = [((0.0, 0.0), theta0[i] + 2 * np.pi / 3 * s, True) for s in (1, 2)]
        for k in range(6):
            sx, sy = isd * np.cos(theta0[i] + np.pi / 3 * k), isd * np.sin(theta0[i] + np.pi / 3 * k)
            cells += [((sx, sy), theta0[i] + 2 * np.pi / 3 * s, False) for s in range(3)]
        xy, d, geff = [], [], []
        for (sx, sy), bore, cosite in cells:
            if layout_rng.random() >= LAYOUT:
                continue
            for _ in range(2):
                for _try in range(50):
                    a = bore + layout_rng.uniform(-np.pi / 3, np.pi / 3)
                    r = np.sqrt(layout_rng.uniform(r_min ** 2, r_max ** 2))
                    x, y = sx + r * np.cos(a), sy + r * np.sin(a)
                    d_us = float(np.hypot(x, y))
                    if d_us < r_min:
                        continue
                    pl_us = pathloss_db(np.array([d_us]), h_bs)[0]
                    pl_own = pl_us if cosite else pathloss_db(np.array([r]), h_bs)[0]
                    g_own, g_us = gain(a - bore), gain(np.arctan2(y, x) - theta0[i])
                    if pl_own - g_own <= pl_us - g_us:
                        if HEADROOM is None:
                            d.append(ALPHA * (pl_own - g_own) + (1 - ALPHA) * pl_ref(h_bs) - pl_us)
                        else:
                            d.append(min(HEADROOM, ALPHA * (pl_own - g_own)) - pl_us
                                     - min(HEADROOM, ALPHA * pl_ref(h_bs)) + pl_ref(h_bs))
                        xy.append((x, y))
                        geff.append(g_us)
                        break
        if not xy:
            continue
        xy, d, geff = np.asarray(xy), np.asarray(d), np.asarray(geff)
        order = np.argsort(-(d + geff))                    # strongest at our antenna first
        top = order[:args.interferers]
        pos[i, :len(top)] = xy[top]; delta[i, :len(top)] = d[top]
        rest[i] = float(np.sum(10 ** ((d + geff)[order[args.interferers:]] / 10)))
        if len(top) < args.interferers:              # unused interferer slots: far away, zero power
            pos[i, len(top):] = (10 * isd, 10 * isd)
    return pos, delta, rest


if args.los != "auto" or args.drop_users or args.interferers:
    set_topology = model.set_topology

    def forced(*topology, **kw):
        if args.los != "auto":
            kw["los"] = args.los == "los"
        if LAYOUT is not None:                       # our pair in our sector, the interferers where the layout put them
            b = int(topology[0].shape[0])
            ut_loc, bs_loc, ut_o, bs_o, ut_v, ins = (np.asarray(v) for v in gen_single_sector_topology(
                b, 2, args.channel.lower(), min_ut_velocity=args.speed, max_ut_velocity=args.speed, indoor_probability=0.))
            pos, kept["layout_delta"], kept["layout_rest"] = layout_interferers(b, bs_o[:, 0, 0], float(bs_loc[0, 0, 2]))
            k = args.interferers
            i_loc = np.concatenate([pos, np.full((b, k, 1), 1.5)], axis=2)
            yaw = layout_rng.uniform(-np.pi, np.pi, (b, k))
            i_o = np.stack([yaw, np.zeros((b, k)), np.zeros((b, k))], axis=2)
            hd = layout_rng.uniform(-np.pi, np.pi, (b, k))
            i_v = np.stack([args.speed * np.cos(hd), args.speed * np.sin(hd), np.zeros((b, k))], axis=2)
            topology = (tf.constant(np.concatenate([ut_loc, i_loc], 1), tf.float32), tf.constant(bs_loc, tf.float32),
                        tf.constant(np.concatenate([ut_o, i_o], 1), tf.float32), tf.constant(bs_o, tf.float32),
                        tf.constant(np.concatenate([ut_v, i_v], 1), tf.float32),
                        tf.constant(np.concatenate([ins, np.zeros((b, k), bool)], 1)))
        elif args.drop_users or args.interferers:   # drop more users: candidates for pairing, or the interferers after the pair
            topology = gen_single_sector_topology(int(topology[0].shape[0]), args.drop_users or 2 + args.interferers, args.channel.lower(),
                                                  min_ut_velocity=args.speed, max_ut_velocity=args.speed,
                                                  indoor_probability=0.)
        return set_topology(*topology[:6], **kw)
    model.set_topology = forced
picked = {"rho": [], "gain_db": [], "rank": []}


def pair_sum_rate(v, no):
    """MMSE sum rate of user 0 with every other user, [b, ant, U, sym, sc] -> [b, U] (bit/s/Hz, mean over PRBs of the
    first symbol; user 0's own entry is -inf)."""
    w = np.moveaxis(v[:, :, :, 0, 6::12], 3, 1)                                   # [b, prb, ant, U], one subcarrier per PRB
    no = np.broadcast_to(np.asarray(no, dtype=np.float64).reshape(-1), (w.shape[0],))[:, None, None, None]
    rate = np.full(w.shape[0::3], -np.inf)
    for j in range(1, w.shape[3]):
        h = w[..., [0, j]]                                                         # [b, prb, ant, 2]
        e = np.linalg.inv(np.eye(2) + np.conj(np.swapaxes(h, 2, 3)) @ h / no)     # MMSE error covariance
        sinr = 1.0 / np.real(np.diagonal(e, axis1=2, axis2=3)) - 1.0               # [b, prb, 2]
        rate[:, j] = np.log2(1.0 + np.maximum(sinr, 0.0)).sum(axis=2).mean(axis=1)
    return rate


def choose_pairs(a, tau, no):
    """[batch, 1, rx_ant, U, 1, path, time] -> indices [batch, 2] of user 0 and its partner (from the first slot)."""
    v = cir_to_ofdm_channel(frequencies, a[..., :symbols], tau, False).numpy()[:, 0, :, :, 0]   # [b, ant, U, sym, sc]
    g = np.mean(np.abs(v) ** 2, axis=(1, 3, 4))                                                  # [b, U]
    num = np.abs(np.sum(np.conj(v[:, :, :1]) * v, axis=1))                                       # [b, U, sym, sc]
    den = np.linalg.norm(v[:, :, :1], axis=1) * np.linalg.norm(v, axis=1) + 1e-12
    rho = (num / den).mean(axis=(2, 3))                                                          # [b, U]
    score = (1 - rho ** 2) * np.minimum(g[:, :1], g)
    score[:, 0] = -np.inf
    if args.pairing == "pf":
        score = pair_sum_rate(v, no)
        j = np.argmax(score, axis=1)
    elif args.pairing == "partner":
        j = np.argmax(score, axis=1)
    elif args.pairing == "minrho":          # the most separable partner whatever its gain
        r = rho.copy(); r[:, 0] = np.inf
        j = np.argmin(r, axis=1)
    else:
        j = np.ones(len(g), dtype=int)
    rank = (score > score[np.arange(len(g)), j][:, None]).sum(axis=1)                            # 0 = the best partner
    picked["rho"] += rho[np.arange(len(g)), j].tolist()
    picked["gain_db"] += (10 * np.log10(g[np.arange(len(g)), j] / g[:, 0] + 1e-30)).tolist()
    picked["rank"] += rank.tolist()
    return tf.constant(np.stack([np.zeros_like(j), j], axis=1), tf.int32)


def time_correlated(channel_inputs):
    """Replaces the generator's channel: samples the model over all transmissions, keeps every response."""
    x, no = channel_inputs
    a, tau = model(tf.shape(x)[0], span, 1.0 / symbol_s)
    if args.drop_users:
        pair = choose_pairs(a, tau, no)
        a = tf.gather(a, pair, axis=3, batch_dims=1)
        tau = tf.gather(tau, pair, axis=2, batch_dims=1)
    if args.interferers:            # users 2.. are the neighbor-cell users
        ai, taui = a[:, :, :, 2:], tau[:, :, 2:]
        a, tau = a[:, :, :, :2], tau[:, :, :2]
        kept["hi"] = [cir_to_ofdm_channel(frequencies, ai[..., t * gap:t * gap + symbols], taui, False) for t in range(len(RVS))]
    hs = [cir_to_ofdm_channel(frequencies, a[..., t * gap:t * gap + symbols], tau, False) for t in range(len(RVS))]
    kept["h"], kept["no"] = hs, no
    return apply_channel([x, hs[0], no]), hs[0]


first.gen._channel = time_correlated
first.gen(1, 10.0)          # build
for _k in picked:
    picked[_k].clear()     # the build call is not a slot


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
MIX = tuple(float(s) for s in args.inr_mix.split(":")) if args.inr_mix else None
assert MIX is None or (args.interferers and args.ebno), "--inr-mix needs --interferers and --ebno"
mix_rng = np.random.default_rng(args.seed + 7919)
for point in POINTS:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        ebno = point if args.ebno else 10.0
        inr = args.inr_db                                    # interference over thermal of this batch (None: --sir-db)
        if MIX:
            inr = None if mix_rng.random() < MIX[0] else float(mix_rng.uniform(MIX[1], MIX[2]))
            ebno = point + (10 * np.log10(1 + 10 ** (inr / 10)) if inr is not None else 0.0)
            info.setdefault("inr", []).append(np.full(n, np.nan if inr is None else inr, dtype=np.float32))
        mute = bool(MIX) and inr is None                     # a batch without interference in a mixed run
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
                x = link.modulate(link.encode(b, rv))         # [batch, user, tx antenna, symbol, subcarrier]
                if args.retx_alone and t >= 1:
                    x = x * tf.constant([1, 0], x.dtype)[None, :, None, None, None]   # data and DMRS of user 1 off
                y = apply_channel([x, hs[t], no])
                if args.interferers and not mute:   # new data every transmission; a random phase per resource element
                    bi = link.gen._source([n, link.params.max_num_tx, link.tx._tb_size])   # stands for the neighbor's DMRS scrambling
                    xi = link.modulate(link.encode(bi, 0))
                    while xi.shape[1] < args.interferers:   # more neighbor users than the link has ports: more draws
                        bj = link.gen._source([n, link.params.max_num_tx, link.tx._tb_size])
                        xi = tf.concat([xi, link.modulate(link.encode(bj, 0))], axis=1)
                    xi = xi[:, :args.interferers]
                    ph = tf.exp(tf.complex(0.0, tf.random.uniform(tf.shape(xi), 0, 2 * np.pi)))
                    if LAYOUT is not None:  # each interferer at its own power, the weaker ones as white noise
                        amp = np.sqrt(10 ** (kept["layout_delta"] / 10))[:, :, None, None, None]
                        y = y + apply_noiseless([xi * ph * tf.constant(amp, xi.dtype), kept["hi"][t]])
                        g_s = np.mean(np.abs(hs[t].numpy()) ** 2, axis=(1, 2, 3, 4, 5, 6))     # served mean channel gain
                        sd = np.sqrt(kept["layout_rest"] * g_s / 2)[:, None, None, None, None].astype(np.float32)
                        y = y + tf.complex(tf.random.normal(tf.shape(y)) * sd, tf.random.normal(tf.shape(y)) * sd)
                    else:
                        sc = ISCALE if inr is None else tf.sqrt(tf.cast(no, tf.float32) * 10 ** (inr / 10) / args.interferers)
                        y = y + apply_noiseless([xi * ph * tf.cast(tf.complex(sc, 0.0), xi.dtype), kept["hi"][t]])
                out[(link.mcs, t)].append(aerial(y))
        info["point"] += [point] * n
        info["corr"].append(np.stack([similarity(h) for h in hs], axis=1))
        flat0 = hs[0].numpy().reshape(n, -1)
        info["change"].append(np.stack([1 - np.abs(np.sum(np.conj(flat0) * h.numpy().reshape(n, -1), axis=1)) /
                                        (np.linalg.norm(flat0, axis=1) * np.linalg.norm(h.numpy().reshape(n, -1), axis=1) + 1e-12)
                                        for h in hs], axis=1))
        info["gain"].append(np.stack([gains(h) for h in hs], axis=1))
        if args.interferers:            # per slot and transmission: served-user mean power over interference power (dB)
            s2 = 0.0 if mute else ISCALE ** 2 if inr is None else float(np.mean(no)) * 10 ** (inr / 10) / args.interferers
            pi = [s2 * np.mean(np.abs(h.numpy()) ** 2, axis=(1, 2, 4, 5, 6)).sum(axis=1) for h in kept["hi"]]
            if LAYOUT is not None:  # the layout's powers: explicit interferers through their channels, plus the rest
                pi = [(10 ** (kept["layout_delta"] / 10) * np.mean(np.abs(h.numpy()) ** 2, axis=(1, 2, 4, 5, 6))).sum(axis=1)
                      + kept["layout_rest"] * np.mean(np.abs(hh.numpy()) ** 2, axis=(1, 2, 3, 4, 5, 6))
                      for h, hh in zip(kept["hi"], hs)]
            pd = [np.mean(np.abs(h.numpy()) ** 2, axis=(1, 2, 3, 4, 5, 6)) for h in hs]
            info.setdefault("sir", []).append(np.stack([10 * np.log10(d / (i + 1e-30)) for d, i in zip(pd, pi)], axis=1))
        done += n
    c = np.concatenate(info["corr"])[-args.slots:, 0]
    print(f"point {point}: user similarity median {np.median(c):.2f}, slots {args.slots}", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
point = np.asarray(info["point"], dtype=np.float32)
corr, change, gain = (np.concatenate(info[k]).astype(np.float32) for k in ("corr", "change", "gain"))
extra = {"sir": np.concatenate(info["sir"]).astype(np.float32)} if args.interferers else {}
if MIX:
    extra["inr"] = np.concatenate(info["inr"])
np.savez(args.output.parent / f"{args.output.name}_info.npz", point=point, corr=corr, change=change, gain=gain, **extra,
         meta=json.dumps({"channel": args.channel, "speed": args.speed, "los": args.los, "rvs": RVS, "mcs": MCS,
                          "gap_ms": args.gap_ms, "gap_symbols": gap, "retx_alone": args.retx_alone,
                          "drop_users": args.drop_users, "pairing": args.pairing, "interferers": args.interferers,
                          "sir_db": args.sir_db if args.interferers and args.inr_db is None and not MIX else None, "inr_db": args.inr_db,
                          "inr_mix": args.inr_mix, "layout": args.layout,
                          "sir_median_db": float(np.median(np.concatenate(info["sir"])[:, 0])) if args.interferers else None,
                          "pair_rho_median": float(np.median(picked["rho"])) if args.drop_users else None,
                          "partner_gain_db_median": float(np.median(picked["gain_db"])) if args.drop_users else None,
                          "partner_rank_mean": float(np.mean(picked["rank"])) if args.drop_users else None, "axis": "ebno" if args.ebno else "snr_db",
                          "num_rx": int(first.params.num_rx_antennas), "seed": args.seed}))
for link in links:
    payload = np.concatenate(bits[link.mcs])
    for t, rv in enumerate(RVS):
        meta = {"base_config": args.base_config, "prbs": args.prbs, "num_tx": 2, "channel": args.channel, "carrier_ghz": FC_GHZ,
                "speed": args.speed, "los": args.los, "mcs_index": link.mcs, "rv": rv, "transmission": t,
                "gap_ms": args.gap_ms, "retx_alone": args.retx_alone and t >= 1,
                "carrier_hz": float(link.params.carrier_frequency),
                "num_rx_ant": int(link.params.num_rx_antennas), "tb_size_bits": int(link.tx._tb_size),
                "num_bits_per_symbol": int(link.tx._num_bits_per_symbol),
                "target_coderate": float(link.tx._target_coderate), "n_cell_id": int(link.params.n_cell_id),
                "slot_number": int(link.params.slot_number), "ebno_db": POINTS, "axis": "ebno" if args.ebno else "snr_db",
                "slots": args.slots, "neural_receiver": None}
        path = args.output.parent / f"{args.output.name}_m{link.mcs}_t{t}.npz"
        np.savez(path, rx=np.concatenate(out[(link.mcs, t)]), payload=payload, ebno=point,
                 nrx_ok=np.zeros(payload.shape[:2], dtype=bool), meta=json.dumps(meta))
        print("saved", path, flush=True)
