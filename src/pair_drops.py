#!/usr/bin/env python3
"""P2, step 1: channel similarity of every pair of users in a 3GPP cell, for many user drops.

Same arrays and model as the HARQ datasets (NVlabs UMi/UMa: 4-antenna cross-polarized base station, single-antenna
users, no path loss or shadowing, outdoor users).  For each drop of U users the frequency response is taken at one time
instant on every 6th subcarrier, and for each pair (i, j) the similarity is the mean over subcarriers of
|h_i^H h_j| / (|h_i||h_j|) -- the same quantity gen_harq.py stores per slot.
Writes <output>.npz: rho[drops, U, U] per U.
usage: pair_drops.py --nrx-repo DIR --channel UMi --users 4,8,16,32 --drops 2000 --output OUT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--nrx-repo", type=Path, required=True)
parser.add_argument("--base-config", default="nrx_large.cfg")
parser.add_argument("--channel", default="UMi")
parser.add_argument("--speed", type=float, default=0.83)
parser.add_argument("--users", default="4,8,16,32")
parser.add_argument("--drops", type=int, default=2000)
parser.add_argument("--batch", type=int, default=50)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--gpu", default="0")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(args.nrx_repo))
os.chdir(args.nrx_repo / "scripts")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.random.set_seed(args.seed)
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, gen_single_sector_topology   # noqa: E402
from sionna.channel.tr38901 import PanelArray, UMi, UMa                                                # noqa: E402
from utils import Parameters                                                                          # noqa: E402

params = Parameters(args.base_config, training=False, system="nrx", num_tx_eval=2)
fc = float(params.carrier_frequency)
rg = params.transmitters[0]._resource_grid
n_rx = int(params.num_rx_antennas)
bs = PanelArray(num_rows_per_panel=1, num_cols_per_panel=n_rx // 2, polarization="dual", polarization_type="cross",
                antenna_pattern="38.901", carrier_frequency=fc)
ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization="single", polarization_type="V",
                antenna_pattern="omni", carrier_frequency=fc)
Model = UMi if args.channel == "UMi" else UMa
model = Model(carrier_frequency=fc, o2i_model="low", bs_array=bs, ut_array=ut, direction="uplink",
              enable_pathloss=False, enable_shadow_fading=False)
freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)[::6]

out = {}
for users in [int(u) for u in args.users.split(",")]:
    rhos = []
    done = 0
    while done < args.drops:
        n = min(args.batch, args.drops - done)
        topo = gen_single_sector_topology(n, users, args.channel.lower(), min_ut_velocity=args.speed,
                                          max_ut_velocity=args.speed, indoor_probability=0.)
        model.set_topology(*topo)
        a, tau = model(1, 1.0 / float(rg.ofdm_symbol_duration))
        h = cir_to_ofdm_channel(freqs, a, tau, False).numpy()      # [n, 1, rx_ant, users, 1, 1, sc]
        v = h[:, 0, :, :, 0, 0, :]                                 # [n, rx_ant, users, sc]
        norm = np.linalg.norm(v, axis=1)                           # [n, users, sc]
        g = np.einsum("baus,bavs->buvs", np.conj(v), v)            # [n, users, users, sc]
        rho = (np.abs(g) / (norm[:, :, None, :] * norm[:, None, :, :] + 1e-12)).mean(axis=-1)
        rhos.append(rho.astype(np.float32))
        done += n
    out[f"rho_u{users}"] = np.concatenate(rhos)
    r = out[f"rho_u{users}"]
    iu = np.triu_indices(users, 1)
    pairs = r[:, iu[0], iu[1]]
    print(f"U={users}: pair similarity median {np.median(pairs):.3f}, best pair per drop median "
          f"{np.median(pairs.min(axis=1)):.3f}", flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output, channel=args.channel, speed=args.speed, **out)
print("saved", args.output)
