#!/usr/bin/env python3
"""Work items of the side pool (exp/side_pool.sh): small classical_rx.py runs (one Eb/No point, one seed, 64 slots)
that the three idle GPUs of the training node take one after another.  Printed as  name|needs|options.
  N5   UMi 3 km/h, 4 RX: conventional (20 and 50 BP iterations), IDD with 3 rounds, LMMSE + K-best, perfect CSI + K-best,
       large and real-time neural receivers; Eb/No 6-12 dB; 40 seeds x 64 slots = 10,240 slots (about 2,000 failed blocks) (needs: smoke + covariance)
  P1   2, 4, 8 RX: conventional and perfect CSI + K-best (the neural receivers exist for 4 RX only); Eb/No grids shifted
       by the array gain (2 RX 8-18 dB, 4 RX 4-14 dB, 8 RX 0-10 dB); 16 seeds x 64 slots per point
Ordered seed by seed so that every experiment has a usable sample early.
"""
N5 = ("--systems baseline_lslin_lmmse,baseline_lslin_lmmse@50,lslin_idd3,baseline_lmmse_kbest,"
      "baseline_perf_csi_kbest,nrx,nrx_rt --slots 64 --batch 4")
P1 = "--systems baseline_lslin_lmmse,baseline_perf_csi_kbest --slots 64 --batch 8"
GRID = {2: (8, 10, 12, 14, 16, 18), 4: (4, 6, 8, 10, 12, 14), 8: (0, 2, 4, 6, 8, 10)}
for i in range(40):
    for e in (6, 8, 10, 12):
        print(f"n5_umi_v3_s{i}_e{e}|n5|{N5} --ebno {e} --seed {500 + i}")
    if i < 16:
        for r, grid in GRID.items():
            for e in grid:
                print(f"p1_ant{r}_v3_s{i}_e{e}|none|{P1} --num-rx {r} --ebno {e} --seed {600 + i}")
