#!/usr/bin/env bash
# N13 for the NRx trained with interference: the same measurement as exp/nrx_lanes.sh (src/nrx_lanes.py, one A100,
# 1-8 concurrent engine copies) with the interference-trained large engine against the published large engine, so that
# the run time of the two recovery receivers is measured the same way (results/n13_lanes_ici.json: "large" is the
# interference-trained engine, "small" the published one).
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
GPU=0 rx python3 /rec/nrx_lanes.py --engine /rec/engines/nrx_large_ici_273prb_2ue.trt --engine2 $ENGINE_LARGE \
    --output /rec/n13_lanes_ici.json || exit 1
mv $REC/src/n13_lanes_ici.json $RESULTS/n13_lanes_ici.json && head -40 $RESULTS/n13_lanes_ici.json
