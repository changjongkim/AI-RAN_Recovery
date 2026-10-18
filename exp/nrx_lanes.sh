#!/usr/bin/env bash
# N13: neural-receiver throughput and latency on one A100 with 1-8 concurrent engine copies (src/nrx_lanes.py)
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
GPU=0 rx python3 /rec/nrx_lanes.py --engine $ENGINE_LARGE --engine2 $ENGINE_SMALL --output /rec/n13_lanes.json || exit 1
mv $REC/src/n13_lanes.json $RESULTS/n13_lanes.json && cat $RESULTS/n13_lanes.json | head -40
