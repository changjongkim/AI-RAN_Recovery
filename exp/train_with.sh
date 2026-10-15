#!/usr/bin/env bash
# One N4 training step on GPU 0 and, at the same time, another experiment on GPUs 1-3 of the same node.
# usage: bash exp/train_with.sh STEP ITERS "SIDE COMMAND"     (the side command must use GPUs 1-3 only)
set -uo pipefail
step=$1; iters=$2; side=${3:-}
cd /pscratch/sd/s/sgkim/kcj/airan_recovery
if [ -n "$side" ]; then
  bash -c "$side" > results/n4/side_step$step.log 2>&1 &
  sp=$!
fi
bash exp/n4_train.sh $step $iters; rc=$?
if [ -n "$side" ]; then
  wait $sp; src=$?
  echo "side task (step $step) exit code $src"; tail -5 results/n4/side_step$step.log
fi
exit $rc
