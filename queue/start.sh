#!/usr/bin/env bash
# Start the queue runner of the recovery project on the login node (it takes its own interactive GPU node).
cd /pscratch/sd/s/sgkim/kcj/airan_recovery || exit 1
rm -f queue/STOP
nohup setsid bash queue/runner.sh > /dev/null 2>&1 &
sleep 2; tail -3 queue/runner.log 2>/dev/null; echo "runner started; progress: tail -f queue/runner.log, finished steps: queue/done"
