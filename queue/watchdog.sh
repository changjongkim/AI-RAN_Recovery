#!/usr/bin/env bash
# Keeps the runners of queue/ and queue_b/ alive on the login node: every 10 minutes, a runner whose lock is free
# (it died) is started again, unless its queue has a STOP file.  Start once: bash queue/watchdog.sh start
R=/pscratch/sd/s/sgkim/kcj/airan_recovery
if [ "${1:-}" = start ]; then
  cd $R && nohup setsid bash queue/watchdog.sh loop > /dev/null 2>&1 &
  echo "watchdog started"; exit 0
fi
exec 8>$R/queue/watchdog.lock
flock -n 8 || { echo "watchdog already running"; exit 0; }
while true; do
  for q in queue queue_b; do
    [ -f $R/$q/STOP ] && continue
    [ -f $R/$q/todo ] || continue
    if ( exec 9>$R/$q/lock; flock -n 9 ); then
      echo "$(date '+%F %T') runner of $q not running: starting it" >> $R/queue/watchdog.log
      ( cd $R && QUEUE_DIR=$R/$q nohup setsid bash queue/runner.sh > /dev/null 2>&1 & )
    fi
  done
  sleep 600
done
