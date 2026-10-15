#!/usr/bin/env bash
# Queue runner (login node).  Runs the steps of queue/todo one after another; a node step is one srun step on the GPU
# node, and there is never more than one at a time.  When the allocation has too little time left for the next
# step, the runner releases it (if the runner took it) and takes a new interactive allocation.
#   start once:  bash /pscratch/sd/s/sgkim/kcj/airan_recovery/queue/start.sh   (state, logs and lock in queue/)
#   Copied from airan_cloudlab (runner_q.sh); this project's runner is independent of the Antiphase queues.
#   todo line:   N|NAME|MINUTES|COMMAND    node step: COMMAND runs on the node from the repository root
#                L|NAME|0|COMMAND          local step on the login node
#   done line:   NAME <tab> exit code <tab> job <tab> start <tab> end
#   queue/STOP:      the runner exits before the next step
#   queue/WAIT_FOR:  path of a log; the runner starts after the line CHAIN_EXIT appears in it
#   queue/nodes:     nodes per allocation (default 1).  Steps run on the first node; every further node runs the side
#                    pool (exp/side_pool.sh, GPUs 0-3) for as long as the allocation lasts and items are left.
# Nothing else may start a step on the node while the runner is alive (queue/lock is held).
ROOT=/pscratch/sd/s/sgkim/kcj/airan_recovery
S=$ROOT
Q=${QUEUE_DIR:-$ROOT/queue}
CUR=$Q/current_job
cd $ROOT || exit 1
exec 9>$Q/lock
flock -n 9 || { echo "runner already running"; exit 1; }
log() { echo "$(date '+%F %T') $*" >> $Q/runner.log; }
left_min() {      # minutes left of job $1; 0 when it is not running
  local s d=0 h=0 m=0 a b c t
  s=$(squeue -h -j "$1" -o "%T %L" 2>/dev/null)
  [ "${s%% *}" = RUNNING ] || { echo 0; return; }
  t=${s#* }
  case $t in *-*) d=${t%%-*}; t=${t#*-} ;; esac
  IFS=: read -r a b c <<<"$t"
  if [ -n "$c" ]; then h=$a; m=$b; elif [ -n "$b" ]; then m=$a; fi
  echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}
want_nodes() { cat $Q/nodes 2>/dev/null || echo 1; }
job_nodes() { squeue -h -j "$1" -o %N 2>/dev/null | xargs -r scontrol show hostnames 2>/dev/null; }
side_nodes() {    # every node of job $1 but the first runs side-pool rounds until the job ends or nothing is left
  local job=$1 node left t0
  for node in $(job_nodes $job | tail -n +2); do
    ( exec 9>&-                         # not the runner: the watchdog must see the lock free if the runner dies
      while true; do
        left=$(left_min $job); [ "$left" -gt 20 ] || break
        t0=$(date +%s)
        srun --jobid=$job --overlap -N1 -n1 -w $node --gpus-per-node=4 --export=ALL \
          bash -c "cd $ROOT && bash exp/side_pool.sh $((left - 10)) '0 1 2 3'" >> $S/logs/side_${node}_j$job.log 2>&1 < /dev/null
        [ $(( $(date +%s) - t0 )) -gt 300 ] || break          # a round that ended at once: nothing left to run
      done ) &
    log "side pool on $node of job $job"
  done
}
allocate() {
  local out job
  while true; do
    [ -f $Q/STOP ] && { log "STOP while waiting for an allocation"; exit 0; }
    out=$(salloc -N $(want_nodes) -C "gpu&hbm80g" -q interactive -A m5320_g -t 04:00:00 --no-shell 2>&1 < /dev/null)
    job=$(echo "$out" | sed -n 's/.*Granted job allocation \([0-9]*\).*/\1/p' | head -1)
    if [ -n "$job" ]; then
      echo $job > $CUR; echo $job >> $Q/jobs; log "allocated job $job ($(want_nodes) nodes)"; sleep 15
      side_nodes $job; return
    fi
    log "salloc failed: $(echo "$out" | tail -1)"; sleep 180
  done
}
log "runner started (pid $$)"
if [ -f $Q/WAIT_FOR ]; then
  f=$(cat $Q/WAIT_FOR); log "waiting for CHAIN_EXIT in $f"
  until grep -q CHAIN_EXIT "$f" 2>/dev/null; do sleep 15; done
  rm -f $Q/WAIT_FOR; sleep 10
fi
while true; do
  [ -f $Q/STOP ] && { log "STOP"; exit 0; }
  line=""
  while IFS= read -r l; do
    case $l in ""|\#*) continue ;; esac
    name=$(echo "$l" | cut -d'|' -f2)
    grep -q "^$name	" $Q/done 2>/dev/null || { line=$l; break; }
  done < $Q/todo
  [ -z "$line" ] && { sleep 60; continue; }
  IFS='|' read -r kind name minutes command <<<"$line"
  start=$(date '+%F %T')
  if [ "$kind" = L ]; then
    log "local step $name"
    bash -c "cd $ROOT && $command" > $Q/local_$name.log 2>&1 < /dev/null; rc=$?; job=-
  else
    job=$(cat $CUR 2>/dev/null); left=$(left_min "$job")
    if [ "$left" -gt 0 ] && [ "$(job_nodes $job | wc -l)" -lt "$(want_nodes)" ]; then
      log "job $job has fewer nodes than $(want_nodes)"; left=0
      grep -qx "$job" $Q/jobs 2>/dev/null && { scancel $job; log "released job $job"; sleep 30; }
    fi
    if [ "$left" -lt $((minutes + 5)) ]; then
      log "job $job has $left min left; step $name needs $minutes"
      if [ "$left" -gt 0 ] && grep -qx "$job" $Q/jobs 2>/dev/null; then scancel $job; log "released job $job"; sleep 30; fi
      allocate; job=$(cat $CUR)
    fi
    log "node step $name on job $job ($minutes min planned)"
    head=$(job_nodes $job | head -1)
    srun --jobid=$job --overlap -N1 -n1 ${head:+-w $head} --gpus-per-node=4 --export=ALL bash -c "cd $ROOT && $command" \
      > $S/logs/q_${name}_j$job.log 2>&1 < /dev/null; rc=$?
  fi
  if [ "$kind" != L ] && [ $rc -ne 0 ] && [ "$(left_min "$job")" -eq 0 ] && ! grep -qx "$name" $Q/retried 2>/dev/null; then
    # the allocation ended under the step: run it once more in a new allocation
    echo "$name" >> $Q/retried; log "step $name was cut by the end of job $job (exit code $rc); it runs again"
    continue
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$job" "$start" "$(date '+%F %T')" >> $Q/done
  log "step $name ended with exit code $rc"
done
