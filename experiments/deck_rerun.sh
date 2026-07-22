#!/usr/bin/env bash
# Full deck dataset re-run: Block 1 (scenario comparison, 1000 runs) then
# Block 2 (responsibility plane, 360 runs). Resumes from cached JSON, so
# re-invoking is safe.
cd /workspaces/talos_crowd_navigation/.claude/worktrees/evaluation
export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:${PYTHONPATH:-}
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export MUJOCO_GL=egl ROS_LOCALHOST_ONLY=1

PROG=experiments/results/deck_progress.log
mkdir -p experiments/results
echo "=== deck re-run started $(date -u +%H:%M:%S) ===" > "$PROG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG"; }

log "Block 1: deck_scenarios (1000 runs)"
python3 experiments/run_sweep.py deck_scenarios --workers 10 >> "$PROG" 2>&1
log "Block 1 done"

log "Block 2: qh_zoom (360 runs)"
python3 experiments/run_sweep.py qh_zoom --workers 10 >> "$PROG" 2>&1
log "Block 2 done"

log "=== ALL DECK RUNS DONE ==="
