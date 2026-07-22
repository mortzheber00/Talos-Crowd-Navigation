#!/usr/bin/env bash
# Full post-merge re-run. Runs sequentially so qa_solo timing is uncontended and
# so the deck-critical sweeps (qc, qf, qh_hard) finish first. Each run_sweep call
# resumes from cached JSON, so re-invoking this script is safe.
cd /workspaces/talos_crowd_navigation/.claude/worktrees/evaluation
export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:${PYTHONPATH:-}
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export MUJOCO_GL=egl ROS_LOCALHOST_ONLY=1

PROG=experiments/results/rerun_progress.log
mkdir -p experiments/results
echo "=== rerun started $(date -u +%H:%M:%S) ===" > "$PROG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG"; }

# --- 0. GIF timeseries: 6 short runs, common 18 s clock, no early stop -------
log "gif_runs: 6 scenarios"
dom=60
for scen in empty head_on crossing corridor circle random; do
  ROS_DOMAIN_ID=$dom timeout 400 ros2 run bullet_sims walking --ros-args \
    -p render:=false -p scenario:=$scen -p seed:=1 \
    -p early_stop:=false -p walk_duration:=18.0 \
    -p out_dir:=experiments/results/gif_runs -p run_id:=gif_$scen \
    > experiments/results/gif_runs_$scen.log 2>&1 &
  dom=$((dom+1))
done
wait
log "gif_runs done"

# --- 1. uncontended timing (workers 1) --------------------------------------
log "qa_solo (workers 1)"
python3 experiments/run_sweep.py qa_solo --workers 1 >> "$PROG" 2>&1
log "qa_solo done"

# --- 2. deck-critical, then the rest (workers 10) ---------------------------
# Omitted on purpose:
#   qh_hard    -- responsibility plane at 48 peds; implausible density, and the
#                 hardened n=10 baseline already fails (re-target once chosen).
#   qab_timing -- contended timing is not quoted (qa_solo covers it); its only
#                 unique output was Q-B velocity tracking (re-run if wanted).
for sw in qc_scenarios qf_perception \
          qd_responsibility qe_density qb2_speed \
          qg_cooperation qg2_ped_responsibility qh_reciprocity qs_stress; do
  log "$sw start"
  python3 experiments/run_sweep.py $sw --workers 10 >> "$PROG" 2>&1
  log "$sw done"
done
log "=== ALL SWEEPS DONE ==="
