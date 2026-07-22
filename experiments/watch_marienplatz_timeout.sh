#!/usr/bin/env bash
# Run INSIDE the devcontainer. Replay a nominal Marienplatz "did not reach" run
# with the viewer on. The robot reaches ~1.15 m from the goal but cannot close it
# out in 45 s -- the goal-sharing crowd clogs the destination. Seeds 0 and 6 are
# the two that time out.
# Usage: ./experiments/watch_marienplatz_timeout.sh [seed]   (0 or 6, default 0)
SEED="${1:-0}"
export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:$PYTHONPATH
source /opt/ros/jazzy/setup.bash
cd /workspaces/talos_crowd_navigation/.claude/worktrees/evaluation
source install/setup.bash
ros2 run bullet_sims walking --ros-args \
  -p render:=true -p early_stop:=false \
  -p scenario:=random -p seed:="${SEED}" -p n_peds:=10 \
  -p robot_responsibility:=0.2 -p ped_responsibility:=0.3 \
  -p goal_share:=0.4 -p crowd_extent:=5.0 \
  -p safety_margin:=0.2 -p desired_speed:=1.2 -p max_speed:=1.5 \
  -p crowd_avoids_robot:=true -p pedestrian_sensor_data:=false \
  -p walk_duration:=45.0
