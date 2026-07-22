#!/usr/bin/env bash
# Run INSIDE the devcontainer. Replay a qh_hard "did not reach" run with the
# viewer on, to see the passive robot (robot_responsibility=0) bog down in the
# 48-pedestrian jam. (Chaotic and slow to watch -- the Marienplatz script is the
# more useful one.)
# Usage: ./experiments/watch_qhh_timeout.sh [seed]   (seed 0-9, default 0)
SEED="${1:-0}"
export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:$PYTHONPATH
source /opt/ros/jazzy/setup.bash
cd /workspaces/talos_crowd_navigation/.claude/worktrees/evaluation
source install/setup.bash
ros2 run bullet_sims walking --ros-args \
  -p render:=true -p early_stop:=false \
  -p scenario:=random -p seed:="${SEED}" -p n_peds:=48 \
  -p robot_responsibility:=0.0 -p ped_responsibility:=0.0 \
  -p goal_share:=0.4 -p crowd_extent:=5.0 \
  -p safety_margin:=0.2 -p desired_speed:=1.2 -p max_speed:=1.5 \
  -p crowd_avoids_robot:=true -p pedestrian_sensor_data:=false \
  -p walk_duration:=45.0
