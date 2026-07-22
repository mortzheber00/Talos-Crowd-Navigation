#!/usr/bin/env bash
# Run INSIDE the devcontainer. Watch the Marienplatz (random) crowd at a chosen
# density with the MuJoCo viewer on, to judge how many pedestrians is a sensible
# hard point for the failure-boundary plane.
#
# Usage: ./experiments/watch_density.sh [n_peds] [seed] [robot_resp] [ped_resp]
#   n_peds      number of pedestrians   (default 24)
#   seed        RNG seed                (default 0)
#   robot_resp  robot ORCA share        (default 0.2)
#   ped_resp    crowd ORCA share        (default 0.3;  set 0.0 for a non-yielding crowd)
#
# Examples:
#   ./experiments/watch_density.sh 16          # 16 peds, cooperative crowd
#   ./experiments/watch_density.sh 24 3         # 24 peds, seed 3
#   ./experiments/watch_density.sh 24 0 0.2 0.0 # 24 peds, crowd yields nothing
NPEDS="${1:-24}"
SEED="${2:-0}"
RROBOT="${3:-0.2}"
RPED="${4:-0.3}"

export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:$PYTHONPATH
source /opt/ros/jazzy/setup.bash
cd /workspaces/talos_crowd_navigation/.claude/worktrees/evaluation
source install/setup.bash

# random n=16 seed=4 rr=0.2 pr=0.3  n_coll=12 min_sep=-0.5953166300710155

echo "[watch] n_peds=${NPEDS} seed=${SEED} robot_resp=${RROBOT} ped_resp=${RPED}"
ros2 run bullet_sims walking --ros-args \
  -p render:=true -p early_stop:=false \
  -p scenario:=random -p seed:="${SEED}" -p n_peds:="${NPEDS}" \
  -p robot_responsibility:="${RROBOT}" -p ped_responsibility:="${RPED}" \
  -p goal_share:=0.4 -p crowd_extent:=5.0 \
  -p safety_margin:=0.2 -p desired_speed:=1.2 -p max_speed:=1.5 \
  -p crowd_avoids_robot:=true -p pedestrian_sensor_data:=false \
  -p walk_duration:=45.0
