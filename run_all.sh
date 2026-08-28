#!/usr/bin/env bash
# End-to-end runner for the EKF -> ROS pipeline described in instructions.md
# and TESTING.md:
#   1. Run the test suite (Levels 1/2/4/5 — no Docker required).
#   2. Open the message pipeline: start a ROS Noetic container running
#      rosbridge_websocket on :9090 (stock image by default; pass --catkin
#      to build/run the companion catkin_ws image instead).
#   3. Do the actual data transfer: run scripts/e2e_check.py --ros, which
#      drives both hook points (run_ekf_rollout + OnlineEKF) and streams
#      live nav_msgs/Odometry to rosbridge, then verify the topics landed.
#
# Usage:
#   ./run_all.sh                          # stock image, stub sim, synthetic data
#   ./run_all.sh --model best_model.pt --data-dir path/to/traj_6
#   ./run_all.sh --catkin                 # use the catkin_ws image instead
#                                          # (also starts foxglove_bridge on
#                                          # :8765 -- see the Foxglove note
#                                          # printed at the end)
#   ./run_all.sh --keep                   # leave the container running afterwards
#   ./run_all.sh --file-only --data-dir path/to/traj_6
#                                          # skip Docker/rosbridge/live streaming
#                                          # entirely; just replay the EKF from
#                                          # source data and write the rollout
#                                          # file(s) (rollout_ekf.txt /
#                                          # rollout_ekf_online.txt)
#
# Env vars (see instructions.md "Configuration"):
#   ROSBRIDGE_URL   default ws://localhost:9090
#   CATKIN_WS       path to the catkin_ws checkout (only with --catkin)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONTAINER_NAME="rosbridge"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
export ROSBRIDGE_URL="${ROSBRIDGE_URL:-ws://localhost:${ROSBRIDGE_PORT}}"

USE_CATKIN=0
KEEP=0
FILE_ONLY=0
MODEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --catkin) USE_CATKIN=1; shift ;;
    --keep) KEEP=1; shift ;;
    --file-only) FILE_ONLY=1; shift ;;
    --model) MODEL_ARGS+=(--model "$2"); shift 2 ;;
    --data-dir) MODEL_ARGS+=(--data-dir "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$FILE_ONLY" -eq 1 && "$USE_CATKIN" -eq 1 ]]; then
  echo "--file-only and --catkin are mutually exclusive." >&2
  exit 1
fi

log() { printf '\n=== %s ===\n' "$1"; }

cleanup() {
  if [[ "$KEEP" -eq 0 ]]; then
    log "Cleanup"
    if [[ "$FILE_ONLY" -eq 0 ]]; then
      if [[ "$USE_CATKIN" -eq 1 ]]; then
        docker compose -f docker/docker-compose.ros-noetic.yml down || true
      else
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
      fi
    fi
  else
    echo "--keep set: leaving the ROS container running (if any)."
  fi
}
trap cleanup EXIT

# --- 1. Tests -----------------------------------------------------------
log "1/3 Running test suite"
python3 -m pytest tests/ -q

if [[ "$FILE_ONLY" -eq 1 ]]; then
  # Bypass the live rosbridge pipeline entirely: replay the EKF from source
  # data (or synthetic data if --data-dir was not given) and just write the
  # rollout file(s). No Docker, no websocket.
  log "2/2 Replaying EKF from source data (file-only, no live ROS stream)"
  python3 scripts/e2e_check.py "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"
  log "Done"
  exit 0
fi

# --- 2. Open the message pipeline ---------------------------------------
log "2/3 Starting ROS Noetic + rosbridge"

if [[ "$USE_CATKIN" -eq 1 ]]; then
  : "${CATKIN_WS:?--catkin requires CATKIN_WS to point at your catkin_ws checkout}"
  export CATKIN_WS
  docker compose -f docker/docker-compose.ros-noetic.yml up -d --build
  READY_CMD=(docker compose -f docker/docker-compose.ros-noetic.yml logs)
  EXEC_CMD=(docker compose -f docker/docker-compose.ros-noetic.yml exec -T ros-noetic bash -lc)
else
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER_NAME" -p "${ROSBRIDGE_PORT}:9090" \
    ros:noetic-ros-base bash -lc '
      source /opt/ros/noetic/setup.bash
      apt-get update -qq && apt-get install -y -qq --no-install-recommends ros-noetic-rosbridge-server
      roscore & sleep 6
      roslaunch --wait rosbridge_server rosbridge_websocket.launch address:=0.0.0.0 port:=9090
    '
  READY_CMD=(docker logs "$CONTAINER_NAME")
  EXEC_CMD=(docker exec "$CONTAINER_NAME" bash -lc)
fi

echo "Waiting for rosbridge to come up (first run installs packages, can take a few minutes)..."
for i in $(seq 1 90); do
  if "${READY_CMD[@]}" 2>&1 | grep -q "Rosbridge WebSocket server started"; then
    echo "rosbridge is up."
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    echo "Timed out waiting for rosbridge to start." >&2
    exit 1
  fi
  sleep 5
done

# --- 3. Actual data transfer ---------------------------------------------
log "3/3 Streaming EKF state to ROS"
python3 scripts/e2e_check.py --ros "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"

log "Verifying topics"
"${EXEC_CMD[@]}" 'source /opt/ros/noetic/setup.bash; rostopic list | grep tensegrity'

if [[ "$USE_CATKIN" -eq 1 ]]; then
  cat <<EOF

Foxglove: open https://app.foxglove.dev, connect to ws://localhost:${FOXGLOVE_PORT},
then add a 3D panel -- it renders nav_msgs/Odometry natively. Rods should be
~0.325 m long.
EOF
else
  cat <<EOF

Foxglove note: foxglove_bridge is only started by the catkin_ws image. Re-run
with --catkin (and CATKIN_WS set) to get it on ws://localhost:${FOXGLOVE_PORT},
then open https://app.foxglove.dev and connect to that URL.
EOF
fi

log "Done"
