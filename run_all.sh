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
#   ./run_all.sh --stream                 # ONE command for a live Foxglove view:
#                                          # tests -> rosbridge -> continuous
#                                          # publish until Ctrl-C. Reuses a running
#                                          # container and leaves it up for reuse.
#   ./run_all.sh --stream --rate 20 --model best.pt --data-dir path/to/traj_6
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
# Flags: --stream (continuous live publish for Foxglove), --rate <hz> (with
# --stream, default 15), --keep, --catkin, --file-only, --model, --data-dir.
#
# Env vars (see instructions.md "Configuration"):
#   ROSBRIDGE_URL    default ws://localhost:9090
#   CATKIN_WS        path to the catkin_ws checkout (only with --catkin)
#   DOCKER_PLATFORM  image platform for the stock rosbridge container
#                    (default linux/amd64 so ROS Noetic runs on Apple Silicon
#                    via emulation; set DOCKER_PLATFORM= empty for host arch)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONTAINER_NAME="rosbridge"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
# Default to amd64: ROS Noetic publishes no arm64 image, so on Apple Silicon the
# host-arch pull fails; use '-' (not ':-') so an explicit empty value opts out.
DOCKER_PLATFORM="${DOCKER_PLATFORM-linux/amd64}"
export ROSBRIDGE_URL="${ROSBRIDGE_URL:-ws://localhost:${ROSBRIDGE_PORT}}"

USE_CATKIN=0
KEEP=0
FILE_ONLY=0
STREAM=0
RATE=15
MODEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --catkin) USE_CATKIN=1; shift ;;
    --keep) KEEP=1; shift ;;
    --file-only) FILE_ONLY=1; shift ;;
    --stream) STREAM=1; shift ;;
    --rate) RATE="$2"; shift 2 ;;
    --model) MODEL_ARGS+=(--model "$2"); shift 2 ;;
    --data-dir) MODEL_ARGS+=(--data-dir "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$FILE_ONLY" -eq 1 && "$USE_CATKIN" -eq 1 ]]; then
  echo "--file-only and --catkin are mutually exclusive." >&2
  exit 1
fi
if [[ "$STREAM" -eq 1 && "$FILE_ONLY" -eq 1 ]]; then
  echo "--stream needs the live ROS pipeline; it is incompatible with --file-only." >&2
  exit 1
fi
# --stream keeps the container up so re-running is instant (it is reused, not
# rebuilt). Remove it yourself with: docker rm -f rosbridge
if [[ "$STREAM" -eq 1 ]]; then
  KEEP=1
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
elif docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  # Already running (e.g. left up by a previous --stream/--keep run): reuse it so
  # we skip the slow first-boot apt install of rosbridge.
  echo "Reusing the already-running '$CONTAINER_NAME' container."
  READY_CMD=(docker logs "$CONTAINER_NAME")
  EXEC_CMD=(docker exec "$CONTAINER_NAME" bash -lc)
else
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  # ROS Noetic is x86-only; on Apple Silicon (Colima/Docker Desktop) force the
  # amd64 image so it runs under emulation instead of failing to find an arm64
  # manifest. Override with DOCKER_PLATFORM= (empty) to use the host arch.
  docker run -d --name "$CONTAINER_NAME" -p "${ROSBRIDGE_PORT}:9090" \
    ${DOCKER_PLATFORM:+--platform "$DOCKER_PLATFORM"} \
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
if [[ "$STREAM" -eq 1 ]]; then
  log "3/3 Live-streaming EKF state to ROS (continuous)"
  cat <<EOF

Connect Foxglove NOW, then leave this running:
  1. Foxglove -> Open connection -> "Rosbridge (ROS 1)" -> ws://localhost:${ROSBRIDGE_PORT}
  2. Add a 3D panel; in its settings set Fixed frame = "world".
  3. Enable the /tensegrity/<rod>/odom topics and zoom in (rods ~0.325 m, near
     the origin). /tf (world -> rod) is broadcast so the panel can place them.
  4. Raw Messages panels on /tensegrity/ekf/covariance and /tensegrity/ekf/jacobian
     show the 36x36 matrices.

Streaming at ${RATE} Hz -- press Ctrl-C to stop (the container is left running for
next time; remove it with: docker rm -f ${CONTAINER_NAME}).
EOF
  python3 scripts/stream_live.py --rate "$RATE" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"
  log "Done"
  exit 0
fi

log "3/3 Streaming EKF state to ROS"
python3 scripts/e2e_check.py --ros "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"

log "Verifying topics"
"${EXEC_CMD[@]}" 'source /opt/ros/noetic/setup.bash; rostopic list | grep tensegrity'

cat <<EOF

Live view: the step above publishes only a few frames and then exits, so a viewer
that connects afterward sees nothing (Odometry is not latched). For a live
Foxglove/RViz view, run everything in one shot with:

  ./run_all.sh --stream

or, against this already-running container, just:

  ROSBRIDGE_URL=${ROSBRIDGE_URL} python3 scripts/stream_live.py --rate ${RATE}

Then in Foxglove connect to ws://localhost:${ROSBRIDGE_PORT} (connection type
"Rosbridge (ROS 1)"), add a 3D panel, set its fixed frame to "world", and enable
the /tensegrity/<rod>/odom topics (stream_live also broadcasts /tf so the panel
can place them). Positions are in meters (rods ~0.325 m, near the origin -- zoom
in). The synthetic trajectory is a visibility aid; pass --model/--data-dir for
real motion.
EOF
if [[ "$USE_CATKIN" -eq 1 ]]; then
  cat <<EOF

(catkin image) foxglove_bridge is also available on ws://localhost:${FOXGLOVE_PORT}.
EOF
fi

log "Done"
