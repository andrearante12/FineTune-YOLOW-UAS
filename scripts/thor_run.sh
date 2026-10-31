#!/usr/bin/env bash
# Run a command inside the Ultralytics arm64 container on Jetson Thor, with the
# repo mounted at /work and a persistent cache (CLIP weights, ultralytics settings).
# Usage: scripts/thor_run.sh python3 scripts/evaluate.py ...
set -euo pipefail

IMAGE="${THOR_IMAGE:-yoloworld-thor}"  # built from Dockerfile.thor
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${THOR_CACHE:-$HOME/.cache/yoloworld_thor}"
mkdir -p "$CACHE"

if ! docker info >/dev/null 2>&1; then
    # docker group membership not active in this login session yet; sg runs the
    # command through sh, so base64 the bash-quoted argv to survive any quoting
    cmd="$(printf '%q ' "$0" "$@" | base64 -w0)"
    exec sg docker -c "bash -c \"eval \\\"\\\$(echo $cmd | base64 -d)\\\"\""
fi

exec docker run --rm --runtime nvidia --ipc=host ${THOR_NAME:+--name "$THOR_NAME"} \
    -u "$(id -u):$(id -g)" -e HOME=/cache -v "$CACHE:/cache" \
    -v "$REPO:/work" -w /work \
    "$IMAGE" "$@"
