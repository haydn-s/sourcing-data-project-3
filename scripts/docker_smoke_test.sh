#!/bin/sh
# End-to-end check of the Docker image (make docker-smoke, also run in CI):
# generate the demo data and train inside the container, then serve the model
# the way Lambda runs it (non-root user, read-only filesystem except /tmp) and
# call /health and /predict.
set -eu

IMAGE="${IMAGE:-project3}"
PORT="${SMOKE_PORT:-8081}"
NAME="project3-smoke-$$"
WORK="$(mktemp -d)"
RUN_AS="$(id -u):$(id -g)"   # files in $WORK stay owned by you, and nothing runs as root

cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "--> API container logs:"; docker logs "$NAME" || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -rf "$WORK" || true
  exit "$status"
}
trap cleanup EXIT
mkdir -p "$WORK/data" "$WORK/models"

echo "--> Generating demo data and training in the container"
docker run --rm --user "$RUN_AS" -v "$WORK/data:/app/data" "$IMAGE" python -m src.demo_data
docker run --rm --user "$RUN_AS" -e WANDB_MODE=disabled \
  -v "$WORK/data:/app/data" -v "$WORK/models:/app/models" "$IMAGE" train

echo "--> Serving with a read-only filesystem, as on Lambda"
docker run -d --name "$NAME" --user "$RUN_AS" --read-only --tmpfs /tmp \
  -p "$PORT:8080" -v "$WORK/models:/app/models:ro" "$IMAGE" >/dev/null

for _ in $(seq 1 60); do
  curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://localhost:$PORT/health"; echo
curl -fsS -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" -d @tests/sample_request.json; echo
echo "--> Docker smoke test passed"
