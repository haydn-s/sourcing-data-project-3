#!/bin/sh
# Check of the Docker image (make docker-smoke): the image builds, its
# packages import, and the data pipeline commands start, as a non-root user.
# It's out of CI until the ML pipeline produces a model.
#
# The train-and-serve check (train inside the container, then serve with a
# read-only filesystem and call /health and /predict) needs a trained model,
# so it comes back with the ML pipeline. See ml_pipeline/README.md.
set -eu

IMAGE="${IMAGE:-project3}"
RUN_AS="$(id -u):$(id -g)"   # nothing runs as root

echo "--> Importing the packages in the container"
docker run --rm --user "$RUN_AS" "$IMAGE" \
  python -c "import api.main, data_pipeline.ingest, data_pipeline.prepare, data_pipeline.push, ml_pipeline"

echo "--> Starting each data pipeline command"
for stage in ingest prepare push; do
  docker run --rm --user "$RUN_AS" "$IMAGE" python -m "data_pipeline.$stage" --help >/dev/null
done
echo "--> Docker smoke test passed"
