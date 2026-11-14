#!/bin/sh
# Container entry point.
#   serve                   run the API on $PORT (the default command)
#   train [ARGS...]         run the training pipeline; ARGS go to src.train
#   anything else           run it as a command, e.g. python -m src.demo_data
set -eu

command="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi

case "$command" in
  serve) exec uvicorn api.main:create_app --factory --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  train) exec python -m src.train "$@" ;;
  *) exec "$command" "$@" ;;
esac
