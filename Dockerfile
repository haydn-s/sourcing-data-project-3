# One image for the whole project:
#   docker run -p 8080:8080 project3                  serve the API (the default; needs a model)
#   docker run project3 python -m data_pipeline.push  run any other command
# A `train` command comes with the ML pipeline (ml_pipeline/README.md).
#
# The same image runs on Google Cloud Run (make deploy). Cloud Run sets $PORT,
# which the API listens on, and runs only linux/amd64 images, so build for that
# platform on every machine, Apple Silicon included:
#   make docker-build
# (docker buildx build --platform linux/amd64 --provenance=false --sbom=false ...)

# Python matches PyDev and CI (3.12). Dependabot proposes patch updates only;
# move all three to a new minor version together, once its wheels exist.
FROM python:3.12.15-slim

# A container run with --user has no writable home directory, so runtime files
# go to /tmp: no .pyc files, and matplotlib's and W&B's files go there.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    WANDB_DIR=/tmp \
    WANDB_DATA_DIR=/tmp/wandb-data \
    WANDB_CACHE_DIR=/tmp/wandb-cache \
    WANDB_CONFIG_DIR=/tmp/wandb-config \
    PORT=8080

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY entrypoint.sh config.yaml ./
COPY shared/ shared/
COPY data_pipeline/ data_pipeline/
COPY ml_pipeline/ ml_pipeline/
COPY api/ api/

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve"]
