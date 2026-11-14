# One image for the whole project:
#   docker run project3 train [--config FILE] [--set KEY=VALUE]   run the training pipeline
#   docker run -p 8080:8080 project3                               serve the API (the default)
#
# The same image runs on AWS Lambda. The Lambda Web Adapter (copied in below)
# turns Lambda invocations into HTTP requests to uvicorn, and does nothing
# outside Lambda. Lambda needs a single-platform image without attestations:
#   make docker-build
# (docker buildx build --platform linux/amd64 --provenance=false --sbom=false ...)

# Python matches PyDev. Dependabot proposes base image updates.
FROM python:3.12.15-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

# Lambda's filesystem is read-only except /tmp, and a container run with
# --user has no writable home directory, so nothing may write elsewhere at
# runtime: no .pyc files, and matplotlib's and W&B's files go to /tmp.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    WANDB_DIR=/tmp \
    WANDB_DATA_DIR=/tmp/wandb-data \
    WANDB_CACHE_DIR=/tmp/wandb-cache \
    WANDB_CONFIG_DIR=/tmp/wandb-config \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY entrypoint.sh config.yaml ./
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve"]
