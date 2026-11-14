"""Move files between local disk and S3.

Reads work without AWS credentials when the object is public (our processed
data is), so anyone can rerun training. Writes need credentials; without them,
uploads are skipped with a warning instead of failing the run.

boto3 is imported lazily so the API doesn't pay for it when serving a local model.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def is_s3_uri(uri: str | Path) -> bool:
    return str(uri).startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split `s3://bucket/key/path` into `("bucket", "key/path")`."""
    if not is_s3_uri(uri):
        raise ValueError(f"not an s3:// URI: '{uri}'")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI needs a bucket and a key: '{uri}'")
    return bucket, key


def join_uri(prefix: str | Path, *parts: str) -> str:
    """Join path parts onto a local path or an s3:// prefix."""
    if is_s3_uri(prefix):
        return "/".join([str(prefix).rstrip("/"), *(p.strip("/") for p in parts)])
    return str(Path(prefix).joinpath(*parts))


def has_aws_credentials() -> bool:
    import boto3
    from botocore.exceptions import BotoCoreError

    try:
        return boto3.Session().get_credentials() is not None
    except BotoCoreError as err:  # e.g. AWS_PROFILE names a profile that isn't configured
        log.warning("Ignoring AWS credentials: %s", err)
        return False


def _s3_client(*, signed: bool):
    import boto3

    if signed:
        return boto3.client("s3")
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download(uri: str, dest: Path) -> Path:
    """Copy `uri` (local path or s3://) to `dest` and return `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not is_s3_uri(uri):
        if Path(uri).resolve() != dest.resolve():
            shutil.copyfile(uri, dest)
        return dest
    bucket, key = parse_s3_uri(uri)
    signed = has_aws_credentials()
    log.info("Downloading %s%s", uri, "" if signed else " (anonymous: no AWS credentials)")
    _s3_client(signed=signed).download_file(bucket, key, str(dest))
    return dest


def upload(src: Path, uri: str) -> bool:
    """Copy `src` to `uri` (local path or s3://). Returns False if skipped for lack of credentials."""
    if not is_s3_uri(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, uri)
        return True
    if not has_aws_credentials():
        log.warning("No AWS credentials, so %s was not uploaded to %s", src, uri)
        return False
    bucket, key = parse_s3_uri(uri)
    _s3_client(signed=True).upload_file(str(src), bucket, key)
    log.info("Uploaded %s to %s", src, uri)
    return True
