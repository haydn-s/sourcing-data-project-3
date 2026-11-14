"""Move files between local disk and cloud storage: Amazon S3 (s3://) or Google Cloud Storage (gs://).

Reads work without credentials when the object is public (our processed data
is), so anyone can rerun training. Writes need credentials; without them,
uploads are skipped with a warning instead of failing the run.

- S3 signs requests with your AWS profile (AWS_PROFILE), or with the function's
  IAM role on Lambda.
- Cloud Storage uses Application Default Credentials. Set them up once with
  `gcloud auth application-default login`.

The cloud SDKs are imported lazily so the API doesn't pay for them when serving a local model.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import warnings
from pathlib import Path

log = logging.getLogger(__name__)


def is_s3_uri(uri: str | Path) -> bool:
    return str(uri).startswith("s3://")


def is_gcs_uri(uri: str | Path) -> bool:
    return str(uri).startswith("gs://")


def is_remote_uri(uri: str | Path) -> bool:
    return is_s3_uri(uri) or is_gcs_uri(uri)


def split_uri(uri: str) -> tuple[str, str]:
    """Split `s3://bucket/key/path` or `gs://bucket/key/path` into `("bucket", "key/path")`."""
    if not is_remote_uri(uri):
        raise ValueError(f"not an s3:// or gs:// URI: '{uri}'")
    bucket, _, key = str(uri).split("://", 1)[1].partition("/")
    if not bucket or not key:
        raise ValueError(f"cloud URI needs a bucket and a key: '{uri}'")
    return bucket, key


def public_url(uri: str) -> str:
    """The HTTPS address of a public object, which reads with no SDK or credentials."""
    bucket, key = split_uri(uri)
    if is_gcs_uri(uri):
        return f"https://storage.googleapis.com/{bucket}/{key}"
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def join_uri(prefix: str | Path, *parts: str) -> str:
    """Join path parts onto a local path or a cloud prefix."""
    if is_remote_uri(prefix):
        return "/".join([str(prefix).rstrip("/"), *(p.strip("/") for p in parts)])
    return str(Path(prefix).joinpath(*parts))


def download(uri: str, dest: Path) -> Path:
    """Copy `uri` (local path, s3://, or gs://) to `dest` and return `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_s3_uri(uri):
        return _download_s3(uri, dest)
    if is_gcs_uri(uri):
        return _download_gcs(uri, dest)
    if Path(uri).resolve() != dest.resolve():
        shutil.copyfile(uri, dest)
    return dest


def upload(src: Path, uri: str) -> bool:
    """Copy `src` to `uri` (local path, s3://, or gs://). Returns False if skipped for lack of credentials."""
    if is_s3_uri(uri):
        return _upload_s3(src, uri)
    if is_gcs_uri(uri):
        return _upload_gcs(src, uri)
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, uri)
    return True


# Amazon S3


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


def _download_s3(uri: str, dest: Path) -> Path:
    bucket, key = split_uri(uri)
    signed = has_aws_credentials()
    log.info("Downloading %s%s", uri, "" if signed else " (anonymous: no AWS credentials)")
    _s3_client(signed=signed).download_file(bucket, key, str(dest))
    return dest


def _upload_s3(src: Path, uri: str) -> bool:
    if not has_aws_credentials():
        log.warning("No AWS credentials, so %s was not uploaded to %s", src, uri)
        return False
    bucket, key = split_uri(uri)
    _s3_client(signed=True).upload_file(str(src), bucket, key)
    log.info("Uploaded %s to %s", src, uri)
    return True


# Google Cloud Storage


@functools.cache
def has_gcp_credentials() -> bool:
    """Whether Application Default Credentials are set up. Cached: the check can be slow."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        with warnings.catch_warnings():
            # gcloud user credentials without a quota project warn about it, but
            # Cloud Storage bills the bucket's project, so it doesn't apply.
            warnings.simplefilter("ignore", UserWarning)
            google.auth.default()
    except DefaultCredentialsError:
        return False
    return True


def _gcs_client(*, signed: bool):
    from google.cloud import storage as gcs

    if not signed:
        return gcs.Client.create_anonymous_client()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the quota-project warning, as above
        # Reading and writing objects doesn't need a project, so None is fine.
        return gcs.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _download_gcs(uri: str, dest: Path) -> Path:
    from google.api_core import exceptions

    bucket, key = split_uri(uri)
    signed = has_gcp_credentials()
    log.info("Downloading %s%s", uri, "" if signed else " (anonymous: no Google Cloud credentials)")
    blob = _gcs_client(signed=signed).bucket(bucket).blob(key)
    try:
        blob.download_to_filename(str(dest))
    except exceptions.NotFound as err:
        raise FileNotFoundError(f"{uri} doesn't exist") from err
    except (exceptions.Forbidden, exceptions.Unauthorized) as err:
        hint = (
            "" if signed else " It isn't public, so sign in: gcloud auth application-default login"
        )
        raise PermissionError(f"no permission to read {uri}.{hint}") from err
    return dest


def _upload_gcs(src: Path, uri: str) -> bool:
    if not has_gcp_credentials():
        log.warning(
            "No Google Cloud credentials, so %s was not uploaded to %s. "
            "Sign in with: gcloud auth application-default login",
            src,
            uri,
        )
        return False
    bucket, key = split_uri(uri)
    blob = _gcs_client(signed=True).bucket(bucket).blob(key)
    # Public objects are cached for up to an hour by default. no-cache makes
    # readers get the new file as soon as it's uploaded.
    blob.cache_control = "no-cache"
    blob.upload_from_filename(str(src))
    log.info("Uploaded %s to %s", src, uri)
    return True
