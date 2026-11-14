"""Move files between local disk and Google Cloud Storage (gs:// URIs).

Our bucket is publicly readable, so downloads try an anonymous request first:
anyone can rerun training, and the API on Lambda can load the model, without a
Google login. Only a private object falls back to your credentials. Uploads
need credentials; without them, they're skipped with a warning instead of
failing the run.

Credentials are Application Default Credentials. Set them up once with
`gcloud auth application-default login`.

The Google SDK is imported lazily so the API doesn't pay for it when serving a local model.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)

SIGN_IN = "gcloud auth application-default login"


def is_gcs_uri(uri: str | Path) -> bool:
    return str(uri).startswith("gs://")


def split_uri(uri: str) -> tuple[str, str]:
    """Split `gs://bucket/key/path` into `("bucket", "key/path")`."""
    if not is_gcs_uri(uri):
        raise ValueError(f"not a gs:// URI: '{uri}'")
    bucket, _, key = str(uri).removeprefix("gs://").partition("/")
    if not bucket or not key:
        raise ValueError(f"gs:// URI needs a bucket and a key: '{uri}'")
    return bucket, key


def public_url(uri: str) -> str:
    """The HTTPS address of a public object, which reads with no SDK or credentials."""
    bucket, key = split_uri(uri)
    return f"https://storage.googleapis.com/{bucket}/{key}"


def join_uri(prefix: str | Path, *parts: str) -> str:
    """Join path parts onto a local path or a gs:// prefix."""
    if is_gcs_uri(prefix):
        return "/".join([str(prefix).rstrip("/"), *(p.strip("/") for p in parts)])
    return str(Path(prefix).joinpath(*parts))


def download(uri: str, dest: Path) -> Path:
    """Copy `uri` (local path or gs://) to `dest` and return `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_gcs_uri(uri):
        return _download_gcs(uri, dest)
    if Path(uri).resolve() != dest.resolve():
        shutil.copyfile(uri, dest)
    return dest


def upload(src: Path, uri: str) -> bool:
    """Copy `src` to `uri` (local path or gs://). Returns False if skipped for lack of credentials."""
    if is_gcs_uri(uri):
        return _upload_gcs(src, uri)
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, uri)
    return True


@contextlib.contextmanager
def _quiet_google_auth() -> Iterator[None]:
    """Hide google-auth's warnings about a missing project or quota project.

    Neither applies: reading and writing objects needs no project, and Cloud
    Storage bills the bucket's own project.
    """
    logger = logging.getLogger("google.auth._default")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield
    finally:
        logger.setLevel(level)


@functools.cache
def has_gcp_credentials() -> bool:
    """Whether Application Default Credentials are set up. Cached: the check can be slow."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        with _quiet_google_auth():
            google.auth.default()
    except DefaultCredentialsError:
        return False
    return True


def _client(*, signed: bool):
    from google.cloud import storage as gcs

    if not signed:
        return gcs.Client.create_anonymous_client()
    with _quiet_google_auth():
        return gcs.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _download_gcs(uri: str, dest: Path) -> Path:
    from google.api_core import exceptions

    bucket, key = split_uri(uri)
    log.info("Downloading %s", uri)
    try:
        try:
            _client(signed=False).bucket(bucket).blob(key).download_to_filename(str(dest))
        except (exceptions.Forbidden, exceptions.Unauthorized) as err:
            if not has_gcp_credentials():
                raise PermissionError(f"{uri} isn't public. Sign in to read it: {SIGN_IN}") from err
            log.info("%s isn't public, so reading it with your Google credentials", uri)
            _client(signed=True).bucket(bucket).blob(key).download_to_filename(str(dest))
    except exceptions.NotFound as err:
        raise FileNotFoundError(f"{uri} doesn't exist") from err
    except (exceptions.Forbidden, exceptions.Unauthorized) as err:
        raise PermissionError(f"your Google account can't read {uri}") from err
    return dest


def _upload_gcs(src: Path, uri: str) -> bool:
    if not has_gcp_credentials():
        log.warning(
            "No Google Cloud credentials, so %s was not uploaded to %s. Run: %s", src, uri, SIGN_IN
        )
        return False
    bucket, key = split_uri(uri)
    blob = _client(signed=True).bucket(bucket).blob(key)
    # Public objects are cached for up to an hour by default. no-cache makes
    # readers get the new file as soon as it's uploaded.
    blob.cache_control = "no-cache"
    blob.upload_from_filename(str(src))
    log.info("Uploaded %s to %s", src, uri)
    return True
