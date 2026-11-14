"""Stage 1 of the data pipeline: fetch each source in the config and save a raw snapshot.

Each source in `ingest.sources` is fetched according to its `type`:

- `file`: a CSV or Parquet file at a local path, an http(s) URL, or an s3:// or gs:// URI.
- `http`: a GET request to a web API that returns JSON records or CSV.
- anything else: a function registered below with `@source_type("<name>")`, for
  APIs that need paging, several requests, or other special handling.

Values written as `${NAME}` in a source's uri, url, params, headers, or options
come from environment variables (.env locally), so API keys stay out of config.yaml.

Snapshots are saved unchanged, as CSV, in `ingest.raw_dir`, and uploaded to
`ingest.remote_prefix` when it's set. A manifest.json next to them records where
each source came from, when it was fetched, its size, and a checksum. Cleaning
happens in the next stage (src/prepare.py), so changing it never needs the APIs again.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PipelineConfig, SourceConfig
from src.data import file_sha256, read_table
from src.storage import download, is_remote_uri, join_uri, upload

log = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
USER_AGENT = "sourcing-data-project-3 data pipeline (python-requests)"
TIMEOUT_SECONDS = 60

Fetcher = Callable[[SourceConfig], pd.DataFrame]
FETCHERS: dict[str, Fetcher] = {}


def source_type(name: str) -> Callable[[Fetcher], Fetcher]:
    """Register a function that fetches `type: <name>` sources and returns their rows."""

    def register(fetch: Fetcher) -> Fetcher:
        FETCHERS[name] = fetch
        return fetch

    return register


def snapshot_file(source_name: str) -> str:
    return f"{source_name}.csv"


# Environment variables and secrets

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any, *, source: str) -> Any:
    """Replace each `${NAME}` in strings, including inside dicts and lists, with that environment variable."""
    if isinstance(value, str):

        def lookup(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"source '{source}' needs the environment variable {name}. "
                    "Add it to .env (see .env.example)"
                )
            return os.environ[name]

        return _ENV_REFERENCE.sub(lookup, value)
    if isinstance(value, dict):
        return {key: expand_env(item, source=source) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item, source=source) for item in value]
    return value


def secrets_in(source: SourceConfig) -> list[str]:
    """The environment values a source refers to, so error messages can hide them."""
    text = json.dumps(source.model_dump(mode="json"))
    names = set(_ENV_REFERENCE.findall(text))
    return [os.environ[name] for name in names if os.environ.get(name)]


def scrub(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


# Built-in source types


@source_type("file")
def fetch_file(source: SourceConfig) -> pd.DataFrame:
    uri = expand_env(source.uri, source=source.name)
    if is_remote_uri(uri):
        with tempfile.TemporaryDirectory() as tmp:
            return read_table(download(uri, Path(tmp) / Path(uri).name))
    return read_table(uri)


@source_type("http")
def fetch_http(source: SourceConfig) -> pd.DataFrame:
    response = http_get(
        expand_env(source.url, source=source.name),
        params=expand_env(source.params, source=source.name),
        headers=expand_env(source.headers, source=source.name),
        secrets=secrets_in(source),
    )
    if source.format == "csv":
        return pd.read_csv(io.StringIO(response.text))
    return records_frame(response.json(), source.records, source=source.name)


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    secrets: Iterable[str] = (),
):
    """GET `url`, retrying rate limits and server errors. Error messages hide `secrets`."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=3,
        backoff_factor=1,  # waits 1, 2, then 4 seconds
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,  # return the last response so its status can be reported
    )
    session = requests.Session()
    for prefix in ("https://", "http://"):
        session.mount(prefix, HTTPAdapter(max_retries=retry))

    secrets = list(secrets)
    try:
        response = session.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as err:
        # `from None`: the original exception's message includes the full URL,
        # query parameters (and so any API key) included.
        raise ConnectionError(scrub(f"GET {url} failed: {err}", secrets)) from None
    if not response.ok:
        detail = response.text[:300].strip()
        raise ConnectionError(
            scrub(f"GET {url} returned HTTP {response.status_code}: {detail}", secrets)
        )
    return response


def records_frame(payload: Any, path: str | None, *, source: str) -> pd.DataFrame:
    """Turn the records in a JSON response into rows. `path` is dotted, e.g. `results.items`.

    The records can be a list of objects (one per row), or an object of
    equal-length lists (one per column), as some APIs like Open-Meteo return.
    """
    records = payload
    for key in path.split(".") if path else []:
        if not isinstance(records, dict) or key not in records:
            raise ValueError(f"source '{source}': the response has no '{key}' (records: {path})")
        records = records[key]
    if isinstance(records, list):
        return pd.json_normalize(records)
    if _is_columns(records):
        return pd.DataFrame(records)
    where = f" at '{path}'" if path else ""
    raise ValueError(
        f"source '{source}': expected a list of records{where}, got a "
        f"{type(records).__name__}. Set `records` to the key that holds them"
    )


def _is_columns(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    lists = [item for item in value.values() if isinstance(item, list)]
    return len(lists) == len(value) and len({len(item) for item in lists}) == 1


# Snapshots and the manifest


def read_manifest(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def ingest(cfg: PipelineConfig, only: Iterable[str] | None = None) -> dict[str, Any]:
    """Fetch every source (or just those named in `only`) and save their snapshots.

    Returns the updated manifest. It's written after each source, so a failure
    partway through keeps the snapshots that did succeed.
    """
    sources = _selected_sources(cfg, only)
    raw_dir = cfg.ingest.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(raw_dir)

    for source in sources:
        manifest[source.name] = ingest_source(source, raw_dir)
        (raw_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
        if cfg.ingest.remote_prefix:
            for name in (snapshot_file(source.name), MANIFEST_FILE):
                upload(raw_dir / name, join_uri(cfg.ingest.remote_prefix, name))
    return manifest


def ingest_source(source: SourceConfig, raw_dir: Path) -> dict[str, Any]:
    """Fetch one source, save its snapshot, and return its manifest entry."""
    log.info("Fetching %s (type %s)", source.name, source.type)
    frame = FETCHERS[source.type](source)
    if frame.empty:
        log.warning("Source %s returned no rows", source.name)

    path = raw_dir / snapshot_file(source.name)
    frame.to_csv(path, index=False)
    log.info("Saved %d rows x %d columns to %s", len(frame), frame.shape[1], path)
    return {
        "type": source.type,
        # As written in the config, so ${NAME} placeholders stay unexpanded.
        "location": source.url or source.uri,
        "params": source.params,
        "options": source.options,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "sha256": file_sha256(path),
    }


def _selected_sources(cfg: PipelineConfig, only: Iterable[str] | None) -> list[SourceConfig]:
    sources = cfg.ingest.sources
    if not sources:
        raise ValueError("no sources to ingest. Add them under ingest.sources in the config")
    if only:
        wanted = set(only)
        unknown = sorted(wanted - {source.name for source in sources})
        if unknown:
            raise ValueError(
                f"no source named {unknown} in the config. "
                f"Its sources are: {[source.name for source in sources]}"
            )
        sources = [source for source in sources if source.name in wanted]
    # Check every type before fetching anything, since some APIs are slow.
    for source in sources:
        if source.type not in FETCHERS:
            raise ValueError(
                f"source '{source.name}' has type '{source.type}', which isn't registered. "
                f"Types: {', '.join(sorted(FETCHERS))}"
            )
    return sources


# Dataset-specific source types go below, registered with @source_type("<name>").
# Use http_get() for requests and expand_env() for any ${NAME} values in
# source.options. For example, an API that returns 1,000 rows per page:
#
# @source_type("paged_api")
# def fetch_paged_api(source: SourceConfig) -> pd.DataFrame:
#     url = expand_env(source.url, source=source.name)
#     pages, offset = [], 1
#     while True:
#         params = {**expand_env(source.params, source=source.name), "offset": offset}
#         payload = http_get(url, params=params, secrets=secrets_in(source)).json()
#         pages.append(records_frame(payload, source.records, source=source.name))
#         if len(pages[-1]) < 1000:
#             return pd.concat(pages, ignore_index=True)
#         offset += 1000
