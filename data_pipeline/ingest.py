"""Data pipeline, stage 1 of 3: fetch each source in the config and save a raw snapshot.

    python -m data_pipeline.ingest --config config.yaml
    python -m data_pipeline.ingest --source colonies     # just one source

Each source in `ingest.sources` is fetched according to its `type`:

- `file`: a CSV or Parquet file at a local path, an http(s) URL, or a gs:// URI.
- `http`: a GET request to a web API that returns JSON records or CSV.
- anything else: a function registered below with `@source_type("<name>")`, for
  APIs that need paging, several requests, or other special handling.

Values written as `${NAME}` in a source's uri, url, params, headers, or options
come from environment variables (.env locally), so API keys stay out of config.yaml.

Snapshots are saved unchanged, as CSV, in `ingest.raw_dir`, and uploaded to
`ingest.remote_prefix` when it's set. A manifest.json next to them records where
each source came from, when it was fetched, its size, and a checksum. Cleaning
happens in the next stage (data_pipeline/prepare.py), so changing it never needs the APIs again.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shared.cli import config_parser, load
from shared.config import PipelineConfig, SourceConfig
from shared.storage import download, is_gcs_uri, join_uri, upload
from shared.tables import file_sha256, read_table

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
            if not os.environ.get(name):  # unset, or blank in .env
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


def env_names(source: SourceConfig) -> list[str]:
    """The environment variables a source refers to as `${NAME}`."""
    return sorted(set(_ENV_REFERENCE.findall(json.dumps(source.model_dump(mode="json")))))


def secrets_in(source: SourceConfig) -> list[str]:
    """The environment values a source refers to, so they can be hidden from errors and snapshots."""
    return [os.environ[name] for name in env_names(source) if os.environ.get(name)]


def scrub(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def scrub_frame(frame: pd.DataFrame, secrets: Iterable[str]) -> tuple[pd.DataFrame, int]:
    """Replace secrets inside text cells with ***. Returns the frame and how many cells changed.

    Some APIs echo the request, API key included, into their records, and
    snapshots are uploaded to a public bucket.
    """
    secrets = [secret for secret in secrets if secret]
    if not secrets:
        return frame, 0
    frame, changed = frame.copy(), 0
    for column in frame.columns:
        values = frame[column]
        if not pd.api.types.is_string_dtype(values.dtype):
            continue
        leaks = values.map(lambda v: isinstance(v, str) and any(s in v for s in secrets))
        if leaks.any():
            changed += int(leaks.sum())
            frame[column] = values.map(lambda v: scrub(v, secrets) if isinstance(v, str) else v)
    return frame, changed


# Built-in source types


@source_type("file")
def fetch_file(source: SourceConfig) -> pd.DataFrame:
    uri = expand_env(source.uri, source=source.name)
    if is_gcs_uri(uri):
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
        return pd.read_csv(io.StringIO(response_text(response)))
    return records_frame(response.json(), source.records, source=source.name)


def response_text(response) -> str:
    """The body as text, read as UTF-8 unless the server names a charset.

    Without one, requests decodes text/* responses as ISO-8859-1, which garbles
    UTF-8 names like "Québec".
    """
    if "charset" in response.headers.get("Content-Type", "").lower():
        return response.text
    return response.content.decode("utf-8-sig", errors="replace")


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
    secrets = list(secrets)
    with requests.Session() as session:
        for prefix in ("https://", "http://"):
            session.mount(prefix, HTTPAdapter(max_retries=retry))
        try:
            response = session.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as err:
            # `from None`: the original exception's message includes the full
            # URL, query parameters (and so any API key) included.
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
    """An object whose values are equal-length lists of plain values (not records)."""
    if not isinstance(value, dict) or not value:
        return False
    lists = [item for item in value.values() if isinstance(item, list)]
    if len(lists) != len(value) or len({len(item) for item in lists}) != 1:
        return False
    # {"data": [{...}, {...}]} is a list of records under a key, not a column.
    return not any(isinstance(cell, dict | list) for column in lists for cell in column)


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
    if frame.shape[1] == 0:
        raise ValueError(f"source '{source.name}' returned no data. Check its url and params")
    if frame.empty:
        log.warning("Source %s returned no rows", source.name)
    frame, scrubbed = scrub_frame(frame, secrets_in(source))
    if scrubbed:
        log.warning(
            "Source %s echoed a secret (an API key or token) in %d cells. They're saved as ***",
            source.name,
            scrubbed,
        )

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
    # Check every type and environment variable before fetching anything, since some APIs are slow.
    for source in sources:
        if source.type not in FETCHERS:
            raise ValueError(
                f"source '{source.name}' has type '{source.type}', which isn't registered. "
                f"Types: {', '.join(sorted(FETCHERS))}"
            )
    missing = sorted({name for s in sources for name in env_names(s) if not os.environ.get(name)})
    if missing:
        raise ValueError(
            f"the sources need environment variables that aren't set: {', '.join(missing)}. "
            "Add them to .env (see .env.example)"
        )
    return sources


# Dataset-specific source types go below, registered with @source_type("<name>").
# Use http_get() for requests and expand_env() for any ${NAME} values.


def _required_url(source: SourceConfig) -> str:
    if not source.url:
        raise ValueError(f"source '{source.name}' has type {source.type}, so it needs a url")
    return expand_env(source.url, source=source.name)


NOAA_CDO_PAGE_SIZE = 1000  # the most results CDO returns per request
NOAA_CDO_PAUSE_SECONDS = 0.25  # CDO allows 5 requests a second


@source_type("noaa_cdo")
def fetch_noaa_cdo(source: SourceConfig) -> pd.DataFrame:
    """NOAA Climate Data Online (API v2), paging through every result.

    `url` is the endpoint (e.g. .../cdo-web/api/v2/data) and `params` its query
    (datasetid, stationid or locationid, startdate, enddate, ...). The token
    goes in `headers: {token: "${NOAA_CDO_TOKEN}"}`. A list value, like several
    stationid or datatypeid, is sent as repeated parameters, which is how CDO
    takes them.

    CDO also caps a query's date range (a year for daily data, ten years for
    yearly), so `options.chunk_years: N` splits startdate..enddate into
    windows of N calendar years and queries each one.
    """
    url = _required_url(source)
    params = expand_env(source.params, source=source.name)
    headers = expand_env(source.headers, source=source.name)
    secrets = secrets_in(source)
    windows = date_windows(params, source.options.get("chunk_years"), source=source.name)
    frames = []
    for start, end in windows:
        query = params if start is None else {**params, "startdate": start, "enddate": end}
        if start is not None:
            log.info("%s: %s to %s", source.name, start, end)
        frames.append(_noaa_cdo_pages(url, query, headers, secrets, source.name))
    return pd.concat(frames, ignore_index=True)


def _noaa_cdo_pages(
    url: str, params: dict[str, Any], headers: dict[str, str], secrets: list[str], name: str
) -> pd.DataFrame:
    pages, offset = [], 1
    while True:
        page = {**params, "limit": NOAA_CDO_PAGE_SIZE, "offset": offset}
        payload = http_get(url, params=page, headers=headers, secrets=secrets).json()
        time.sleep(NOAA_CDO_PAUSE_SECONDS)
        # CDO answers a query that matches nothing with an empty object.
        results = payload.get("results", [])
        pages.append(pd.json_normalize(results))
        total = payload.get("metadata", {}).get("resultset", {}).get("count", 0)
        offset += NOAA_CDO_PAGE_SIZE
        if not results or offset > total:
            return pd.concat(pages, ignore_index=True)
        log.info("%s: fetched %d of %d results", name, offset - 1, total)


def date_windows(
    params: dict[str, Any], chunk_years: int | None, *, source: str
) -> list[tuple[str | None, str | None]]:
    """Split params' startdate..enddate into windows of `chunk_years` calendar years.

    Returns [(None, None)], meaning one query as configured, when chunk_years isn't set.
    """
    if not chunk_years:
        return [(None, None)]
    try:
        first = date.fromisoformat(str(params["startdate"])[:10])
        last = date.fromisoformat(str(params["enddate"])[:10])
    except (KeyError, ValueError) as err:
        raise ValueError(
            f"source '{source}': options.chunk_years needs params startdate and enddate "
            "as YYYY-MM-DD"
        ) from err
    windows, year = [], first.year
    while year <= last.year:
        start = max(first, date(year, 1, 1))
        end = min(last, date(year + int(chunk_years) - 1, 12, 31))
        windows.append((start.isoformat(), end.isoformat()))
        year += int(chunk_years)
    return windows


@source_type("noaa_cdo_stations")
def fetch_noaa_cdo_stations(source: SourceConfig) -> pd.DataFrame:
    """Metadata for the NOAA CDO stations in `options.station_ids`: name, location, elevation.

    `url` is .../cdo-web/api/v2/stations, and the token goes in the headers as
    for noaa_cdo. It asks for each station by id (`<url>/<id>`), since CDO's
    station list ignores a stationid filter.
    """
    station_ids = source.options.get("station_ids")
    if not station_ids:
        raise ValueError(
            f"source '{source.name}' needs options.station_ids, the stations to look up"
        )
    url = _required_url(source).rstrip("/")
    headers = expand_env(source.headers, source=source.name)
    secrets = secrets_in(source)
    records = []
    for station_id in station_ids:
        records.append(http_get(f"{url}/{station_id}", headers=headers, secrets=secrets).json())
        time.sleep(NOAA_CDO_PAUSE_SECONDS)
    return pd.json_normalize(records)


@source_type("earthdata_timeseries")
def fetch_earthdata_timeseries(source: SourceConfig) -> pd.DataFrame:
    """The newest file of a NASA Earthdata collection holding a plain-text time series.

    Made for GRACE's Greenland and Antarctic ice mass (PO.DAAC), whose file name
    changes with every monthly update. `url` and `params` query NASA's public
    catalog (CMR) for the newest file, e.g. params {short_name: ..., sort_key:
    -start_date, page_size: 1}. The download needs an Earthdata token, in
    `headers: {Authorization: "Bearer ${EARTHDATA_TOKEN}"}`. `options.columns`
    names the file's whitespace-separated columns, in order, and lines starting
    with `options.header_prefix` (default HDR) are skipped. A `granule` column
    records which file the rows came from.
    """
    columns = source.options.get("columns")
    if not columns:
        raise ValueError(f"source '{source.name}' needs options.columns, the file's column names")
    granule, data_url = latest_granule(source)
    log.info("%s: the newest file is %s", source.name, data_url)
    response = http_get(
        data_url,
        headers=expand_env(source.headers, source=source.name),
        secrets=secrets_in(source),
    )
    if "html" in response.headers.get("Content-Type", "").lower():
        # Earthdata answers a missing, wrong, or expired token with its sign-in page.
        raise PermissionError(
            f"source '{source.name}': Earthdata sent a sign-in page instead of the file. "
            "Check EARTHDATA_TOKEN in .env, or generate a new one at https://urs.earthdata.nasa.gov"
        )
    text = response_text(response)
    frame = read_text_columns(
        text, columns, header_prefix=source.options.get("header_prefix", "HDR"), source=source.name
    )
    return frame.assign(granule=granule)


def latest_granule(source: SourceConfig) -> tuple[str, str]:
    """Ask CMR for the source's newest granule. Returns its title and download URL."""
    payload = http_get(_required_url(source), params=expand_env(source.params, source=source.name))
    entries = payload.json().get("feed", {}).get("entry", [])
    if not entries:
        raise ValueError(
            f"source '{source.name}': NASA's catalog found no files for {source.params}. "
            "Check params.short_name"
        )
    granule = entries[0]
    # The granule's own download link. Links marked `inherited` come from the
    # collection (search pages, docs) and aren't the file.
    for link in granule.get("links", []):
        href = link.get("href", "")
        is_data = link.get("rel", "").endswith("/data#") and href.startswith("https://")
        if is_data and not link.get("inherited"):
            return granule.get("title", href.rsplit("/", 1)[-1]), href
    raise ValueError(f"source '{source.name}': granule {granule.get('title')} has no download link")


def read_text_columns(
    text: str, columns: list[str], *, header_prefix: str, source: str
) -> pd.DataFrame:
    """Read whitespace-separated columns, skipping blank lines and header lines."""
    rows = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(header_prefix)
    ]
    if not rows:
        raise ValueError(f"source '{source}': the file has no data rows")
    frame = pd.read_csv(io.StringIO("\n".join(rows)), sep=r"\s+", header=None)
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"source '{source}': the file has {frame.shape[1]} columns, but options.columns "
            f"names {len(columns)}: {columns}"
        )
    frame.columns = columns
    return frame


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = config_parser("Fetch the raw data from every source in the config.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        metavar="NAME",
        help="fetch only this source (repeatable). Default: all of them",
    )
    args = parser.parse_args(argv)
    return ingest(load(args), only=args.sources)


if __name__ == "__main__":
    main()
