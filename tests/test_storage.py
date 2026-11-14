from __future__ import annotations

from pathlib import Path

import pytest
from google.api_core import exceptions

from src import storage

# The real function, saved before conftest's autouse fixture replaces it.
real_has_gcp_credentials = storage.has_gcp_credentials


def test_split_uri() -> None:
    assert storage.split_uri("gs://bucket/data/processed/x.csv") == (
        "bucket",
        "data/processed/x.csv",
    )


@pytest.mark.parametrize(
    "uri", ["gs://bucket", "gs://bucket/", "gs:///key", "https://example.org/key", "x.csv"]
)
def test_split_uri_rejects_other_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        storage.split_uri(uri)


def test_join_uri() -> None:
    assert storage.join_uri("gs://bucket/models/", "abc", "model.joblib") == (
        "gs://bucket/models/abc/model.joblib"
    )
    assert storage.join_uri("models", "abc", "model.joblib") == str(Path("models/abc/model.joblib"))


def test_public_url() -> None:
    assert storage.public_url("gs://bucket/processed/x.parquet") == (
        "https://storage.googleapis.com/bucket/processed/x.parquet"
    )


def test_local_upload_and_download_copy_files(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("hello")

    assert storage.upload(src, str(tmp_path / "out" / "b.txt"))
    copied = storage.download(str(tmp_path / "out" / "b.txt"), tmp_path / "back" / "c.txt")

    assert copied.read_text() == "hello"


class FakeGCS:
    """An in-memory stand-in for Cloud Storage: public and private objects, signed and anonymous clients."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.private: set[str] = set()
        self.cache_control: dict[str, str | None] = {}
        self.requests: list[tuple[str, bool]] = []  # (what, signed)

    def client(self, *, signed: bool) -> FakeClient:
        return FakeClient(self, signed)


class FakeClient:
    def __init__(self, gcs: FakeGCS, signed: bool) -> None:
        self.gcs, self.signed = gcs, signed

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)


class FakeBucket:
    def __init__(self, client: FakeClient, name: str) -> None:
        self.client, self.name = client, name

    def blob(self, key: str) -> FakeBlob:
        return FakeBlob(self.client, f"gs://{self.name}/{key}")


class FakeBlob:
    def __init__(self, client: FakeClient, uri: str) -> None:
        self.gcs, self.signed, self.uri = client.gcs, client.signed, uri
        self.cache_control: str | None = None

    def upload_from_filename(self, filename: str) -> None:
        self.gcs.requests.append(("upload", self.signed))
        self.gcs.objects[self.uri] = Path(filename).read_bytes()
        self.gcs.cache_control[self.uri] = self.cache_control

    def download_to_filename(self, filename: str) -> None:
        self.gcs.requests.append(("download", self.signed))
        if self.uri in self.gcs.private and not self.signed:
            raise exceptions.Unauthorized("anonymous caller")
        if self.uri not in self.gcs.objects:
            raise exceptions.NotFound("no such object")
        Path(filename).write_bytes(self.gcs.objects[self.uri])


@pytest.fixture
def gcs(monkeypatch: pytest.MonkeyPatch) -> FakeGCS:
    fake = FakeGCS()
    monkeypatch.setattr(storage, "_client", fake.client)
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: True)
    return fake


def test_round_trip(gcs: FakeGCS, tmp_path: Path) -> None:
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")

    assert storage.upload(src, "gs://bucket/processed/data.csv") is True
    copied = storage.download("gs://bucket/processed/data.csv", tmp_path / "back" / "data.csv")

    assert copied.read_text() == "a,b\n1,2\n"
    # Uploads turn off the public cache, so readers see a new push right away.
    assert gcs.cache_control["gs://bucket/processed/data.csv"] == "no-cache"


def test_public_objects_are_read_anonymously_even_with_credentials(
    gcs: FakeGCS, tmp_path: Path
) -> None:
    gcs.objects["gs://bucket/models/latest/model.joblib"] = b"x"

    storage.download("gs://bucket/models/latest/model.joblib", tmp_path / "model.joblib")

    assert gcs.requests == [("download", False)]


def test_private_objects_fall_back_to_credentials(gcs: FakeGCS, tmp_path: Path) -> None:
    gcs.objects["gs://bucket/private.csv"] = b"secret"
    gcs.private.add("gs://bucket/private.csv")

    copied = storage.download("gs://bucket/private.csv", tmp_path / "private.csv")

    assert copied.read_bytes() == b"secret"
    assert gcs.requests == [("download", False), ("download", True)]


def test_private_objects_without_credentials_explain_how_to_sign_in(
    gcs: FakeGCS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gcs.objects["gs://bucket/private.csv"] = b"secret"
    gcs.private.add("gs://bucket/private.csv")
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)

    with pytest.raises(PermissionError, match="application-default login"):
        storage.download("gs://bucket/private.csv", tmp_path / "x.csv")


def test_missing_object_raises_file_not_found(gcs: FakeGCS, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        storage.download("gs://bucket/nothing.csv", tmp_path / "x.csv")


def test_upload_is_skipped_without_credentials(
    gcs: FakeGCS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)
    src = tmp_path / "data.csv"
    src.write_text("x")

    assert storage.upload(src, "gs://bucket/processed/data.csv") is False
    assert gcs.objects == {}


def test_missing_google_credentials_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_credentials() -> None:
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_credentials)
    real_has_gcp_credentials.cache_clear()
    try:
        assert real_has_gcp_credentials() is False
    finally:
        real_has_gcp_credentials.cache_clear()


def test_tests_cannot_reach_the_real_bucket(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="tried to reach Cloud Storage"):
        storage.download("gs://climate-impact-pipeline-data/raw/x.csv", tmp_path / "x.csv")
