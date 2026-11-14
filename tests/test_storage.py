from __future__ import annotations

from pathlib import Path

import pytest
from google.api_core import exceptions

from src import storage


@pytest.mark.parametrize("scheme", ["s3", "gs"])
def test_split_uri(scheme: str) -> None:
    assert storage.split_uri(f"{scheme}://bucket/data/processed/x.csv") == (
        "bucket",
        "data/processed/x.csv",
    )


@pytest.mark.parametrize(
    "uri", ["s3://bucket", "s3://bucket/", "s3:///key", "gs://bucket", "data/x.csv"]
)
def test_split_uri_rejects_incomplete_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        storage.split_uri(uri)


def test_join_uri() -> None:
    assert storage.join_uri("s3://bucket/models/", "abc", "model.joblib") == (
        "s3://bucket/models/abc/model.joblib"
    )
    assert storage.join_uri("gs://bucket/raw", "a.csv") == "gs://bucket/raw/a.csv"
    assert storage.join_uri("models", "abc", "model.joblib") == str(Path("models/abc/model.joblib"))


def test_public_url() -> None:
    assert storage.public_url("gs://bucket/processed/x.parquet") == (
        "https://storage.googleapis.com/bucket/processed/x.parquet"
    )
    assert storage.public_url("s3://bucket/x.csv") == "https://bucket.s3.amazonaws.com/x.csv"


def test_local_upload_and_download_copy_files(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("hello")

    assert storage.upload(src, str(tmp_path / "out" / "b.txt"))
    copied = storage.download(str(tmp_path / "out" / "b.txt"), tmp_path / "back" / "c.txt")

    assert copied.read_text() == "hello"


# S3


def test_s3_upload_is_skipped_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "has_aws_credentials", lambda: False)
    src = tmp_path / "model.joblib"
    src.write_bytes(b"x")

    assert storage.upload(src, "s3://bucket/models/model.joblib") is False


def test_missing_aws_profile_counts_as_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_PROFILE", "no-such-profile-for-tests")
    assert storage.has_aws_credentials() is False


# Google Cloud Storage, against an in-memory fake of the client


class FakeGCS:
    """Just enough of google.cloud.storage.Client: bucket(name).blob(key) up and down."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.cache_control: dict[str, str | None] = {}
        self.forbidden = False
        self.signed: list[bool] = []

    def client(self, *, signed: bool) -> FakeGCS:
        self.signed.append(signed)
        return self

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)


class FakeBucket:
    def __init__(self, gcs: FakeGCS, name: str) -> None:
        self.gcs, self.name = gcs, name

    def blob(self, key: str) -> FakeBlob:
        return FakeBlob(self.gcs, f"gs://{self.name}/{key}")


class FakeBlob:
    def __init__(self, gcs: FakeGCS, uri: str) -> None:
        self.gcs, self.uri = gcs, uri
        self.cache_control: str | None = None

    def upload_from_filename(self, filename: str) -> None:
        self.gcs.objects[self.uri] = Path(filename).read_bytes()
        self.gcs.cache_control[self.uri] = self.cache_control

    def download_to_filename(self, filename: str) -> None:
        if self.gcs.forbidden:
            raise exceptions.Forbidden("no access")
        if self.uri not in self.gcs.objects:
            raise exceptions.NotFound("no such object")
        Path(filename).write_bytes(self.gcs.objects[self.uri])


@pytest.fixture
def gcs(monkeypatch: pytest.MonkeyPatch) -> FakeGCS:
    fake = FakeGCS()
    monkeypatch.setattr(storage, "_gcs_client", fake.client)
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: True)
    return fake


def test_gcs_round_trip(gcs: FakeGCS, tmp_path: Path) -> None:
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")

    assert storage.upload(src, "gs://bucket/processed/data.csv") is True
    copied = storage.download("gs://bucket/processed/data.csv", tmp_path / "back" / "data.csv")

    assert copied.read_text() == "a,b\n1,2\n"
    # Uploads turn off the public cache, so readers see a new push right away.
    assert gcs.cache_control["gs://bucket/processed/data.csv"] == "no-cache"


def test_gcs_reads_anonymously_without_credentials(
    gcs: FakeGCS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gcs.objects["gs://bucket/processed/data.csv"] = b"x"
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)

    storage.download("gs://bucket/processed/data.csv", tmp_path / "data.csv")

    assert gcs.signed == [False]


def test_gcs_upload_is_skipped_without_credentials(
    gcs: FakeGCS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)
    src = tmp_path / "data.csv"
    src.write_text("x")

    assert storage.upload(src, "gs://bucket/processed/data.csv") is False
    assert gcs.objects == {}


def test_gcs_missing_object_raises_file_not_found(gcs: FakeGCS, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        storage.download("gs://bucket/nothing.csv", tmp_path / "x.csv")


def test_gcs_private_object_explains_how_to_sign_in(
    gcs: FakeGCS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gcs.forbidden = True
    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)

    with pytest.raises(PermissionError, match="application-default login"):
        storage.download("gs://bucket/private.csv", tmp_path / "x.csv")


def test_missing_google_credentials_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_credentials() -> None:
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_credentials)
    storage.has_gcp_credentials.cache_clear()
    try:
        assert storage.has_gcp_credentials() is False
    finally:
        storage.has_gcp_credentials.cache_clear()
