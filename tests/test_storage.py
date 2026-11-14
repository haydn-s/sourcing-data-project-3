from pathlib import Path

import pytest

from src import storage


def test_parse_s3_uri() -> None:
    assert storage.parse_s3_uri("s3://bucket/data/processed/x.csv") == (
        "bucket",
        "data/processed/x.csv",
    )


@pytest.mark.parametrize("uri", ["s3://bucket", "s3://bucket/", "s3:///key", "data/x.csv"])
def test_parse_s3_uri_rejects_incomplete_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        storage.parse_s3_uri(uri)


def test_join_uri() -> None:
    assert storage.join_uri("s3://bucket/models/", "abc", "model.joblib") == (
        "s3://bucket/models/abc/model.joblib"
    )
    assert storage.join_uri("models", "abc", "model.joblib") == str(Path("models/abc/model.joblib"))


def test_local_upload_and_download_copy_files(tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("hello")

    assert storage.upload(src, str(tmp_path / "out" / "b.txt"))
    copied = storage.download(str(tmp_path / "out" / "b.txt"), tmp_path / "back" / "c.txt")

    assert copied.read_text() == "hello"


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
