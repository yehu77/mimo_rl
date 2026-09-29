from __future__ import annotations

import json
from pathlib import Path

import pytest

import mimo_rl.fetch as fetch_module
from mimo_rl.fetch import DATASET, DataFetchError, fetch_code_data


def test_revision_is_required_and_existing_untracked_files_are_not_reused(tmp_path: Path) -> None:
    with pytest.raises(DataFetchError):
        fetch_code_data("not-a-sha", tmp_path)
    (tmp_path / "code.parquet").write_bytes(b"synthetic")
    with pytest.raises(DataFetchError):
        fetch_code_data("a" * 40, tmp_path)


def test_manifest_rejects_another_revision(tmp_path: Path) -> None:
    manifest = {"revision": "a" * 40, "files": []}
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataFetchError):
        fetch_code_data("b" * 40, tmp_path)


def _write_valid_manifest(tmp_path: Path, revision: str = "a" * 40) -> dict:
    files = []
    for name, content in (("code.parquet", b"synthetic parquet"), ("image-mapping.jsonl", b"synthetic mapping\n")):
        target = tmp_path / name
        target.write_bytes(content)
        files.append({"path": name, "revision": revision, "size_bytes": target.stat().st_size, "sha256": fetch_module._sha256(target)})
    manifest = {"dataset": DATASET, "revision": revision, "files": files}
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_verified_cache_is_reused_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_valid_manifest(tmp_path)

    def unexpected_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("verified cache should not download")

    monkeypatch.setattr(fetch_module, "_download", unexpected_download)
    assert fetch_code_data("a" * 40, tmp_path) == manifest


def test_tampered_cached_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_valid_manifest(tmp_path)
    (tmp_path / "code.parquet").write_bytes(b"changed")
    with pytest.raises(DataFetchError, match="does not match manifest"):
        fetch_code_data("a" * 40, tmp_path)


def test_malformed_manifest_is_rejected_before_download(tmp_path: Path) -> None:
    (tmp_path / "download_manifest.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(DataFetchError, match="invalid existing manifest"):
        fetch_code_data("a" * 40, tmp_path)
