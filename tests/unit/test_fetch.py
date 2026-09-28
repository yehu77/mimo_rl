from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimo_rl.fetch import DataFetchError, fetch_code_data


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
