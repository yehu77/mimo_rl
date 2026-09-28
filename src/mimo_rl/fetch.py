"""Download only the two official code-data files at a fixed HF revision."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
FILES = ("code.parquet", "image-mapping.jsonl")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class DataFetchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _url(filename: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{filename}"


def _download(filename: str, revision: str, destination: Path) -> Dict[str, object]:
    request = Request(_url(filename, revision), headers={"User-Agent": "mimo-rl-b001/0.1"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=str(destination.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            source_revision = response.headers.get("x-repo-commit")
            if source_revision and source_revision != revision:
                raise DataFetchError(f"HF resolved {filename} at {source_revision}, expected {revision}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        temporary.replace(destination)
    except DataFetchError:
        temporary.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise DataFetchError(f"download failed for {filename}: {exc}") from exc
    return {"path": filename, "size_bytes": destination.stat().st_size, "sha256": _sha256(destination), "revision": revision}


def fetch_code_data(revision: str, output_dir: Path) -> Dict[str, object]:
    if not REVISION_RE.fullmatch(revision):
        raise DataFetchError("revision must be a 40-character lowercase commit SHA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataFetchError(f"invalid existing manifest {manifest_path}: {exc}") from exc
        if old.get("dataset") != DATASET:
            raise DataFetchError("existing download manifest belongs to another dataset")
        if old.get("revision") != revision:
            raise DataFetchError("existing download manifest belongs to another revision; choose a new output directory")
        files = old.get("files")
        valid_file_entries = isinstance(files, list) and len(files) == len(FILES) and all(
            isinstance(item, dict)
            and item.get("path") in FILES
            and isinstance(item.get("sha256"), str)
            and isinstance(item.get("size_bytes"), int)
            and item.get("revision") == revision
            for item in files
        )
        if not valid_file_entries or {item["path"] for item in files} != set(FILES):
            raise DataFetchError("existing download manifest is malformed or incomplete")
        for item in files:
            target = output_dir / item["path"]
            if not target.exists():
                raise DataFetchError(f"cached file is missing: {item['path']}")
            if target.stat().st_size != item["size_bytes"] or _sha256(target) != item["sha256"]:
                raise DataFetchError(f"cached file does not match manifest: {item['path']}")
        return old
    elif any((output_dir / filename).exists() for filename in FILES):
        raise DataFetchError("data files exist without a manifest; remove or move them before downloading")

    records = [_download(filename, revision, output_dir / filename) for filename in FILES]
    manifest = {"dataset": DATASET, "revision": revision, "files": records}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
