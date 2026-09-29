from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mimo_rl.inspection import DATASET

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "inspect_dataset.py"
REVISION = "a" * 40


def _schema(
    *,
    include_reward: bool = True,
    prompt_type: Any = None,
    reward_type: Any = None,
    extra_type: Any = None,
) -> pa.Schema:
    fields = [
        ("data_source", pa.string()),
        ("ability", pa.string()),
        ("agent_name", pa.string()),
        ("prompt", prompt_type or pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
    ]
    if include_reward:
        fields.append(("reward_model", reward_type or pa.struct([("ground_truth", pa.string()), ("style", pa.string())])))
    fields.append(("extra_info", extra_type or pa.struct([("dataset_type", pa.string()), ("index", pa.int64()), ("instance_id", pa.string()), ("instance_json", pa.string())])))
    return pa.schema(fields)


def _valid_row(instance_id: str = "task-0", image: str = "known-image") -> Dict[str, Any]:
    return {
        "data_source": "synthetic",
        "ability": "code",
        "agent_name": "agent",
        "prompt": [{"content": "solve", "role": "user"}],
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": {"dataset_type": "synthetic", "index": 0, "instance_id": instance_id, "instance_json": json.dumps({"docker_image": image})},
    }


def _write_case(tmp_path: Path, rows: Sequence[Dict[str, Any]], mapping: Iterable[Dict[str, Any]], schema: Optional[pa.Schema] = None) -> tuple[Path, Path, Path]:
    parquet = tmp_path / "code.parquet"
    mapping_path = tmp_path / "image-mapping.jsonl"
    output = tmp_path / "reports"
    schema = schema or _schema()
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), parquet)
    mapping_path.write_text("".join(json.dumps(item) + "\n" for item in mapping), encoding="utf-8")
    return parquet, mapping_path, output


def _run_cli(parquet: Path, mapping: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), "--parquet", str(parquet), "--mapping", str(mapping), "--output-dir", str(output), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _payload(result: subprocess.CompletedProcess[str]) -> Dict[str, Any]:
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def test_cli_valid_native_parquet_is_pass_with_explicit_unverified_source(tmp_path: Path) -> None:
    parquet, mapping, output = _write_case(tmp_path, [_valid_row()], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    result = _run_cli(parquet, mapping, output, "--revision", REVISION)
    payload = _payload(result)
    assert result.returncode == 0
    assert payload["status"] == "PASS"
    assert payload["provenance_status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "name, rows, mapping, schema, code",
    [
        ("duplicate_id", [_valid_row("same"), _valid_row("same")], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], None, "duplicate_instance_id"),
        ("missing_reward_column", [{key: value for key, value in _valid_row().items() if key != "reward_model"}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], _schema(include_reward=False), "missing_required_column"),
        ("null_reward", [{**_valid_row(), "reward_model": None}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], None, "null_reward_model"),
        ("wrong_reward_type", [{**_valid_row(), "reward_model": "wrong"}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], _schema(reward_type=pa.string()), "invalid_reward_model_type"),
        ("wrong_prompt_type", [{**_valid_row(), "prompt": "wrong"}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], _schema(prompt_type=pa.string()), "invalid_prompt_type"),
        ("missing_image", [{**_valid_row(), "extra_info": {"dataset_type": "synthetic", "index": 0, "instance_id": "task-0", "instance_json": json.dumps({})}}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], None, "missing_docker_image"),
        ("empty_image", [{**_valid_row(), "extra_info": {"dataset_type": "synthetic", "index": 0, "instance_id": "task-0", "instance_json": json.dumps({"docker_image": ""})}}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], None, "empty_docker_image"),
        ("wrong_extra_type", [{**_valid_row(), "extra_info": "wrong"}], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}], _schema(extra_type=pa.string()), "invalid_extra_info_type"),
    ],
)
def test_cli_each_native_parquet_error_is_blocking(tmp_path: Path, name: str, rows: Sequence[Dict[str, Any]], mapping: Iterable[Dict[str, Any]], schema: Optional[pa.Schema], code: str) -> None:
    parquet, mapping_path, output = _write_case(tmp_path, rows, mapping, schema)
    result = _run_cli(parquet, mapping_path, output, "--revision", REVISION)
    payload = _payload(result)
    assert result.returncode == 1, name
    assert payload["status"] == "FAIL"
    assert any(item["code"] == code for item in payload["blocking_findings"]), payload


@pytest.mark.parametrize(
    "name, mapping, code",
    [
        ("missing_key", [{"dockerhub_image": "docker.io/example/known:latest"}], "missing_mapping_key"),
        ("empty_key", [{"dataset_image": "", "dockerhub_image": "docker.io/example/known:latest"}], "empty_mapping_key"),
        ("missing_value", [{"dataset_image": "known-image"}], "missing_mapping_value"),
        ("empty_value", [{"dataset_image": "known-image", "dockerhub_image": ""}], "empty_mapping_value"),
        ("duplicate_key", [{"dataset_image": "known-image", "dockerhub_image": "one"}, {"dataset_image": "known-image", "dockerhub_image": "two"}], "duplicate_mapping_key"),
    ],
)
def test_cli_each_mapping_error_is_blocking_without_other_records(tmp_path: Path, name: str, mapping: Iterable[Dict[str, Any]], code: str) -> None:
    parquet, mapping_path, output = _write_case(tmp_path, [], mapping)
    result = _run_cli(parquet, mapping_path, output, "--revision", REVISION)
    payload = _payload(result)
    assert result.returncode == 1, name
    assert any(item["code"] == code for item in payload["blocking_findings"]), payload


def test_cli_preserves_all_repeated_unmatched_record_indices(tmp_path: Path) -> None:
    rows = [_valid_row("task-0", "unknown-image"), _valid_row("task-1", "unknown-image")]
    parquet, mapping, output = _write_case(tmp_path, rows, [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    result = _run_cli(parquet, mapping, output, "--revision", REVISION)
    payload = _payload(result)
    assert result.returncode == 1
    summary = json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["mapping"]["unmatched_instance_images"] == [{"image": "unknown-image", "record_indices": [0, 1], "count": 2}]
    finding = next(item for item in payload["blocking_findings"] if item["code"] == "unmatched_docker_image")
    assert finding["record_indices"] == [0, 1]


def _write_manifest(parquet: Path, mapping: Path, revision: str = REVISION) -> Path:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "files": [
            {"path": "code.parquet", "revision": revision, "size_bytes": parquet.stat().st_size, "sha256": digest(parquet)},
            {"path": "image-mapping.jsonl", "revision": revision, "size_bytes": mapping.stat().st_size, "sha256": digest(mapping)},
        ],
    }
    target = parquet.parent / "download_manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_cli_strict_provenance_verifies_manifest_and_files(tmp_path: Path) -> None:
    parquet, mapping, output = _write_case(tmp_path, [_valid_row()], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    _write_manifest(parquet, mapping)
    result = _run_cli(parquet, mapping, output, "--revision", REVISION, "--strict-provenance")
    payload = _payload(result)
    assert result.returncode == 0
    assert payload["provenance_status"] == "VERIFIED"


def test_cli_strict_provenance_rejects_tampered_input(tmp_path: Path) -> None:
    parquet, mapping, output = _write_case(tmp_path, [_valid_row()], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    _write_manifest(parquet, mapping)
    table = pq.read_table(parquet)
    pq.write_table(table.replace_schema_metadata({b"tamper": b"1"}), parquet)
    result = _run_cli(parquet, mapping, output, "--revision", REVISION, "--strict-provenance")
    payload = _payload(result)
    assert result.returncode == 1
    assert any(item["code"] == "manifest_hash_mismatch" for item in payload["blocking_findings"])


def test_cli_strict_provenance_rejects_missing_or_different_manifest(tmp_path: Path) -> None:
    parquet, mapping, output = _write_case(tmp_path, [_valid_row()], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    result = _run_cli(parquet, mapping, output, "--revision", REVISION, "--strict-provenance")
    payload = _payload(result)
    assert result.returncode == 1
    assert any(item["code"] == "provenance_manifest_missing" for item in payload["blocking_findings"])
    _write_manifest(parquet, mapping, "b" * 40)
    result = _run_cli(parquet, mapping, output, "--revision", REVISION, "--strict-provenance")
    payload = _payload(result)
    assert result.returncode == 1
    assert any(item["code"] == "manifest_revision_mismatch" for item in payload["blocking_findings"])


def test_cli_strict_provenance_rejects_malformed_manifest(tmp_path: Path) -> None:
    parquet, mapping, output = _write_case(tmp_path, [_valid_row()], [{"dataset_image": "known-image", "dockerhub_image": "docker.io/example/known:latest"}])
    (tmp_path / "download_manifest.json").write_text("{bad", encoding="utf-8")
    result = _run_cli(parquet, mapping, output, "--revision", REVISION, "--strict-provenance")
    payload = _payload(result)
    assert result.returncode == 1
    assert any(item["code"] == "invalid_provenance_manifest" for item in payload["blocking_findings"])
