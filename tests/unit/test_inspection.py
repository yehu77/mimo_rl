from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mimo_rl.inspection import _decode_instance, inspect_dataset, write_result


def _write_table(path: Path, rows: list[dict]) -> None:
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("ability", pa.string()),
            ("agent_name", pa.string()),
            ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            ("extra_info", pa.struct([("dataset_type", pa.string()), ("index", pa.int64()), ("instance_id", pa.string()), ("instance_json", pa.string())])),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_string_instance_json_and_mapping_are_reported(tmp_path: Path) -> None:
    parquet = tmp_path / "code.parquet"
    mapping = tmp_path / "image-mapping.jsonl"
    _write_table(
        parquet,
        [
            {
                "data_source": "synthetic",
                "ability": "code",
                "agent_name": "agent",
                "prompt": [{"content": "solve", "role": "user"}],
                "reward_model": {"ground_truth": "", "style": "rule"},
                "extra_info": {"dataset_type": "synthetic", "index": 0, "instance_id": "task-0", "instance_json": json.dumps({"docker_image": "synthetic-image:latest"})},
            }
        ],
    )
    mapping.write_text('{"dataset_image":"synthetic-image:latest","dockerhub_image":"docker.io/example/synthetic:latest"}\n', encoding="utf-8")
    result = inspect_dataset(parquet, mapping, tmp_path / "reports", "a" * 40)
    assert result.schema["rows"] == 1
    assert result.summary["issue_count"] == 0
    assert result.summary["overall_status"] == "PASS"
    assert result.summary["mapping"]["unmatched_instance_images"] == []
    assert result.provenance["status"] == "UNVERIFIED"
    write_result(result, tmp_path / "reports")
    assert (tmp_path / "reports" / "schema.json").exists()
    assert (tmp_path / "reports" / "sample_records_summary.json").exists()
    assert not (tmp_path / "reports" / "sample_records.json").exists()
    write_result(result, tmp_path / "reports", write_samples=True)
    assert (tmp_path / "reports" / "sample_records.json").exists()


def test_object_instance_json_is_checked_without_json_reencoding(tmp_path: Path) -> None:
    issues = []
    decoded = _decode_instance({"docker_image": "synthetic-image:latest"}, 0, issues)
    assert decoded == {"docker_image": "synthetic-image:latest"}
    assert issues == []


def test_malformed_json_empty_prompt_and_duplicate_id_are_retained(tmp_path: Path) -> None:
    parquet = tmp_path / "code.parquet"
    mapping = tmp_path / "image-mapping.jsonl"
    rows = []
    for index, instance_json in enumerate(["{bad", json.dumps({"docker_image": "unknown"})]):
        rows.append(
            {
                "data_source": "synthetic",
                "ability": "code",
                "agent_name": "agent",
                "prompt": [] if index == 0 else [{"content": "ok", "role": "user"}],
                "reward_model": {"ground_truth": "", "style": "rule"},
                "extra_info": {"dataset_type": "synthetic", "index": index, "instance_id": "same", "instance_json": instance_json},
            }
        )
    _write_table(parquet, rows)
    mapping.write_text("not-json\n", encoding="utf-8")
    result = inspect_dataset(parquet, mapping, tmp_path / "reports", "b" * 40)
    codes = {item["code"] for item in result.summary["issues"]}
    assert "malformed_instance_json" in codes
    assert "empty_prompt" in codes
    assert result.summary["identifiers"]["duplicate_ids"] == ["same"]
    assert result.summary["identifiers"]["duplicate_id_records"] == [{"instance_id": "same", "record_indices": [0, 1], "count": 2}]
    assert result.summary["mapping"]["parse_errors"]
    assert result.summary["overall_status"] == "FAIL"
