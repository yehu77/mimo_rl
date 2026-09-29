from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mimo_rl.inspection import DATASET
from mimo_rl.task import MappingEntry, TaskBundle, TaskContractError, load_image_mapping


ROOT = Path(__file__).resolve().parents[2]
REVISION = "a" * 40


def _instance(*, patch: str = "diff --git a/test b/test\n", **updates: object) -> dict:
    value = {
        "dataset_type": "opensource-code",
        "docker_image": "dataset-image:latest",
        "cwd": "/testbed",
        "instance_id": "task-0",
        "problem_statement": "Solve the task",
        "test_patch": patch,
        "test_command": "bash /testbed/check.sh",
        "verifier_timeout_sec": 30,
    }
    value.update(updates)
    return value


def _row(instance: dict | None = None, *, private_extra: object = None) -> dict:
    instance = instance or _instance()
    extra = {
        "dataset_type": "opensource-code",
        "index": 17,
        "instance_id": "task-0",
        "instance_json": json.dumps(instance, ensure_ascii=False),
    }
    if private_extra is not None:
        extra["private"] = private_extra
    return {
        "prompt": [{"role": "user", "content": "Solve the task"}],
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": extra,
    }


def _mapping() -> dict[str, MappingEntry]:
    return {"dataset-image:latest": MappingEntry("dataset-image:latest", "docker.io/example/task:1", (3,))}


def test_string_and_object_instance_json_are_equivalent_and_prompt_is_unchanged() -> None:
    instance = _instance()
    string_bundle = TaskBundle.from_record(_row(instance), source_row_index=2, data_revision=REVISION, mapping=_mapping())
    object_row = _row(instance)
    object_row["extra_info"]["instance_json"] = instance
    object_bundle = TaskBundle.from_record(object_row, source_row_index=2, data_revision=REVISION, mapping=_mapping())
    assert string_bundle.to_solver_payload() == object_bundle.to_solver_payload()
    assert string_bundle.to_solver_payload()["messages"] == [{"role": "user", "content": "Solve the task"}]
    assert string_bundle.to_mimoagent_instance()["test_patch"] == instance["test_patch"]


def test_solver_allowlist_excludes_private_nested_fields_and_repr_redacts_patch() -> None:
    sentinel = "PRIVATE_TEST_SENTINEL"
    bundle = TaskBundle.from_record(
        _row(_instance(patch=sentinel, private_field={"nested": sentinel}), private_extra={"nested": sentinel}),
        source_row_index=0,
        data_revision=REVISION,
        mapping=_mapping(),
    )
    serialized = json.dumps(bundle.to_solver_payload(), ensure_ascii=False)
    assert sentinel not in serialized
    assert set(bundle.to_solver_payload()) == {"task_id", "messages", "problem_statement", "cwd"}
    assert sentinel not in repr(bundle)
    assert sentinel in bundle.to_mimoagent_instance()["test_patch"]


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"cwd": "relative"}, "invalid_cwd"),
        ({"test_patch": ""}, "invalid_test_patch"),
        ({"test_command": ""}, "invalid_test_command"),
        ({"verifier_timeout_sec": 0}, "invalid_verifier_timeout"),
        ({"instance_id": "inner-other"}, "instance_id_conflict"),
    ],
)
def test_core_contract_errors_are_specific(updates: dict, code: str) -> None:
    with pytest.raises(TaskContractError) as exc_info:
        TaskBundle.from_record(_row(_instance(**updates)), source_row_index=4, data_revision=REVISION, mapping=_mapping())
    assert code in {item["code"] for item in exc_info.value.errors}


def test_unmapped_image_and_duplicate_mapping_are_blocking() -> None:
    with pytest.raises(TaskContractError) as exc_info:
        TaskBundle.from_record(_row(_instance(docker_image="unknown:latest")), source_row_index=0, data_revision=REVISION, mapping=_mapping())
    assert {item["code"] for item in exc_info.value.errors} == {"unmapped_docker_image"}

    mapping_path = Path("/tmp") / "mimo-b002-duplicate-mapping.jsonl"
    mapping_path.write_text(
        '{"dataset_image":"x","dockerhub_image":"one"}\n{"dataset_image":"x","dockerhub_image":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskContractError) as exc_info:
        load_image_mapping(mapping_path)
    assert exc_info.value.errors[0]["mapping_line_indices"] == [0, 1]


def test_private_hash_changes_when_only_patch_changes_and_solver_payload_does_not() -> None:
    first = TaskBundle.from_record(_row(_instance(patch="patch-one\n")), source_row_index=0, data_revision=REVISION, mapping=_mapping())
    second = TaskBundle.from_record(_row(_instance(patch="patch-two\n")), source_row_index=0, data_revision=REVISION, mapping=_mapping())
    assert first.to_solver_payload() == second.to_solver_payload()
    assert first.verifier.private_hash != second.verifier.private_hash


def test_contract_conversion_has_no_external_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("task contract must not spawn a process")

    monkeypatch.setattr(subprocess, "run", fail)
    bundle = TaskBundle.from_record(_row(), source_row_index=0, data_revision=REVISION, mapping=_mapping())
    assert bundle.to_solver_payload()["task_id"] == "task-0"
    assert bundle.to_mimoagent_instance()["test_command"] == "bash /testbed/check.sh"


def test_serialization_round_trip_and_input_immutability() -> None:
    row = _row(_instance(), private_extra={"nested": ["value"]})
    before = copy.deepcopy(row)
    bundle = TaskBundle.from_record(row, source_row_index=7, data_revision=REVISION, mapping=_mapping())
    restored = TaskBundle.from_dict(json.loads(bundle.to_json(include_private=True)))
    assert row == before
    assert restored.to_mimoagent_instance() == bundle.to_mimoagent_instance()
    assert restored.runtime.source_extra_info_index == 17


def _write_parquet_case(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    parquet = tmp_path / "code.parquet"
    mapping = tmp_path / "image-mapping.jsonl"
    output = tmp_path / "tasks"
    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            ("extra_info", pa.struct([("dataset_type", pa.string()), ("index", pa.int64()), ("instance_id", pa.string()), ("instance_json", pa.string())])),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), parquet)
    mapping.write_text('{"dataset_image":"dataset-image:latest","dockerhub_image":"docker.io/example/task:1"}\n', encoding="utf-8")
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "files": [
            {"path": "code.parquet", "revision": REVISION, "size_bytes": parquet.stat().st_size, "sha256": __import__("hashlib").sha256(parquet.read_bytes()).hexdigest()},
            {"path": "image-mapping.jsonl", "revision": REVISION, "size_bytes": mapping.stat().st_size, "sha256": __import__("hashlib").sha256(mapping.read_bytes()).hexdigest()},
        ],
    }
    manifest_path = tmp_path / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return parquet, mapping, output


def test_prepare_cli_public_catalog_excludes_private_patch_and_can_write_selected_bundle(tmp_path: Path) -> None:
    parquet, mapping, output = _write_parquet_case(tmp_path, [_row(_instance(patch="PRIVATE_TEST_SENTINEL\n"))])
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_code_tasks.py"),
        "--parquet",
        str(parquet),
        "--mapping",
        str(mapping),
        "--manifest",
        str(parquet.parent / "download_manifest.json"),
        "--revision",
        REVISION,
        "--output-dir",
        str(output),
        "--task-id",
        "task-0",
        "--write-private",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    catalog = (output / "task_catalog.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE_TEST_SENTINEL" not in catalog
    assert "PRIVATE_TEST_SENTINEL" not in (output / "catalog_summary.json").read_text(encoding="utf-8")
    assert "PRIVATE_TEST_SENTINEL" not in (output / "source_validation.json").read_text(encoding="utf-8")
    private_files = list((output / "private").glob("*.json"))
    assert len(private_files) == 1
    assert "PRIVATE_TEST_SENTINEL" in private_files[0].read_text(encoding="utf-8")
