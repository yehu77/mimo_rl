#!/usr/bin/env python3
"""Validate official code records and export a safe task catalog.

The default output contains only IDs, source locations, field summaries and
hashes. Full controller instances are written only with ``--write-private``
and a specific ``--task-id``; they are never sent to a model or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.inspection import inspect_dataset, validation_status
from mimo_rl.task import TaskBundle, TaskContractError, load_image_mapping


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_private_name(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:20] + ".json"


def _failure_entry(row: Mapping[str, Any], row_index: int, errors: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    extra = row.get("extra_info")
    task_id = extra.get("instance_id") if isinstance(extra, Mapping) else None
    return {
        "task_id": task_id if isinstance(task_id, str) else None,
        "source_row_index": row_index,
        "status": "FAILED",
        "errors": [dict(item) for item in errors],
    }


def prepare_tasks(
    parquet_path: Path,
    mapping_path: Path,
    manifest_path: Path,
    revision: str,
    output_dir: Path,
    *,
    task_id: Optional[str] = None,
    write_private: bool = False,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    result = inspect_dataset(
        parquet_path,
        mapping_path,
        output_dir,
        source_revision=revision,
        manifest_path=manifest_path,
        strict_provenance=True,
    )
    source_status = validation_status(result.summary)
    _write_json(output_dir / "source_validation.json", {"schema": result.schema, "summary": result.summary, "provenance": result.provenance})

    catalog: List[Dict[str, Any]] = []
    bundles: Dict[str, TaskBundle] = {}
    contract_errors = Counter()
    unknown_fields = Counter()
    pending_runtime = 0
    mapping: Dict[str, Any] = {}
    mapping_error: Optional[TaskContractError] = None
    try:
        mapping = load_image_mapping(mapping_path)
    except TaskContractError as exc:
        mapping_error = exc

    for row_index, row in enumerate(rows):
        if source_status["status"] != "PASS":
            errors = [_error for _error in source_status["blocking_findings"]]
            catalog.append(_failure_entry(row, row_index, errors or [{"code": "source_validation_failed", "detail": "strict source validation failed"}]))
            contract_errors["source_validation_failed"] += 1
            continue
        if mapping_error is not None:
            catalog.append(_failure_entry(row, row_index, mapping_error.errors))
            for error in mapping_error.errors:
                contract_errors[str(error.get("code", "mapping_error"))] += 1
            continue
        try:
            bundle = TaskBundle.from_record(
                row,
                source_row_index=row_index,
                data_revision=revision,
                mapping=mapping,
            )
        except TaskContractError as exc:
            catalog.append(_failure_entry(row, row_index, exc.errors))
            for error in exc.errors:
                contract_errors[str(error.get("code", "contract_error"))] += 1
            continue
        bundles[bundle.task_id] = bundle
        if bundle.runtime.pending_runtime:
            pending_runtime += 1
        for field in bundle.unknown_fields:
            unknown_fields[field] += 1
        catalog.append({**bundle.public_summary(), "status": "TASK_PREPARED"})

    selected_bundle = None
    selection_error = None
    if task_id is not None:
        selected_bundle = bundles.get(task_id)
        if selected_bundle is None:
            selection_error = {"code": "task_id_not_found", "task_id": task_id}
        elif write_private:
            private_dir = output_dir / "private"
            private_dir.mkdir(parents=True, exist_ok=True)
            private_path = private_dir / _safe_private_name(task_id)
            private_path.write_text(selected_bundle.to_json(include_private=True), encoding="utf-8")
    elif write_private:
        raise ValueError("--write-private requires --task-id")

    with (output_dir / "task_catalog.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for item in catalog:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    prepared_count = sum(item.get("status") == "TASK_PREPARED" for item in catalog)
    failed_count = len(catalog) - prepared_count
    overall_pass = source_status["status"] == "PASS" and mapping_error is None and failed_count == 0 and selection_error is None
    summary = {
        "status": "CONTRACT_VALIDATED" if overall_pass else "CONTRACT_INVALID",
        "source_validation_status": source_status["status"],
        "provenance_status": result.provenance["status"],
        "input_count": len(rows),
        "prepared_count": prepared_count,
        "failed_count": failed_count,
        "input_accounting_ok": len(rows) == prepared_count + failed_count,
        "pending_runtime_count": pending_runtime,
        "unknown_field_counts": dict(sorted(unknown_fields.items())),
        "contract_error_counts": dict(sorted(contract_errors.items())),
        "selected_task_id": task_id,
        "selection_found": selected_bundle is not None if task_id is not None else None,
        "selection_error": selection_error,
        "catalog_path": "task_catalog.jsonl",
        "source_validation_path": "source_validation.json",
        "private_output": "private/" if write_private else None,
        "states": {"prepared": "TASK_PREPARED", "source_ready": "CONTRACT_VALIDATED", "runtime": "PENDING_RUNTIME"},
    }
    _write_json(output_dir / "catalog_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-id", default=None, help="select one exact task ID for an optional private bundle")
    parser.add_argument("--write-private", action="store_true", help="write the selected full controller bundle under output-dir/private")
    args = parser.parse_args()
    try:
        return prepare_tasks(
            args.parquet,
            args.mapping,
            args.manifest,
            args.revision,
            args.output_dir,
            task_id=args.task_id,
            write_private=args.write_private,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
