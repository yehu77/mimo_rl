#!/usr/bin/env python3
"""Inspect local raw trajectory JSON/JSONL without running any backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.trajectory import (  # noqa: E402
    Trajectory,
    TrajectoryContractError,
    check_training_export,
    compile_single_context_trace,
)


def _read_documents(path: Path) -> List[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        documents = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryContractError([{"code": "malformed_json", "detail": str(exc), "line": line_number}]) from exc
            if not isinstance(item, Mapping):
                raise TrajectoryContractError([{"code": "invalid_json_record", "detail": "JSONL record must be an object", "line": line_number}])
            documents.append(item)
        return documents
    if isinstance(value, list):
        if not all(isinstance(item, Mapping) for item in value):
            raise TrajectoryContractError([{"code": "invalid_json_record", "detail": "JSON list entries must be objects"}])
        return list(value)
    if not isinstance(value, Mapping):
        raise TrajectoryContractError([{"code": "invalid_json_record", "detail": "trajectory JSON must be an object or list"}])
    return [value]


def inspect_trajectories(path: Path, *, export_training: bool = False) -> Dict[str, Any]:
    try:
        documents = _read_documents(path)
    except (OSError, TrajectoryContractError) as exc:
        errors = getattr(exc, "errors", [{"code": "input_error", "detail": str(exc)}])
        return {"status": "INVALID", "trajectory_count": 0, "errors": [dict(item) for item in errors], "eligible_for_training": False}

    compiled = []
    errors = []
    source_kinds = set()
    for index, document in enumerate(documents):
        try:
            trace = Trajectory.from_dict(document)
            source_kinds.add(trace.source_kind)
            compiled.append(compile_single_context_trace(trace))
        except TrajectoryContractError as exc:
            for item in exc.errors:
                errors.append({"trajectory_index": index, **dict(item)})

    decision = check_training_export(compiled)
    if errors:
        status = "NOT_TRAINABLE"
    elif source_kinds == {"synthetic"}:
        status = "SYNTHETIC_VALID_NOT_TRAINABLE"
    elif decision["eligible"]:
        status = "TRAINING_EXPORT_ELIGIBLE"
    else:
        status = "NOT_TRAINABLE"
    if export_training and not decision["eligible"]:
        status = "EXPORT_REJECTED"
    result = {
        "status": status,
        "trajectory_count": len(documents),
        "compiled_count": len(compiled),
        "source_kinds": sorted(source_kinds),
        "generated_token_count": sum(record.get("generated_token_count", 0) for record in compiled),
        "eligible_for_training": bool(decision["eligible"] and not errors),
        "export_decision": {"eligible": decision["eligible"], "reason_codes": sorted({item.get("code") for item in decision["reasons"]})},
        "errors": errors,
        "parameter_update": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--export-training", action="store_true", help="apply future-training rejection rules; never runs training")
    args = parser.parse_args()
    result = inspect_trajectories(args.input, export_training=args.export_training)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] in {"SYNTHETIC_VALID_NOT_TRAINABLE", "TRAINING_EXPORT_ELIGIBLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
