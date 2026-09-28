"""Schema and integrity inspection for the official MiMo code records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class InspectionResult:
    schema: Dict[str, Any]
    samples: List[Dict[str, Any]]
    summary: Dict[str, Any]
    provenance: Dict[str, Any]


def _json_value(value: Any) -> Any:
    """Convert Arrow/Python values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    return str(value)


def _arrow_type(node: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"type": str(node), "nullable": bool(getattr(node, "nullable", True))}
    if hasattr(node, "field") and hasattr(node, "num_fields"):
        result["fields"] = [_arrow_type(node.field(i)) for i in range(node.num_fields)]
    elif hasattr(node, "value_type"):
        result["value_type"] = _arrow_type(node.value_type)
    return result


def _file_info(path: Path, source_revision: Optional[str]) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "revision": source_revision,
    }


def _issue(index: int, code: str, detail: str, value: Any = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {"record_index": index, "code": code, "detail": detail}
    if value is not None:
        item["value"] = _json_value(value)
    return item


def _decode_instance(value: Any, index: int, issues: List[Dict[str, Any]]) -> Optional[Mapping[str, Any]]:
    if value is None:
        issues.append(_issue(index, "missing_instance_json", "extra_info.instance_json is null"))
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            issues.append(_issue(index, "malformed_instance_json", str(exc)))
            return None
    elif isinstance(value, Mapping):
        decoded = value
    else:
        issues.append(_issue(index, "invalid_instance_json_type", type(value).__name__))
        return None
    if not isinstance(decoded, Mapping):
        issues.append(_issue(index, "instance_json_not_object", "decoded value is not an object"))
        return None
    return decoded


def _inspect_prompt(value: Any, index: int, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"kind": type(value).__name__, "message_count": 0, "empty": False}
    if value is None:
        issues.append(_issue(index, "missing_prompt", "prompt is null"))
        result["empty"] = True
        return result
    if not isinstance(value, list):
        issues.append(_issue(index, "invalid_prompt_type", "prompt is not a list", type(value).__name__))
        result["empty"] = True
        return result
    result["kind"] = "list"
    result["message_count"] = len(value)
    nonempty = False
    for message_index, message in enumerate(value):
        if not isinstance(message, Mapping):
            issues.append(_issue(index, "invalid_prompt_message", f"message {message_index} is not an object"))
            continue
        content = message.get("content")
        role = message.get("role")
        if not isinstance(content, str):
            issues.append(_issue(index, "invalid_prompt_content", f"message {message_index} content is not a string"))
        elif content.strip():
            nonempty = True
        if not isinstance(role, str):
            issues.append(_issue(index, "invalid_prompt_role", f"message {message_index} role is not a string"))
    result["empty"] = not nonempty
    if result["empty"]:
        issues.append(_issue(index, "empty_prompt", "prompt has no non-whitespace content"))
    return result


def _mapping_inspection(path: Path, instance_images: Sequence[Tuple[int, Optional[str]]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    key_counts: Counter[str] = Counter()
    keys: List[str] = []
    field_types: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.rstrip("\n")
            if not text.strip():
                parse_errors.append(_issue(index, "empty_mapping_line", "line is empty"))
                continue
            try:
                record = json.loads(text)
            except ValueError as exc:
                parse_errors.append(_issue(index, "malformed_mapping_json", str(exc)))
                continue
            if not isinstance(record, Mapping):
                parse_errors.append(_issue(index, "mapping_record_not_object", "JSON value is not an object"))
                continue
            record_json = _json_value(record)
            records.append(record_json)
            for field, value in record.items():
                field_types.setdefault(str(field), set()).add(type(value).__name__)
            dataset_key = record.get("dataset_image")
            if dataset_key is None:
                parse_errors.append(_issue(index, "missing_mapping_key", "dataset_image is missing"))
                continue
            if not isinstance(dataset_key, str):
                parse_errors.append(_issue(index, "invalid_mapping_key", "dataset_image is not a string", dataset_key))
                continue
            key_counts[dataset_key] += 1
            keys.append(dataset_key)
            value = record.get("dockerhub_image")
            if value is None:
                parse_errors.append(_issue(index, "missing_mapping_value", "dockerhub_image is missing"))
            elif not isinstance(value, str):
                parse_errors.append(_issue(index, "invalid_mapping_value", "dockerhub_image is not a string", value))
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    mapping_keys = set(keys)
    used_images = {image for _, image in instance_images if image}
    unmatched = sorted(used_images - mapping_keys)
    unused = sorted(mapping_keys - used_images)
    return {
        "path": str(path),
        "record_count": len(records),
        "schema": sorted({key for record in records for key in record}),
        "field_types": {key: sorted(values) for key, values in sorted(field_types.items())},
        "parse_errors": parse_errors,
        "duplicate_keys": duplicate_keys,
        "duplicate_key_count": len(duplicate_keys),
        "key_field": "dataset_image",
        "value_field": "dockerhub_image",
        "matched_instance_image_count": len(used_images & mapping_keys),
        "unmatched_instance_images": unmatched,
        "unused_mapping_keys": unused,
        "sample_records": records[:3],
    }


def inspect_dataset(
    parquet_path: Path,
    mapping_path: Path,
    output_dir: Path,
    source_revision: Optional[str] = None,
    sample_size: int = 3,
) -> InspectionResult:
    """Inspect a parquet table and image mapping without modifying source files."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks
        raise RuntimeError("Parquet inspection requires pyarrow; install the dev extra") from exc

    parquet_path = Path(parquet_path)
    mapping_path = Path(mapping_path)
    output_dir = Path(output_dir)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    schema_fields = []
    for field in table.schema:
        schema_fields.append({"name": field.name, **_arrow_type(field.type), "nullable": field.nullable})
    schema = {
        "format": "parquet",
        "rows": table.num_rows,
        "columns": schema_fields,
        "column_names": table.column_names,
        "required_columns_present": {
            key: key in table.column_names for key in ("prompt", "reward_model", "extra_info")
        },
    }

    issues: List[Dict[str, Any]] = []
    id_counts: Counter[str] = Counter()
    instance_images: List[Tuple[int, Optional[str]]] = []
    prompt_stats = {"empty_count": 0, "message_count": Counter()}
    instance_json_kinds: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        prompt_result = _inspect_prompt(row.get("prompt"), index, issues)
        prompt_stats["message_count"][str(prompt_result["message_count"])] += 1
        if prompt_result["empty"]:
            prompt_stats["empty_count"] += 1
        extra = row.get("extra_info")
        if extra is None:
            issues.append(_issue(index, "missing_extra_info", "extra_info is null"))
            extra = {}
        if not isinstance(extra, Mapping):
            issues.append(_issue(index, "invalid_extra_info_type", "extra_info is not an object"))
            extra = {}
        instance_id = extra.get("instance_id")
        if instance_id is None or (isinstance(instance_id, str) and not instance_id.strip()):
            issues.append(_issue(index, "missing_instance_id", "extra_info.instance_id is missing or empty"))
        elif not isinstance(instance_id, str):
            issues.append(_issue(index, "invalid_instance_id", "extra_info.instance_id is not a string", instance_id))
        else:
            id_counts[instance_id] += 1
        decoded = _decode_instance(extra.get("instance_json"), index, issues)
        instance_json_kinds["object" if decoded is not None else "invalid"] += 1
        image = decoded.get("docker_image") if decoded else None
        if image is not None and not isinstance(image, str):
            issues.append(_issue(index, "invalid_docker_image", "docker_image is not a string", image))
            image = None
        instance_images.append((index, image))
        if len(samples) < sample_size:
            samples.append(_json_value({"record_index": index, **row}))

    mapping = _mapping_inspection(mapping_path, instance_images)
    duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
    missing_required = [key for key, present in schema["required_columns_present"].items() if not present]
    summary = {
        "parquet_rows": table.num_rows,
        "parquet_columns": table.column_names,
        "required_column_missing": missing_required,
        "prompt": {
            "empty_count": prompt_stats["empty_count"],
            "message_count_distribution": dict(sorted(prompt_stats["message_count"].items())),
        },
        "reward_model": {
            "present": "reward_model" in table.column_names,
            "null_count": sum(row.get("reward_model") is None for row in rows),
        },
        "extra_info": {
            "present": "extra_info" in table.column_names,
            "instance_json_kinds": dict(instance_json_kinds),
        },
        "identifiers": {
            "missing_id_count": sum(1 for item in issues if item["code"] == "missing_instance_id"),
            "duplicate_ids": duplicate_ids,
            "duplicate_id_count": len(duplicate_ids),
        },
        "issues": issues,
        "issue_count": len(issues),
        "mapping": {key: value for key, value in mapping.items() if key != "sample_records"},
    }
    provenance = {
        "dataset": "XiaomiMiMo/MiMo-V2.6-RL-oss",
        "revision": source_revision,
        "source_urls": {
            "code.parquet": f"https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/{source_revision or 'UNKNOWN'}/code.parquet",
            "image-mapping.jsonl": f"https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/{source_revision or 'UNKNOWN'}/image-mapping.jsonl",
        },
        "files": {
            "code.parquet": _file_info(parquet_path, source_revision),
            "image-mapping.jsonl": _file_info(mapping_path, source_revision),
        },
    }
    return InspectionResult(schema=schema, samples=samples, summary=summary, provenance=provenance)


def write_result(result: InspectionResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "schema.json": result.schema,
        "sample_records.json": result.samples,
        "validation_summary.json": result.summary,
        "provenance.json": result.provenance,
    }
    for name, payload in outputs.items():
        target = output_dir / name
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
