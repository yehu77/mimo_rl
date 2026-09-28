"""Schema, provenance, and integrity inspection for official MiMo records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
EXPECTED_FILES = ("code.parquet", "image-mapping.jsonl")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


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
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "revision": source_revision,
    }


def _issue(index: Optional[int], code: str, detail: str, value: Any = None, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"record_index": index, "code": code, "detail": detail}
    if value is not None:
        item["value"] = _json_value(value)
    item.update({key: _json_value(val) for key, val in extra.items()})
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


def _inspect_reward_model(value: Any, index: int, issues: List[Dict[str, Any]]) -> None:
    if value is None:
        issues.append(_issue(index, "null_reward_model", "reward_model is null"))
    elif not isinstance(value, Mapping):
        issues.append(_issue(index, "invalid_reward_model_type", "reward_model is not an object", type(value).__name__))


def _mapping_inspection(
    path: Path, instance_images: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    key_counts: Counter[str] = Counter()
    key_lines: Dict[str, List[int]] = defaultdict(list)
    field_types: Dict[str, set] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            text = line.rstrip("\n")
            if not text.strip():
                parse_errors.append(_issue(line_index, "empty_mapping_line", "mapping line is empty", source="mapping"))
                continue
            try:
                record = json.loads(text)
            except ValueError as exc:
                parse_errors.append(_issue(line_index, "malformed_mapping_json", str(exc), source="mapping"))
                continue
            if not isinstance(record, Mapping):
                parse_errors.append(_issue(line_index, "mapping_record_not_object", "JSON value is not an object", source="mapping"))
                continue
            record_json = _json_value(record)
            records.append(record_json)
            for field, value in record.items():
                field_types.setdefault(str(field), set()).add(type(value).__name__)
            dataset_key = record.get("dataset_image")
            if dataset_key is None:
                parse_errors.append(_issue(line_index, "missing_mapping_key", "dataset_image is missing", source="mapping"))
                continue
            if not isinstance(dataset_key, str):
                parse_errors.append(_issue(line_index, "invalid_mapping_key", "dataset_image is not a string", dataset_key, source="mapping"))
                continue
            if not dataset_key.strip():
                parse_errors.append(_issue(line_index, "empty_mapping_key", "dataset_image is empty", source="mapping"))
            key_counts[dataset_key] += 1
            key_lines[dataset_key].append(line_index)
            value = record.get("dockerhub_image")
            if value is None:
                parse_errors.append(_issue(line_index, "missing_mapping_value", "dockerhub_image is missing", source="mapping"))
            elif not isinstance(value, str):
                parse_errors.append(_issue(line_index, "invalid_mapping_value", "dockerhub_image is not a string", value, source="mapping"))
            elif not value.strip():
                parse_errors.append(_issue(line_index, "empty_mapping_value", "dockerhub_image is empty", source="mapping"))
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    duplicate_key_records = [
        {"dataset_image": key, "mapping_line_indices": key_lines[key], "count": key_counts[key]}
        for key in duplicate_keys
    ]
    mapping_keys = {key for key in key_counts if key.strip()}
    unmatched_by_image: Dict[str, List[int]] = defaultdict(list)
    valid_image_records = []
    for observation in instance_images:
        if observation["status"] == "valid":
            image = observation["image"]
            valid_image_records.append(observation)
            if image not in mapping_keys:
                unmatched_by_image[image].append(observation["record_index"])
    unmatched = [
        {"image": image, "record_indices": indices, "count": len(indices)}
        for image, indices in sorted(unmatched_by_image.items())
    ]
    unmatched_issues = [
        _issue(index, "unmatched_docker_image", "docker_image has no mapping entry", image=image)
        for item in unmatched
        for index in item["record_indices"]
    ]
    used_keys = {observation["image"] for observation in valid_image_records}
    unused = sorted(mapping_keys - used_keys)
    return {
        "path": str(path),
        "record_count": len(records),
        "schema": sorted({key for record in records for key in record}),
        "field_types": {key: sorted(values) for key, values in sorted(field_types.items())},
        "parse_errors": parse_errors,
        "duplicate_keys": duplicate_keys,
        "duplicate_key_records": duplicate_key_records,
        "duplicate_key_count": len(duplicate_keys),
        "key_field": "dataset_image",
        "value_field": "dockerhub_image",
        "matched_instance_image_count": sum(1 for observation in valid_image_records if observation["image"] in mapping_keys),
        "valid_instance_image_count": len(valid_image_records),
        "unmatched_instance_images": unmatched,
        "unused_mapping_keys": unused,
        "blocking_issues": parse_errors + [
            _issue(index, "duplicate_mapping_key", "dataset_image occurs more than once", dataset_image=item["dataset_image"], source="mapping")
            for item in duplicate_key_records
            for index in item["mapping_line_indices"]
        ] + unmatched_issues,
        "sample_records": records[:3],
    }


def _auto_manifest(parquet_path: Path, mapping_path: Path) -> Optional[Path]:
    candidates = []
    for candidate in (parquet_path.parent / "download_manifest.json", mapping_path.parent / "download_manifest.json"):
        if candidate not in candidates and candidate.exists():
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _validate_manifest(
    manifest_path: Optional[Path],
    parquet_path: Path,
    mapping_path: Path,
    declared_revision: Optional[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    files = {"code.parquet": parquet_path, "image-mapping.jsonl": mapping_path}
    if manifest_path is None:
        return {
            "status": "UNVERIFIED",
            "manifest_path": None,
            "revision": None,
            "declared_revision": declared_revision,
            "checks": ["manifest_not_found"],
        }, []
    manifest_info: Dict[str, Any] = {
        "status": "INVALID",
        "manifest_path": str(manifest_path),
        "revision": None,
        "declared_revision": declared_revision,
        "checks": [],
    }
    errors: List[Dict[str, Any]] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(_issue(None, "invalid_provenance_manifest", str(exc), source="manifest"))
        manifest_info["checks"].append("manifest_parse_failed")
        return manifest_info, errors
    if not isinstance(manifest, Mapping):
        errors.append(_issue(None, "invalid_provenance_manifest", "manifest is not an object", source="manifest"))
        return manifest_info, errors
    if manifest.get("dataset") != DATASET:
        errors.append(_issue(None, "manifest_dataset_mismatch", "manifest dataset does not match expected dataset", manifest.get("dataset"), source="manifest"))
    revision = manifest.get("revision")
    manifest_info["revision"] = revision
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        errors.append(_issue(None, "manifest_revision_invalid", "manifest revision is not a 40-character SHA", revision, source="manifest"))
    if declared_revision is not None and declared_revision != revision:
        errors.append(_issue(None, "manifest_revision_mismatch", "declared revision differs from manifest", declared_revision, manifest_revision=revision, source="manifest"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        errors.append(_issue(None, "manifest_files_invalid", "manifest files is not a list", source="manifest"))
        entries = []
    entry_paths = [item.get("path") for item in entries if isinstance(item, Mapping) and isinstance(item.get("path"), str)]
    if len(entry_paths) != len(set(entry_paths)):
        errors.append(_issue(None, "manifest_files_duplicate", "manifest contains duplicate file entries", source="manifest"))
    by_name = {item.get("path"): item for item in entries if isinstance(item, Mapping)}
    if set(by_name) != set(EXPECTED_FILES):
        errors.append(_issue(None, "manifest_files_mismatch", "manifest must contain exactly code.parquet and image-mapping.jsonl", sorted(by_name), source="manifest"))
    for name, path in files.items():
        if path.name != name:
            errors.append(_issue(None, "manifest_input_name_mismatch", f"input filename must be {name}", path.name, file=name, source="manifest"))
        entry = by_name.get(name)
        if entry is None:
            continue
        if entry.get("revision") != revision:
            errors.append(_issue(None, "manifest_file_revision_mismatch", f"{name} revision differs from manifest", entry.get("revision"), file=name, source="manifest"))
        actual = _file_info(path, revision if isinstance(revision, str) else None)
        if entry.get("size_bytes") != actual["size_bytes"]:
            errors.append(_issue(None, "manifest_size_mismatch", f"{name} size differs from manifest", entry.get("size_bytes"), actual_size_bytes=actual["size_bytes"], file=name, source="manifest"))
        if entry.get("sha256") != actual["sha256"]:
            errors.append(_issue(None, "manifest_hash_mismatch", f"{name} SHA-256 differs from manifest", entry.get("sha256"), actual_sha256=actual["sha256"], file=name, source="manifest"))
    if errors:
        manifest_info["checks"].append("manifest_validation_failed")
    else:
        manifest_info["status"] = "VERIFIED"
        manifest_info["checks"].append("manifest_and_files_match")
    return manifest_info, errors


def _sample_summary(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def summarize(value: Any) -> Dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, str):
            return {"type": "string", "length": len(value), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
        if isinstance(value, Mapping):
            return {"type": "object", "keys": sorted(str(key) for key in value)}
        if isinstance(value, list):
            return {"type": "list", "length": len(value)}
        return {"type": type(value).__name__}

    return [
        {"record_index": sample.get("record_index"), "fields": {key: summarize(value) for key, value in sample.items() if key != "record_index"}}
        for sample in samples
    ]


def validation_status(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the single fatal/warning status used by both reports and the CLI."""
    issues = list(summary.get("issues", []))
    warnings = list(summary.get("warnings", []))
    groups: Dict[str, Dict[str, Any]] = {}
    for item in issues:
        code = str(item.get("code", "unknown"))
        group = groups.setdefault(code, {"code": code, "count": 0, "record_indices": []})
        group["count"] += 1
        index = item.get("record_index")
        if index is not None and index not in group["record_indices"]:
            group["record_indices"].append(index)
    findings = sorted(groups.values(), key=lambda item: item["code"])
    return {
        "status": "PASS" if not issues else "FAIL",
        "blocking_count": len(issues),
        "warning_count": len(warnings),
        "blocking_findings": findings,
    }


def inspect_dataset(
    parquet_path: Path,
    mapping_path: Path,
    output_dir: Path,
    source_revision: Optional[str] = None,
    sample_size: int = 3,
    manifest_path: Optional[Path] = None,
    strict_provenance: bool = False,
) -> InspectionResult:
    """Inspect Parquet/JSONL inputs and preserve row-level validation evidence."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet inspection requires pyarrow; install the dev extra") from exc

    parquet_path = Path(parquet_path)
    mapping_path = Path(mapping_path)
    output_dir = Path(output_dir)
    if source_revision is not None and not REVISION_RE.fullmatch(source_revision):
        declared_revision_issue = _issue(None, "invalid_declared_revision", "revision must be a 40-character lowercase SHA", source_revision, source="caller")
    else:
        declared_revision_issue = None
    resolved_manifest = Path(manifest_path) if manifest_path is not None else _auto_manifest(parquet_path, mapping_path)
    provenance, provenance_issues = _validate_manifest(resolved_manifest, parquet_path, mapping_path, source_revision)
    if strict_provenance and provenance["status"] == "UNVERIFIED":
        provenance_issues.append(_issue(None, "provenance_manifest_missing", "strict provenance requires a matching download manifest", source="manifest"))
    if declared_revision_issue is not None:
        provenance_issues.append(declared_revision_issue)
        provenance["status"] = "INVALID"

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    schema_fields = []
    for field in table.schema:
        schema_fields.append({"name": field.name, **_arrow_type(field.type), "nullable": field.nullable})
    required_columns = ("prompt", "reward_model", "extra_info")
    required_present = {key: key in table.column_names for key in required_columns}
    schema = {
        "format": "parquet",
        "rows": table.num_rows,
        "columns": schema_fields,
        "column_names": table.column_names,
        "required_columns_present": required_present,
    }

    issues: List[Dict[str, Any]] = list(provenance_issues)
    for key, present in required_present.items():
        if not present:
            issues.append(_issue(None, "missing_required_column", f"required column {key} is missing", key, field=key, source="parquet_schema"))
    id_indices: Dict[str, List[int]] = defaultdict(list)
    instance_images: List[Dict[str, Any]] = []
    prompt_stats = {"empty_count": 0, "message_count": Counter()}
    instance_json_kinds: Counter[str] = Counter()
    image_status_counts: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        if required_present["prompt"]:
            prompt_result = _inspect_prompt(row.get("prompt"), index, issues)
            prompt_stats["message_count"][str(prompt_result["message_count"])] += 1
            if prompt_result["empty"]:
                prompt_stats["empty_count"] += 1
        if required_present["reward_model"]:
            _inspect_reward_model(row.get("reward_model"), index, issues)
        if required_present["extra_info"]:
            extra = row.get("extra_info")
            if extra is None:
                issues.append(_issue(index, "null_extra_info", "extra_info is null"))
                extra = None
            elif not isinstance(extra, Mapping):
                issues.append(_issue(index, "invalid_extra_info_type", "extra_info is not an object", type(extra).__name__))
                extra = None
            if extra is not None:
                instance_id = extra.get("instance_id")
                if instance_id is None or (isinstance(instance_id, str) and not instance_id.strip()):
                    issues.append(_issue(index, "missing_instance_id", "extra_info.instance_id is missing or empty"))
                elif not isinstance(instance_id, str):
                    issues.append(_issue(index, "invalid_instance_id", "extra_info.instance_id is not a string", instance_id))
                else:
                    id_indices[instance_id].append(index)
                decoded = _decode_instance(extra.get("instance_json"), index, issues)
                instance_json_kinds["object" if decoded is not None else "invalid"] += 1
                if decoded is not None:
                    if "docker_image" not in decoded or decoded.get("docker_image") is None:
                        issues.append(_issue(index, "missing_docker_image", "instance_json has no docker_image"))
                        observation = {"record_index": index, "instance_id": instance_id, "status": "missing", "image": None}
                    elif not isinstance(decoded.get("docker_image"), str):
                        issues.append(_issue(index, "invalid_docker_image", "docker_image is not a string", decoded.get("docker_image")))
                        observation = {"record_index": index, "instance_id": instance_id, "status": "invalid", "image": None}
                    elif not decoded["docker_image"].strip():
                        issues.append(_issue(index, "empty_docker_image", "docker_image is empty"))
                        observation = {"record_index": index, "instance_id": instance_id, "status": "empty", "image": ""}
                    else:
                        observation = {"record_index": index, "instance_id": instance_id, "status": "valid", "image": decoded["docker_image"]}
                else:
                    observation = {"record_index": index, "instance_id": instance_id, "status": "unavailable", "image": None}
                image_status_counts[observation["status"]] += 1
                instance_images.append(observation)
        if len(samples) < sample_size:
            samples.append(_json_value({"record_index": index, **row}))

    duplicate_ids = sorted(key for key, indices in id_indices.items() if len(indices) > 1)
    duplicate_id_records = [
        {"instance_id": key, "record_indices": id_indices[key], "count": len(id_indices[key])}
        for key in duplicate_ids
    ]
    for item in duplicate_id_records:
        for index in item["record_indices"]:
            issues.append(_issue(index, "duplicate_instance_id", "instance_id occurs more than once", item["instance_id"]))
    mapping = _mapping_inspection(mapping_path, instance_images)
    issues.extend(mapping["blocking_issues"])
    missing_required = [key for key, present in required_present.items() if not present]
    warnings: List[Dict[str, Any]] = []
    if mapping["unused_mapping_keys"]:
        warnings.append({"code": "unused_mapping_keys", "count": len(mapping["unused_mapping_keys"]), "keys": mapping["unused_mapping_keys"]})
    if provenance["status"] == "UNVERIFIED":
        warnings.append({"code": "unverified_provenance", "detail": "no matching download manifest was provided"})
    summary: Dict[str, Any] = {
        "parquet_rows": table.num_rows,
        "parquet_columns": table.column_names,
        "required_column_missing": missing_required,
        "prompt": {
            "empty_count": prompt_stats["empty_count"],
            "message_count_distribution": dict(sorted(prompt_stats["message_count"].items())),
        },
        "reward_model": {
            "present": required_present["reward_model"],
            "null_count": sum(item["code"] == "null_reward_model" for item in issues),
            "invalid_type_count": sum(item["code"] == "invalid_reward_model_type" for item in issues),
        },
        "extra_info": {
            "present": required_present["extra_info"],
            "instance_json_kinds": dict(instance_json_kinds),
        },
        "images": {
            "status_counts": dict(image_status_counts),
            "error_record_indices": {
                status: [item["record_index"] for item in instance_images if item["status"] == status]
                for status in ("missing", "empty", "invalid", "unavailable")
                if any(item["status"] == status for item in instance_images)
            },
        },
        "identifiers": {
            "missing_id_count": sum(1 for item in issues if item["code"] == "missing_instance_id"),
            "duplicate_ids": duplicate_ids,
            "duplicate_id_records": duplicate_id_records,
            "duplicate_id_count": len(duplicate_ids),
        },
        "provenance_status": provenance["status"],
        "issues": issues,
        "issue_count": len(issues),
        "warnings": warnings,
        "warning_count": len(warnings),
        "mapping": {key: value for key, value in mapping.items() if key not in ("sample_records", "blocking_issues")},
    }
    status = validation_status(summary)
    summary["overall_status"] = status["status"]
    summary["blocking_findings"] = status["blocking_findings"]
    revision = provenance.get("revision") if provenance["status"] == "VERIFIED" else None
    provenance["dataset"] = DATASET
    provenance["source_urls"] = {
        name: f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}" if revision else None
        for name in EXPECTED_FILES
    }
    provenance["files"] = {
        "code.parquet": _file_info(parquet_path, revision),
        "image-mapping.jsonl": _file_info(mapping_path, revision),
    }
    provenance["strict_mode"] = strict_provenance
    return InspectionResult(schema=schema, samples=samples, summary=summary, provenance=provenance)


def write_result(result: InspectionResult, output_dir: Path, write_samples: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "schema.json": result.schema,
        "sample_records_summary.json": _sample_summary(result.samples),
        "validation_summary.json": result.summary,
        "provenance.json": result.provenance,
    }
    if write_samples:
        outputs["sample_records.json"] = result.samples
    for name, payload in outputs.items():
        target = output_dir / name
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
