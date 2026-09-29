"""Small, dependency-free task contracts for the B002/B003 data boundary.

The adapter keeps three audiences separate:

* :class:`SolverTask` contains the allowlisted request visible to a model;
* :class:`RuntimeSpec` contains source and image routing metadata for a future
  controller;
* :class:`VerifierSpec` contains the private official grading inputs.

This module only parses and serializes data. It never resolves paths, runs
commands, starts containers, imports an upstream framework, or contacts a
model service.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


OFFICIAL_FIELDS = (
    "dataset_type",
    "docker_image",
    "cwd",
    "instance_id",
    "problem_statement",
    "test_patch",
    "test_command",
    "verifier_timeout_sec",
)
TASK_SCHEMA_VERSION = 1
_RUNTIME_PENDING_FIELDS = ("repo_identity", "base_ref", "image_digest")
_MESSAGE_FIELDS = ("role", "content")
_MAPPING_FIELDS = ("dataset_image", "dockerhub_image")
_EVIDENCE_FIELDS = ("record_index", "extra_info_index", "mapping_line_indices")


def _clone(value: Any) -> Any:
    """Deep-copy JSON-shaped values without invoking user code."""
    return copy.deepcopy(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _summary(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "string", "length": len(value), "sha256": _sha256_text(value)}
    if isinstance(value, Mapping):
        return {"type": "object", "keys": sorted(str(key) for key in value)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(code: str, detail: str, **evidence: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {"code": code, "detail": detail}
    for key, value in evidence.items():
        if value is not None:
            item[key] = _clone(value)
    return item


class TaskContractError(ValueError):
    """A deterministic collection of input contract errors."""

    def __init__(self, errors: Sequence[Mapping[str, Any]]):
        self.errors = [dict(item) for item in errors]
        codes = ", ".join(str(item.get("code", "unknown")) for item in self.errors)
        super().__init__(codes or "task contract is invalid")


@dataclass(frozen=True)
class MappingEntry:
    dataset_image: str
    dockerhub_image: str
    line_indices: Tuple[int, ...]


def load_image_mapping(path: Path) -> Dict[str, MappingEntry]:
    """Parse the official JSONL mapping and retain every source line index."""
    path = Path(path)
    entries: Dict[str, MappingEntry] = {}
    errors: List[Dict[str, Any]] = []
    lines: Dict[str, List[int]] = {}
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            text = raw_line.rstrip("\n")
            if not text.strip():
                errors.append(_error("empty_mapping_line", "mapping line is empty", mapping_line_index=line_index))
                continue
            try:
                record = json.loads(text)
            except (TypeError, ValueError) as exc:
                errors.append(_error("malformed_mapping_json", str(exc), mapping_line_index=line_index))
                continue
            if not isinstance(record, Mapping):
                errors.append(_error("mapping_record_not_object", "mapping JSON value is not an object", mapping_line_index=line_index))
                continue
            key = record.get("dataset_image")
            value = record.get("dockerhub_image")
            if not isinstance(key, str) or not key.strip():
                errors.append(_error("invalid_mapping_key", "dataset_image must be a non-empty string", mapping_line_index=line_index))
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(_error("invalid_mapping_value", "dockerhub_image must be a non-empty string", mapping_line_index=line_index))
                continue
            lines.setdefault(key, []).append(line_index)
            values[key] = value
    for key, line_indices in sorted(lines.items()):
        if len(line_indices) != 1:
            errors.append(_error("duplicate_mapping_key", "dataset_image occurs more than once", dataset_image=key, mapping_line_indices=line_indices))
        else:
            entries[key] = MappingEntry(key, values[key], tuple(line_indices))
    if errors:
        raise TaskContractError(errors)
    return entries


def _decode_instance(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise TaskContractError([_error("malformed_instance_json", str(exc))]) from exc
    if not isinstance(value, Mapping):
        raise TaskContractError([_error("invalid_instance_json", "instance_json must be an object or JSON object string")])
    return value


def _validate_string(value: Any, code: str, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        detail = f"{field} must be a non-empty string" if nonempty else f"{field} must be a string"
        raise TaskContractError([_error(code, detail, field=field)])
    return value


def _validate_cwd(value: Any) -> str:
    value = _validate_string(value, "invalid_cwd", "cwd")
    if not value.startswith("/") or "\x00" in value:
        raise TaskContractError([_error("invalid_cwd", "cwd must be an absolute POSIX path", field="cwd")])
    return value


def _validate_timeout(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskContractError([_error("invalid_verifier_timeout", "verifier_timeout_sec must be a positive finite number")])
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise TaskContractError([_error("invalid_verifier_timeout", "verifier_timeout_sec must be a positive finite number")])
    return value


def _messages(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise TaskContractError([_error("invalid_prompt", "prompt must be a list of message objects")])
    result: List[Dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TaskContractError([_error("invalid_prompt_message", f"prompt message {index} is not an object")])
        if not isinstance(message.get("role"), str) or not message.get("role", "").strip() or not isinstance(message.get("content"), str):
            raise TaskContractError([_error("invalid_prompt_message", f"prompt message {index} needs string role/content")])
        extra_fields = sorted(str(key) for key in message if key not in _MESSAGE_FIELDS)
        if extra_fields:
            raise TaskContractError(
                [_error("unsupported_prompt_fields", "prompt message contains fields outside the solver allowlist", message_index=index, fields=extra_fields)]
            )
        result.append({"role": message["role"], "content": message["content"]})
    if not result:
        raise TaskContractError([_error("empty_prompt", "prompt contains no messages")])
    return result


def select_task_text(messages: Sequence[Mapping[str, Any]], problem_statement: Any, raw_prompt: Any = None) -> Tuple[str, str]:
    """Apply the fixed runner priority without rewriting either source text."""
    try:
        messages = _messages(list(messages))
    except TypeError as exc:
        raise TaskContractError([_error("invalid_prompt", "messages must be a list or tuple of message objects")]) from exc
    if raw_prompt is not None and not isinstance(raw_prompt, str):
        raise TaskContractError([_error("unsupported_raw_prompt", "raw_prompt must be a string when present")])
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt, "raw_prompt"
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"], "prompt_user"
    return _validate_string(problem_statement, "invalid_problem_statement", "problem_statement"), "problem_statement"


def _validate_selected_task_text(messages: Sequence[Mapping[str, Any]], problem_statement: str, selected: Any, source: Any, raw_prompt: Any = None) -> Tuple[str, str]:
    selected = _validate_string(selected, "invalid_selected_task_text", "selected_task_text")
    source = _validate_string(source, "invalid_task_text_source", "task_text_source")
    if source == "raw_prompt" and (not isinstance(raw_prompt, str) or not raw_prompt.strip() or selected != raw_prompt):
        raise TaskContractError([_error("task_text_conflict", "raw_prompt task text lacks matching source evidence", task_text_source=source)])
    expected, expected_source = select_task_text(messages, problem_statement, raw_prompt)
    if source != expected_source or selected != expected:
        raise TaskContractError([_error("task_text_conflict", "selected task text does not match its declared source", task_text_source=source)])
    return selected, source


@dataclass(frozen=True)
class SolverTask:
    task_id: str
    messages: Tuple[Mapping[str, Any], ...]
    problem_statement: str
    cwd: str
    selected_task_text: str = ""
    task_text_source: str = ""
    raw_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        task_id = _validate_string(self.task_id, "invalid_task_id", "task_id")
        problem_statement = _validate_string(self.problem_statement, "invalid_problem_statement", "problem_statement")
        cwd = _validate_cwd(self.cwd)
        try:
            messages = tuple(_messages(list(self.messages)))
        except TypeError as exc:
            raise TaskContractError([_error("invalid_prompt", "messages must be a list or tuple of message objects")]) from exc
        selected = self.selected_task_text
        source = self.task_text_source
        if self.raw_prompt is not None and not isinstance(self.raw_prompt, str):
            raise TaskContractError([_error("unsupported_raw_prompt", "raw_prompt must be a string when present")])
        if not selected and not source:
            selected, source = select_task_text(messages, problem_statement, self.raw_prompt)
        selected, source = _validate_selected_task_text(messages, problem_statement, selected, source, self.raw_prompt)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "problem_statement", problem_statement)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "selected_task_text", selected)
        object.__setattr__(self, "task_text_source", source)
        object.__setattr__(self, "raw_prompt", self.raw_prompt)

    def to_solver_payload(self) -> Dict[str, Any]:
        """Return one canonical task text for a future model request."""
        messages = tuple(_messages(list(self.messages)))
        selected, source = _validate_selected_task_text(messages, self.problem_statement, self.selected_task_text, self.task_text_source, self.raw_prompt)
        return {
            "task_id": self.task_id,
            "selected_task_text": selected,
            "cwd": self.cwd,
            "task_text_source": source,
        }


@dataclass(frozen=True)
class RuntimeSpec:
    dataset_image: str
    dockerhub_image: str
    cwd: str
    data_revision: Optional[str]
    source_row_index: int
    source_extra_info_index: Any
    mapping_line_indices: Tuple[int, ...]
    repo_identity: Optional[str]
    base_ref: Optional[str]
    image_digest: Optional[str]
    pending_runtime: Tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_string(self.dataset_image, "invalid_dataset_image", "dataset_image")
        _validate_string(self.dockerhub_image, "invalid_dockerhub_image", "dockerhub_image")
        _validate_cwd(self.cwd)
        if self.data_revision is not None and (not isinstance(self.data_revision, str) or not self.data_revision):
            raise TaskContractError([_error("invalid_data_revision", "data_revision must be null or a non-empty string")])
        if isinstance(self.source_row_index, bool) or not isinstance(self.source_row_index, int) or self.source_row_index < 0:
            raise TaskContractError([_error("invalid_source_row_index", "source_row_index must be a non-negative integer")])
        if not isinstance(self.mapping_line_indices, tuple) or not self.mapping_line_indices or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.mapping_line_indices):
            raise TaskContractError([_error("invalid_mapping_line_indices", "mapping_line_indices must be a non-empty tuple of non-negative integers")])
        for field in _RUNTIME_PENDING_FIELDS:
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskContractError([_error("invalid_runtime_declaration", f"{field} must be null or a non-empty declared string")])
        if tuple(self.pending_runtime) != _RUNTIME_PENDING_FIELDS:
            raise TaskContractError([_error("pending_runtime_mismatch", "pending_runtime must remain PENDING until runtime evidence is observed")])


@dataclass(frozen=True, repr=False)
class VerifierSpec:
    test_patch: str
    test_command: str
    verifier_timeout_sec: int | float

    def __post_init__(self) -> None:
        _validate_string(self.test_patch, "invalid_test_patch", "test_patch")
        _validate_string(self.test_command, "invalid_test_command", "test_command")
        _validate_timeout(self.verifier_timeout_sec)

    @property
    def private_hash(self) -> str:
        payload = {
            "test_patch": self.test_patch,
            "test_command": self.test_command,
            "verifier_timeout_sec": self.verifier_timeout_sec,
        }
        return _sha256_text(_json_dump(payload))

    def __repr__(self) -> str:
        return (
            "VerifierSpec(test_patch=<redacted "
            f"{len(self.test_patch)} chars>, test_command=<redacted "
            f"{len(self.test_command)} chars>, verifier_timeout_sec={self.verifier_timeout_sec!r}, "
            f"private_hash={self.private_hash!r})"
        )

    def public_summary(self) -> Dict[str, Any]:
        return {
            "test_patch": _summary(self.test_patch),
            "test_command": _summary(self.test_command),
            "verifier_timeout_sec": self.verifier_timeout_sec,
            "private_hash": self.private_hash,
        }

    def to_private_dict(self) -> Dict[str, Any]:
        return {
            "test_patch": self.test_patch,
            "test_command": self.test_command,
            "verifier_timeout_sec": self.verifier_timeout_sec,
        }


@dataclass(frozen=True, repr=False)
class TaskBundle:
    solver: SolverTask
    runtime: RuntimeSpec
    verifier: VerifierSpec
    unknown_fields: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.solver.task_id != self.task_id:
            raise TaskContractError([_error("task_id_conflict", "bundle and solver task_id differ")])
        if self.solver.cwd != self.runtime.cwd:
            raise TaskContractError([_error("cwd_conflict", "solver and runtime cwd differ")])
        if not isinstance(self.unknown_fields, tuple) or any(not isinstance(item, str) for item in self.unknown_fields):
            raise TaskContractError([_error("invalid_unknown_fields", "unknown_fields must be a tuple of strings")])

    @property
    def task_id(self) -> str:
        return self.solver.task_id

    def __repr__(self) -> str:
        return f"TaskBundle(task_id={self.task_id!r}, source_row_index={self.runtime.source_row_index}, verifier={self.verifier!r})"

    def to_solver_payload(self) -> Dict[str, Any]:
        return self.solver.to_solver_payload()

    def to_mimoagent_instance(self) -> Dict[str, Any]:
        """Return the private controller object expected by OpenSourceCodeEnvironment."""
        return {
            "dataset_type": "opensource-code",
            "docker_image": self.runtime.dockerhub_image,
            "cwd": self.runtime.cwd,
            "instance_id": self.task_id,
            "problem_statement": self.solver.problem_statement,
            "test_patch": self.verifier.test_patch,
            "test_command": self.verifier.test_command,
            "verifier_timeout_sec": self.verifier.verifier_timeout_sec,
        }

    def public_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_row_index": self.runtime.source_row_index,
            "source_extra_info_index": _clone(self.runtime.source_extra_info_index),
            "dataset_image": self.runtime.dataset_image,
            "dockerhub_image": self.runtime.dockerhub_image,
            "cwd": _summary(self.runtime.cwd),
            "data_revision": self.runtime.data_revision,
            "mapping_line_indices": list(self.runtime.mapping_line_indices),
            "field_presence": {field: True for field in OFFICIAL_FIELDS},
            "unknown_fields": list(self.unknown_fields),
            "runtime_status": "PENDING_RUNTIME" if self.runtime.pending_runtime else "SOURCE_CONTRACT_READY",
            "pending_runtime": list(self.runtime.pending_runtime),
            "solver": {
                "task_id": self.task_id,
                "messages": [{key: _summary(value) for key, value in message.items()} for message in self.solver.messages],
                "problem_statement": _summary(self.solver.problem_statement),
                "cwd": _summary(self.solver.cwd),
                "selected_task_text": _summary(self.solver.selected_task_text),
                "task_text_source": self.solver.task_text_source,
            },
            "verifier": self.verifier.public_summary(),
        }

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        payload = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": self.task_id,
            "source_row_index": self.runtime.source_row_index,
            "source_extra_info_index": _clone(self.runtime.source_extra_info_index),
            "runtime": {
                "dataset_image": self.runtime.dataset_image,
                "dockerhub_image": self.runtime.dockerhub_image,
                "cwd": self.runtime.cwd,
                "data_revision": self.runtime.data_revision,
                "mapping_line_indices": list(self.runtime.mapping_line_indices),
                "repo_identity": self.runtime.repo_identity,
                "base_ref": self.runtime.base_ref,
                "image_digest": self.runtime.image_digest,
                "pending_runtime": list(self.runtime.pending_runtime),
            },
            "unknown_fields": list(self.unknown_fields),
        }
        if include_private:
            payload["solver"] = {
                "task_id": self.task_id,
                "messages": _clone(list(_messages(list(self.solver.messages)))),
                "problem_statement": self.solver.problem_statement,
                "cwd": self.solver.cwd,
                "selected_task_text": self.solver.selected_task_text,
                "task_text_source": self.solver.task_text_source,
                "raw_prompt": self.solver.raw_prompt,
            }
            payload["verifier"] = self.verifier.to_private_dict()
            payload["mimoagent_instance"] = self.to_mimoagent_instance()
        else:
            payload["summary"] = self.public_summary()
        return payload

    def to_json(self, *, include_private: bool = False) -> str:
        return json.dumps(self.to_dict(include_private=include_private), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskBundle":
        if not isinstance(payload, Mapping):
            raise TaskContractError([_error("invalid_bundle_serialization", "bundle must be an object")])
        if payload.get("schema_version") != TASK_SCHEMA_VERSION:
            raise TaskContractError([_error("unsupported_schema_version", "unsupported or missing task bundle schema_version")])
        solver = payload.get("solver")
        runtime = payload.get("runtime")
        verifier = payload.get("verifier")
        if not isinstance(solver, Mapping) or not isinstance(runtime, Mapping) or not isinstance(verifier, Mapping):
            raise TaskContractError([_error("invalid_bundle_serialization", "private bundle requires solver, runtime, and verifier objects")])
        top_id = _validate_string(payload.get("task_id"), "invalid_task_id", "task_id")
        solver_id = _validate_string(solver.get("task_id"), "invalid_task_id", "solver.task_id")
        if top_id != solver_id:
            raise TaskContractError([_error("task_id_conflict", "top-level and solver task_id differ")])
        messages = tuple(_messages(solver.get("messages")))
        problem_statement = _validate_string(solver.get("problem_statement"), "invalid_problem_statement", "problem_statement")
        solver_cwd = _validate_cwd(solver.get("cwd"))
        selected = solver.get("selected_task_text")
        source = solver.get("task_text_source")
        raw_prompt = solver.get("raw_prompt")
        if raw_prompt is not None and not isinstance(raw_prompt, str):
            raise TaskContractError([_error("unsupported_raw_prompt", "raw_prompt must be a string when present")])
        if selected is None and source is None:
            selected, source = select_task_text(messages, problem_statement, raw_prompt)
        selected, source = _validate_selected_task_text(messages, problem_statement, selected, source, raw_prompt)
        runtime_cwd = _validate_cwd(runtime.get("cwd"))
        if solver_cwd != runtime_cwd:
            raise TaskContractError([_error("cwd_conflict", "solver and runtime cwd differ")])
        source_row_index = payload.get("source_row_index")
        if isinstance(source_row_index, bool) or not isinstance(source_row_index, int) or source_row_index < 0:
            raise TaskContractError([_error("invalid_source_row_index", "source_row_index must be a non-negative integer")])
        mapping_indices = runtime.get("mapping_line_indices")
        if not isinstance(mapping_indices, list) or not mapping_indices or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in mapping_indices):
            raise TaskContractError([_error("invalid_mapping_line_indices", "mapping_line_indices must be a non-empty list of non-negative integers")])
        runtime_pending = runtime.get("pending_runtime")
        if not isinstance(runtime_pending, list) or any(not isinstance(item, str) for item in runtime_pending):
            raise TaskContractError([_error("invalid_pending_runtime", "pending_runtime must be a list of strings")])
        runtime_values = {field: runtime.get(field) for field in _RUNTIME_PENDING_FIELDS}
        for field, value in runtime_values.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskContractError([_error("invalid_runtime_declaration", f"{field} must be null or a non-empty declared string")])
        data_revision = runtime.get("data_revision")
        if data_revision is not None and (not isinstance(data_revision, str) or not data_revision):
            raise TaskContractError([_error("invalid_data_revision", "data_revision must be null or a non-empty string")])
        verifier_patch = verifier.get("test_patch")
        verifier_command = verifier.get("test_command")
        timeout = _validate_timeout(verifier.get("verifier_timeout_sec"))
        verifier_spec = VerifierSpec(verifier_patch, verifier_command, timeout)
        unknown = payload.get("unknown_fields", [])
        if not isinstance(unknown, list) or any(not isinstance(item, str) for item in unknown):
            raise TaskContractError([_error("invalid_unknown_fields", "unknown_fields must be a list of strings")])
        bundle = cls(
            solver=SolverTask(task_id=top_id, messages=messages, problem_statement=problem_statement, cwd=solver_cwd, selected_task_text=selected, task_text_source=source, raw_prompt=raw_prompt),
            runtime=RuntimeSpec(dataset_image=_validate_string(runtime.get("dataset_image"), "invalid_dataset_image", "dataset_image"), dockerhub_image=_validate_string(runtime.get("dockerhub_image"), "invalid_dockerhub_image", "dockerhub_image"), cwd=runtime_cwd, data_revision=data_revision, source_row_index=source_row_index, source_extra_info_index=_clone(payload.get("source_extra_info_index")), mapping_line_indices=tuple(mapping_indices), repo_identity=runtime_values["repo_identity"], base_ref=runtime_values["base_ref"], image_digest=runtime_values["image_digest"], pending_runtime=tuple(runtime_pending)),
            verifier=verifier_spec,
            unknown_fields=tuple(unknown),
        )
        controller = payload.get("mimoagent_instance")
        if controller is not None:
            if not isinstance(controller, Mapping) or dict(controller) != bundle.to_mimoagent_instance():
                raise TaskContractError([_error("controller_instance_conflict", "mimoagent_instance does not match the bundle fields")])
        return bundle

    @classmethod
    def from_record(
        cls,
        row: Mapping[str, Any],
        *,
        source_row_index: int,
        data_revision: Optional[str],
        mapping: Mapping[str, MappingEntry],
    ) -> "TaskBundle":
        errors: List[Dict[str, Any]] = []
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping):
            raise TaskContractError([_error("invalid_extra_info", "extra_info must be an object", record_index=source_row_index)])
        outer_id = extra.get("instance_id")
        if not isinstance(outer_id, str) or not outer_id.strip():
            errors.append(_error("missing_outer_instance_id", "extra_info.instance_id must be a non-empty string", record_index=source_row_index))
        raw_instance = extra.get("instance_json")
        try:
            instance = _decode_instance(raw_instance)
        except TaskContractError as exc:
            errors.extend(exc.errors)
            instance = {}
        if not isinstance(instance, Mapping):  # defensive for type checkers
            instance = {}
        inner_id = instance.get("instance_id")
        task_id = outer_id if isinstance(outer_id, str) and outer_id.strip() else inner_id
        if not isinstance(inner_id, str) or not inner_id.strip():
            errors.append(_error("missing_instance_id", "instance_json.instance_id must be a non-empty string", record_index=source_row_index))
        if isinstance(outer_id, str) and isinstance(inner_id, str) and outer_id != inner_id:
            errors.append(_error("instance_id_conflict", "outer and inner instance_id differ", record_index=source_row_index, outer_instance_id=outer_id, inner_instance_id=inner_id))
        if not isinstance(task_id, str) or not task_id.strip():
            task_id = f"row-{source_row_index}"

        dataset_type = instance.get("dataset_type")
        if dataset_type != "opensource-code":
            errors.append(_error("invalid_dataset_type", "instance_json.dataset_type must be opensource-code", record_index=source_row_index))
        docker_image = instance.get("docker_image")
        if not isinstance(docker_image, str) or not docker_image.strip():
            errors.append(_error("invalid_docker_image", "instance_json.docker_image must be a non-empty string", record_index=source_row_index))
        cwd = instance.get("cwd")
        if not isinstance(cwd, str) or not cwd.strip():
            errors.append(_error("invalid_cwd", "cwd must be a non-empty POSIX path", record_index=source_row_index))
        elif not cwd.startswith("/") or "\x00" in cwd:
            errors.append(_error("invalid_cwd", "cwd must be an absolute POSIX path", record_index=source_row_index))
        problem_statement = instance.get("problem_statement")
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            errors.append(_error("invalid_problem_statement", "problem_statement must be a non-empty string", record_index=source_row_index))
        test_patch = instance.get("test_patch")
        if not isinstance(test_patch, str) or not test_patch:
            errors.append(_error("invalid_test_patch", "test_patch must be a non-empty string", record_index=source_row_index))
        test_command = instance.get("test_command")
        if not isinstance(test_command, str) or not test_command.strip():
            errors.append(_error("invalid_test_command", "test_command must be a non-empty string", record_index=source_row_index))
        timeout = instance.get("verifier_timeout_sec")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            errors.append(_error("invalid_verifier_timeout", "verifier_timeout_sec must be a positive finite number", record_index=source_row_index))

        mapping_entry = mapping.get(docker_image) if isinstance(docker_image, str) else None
        if mapping_entry is None:
            errors.append(_error("unmapped_docker_image", "docker_image has no unique mapping entry", record_index=source_row_index, image=docker_image))

        if errors:
            raise TaskContractError(errors)

        messages = _messages(row.get("prompt"))
        raw_prompt = row.get("raw_prompt") if "raw_prompt" in row else None
        try:
            selected_task_text, task_text_source = select_task_text(messages, problem_statement, raw_prompt)
        except TaskContractError as exc:
            raise exc
        runtime_values = {
            "repo_identity": instance.get("repo_identity"),
            "base_ref": instance.get("base_ref"),
            "image_digest": instance.get("image_digest"),
        }
        for field, value in runtime_values.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskContractError([_error("invalid_runtime_declaration", f"{field} must be null or a non-empty declared string", record_index=source_row_index)])
        pending = _RUNTIME_PENDING_FIELDS
        unknown_fields = tuple(sorted(str(key) for key in instance if key not in OFFICIAL_FIELDS and key not in runtime_values))
        return cls(
            solver=SolverTask(
                task_id=task_id,
                messages=tuple(messages),
                problem_statement=problem_statement,
                cwd=cwd,
                selected_task_text=selected_task_text,
                task_text_source=task_text_source,
                raw_prompt=raw_prompt,
            ),
            runtime=RuntimeSpec(
                dataset_image=docker_image,
                dockerhub_image=mapping_entry.dockerhub_image,
                cwd=cwd,
                data_revision=data_revision,
                source_row_index=source_row_index,
                source_extra_info_index=_clone(extra.get("index")),
                mapping_line_indices=mapping_entry.line_indices,
                repo_identity=runtime_values["repo_identity"],
                base_ref=runtime_values["base_ref"],
                image_digest=runtime_values["image_digest"],
                pending_runtime=pending,
            ),
            verifier=VerifierSpec(
                test_patch=test_patch,
                test_command=test_command,
                verifier_timeout_sec=timeout,
            ),
            unknown_fields=unknown_fields,
        )


__all__ = [
    "OFFICIAL_FIELDS",
    "MappingEntry",
    "RuntimeSpec",
    "SolverTask",
    "TaskBundle",
    "TaskContractError",
    "VerifierSpec",
    "load_image_mapping",
    "select_task_text",
]
