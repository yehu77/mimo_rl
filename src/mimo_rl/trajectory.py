"""Dependency-free raw trajectory contracts for B003.

This module records tokens and the probabilities returned by a generation
backend. It never tokenizes text, calls a model, executes tools, or instantiates
slime ``Sample`` objects.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TRAJECTORY_SCHEMA_VERSION = 1
_SOURCE_KINDS = ("synthetic", "captured")


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _error(code: str, detail: str, **evidence: Any) -> Dict[str, Any]:
    item = {"code": code, "detail": detail}
    item.update({key: _clone(value) for key, value in evidence.items() if value is not None})
    return item


class TrajectoryContractError(ValueError):
    """Structured trajectory validation errors."""

    def __init__(self, errors: Sequence[Mapping[str, Any]]):
        self.errors = [dict(item) for item in errors]
        super().__init__(", ".join(str(item.get("code", "unknown")) for item in self.errors) or "trajectory contract is invalid")


def _string(value: Any, code: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryContractError([_error(code, f"{field} must be a non-empty string", field=field)])
    return value


def _index(value: Any, code: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrajectoryContractError([_error(code, f"{field} must be a non-negative integer", field=field)])
    return value


def _tokens(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TrajectoryContractError([_error("invalid_token_ids", f"{field} must be a list of token IDs", field=field)])
    result = []
    for position, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise TrajectoryContractError([_error("invalid_token_id", f"{field}[{position}] must be a non-negative integer", field=field, position=position)])
        result.append(token)
    return tuple(result)


def _log_probs(value: Any, output_length: int) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TrajectoryContractError([_error("invalid_log_probs", "rollout_log_probs must be a list or null", field="rollout_log_probs")])
    if len(value) != output_length:
        raise TrajectoryContractError([_error("log_probs_length_mismatch", "rollout_log_probs must align with output_token_ids", expected=output_length, actual=len(value))])
    result = []
    for position, probability in enumerate(value):
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
            raise TrajectoryContractError([_error("invalid_log_prob", "log probability must be finite", position=position)])
        result.append(float(probability))
    return tuple(result)


@dataclass(frozen=True)
class GenerationEvent:
    event_id: str
    request_id: str
    step_index: int
    input_token_ids: Tuple[int, ...]
    output_token_ids: Tuple[int, ...]
    rollout_log_probs: Optional[Tuple[float, ...]]
    policy_version: str
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        event_id = _string(self.event_id, "invalid_event_id", "event_id")
        request_id = _string(self.request_id, "invalid_request_id", "request_id")
        step_index = _index(self.step_index, "invalid_step_index", "step_index")
        inputs = _tokens(self.input_token_ids, "input_token_ids")
        outputs = _tokens(self.output_token_ids, "output_token_ids")
        probabilities = _log_probs(self.rollout_log_probs, len(outputs))
        policy_version = _string(self.policy_version, "invalid_policy_version", "policy_version")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise TrajectoryContractError([_error("invalid_finish_reason", "finish_reason must be a string or null")])
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "input_token_ids", inputs)
        object.__setattr__(self, "output_token_ids", outputs)
        object.__setattr__(self, "rollout_log_probs", probabilities)
        object.__setattr__(self, "policy_version", policy_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "step_index": self.step_index,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "rollout_log_probs": None if self.rollout_log_probs is None else list(self.rollout_log_probs),
            "policy_version": self.policy_version,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationEvent":
        if not isinstance(value, Mapping):
            raise TrajectoryContractError([_error("invalid_generation_event", "generation event must be an object")])
        unknown = sorted(set(value) - {"event_id", "request_id", "step_index", "input_token_ids", "output_token_ids", "rollout_log_probs", "policy_version", "finish_reason"})
        if unknown:
            raise TrajectoryContractError([_error("unsupported_generation_event_fields", "generation event contains unsupported fields", fields=unknown)])
        return cls(
            event_id=value.get("event_id"),
            request_id=value.get("request_id"),
            step_index=value.get("step_index"),
            input_token_ids=value.get("input_token_ids"),
            output_token_ids=value.get("output_token_ids"),
            rollout_log_probs=value.get("rollout_log_probs"),
            policy_version=value.get("policy_version"),
            finish_reason=value.get("finish_reason"),
        )


@dataclass(frozen=True)
class ToolEvent:
    event_id: str
    step_index: int
    tool_call_id: str
    result_summary: Optional[str] = None
    trainable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _string(self.event_id, "invalid_tool_event_id", "event_id"))
        object.__setattr__(self, "step_index", _index(self.step_index, "invalid_step_index", "step_index"))
        object.__setattr__(self, "tool_call_id", _string(self.tool_call_id, "invalid_tool_call_id", "tool_call_id"))
        if self.result_summary is not None and not isinstance(self.result_summary, str):
            raise TrajectoryContractError([_error("invalid_tool_result", "result_summary must be a string or null")])
        if not isinstance(self.trainable, bool):
            raise TrajectoryContractError([_error("invalid_tool_trainable", "tool event trainable must be boolean")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "step_index": self.step_index,
            "tool_call_id": self.tool_call_id,
            "result_summary": self.result_summary,
            "trainable": self.trainable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolEvent":
        if not isinstance(value, Mapping):
            raise TrajectoryContractError([_error("invalid_tool_event", "tool event must be an object")])
        unknown = sorted(set(value) - {"event_id", "step_index", "tool_call_id", "result_summary", "trainable"})
        if unknown:
            raise TrajectoryContractError([_error("unsupported_tool_event_fields", "tool event contains unsupported fields", fields=unknown)])
        return cls(
            event_id=value.get("event_id"),
            step_index=value.get("step_index"),
            tool_call_id=value.get("tool_call_id"),
            result_summary=value.get("result_summary"),
            trainable=value.get("trainable", False),
        )


def _sampling_config(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryContractError([_error("invalid_sampling_config", "sampling_config must be an object")])
    config = dict(value)
    temperature = config.get("temperature")
    top_p = config.get("top_p")
    top_k = config.get("top_k", 0)
    repetition_penalty = config.get("repetition_penalty", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) != 1.0:
        raise TrajectoryContractError([_error("unsupported_sampling_config", "B003 requires temperature=1.0")])
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or float(top_p) != 1.0:
        raise TrajectoryContractError([_error("unsupported_sampling_config", "B003 requires top_p=1.0")])
    if isinstance(top_k, bool) or top_k not in (None, 0):
        raise TrajectoryContractError([_error("unsupported_sampling_config", "B003 requires full-vocabulary top_k=0")])
    if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or float(repetition_penalty) != 1.0:
        raise TrajectoryContractError([_error("unsupported_sampling_config", "B003 requires repetition_penalty=1.0")])
    if config.get("probability_mode") != "full_vocab":
        raise TrajectoryContractError([_error("unsupported_sampling_config", "probability_mode must be full_vocab")])
    return _clone(config)


@dataclass(frozen=True)
class Trajectory:
    schema_version: int
    source_kind: str
    task_id: str
    group_id: str
    rollout_id: str
    context_id: str
    policy_name: str
    policy_version: str
    tokenizer_fingerprint: str
    template_fingerprint: str
    harness_fingerprint: str
    sampling_config: Mapping[str, Any]
    generation_events: Tuple[GenerationEvent, ...]
    tool_events: Tuple[ToolEvent, ...] = ()
    outcome: Optional[Mapping[str, Any]] = None
    error_category: Optional[str] = None
    expected_group_size: Optional[int] = None
    capture_evidence: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise TrajectoryContractError([_error("unsupported_schema_version", "unsupported trajectory schema_version")])
        if self.source_kind not in _SOURCE_KINDS:
            raise TrajectoryContractError([_error("invalid_source_kind", "source_kind must be synthetic or captured")])
        for field in ("task_id", "group_id", "rollout_id", "context_id", "policy_name", "policy_version", "tokenizer_fingerprint", "template_fingerprint", "harness_fingerprint"):
            _string(getattr(self, field), "invalid_identity", field)
        if self.source_kind == "synthetic" and not self.policy_name.startswith("synthetic"):
            raise TrajectoryContractError([_error("synthetic_policy_identity", "synthetic trajectory must use a synthetic policy identity")])
        if self.capture_evidence is not None and not isinstance(self.capture_evidence, str):
            raise TrajectoryContractError([_error("invalid_capture_evidence", "capture_evidence must be a string or null")])
        if not isinstance(self.sampling_config, Mapping):
            raise TrajectoryContractError([_error("invalid_sampling_config", "sampling_config must be an object")])
        config = _sampling_config(self.sampling_config)
        if not isinstance(self.generation_events, (list, tuple)) or not self.generation_events:
            raise TrajectoryContractError([_error("missing_generation_events", "trajectory requires generation events")])
        events = tuple(event if isinstance(event, GenerationEvent) else GenerationEvent.from_dict(event) for event in self.generation_events)
        event_ids = set()
        request_ids = set()
        for expected_step, event in enumerate(events):
            if event.step_index != expected_step:
                raise TrajectoryContractError([_error("event_order_error", "generation events must be ordered and contiguous", event_id=event.event_id, step_index=event.step_index)])
            if event.event_id in event_ids or event.request_id in request_ids:
                raise TrajectoryContractError([_error("duplicate_event_id", "event_id and request_id must be unique", event_id=event.event_id, step_index=event.step_index)])
            event_ids.add(event.event_id)
            request_ids.add(event.request_id)
            if event.policy_version != self.policy_version:
                raise TrajectoryContractError([_error("policy_version_conflict", "generation event policy_version differs from trajectory", event_id=event.event_id)])
        tools = tuple(tool if isinstance(tool, ToolEvent) else ToolEvent.from_dict(tool) for tool in (self.tool_events or ()))
        if self.outcome is not None and not isinstance(self.outcome, Mapping):
            raise TrajectoryContractError([_error("invalid_outcome", "outcome must be an object or null")])
        if self.error_category is not None and not isinstance(self.error_category, str):
            raise TrajectoryContractError([_error("invalid_error_category", "error_category must be a string or null")])
        if self.expected_group_size is not None:
            _index(self.expected_group_size, "invalid_group_size", "expected_group_size")
            if self.expected_group_size == 0:
                raise TrajectoryContractError([_error("invalid_group_size", "expected_group_size must be positive")])
        object.__setattr__(self, "sampling_config", config)
        object.__setattr__(self, "generation_events", events)
        object.__setattr__(self, "tool_events", tools)
        object.__setattr__(self, "outcome", None if self.outcome is None else _clone(dict(self.outcome)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "context_id": self.context_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "template_fingerprint": self.template_fingerprint,
            "harness_fingerprint": self.harness_fingerprint,
            "sampling_config": _clone(dict(self.sampling_config)),
            "generation_events": [event.to_dict() for event in self.generation_events],
            "tool_events": [event.to_dict() for event in self.tool_events],
            "outcome": None if self.outcome is None else _clone(dict(self.outcome)),
            "error_category": self.error_category,
            "expected_group_size": self.expected_group_size,
            "capture_evidence": self.capture_evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trajectory":
        if not isinstance(value, Mapping):
            raise TrajectoryContractError([_error("invalid_trajectory", "trajectory must be an object")])
        unknown = sorted(set(value) - {"schema_version", "source_kind", "task_id", "group_id", "rollout_id", "context_id", "policy_name", "policy_version", "tokenizer_fingerprint", "template_fingerprint", "harness_fingerprint", "sampling_config", "generation_events", "tool_events", "outcome", "error_category", "expected_group_size", "capture_evidence"})
        if unknown:
            raise TrajectoryContractError([_error("unsupported_trajectory_fields", "trajectory contains unsupported fields", fields=unknown)])
        events = value.get("generation_events")
        tools = value.get("tool_events", [])
        if not isinstance(events, list) or not isinstance(tools, list):
            raise TrajectoryContractError([_error("invalid_event_list", "generation_events and tool_events must be lists")])
        return cls(
            schema_version=value.get("schema_version"),
            source_kind=value.get("source_kind"),
            task_id=value.get("task_id"),
            group_id=value.get("group_id"),
            rollout_id=value.get("rollout_id"),
            context_id=value.get("context_id"),
            policy_name=value.get("policy_name"),
            policy_version=value.get("policy_version"),
            tokenizer_fingerprint=value.get("tokenizer_fingerprint"),
            template_fingerprint=value.get("template_fingerprint"),
            harness_fingerprint=value.get("harness_fingerprint"),
            sampling_config=value.get("sampling_config"),
            generation_events=tuple(GenerationEvent.from_dict(item) for item in events),
            tool_events=tuple(ToolEvent.from_dict(item) for item in tools),
            outcome=value.get("outcome"),
            error_category=value.get("error_category"),
            expected_group_size=value.get("expected_group_size"),
            capture_evidence=value.get("capture_evidence"),
        )


def compile_single_context_trace(trace: Trajectory) -> Dict[str, Any]:
    """Compile recorded append-only calls into response-aligned fields."""
    if not isinstance(trace, Trajectory):
        raise TrajectoryContractError([_error("invalid_trace", "compile expects a Trajectory")])
    first = trace.generation_events[0]
    tokens = list(first.input_token_ids)
    prompt_length = len(tokens)
    previous_full = list(first.input_token_ids)
    loss_mask: List[int] = []
    rollout_log_probs: List[float] = []
    generated_event_ids: List[Optional[str]] = []
    for event in trace.generation_events:
        current_input = list(event.input_token_ids)
        if len(current_input) < len(previous_full):
            raise TrajectoryContractError([_error("history_truncated", "generation input is shorter than the prior full context", event_id=event.event_id, step_index=event.step_index)])
        if current_input[: len(previous_full)] != previous_full:
            raise TrajectoryContractError([_error("history_prefix_mismatch", "generation input does not preserve the prior full context", event_id=event.event_id, step_index=event.step_index)])
        observation = current_input[len(previous_full) :]
        if observation:
            tokens.extend(observation)
            loss_mask.extend([0] * len(observation))
            rollout_log_probs.extend([0.0] * len(observation))
            generated_event_ids.extend([None] * len(observation))
        if event.output_token_ids:
            if event.rollout_log_probs is None:
                raise TrajectoryContractError([_error("missing_generation_log_probs", "model-generated tokens require their original log probabilities", event_id=event.event_id, step_index=event.step_index)])
            tokens.extend(event.output_token_ids)
            loss_mask.extend([1] * len(event.output_token_ids))
            rollout_log_probs.extend(event.rollout_log_probs)
            generated_event_ids.extend([event.event_id] * len(event.output_token_ids))
        previous_full = current_input + list(event.output_token_ids)
    if not any(loss_mask):
        raise TrajectoryContractError([_error("no_generated_tokens", "trajectory has no model-generated tokens", field="output_token_ids")])
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "source_kind": trace.source_kind,
        "task_id": trace.task_id,
        "group_id": trace.group_id,
        "rollout_id": trace.rollout_id,
        "context_id": trace.context_id,
        "policy_name": trace.policy_name,
        "policy_version": trace.policy_version,
        "tokenizer_fingerprint": trace.tokenizer_fingerprint,
        "template_fingerprint": trace.template_fingerprint,
        "harness_fingerprint": trace.harness_fingerprint,
        "tokens": tokens,
        "prompt_length": prompt_length,
        "response_length": len(loss_mask),
        "generated_token_count": sum(loss_mask),
        "loss_mask": loss_mask,
        "rollout_log_probs": rollout_log_probs,
        "generated_event_ids": generated_event_ids,
        "sampling_config": _clone(dict(trace.sampling_config)),
        "outcome": None if trace.outcome is None else _clone(dict(trace.outcome)),
        "error_category": trace.error_category,
        "expected_group_size": trace.expected_group_size,
        "capture_evidence": trace.capture_evidence,
        "eligible_for_training": trace.source_kind == "captured",
        "validation_scope": "synthetic" if trace.source_kind == "synthetic" else "captured_contract",
        "parameter_update": False,
    }


def _check_compiled_record_shape(record: Mapping[str, Any], record_index: int, reasons: List[Dict[str, Any]]) -> None:
    """Validate the response-aligned arrays before a future training export."""
    tokens = record.get("tokens")
    loss_mask = record.get("loss_mask")
    probabilities = record.get("rollout_log_probs")
    if not isinstance(tokens, list):
        reasons.append(_error("invalid_tokens", "training record tokens must be a list", record_index=record_index))
    else:
        for position, token in enumerate(tokens):
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                reasons.append(_error("invalid_token_id", "training record token must be a non-negative integer", record_index=record_index, position=position))
                break
    if not isinstance(loss_mask, list):
        reasons.append(_error("invalid_loss_mask", "training record loss_mask must be a list", record_index=record_index))
    if not isinstance(probabilities, list):
        reasons.append(_error("invalid_rollout_log_probs", "training record rollout_log_probs must be a list", record_index=record_index))
    prompt_length = record.get("prompt_length")
    response_length = record.get("response_length")
    generated_count = record.get("generated_token_count")
    valid_prompt = isinstance(prompt_length, int) and not isinstance(prompt_length, bool) and prompt_length >= 0
    valid_response = isinstance(response_length, int) and not isinstance(response_length, bool) and response_length >= 0
    if not valid_prompt:
        reasons.append(_error("invalid_prompt_length", "prompt_length must be a non-negative integer", record_index=record_index))
    if not valid_response:
        reasons.append(_error("invalid_response_length", "response_length must be a non-negative integer", record_index=record_index))
    if isinstance(loss_mask, list) and valid_response and len(loss_mask) != response_length:
        reasons.append(_error("loss_mask_length_mismatch", "loss_mask must equal response_length", record_index=record_index, expected=response_length, actual=len(loss_mask)))
    if isinstance(probabilities, list) and valid_response and len(probabilities) != response_length:
        reasons.append(_error("rollout_log_probs_length_mismatch", "rollout_log_probs must equal response_length", record_index=record_index, expected=response_length, actual=len(probabilities)))
    if isinstance(tokens, list) and valid_prompt and valid_response and len(tokens) != prompt_length + response_length:
        reasons.append(_error("tokens_length_mismatch", "tokens must equal prompt_length plus response_length", record_index=record_index, expected=prompt_length + response_length, actual=len(tokens)))
    if isinstance(loss_mask, list) and isinstance(probabilities, list) and len(loss_mask) == len(probabilities):
        generated = 0
        for position, (mask, probability) in enumerate(zip(loss_mask, probabilities)):
            if isinstance(mask, bool) or mask not in (0, 1):
                reasons.append(_error("invalid_loss_mask_value", "loss_mask values must be 0 or 1", record_index=record_index, position=position))
                continue
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
                reasons.append(_error("invalid_rollout_log_prob", "rollout_log_probs values must be finite numbers", record_index=record_index, position=position))
                continue
            if mask == 0 and float(probability) != 0.0:
                reasons.append(_error("observation_log_prob_nonzero", "non-generated positions must use a 0.0 probability placeholder", record_index=record_index, position=position))
            if mask == 1:
                generated += 1
        if isinstance(generated_count, int) and not isinstance(generated_count, bool) and generated_count >= 0:
            if generated_count != generated:
                reasons.append(_error("generated_count_mismatch", "generated_token_count must equal sum(loss_mask)", record_index=record_index, expected=generated, actual=generated_count))
        else:
            reasons.append(_error("invalid_generated_count", "generated_token_count must be a non-negative integer", record_index=record_index))


def check_training_export(records: Sequence[Mapping[str, Any]], expected_group_size: Optional[int] = None) -> Dict[str, Any]:
    """Return future-training eligibility without computing rewards or gradients."""
    reasons: List[Dict[str, Any]] = []
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    rollout_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            reasons.append(_error("invalid_training_record", "training record must be an object", record_index=index))
            continue
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            reasons.append(_error("missing_group_id", "training record has no group_id", record_index=index))
        else:
            groups[group_id].append(record)
        _check_compiled_record_shape(record, index, reasons)
        if record.get("source_kind") != "captured":
            reasons.append(_error("synthetic_not_trainable", "synthetic or unknown source cannot be exported for training", record_index=index))
        if record.get("capture_evidence") != "engine_raw":
            reasons.append(_error("missing_capture_evidence", "captured flag is not engine evidence", record_index=index))
        if record.get("eligible_for_training") is not True:
            reasons.append(_error("record_not_eligible", "record is not eligible_for_training", record_index=index))
        if not isinstance(record.get("policy_version"), str) or not record.get("policy_version"):
            reasons.append(_error("missing_policy_version", "training record has no verified policy_version", record_index=index))
        rollout_id = record.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            reasons.append(_error("missing_rollout_id", "training record has no valid rollout_id", record_index=index))
        elif rollout_id in rollout_ids:
            reasons.append(_error("duplicate_rollout_id", "rollout_id must be unique", record_index=index))
        elif isinstance(rollout_id, str) and rollout_id:
            rollout_ids.add(rollout_id)
        if record.get("outcome") is None:
            reasons.append(_error("incomplete_outcome", "training record has no recorded verifier outcome", record_index=index))
        if record.get("error_category") == "INFRA_ERROR":
            reasons.append(_error("infra_error_not_trainable", "INFRA_ERROR cannot be exported as a normal sample", record_index=index))
    for group_id, group in groups.items():
        expected = expected_group_size
        if expected is None:
            expected_values = {record.get("expected_group_size") for record in group}
            if len(expected_values) == 1:
                expected = next(iter(expected_values))
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            reasons.append(_error("missing_expected_group_size", "group requires an expected_group_size", group_id=group_id))
        elif len(group) != expected:
            reasons.append(_error("incomplete_group", "group does not contain expected rollout count", group_id=group_id, expected=expected, actual=len(group)))
        identity_fields = ("task_id", "context_id", "policy_name", "policy_version", "tokenizer_fingerprint", "template_fingerprint", "harness_fingerprint")
        for field in identity_fields:
            values = [record.get(field) for record in group]
            if any(not isinstance(value, str) or not value for value in values):
                reasons.append(_error("invalid_group_identity", f"group {field} must be non-empty strings", group_id=group_id, field=field))
                continue
            unique_values = set(values)
            if len(unique_values) != 1:
                reasons.append(_error("group_identity_conflict", f"group {field} is inconsistent", group_id=group_id, field=field))
    return {"eligible": not reasons, "reasons": reasons, "parameter_update": False}


def export_training_records(records: Sequence[Mapping[str, Any]], expected_group_size: Optional[int] = None) -> List[Dict[str, Any]]:
    decision = check_training_export(records, expected_group_size=expected_group_size)
    if not decision["eligible"]:
        raise TrajectoryContractError(decision["reasons"])
    return [_clone(dict(record)) for record in records]


__all__ = [
    "GenerationEvent",
    "ToolEvent",
    "Trajectory",
    "TrajectoryContractError",
    "TRAJECTORY_SCHEMA_VERSION",
    "check_training_export",
    "compile_single_context_trace",
    "export_training_records",
]
