from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from mimo_rl.trajectory import (
    GenerationEvent,
    ToolEvent,
    Trajectory,
    TrajectoryContractError,
    check_training_export,
    compile_single_context_trace,
)


ROOT = Path(__file__).resolve().parents[2]
SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "probability_mode": "full_vocab",
}


def _event(step: int, input_ids: list[int], output_ids: list[int], probs: list[float] | None, *, policy: str = "synthetic-policy-v1", event_id: str | None = None) -> GenerationEvent:
    return GenerationEvent(
        event_id=event_id or f"event-{step}",
        request_id=f"request-{step}",
        step_index=step,
        input_token_ids=tuple(input_ids),
        output_token_ids=tuple(output_ids),
        rollout_log_probs=None if probs is None else tuple(probs),
        policy_version=policy,
        finish_reason="stop",
    )


def _trace(*events: GenerationEvent, source: str = "synthetic", group: str = "group-0", rollout: str = "rollout-0", outcome: dict | None = None, expected: int | None = None, capture_evidence: str | None = None) -> Trajectory:
    return Trajectory(
        schema_version=1,
        source_kind=source,
        task_id="task-0",
        group_id=group,
        rollout_id=rollout,
        context_id="context-0",
        policy_name="synthetic-policy-v1" if source == "synthetic" else "policy-9b",
        policy_version="synthetic-policy-v1" if source == "synthetic" else "policy-v1",
        tokenizer_fingerprint="tok-fp",
        template_fingerprint="template-fp",
        harness_fingerprint="harness-fp",
        sampling_config=SAMPLING,
        generation_events=events,
        outcome=outcome,
        expected_group_size=expected,
        capture_evidence=capture_evidence,
    )


def test_manual_two_turn_gold_fixture_matches_exact_arrays() -> None:
    trace = _trace(
        _event(0, [10, 11], [21, 22], [-0.21, -0.22]),
        _event(1, [10, 11, 21, 22, 31, 32, 33], [41], [-0.41]),
    )
    compiled = compile_single_context_trace(trace)
    assert compiled["tokens"] == [10, 11, 21, 22, 31, 32, 33, 41]
    assert compiled["prompt_length"] == 2
    assert compiled["response_length"] == 6
    assert compiled["generated_token_count"] == 3
    assert compiled["loss_mask"] == [1, 1, 0, 0, 0, 1]
    assert compiled["rollout_log_probs"] == [-0.21, -0.22, 0.0, 0.0, 0.0, -0.41]
    assert compiled["generated_event_ids"] == ["event-0", "event-0", None, None, None, "event-1"]
    assert compiled["eligible_for_training"] is False
    assert compiled["parameter_update"] is False


def test_three_turn_empty_observation_and_roundtrip_preserve_raw_calls() -> None:
    trace = _trace(
        _event(0, [1], [2], [-0.2]),
        _event(1, [1, 2], [3], [-0.3]),
        _event(2, [1, 2, 3, 4], [5], [-0.5]),
    )
    assert compile_single_context_trace(trace)["tokens"] == [1, 2, 3, 4, 5]
    restored = Trajectory.from_dict(json.loads(trace.to_json()))
    assert restored.to_dict() == trace.to_dict()


@pytest.mark.parametrize(
    "factory,code",
    [
        (lambda: _event(0, [1], [2], None), "missing_generation_log_probs"),
        (lambda: GenerationEvent("e", "r", 0, (1,), (2, 3), (-0.2,), "v"), "log_probs_length_mismatch"),
        (lambda: GenerationEvent("e", "r", 0, (1,), (2,), (math.nan,), "v"), "invalid_log_prob"),
        (lambda: GenerationEvent("e", "r", 0, (1,), (True,), (0.2,), "v"), "invalid_token_id"),
        (lambda: GenerationEvent("e", "r", 0, (1,), (-1,), (0.2,), "v"), "invalid_token_id"),
    ],
)
def test_generation_probability_and_token_negatives(factory, code: str) -> None:
    with pytest.raises(TrajectoryContractError) as exc_info:
        value = factory()
        if code == "missing_generation_log_probs":
            compile_single_context_trace(_trace(value))
    assert code in {item["code"] for item in exc_info.value.errors}


def test_append_only_prefix_changes_and_truncation_are_rejected() -> None:
    mismatch = _trace(_event(0, [10], [11], [-0.1]), _event(1, [10, 99], [12], [-0.2]))
    with pytest.raises(TrajectoryContractError) as exc_info:
        compile_single_context_trace(mismatch)
    assert "history_prefix_mismatch" in {item["code"] for item in exc_info.value.errors}

    truncated = _trace(_event(0, [10], [11], [-0.1]), _event(1, [], [12], [-0.2]))
    with pytest.raises(TrajectoryContractError) as exc_info:
        compile_single_context_trace(truncated)
    assert "history_truncated" in {item["code"] for item in exc_info.value.errors}


def test_text_only_trace_is_not_a_trainable_generation() -> None:
    trace = _trace(_event(0, [10, 11], [], None))
    with pytest.raises(TrajectoryContractError) as exc_info:
        compile_single_context_trace(trace)
    assert "no_generated_tokens" in {item["code"] for item in exc_info.value.errors}


def test_event_identity_order_and_policy_are_strict() -> None:
    with pytest.raises(TrajectoryContractError):
        _trace(_event(1, [1], [2], [-0.2]))
    with pytest.raises(TrajectoryContractError):
        _trace(_event(0, [1], [2], [-0.2]), _event(1, [1, 2], [3], [-0.3], event_id="event-0"))
    with pytest.raises(TrajectoryContractError) as exc_info:
        _trace(_event(0, [1], [2], [-0.2], policy="other"))
    assert "policy_version_conflict" in {item["code"] for item in exc_info.value.errors}


def test_tool_trainable_flag_cannot_make_observation_tokens_trainable() -> None:
    trace = _trace(
        _event(0, [10], [11], [-0.1]),
        _event(1, [10, 11, 90, 91], [12], [-0.2]),
    )
    trace = Trajectory(**{**trace.__dict__, "tool_events": (ToolEvent("tool-0", 1, "call-0", "result", True),)})
    compiled = compile_single_context_trace(trace)
    assert compiled["tokens"] == [10, 11, 90, 91, 12]
    assert compiled["loss_mask"] == [1, 0, 0, 1]
    assert compiled["rollout_log_probs"] == [-0.1, 0.0, 0.0, -0.2]


def test_export_rejects_synthetic_and_accepts_only_complete_consistent_captured_group() -> None:
    synthetic = compile_single_context_trace(_trace(_event(0, [1], [2], [-0.2])))
    decision = check_training_export([synthetic], expected_group_size=1)
    assert decision["eligible"] is False
    assert any(item["code"] == "synthetic_not_trainable" for item in decision["reasons"])

    records = []
    for index in range(2):
        trace = _trace(
            _event(0, [1], [2 + index], [-0.2 - index], policy="policy-v1"),
            source="captured",
            group="group-real",
            rollout=f"rollout-{index}",
            outcome={"status": "success"},
            expected=2,
            capture_evidence="engine_raw",
        )
        records.append(compile_single_context_trace(trace))
    assert check_training_export(records)["eligible"] is True

    incomplete = check_training_export(records[:1])
    assert incomplete["eligible"] is False
    assert any(item["code"] == "incomplete_group" for item in incomplete["reasons"])
    bad = copy.deepcopy(records)
    bad[1]["policy_version"] = "other"
    assert any(item["code"] == "group_identity_conflict" for item in check_training_export(bad)["reasons"])


def test_export_rechecks_arrays_types_and_single_context_identity() -> None:
    records = []
    for index in range(2):
        records.append(
            compile_single_context_trace(
                _trace(
                    _event(0, [1], [2 + index], [-0.2 - index], policy="policy-v1"),
                    source="captured",
                    group="group-contract",
                    rollout=f"rollout-contract-{index}",
                    outcome={"status": "success"},
                    expected=2,
                    capture_evidence="engine_raw",
                )
            )
        )

    wrong_probability_length = copy.deepcopy(records)
    wrong_probability_length[0]["rollout_log_probs"] = []
    assert "rollout_log_probs_length_mismatch" in {item["code"] for item in check_training_export(wrong_probability_length)["reasons"]}

    wrong_mask_type = copy.deepcopy(records)
    wrong_mask_type[0]["loss_mask"][0] = True
    assert "invalid_loss_mask_value" in {item["code"] for item in check_training_export(wrong_mask_type)["reasons"]}

    mixed_context = copy.deepcopy(records)
    mixed_context[1]["context_id"] = "other-context"
    assert "group_identity_conflict" in {item["code"] for item in check_training_export(mixed_context)["reasons"]}

    unhashable_rollout_id = copy.deepcopy(records)
    unhashable_rollout_id[0]["rollout_id"] = ["bad"]
    assert "missing_rollout_id" in {item["code"] for item in check_training_export(unhashable_rollout_id)["reasons"]}


def test_export_rejects_infrastructure_error_even_with_complete_group() -> None:
    records = []
    for index in range(2):
        records.append(
            compile_single_context_trace(
                _trace(
                    _event(0, [1], [2 + index], [-0.2 - index], policy="policy-v1"),
                    source="captured",
                    group="group-infra",
                    rollout=f"rollout-infra-{index}",
                    outcome={"status": "failed"},
                    expected=2,
                    capture_evidence="engine_raw",
                )
            )
        )
    records[0]["error_category"] = "INFRA_ERROR"
    decision = check_training_export(records)
    assert decision["eligible"] is False
    assert any(item["code"] == "infra_error_not_trainable" for item in decision["reasons"])


def test_cli_can_inspect_synthetic_but_does_not_export_it(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(_trace(_event(0, [1], [2], [-0.2])).to_json(), encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts" / "inspect_trajectory.py"), "--input", str(trace_path)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "SYNTHETIC_VALID_NOT_TRAINABLE"
    assert payload["eligible_for_training"] is False

    result = subprocess.run(command + ["--export-training"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "EXPORT_REJECTED" in result.stdout


def test_cli_rejects_missing_generation_probability_without_zero_fill(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-probability.json"
    trace_path.write_text(_trace(_event(0, [1], [2], None)).to_json(), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_trajectory.py"), "--input", str(trace_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "NOT_TRAINABLE"
    assert any(item["code"] == "missing_generation_log_probs" for item in payload["errors"])
