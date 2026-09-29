from __future__ import annotations

import math

import pytest

from mimo_rl.reward import RewardContractError, normalize_mimoagent_reward, validate_verifier_outcome


def test_normalize_official_pass_and_test_fail_without_executing_verifier() -> None:
    passed = normalize_mimoagent_reward(1.0, {"verifier_returncode": 0, "resolved": True, "test_duration": 0.1})
    assert passed.to_dict()["status"] == "PASS"
    assert passed.to_dict()["reward"] == 1.0

    failed = normalize_mimoagent_reward(0.0, {"verifier_returncode": 1, "resolved": False, "test_duration": 0.1})
    assert failed.to_dict()["status"] == "TEST_FAIL"
    assert failed.to_dict()["reward"] == 0.0


@pytest.mark.parametrize(
    "raw_reward,extra,expected",
    [
        (0.0, {"error_category": "reward/testbed_corrupted"}, "INFRA_ERROR"),
        (0.0, {"transport_error": True}, "INFRA_ERROR"),
        (0.0, {"agent_status": "InfraError"}, "INFRA_ERROR"),
        (0.0, {"model_patch": "no verifier result"}, "UNKNOWN"),
        (0.0, {"verifier_returncode": 0, "resolved": False}, "UNKNOWN"),
        (0.0, {"verifier_returncode": 1, "resolved": True}, "UNKNOWN"),
        (math.nan, {"verifier_returncode": 1, "resolved": False}, "UNKNOWN"),
    ],
)
def test_normalize_keeps_infrastructure_and_information_gaps_separate(raw_reward: float, extra: dict, expected: str) -> None:
    assert normalize_mimoagent_reward(raw_reward, extra).status == expected


@pytest.mark.parametrize(
    "outcome,code",
    [
        ({}, "unknown_outcome_status"),
        ({"status": "pending"}, "unknown_outcome_status"),
        ({"status": "PASS", "raw_reward": True}, "invalid_outcome_reward"),
        ({"status": "PASS", "raw_reward": 1.0, "reward": 1.0, "verifier_returncode": 1, "resolved": True, "completion_evidence": "official_verifier_returncode"}, "outcome_fields_conflict"),
    ],
)
def test_outcome_contract_rejects_incomplete_or_conflicting_results(outcome: dict, code: str) -> None:
    with pytest.raises(RewardContractError) as exc_info:
        validate_verifier_outcome(outcome)
    assert code in {item["code"] for item in exc_info.value.errors}


def test_validated_infrastructure_outcome_cannot_be_trainable() -> None:
    value = normalize_mimoagent_reward(0.0, {"error_category": "reward/testbed_corrupted"}).to_dict()
    assert value["status"] == "INFRA_ERROR"
    assert value["reward"] is None
    assert value["raw_error_category"] == "reward/testbed_corrupted"
