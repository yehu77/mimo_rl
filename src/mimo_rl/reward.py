"""Pure adaptation of the fixed MiMo-Agent reward result contract.

This module only validates and classifies an already returned ``(reward,
extra)`` pair.  It never runs a verifier, starts a container, or infers a
model result from a log message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


OUTCOME_SCHEMA_VERSION = 1
OUTCOME_STATUSES = ("PASS", "TEST_FAIL", "INFRA_ERROR", "NOT_RUN", "UNKNOWN")
_INFRA_CATEGORIES = {
    "reward/testbed_corrupted",
    "REWARD_TESTBED_CORRUPTED",
    "transport_error",
}


def _reason(code: str, detail: str, **extra: Any) -> Dict[str, Any]:
    value = {"code": code, "detail": detail}
    value.update({key: item for key, item in extra.items() if item is not None})
    return value


class RewardContractError(ValueError):
    def __init__(self, errors: Sequence[Mapping[str, Any]]):
        self.errors = [dict(item) for item in errors]
        super().__init__(", ".join(item.get("code", "invalid_outcome") for item in self.errors))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RewardContractError([_reason("invalid_outcome_reward", f"{field} must be a finite number", field=field)])
    return float(value)


def _returncode(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RewardContractError([_reason("invalid_verifier_returncode", "verifier_returncode must be an integer or null")])
    return value


@dataclass(frozen=True)
class VerifierOutcome:
    """Project-owned, exportable outcome schema; it is not an upstream type."""

    status: str
    raw_reward: Optional[float]
    reward: Optional[float]
    raw_error_category: Optional[str]
    verifier_returncode: Optional[int]
    resolved: Optional[bool]
    completion_evidence: Optional[str]
    schema_version: int = OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OUTCOME_SCHEMA_VERSION:
            raise RewardContractError([_reason("unsupported_outcome_schema", "unsupported outcome schema_version")])
        if self.status not in OUTCOME_STATUSES:
            raise RewardContractError([_reason("unknown_outcome_status", "outcome status is not supported", status=self.status)])
        if self.raw_error_category is not None and (not isinstance(self.raw_error_category, str) or not self.raw_error_category.strip()):
            raise RewardContractError([_reason("invalid_error_category", "raw_error_category must be a non-empty string or null")])
        if self.completion_evidence is not None and (not isinstance(self.completion_evidence, str) or not self.completion_evidence.strip()):
            raise RewardContractError([_reason("invalid_completion_evidence", "completion_evidence must be a non-empty string or null")])
        if self.verifier_returncode is not None and (isinstance(self.verifier_returncode, bool) or not isinstance(self.verifier_returncode, int)):
            raise RewardContractError([_reason("invalid_verifier_returncode", "verifier_returncode must be an integer or null")])
        if self.resolved is not None and not isinstance(self.resolved, bool):
            raise RewardContractError([_reason("invalid_resolved", "resolved must be boolean or null")])
        for field in ("raw_reward", "reward"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise RewardContractError([_reason("invalid_outcome_reward", f"{field} must be a finite number or null", field=field)])
        if self.status in ("PASS", "TEST_FAIL"):
            if self.raw_reward is None or self.reward is None or self.verifier_returncode is None or self.resolved is None:
                raise RewardContractError([_reason("incomplete_verifier_outcome", "completed outcome requires reward, returncode and resolved")])
            if self.completion_evidence != "official_verifier_returncode":
                raise RewardContractError([_reason("missing_completion_evidence", "completed outcome requires official verifier evidence")])
            expected_status = "PASS" if self.resolved and self.verifier_returncode == 0 and float(self.raw_reward) == 1.0 else "TEST_FAIL" if (not self.resolved and self.verifier_returncode != 0 and float(self.raw_reward) == 0.0) else None
            if expected_status != self.status or float(self.reward) != float(self.raw_reward):
                raise RewardContractError([_reason("outcome_fields_conflict", "status, reward, verifier_returncode and resolved are inconsistent")])
        elif self.status == "INFRA_ERROR" and not self.raw_error_category:
            raise RewardContractError([_reason("missing_infrastructure_category", "INFRA_ERROR requires raw_error_category")])
        elif self.reward is not None:
            raise RewardContractError([_reason("nonterminal_reward", "non-completed outcome must not expose a trainable reward")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "raw_reward": self.raw_reward,
            "reward": self.reward,
            "raw_error_category": self.raw_error_category,
            "verifier_returncode": self.verifier_returncode,
            "resolved": self.resolved,
            "completion_evidence": self.completion_evidence,
        }


def validate_verifier_outcome(value: Any) -> VerifierOutcome:
    if not isinstance(value, Mapping):
        raise RewardContractError([_reason("invalid_outcome", "outcome must be an object")])
    allowed = {"schema_version", "status", "raw_reward", "reward", "raw_error_category", "verifier_returncode", "resolved", "completion_evidence"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RewardContractError([_reason("unsupported_outcome_fields", "outcome contains unsupported fields", fields=unknown)])
    try:
        raw_reward = None if value.get("raw_reward") is None else _finite_number(value.get("raw_reward"), "raw_reward")
        reward = None if value.get("reward") is None else _finite_number(value.get("reward"), "reward")
        return VerifierOutcome(
            schema_version=value.get("schema_version", OUTCOME_SCHEMA_VERSION),
            status=value.get("status"),
            raw_reward=raw_reward,
            reward=reward,
            raw_error_category=value.get("raw_error_category"),
            verifier_returncode=_returncode(value.get("verifier_returncode")),
            resolved=value.get("resolved"),
            completion_evidence=value.get("completion_evidence"),
        )
    except RewardContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise RewardContractError([_reason("invalid_outcome", str(exc))]) from exc


def normalize_mimoagent_reward(raw_reward: Any, extra: Any) -> VerifierOutcome:
    """Normalize official ``calculate_reward`` output without executing it."""
    if not isinstance(extra, Mapping):
        return VerifierOutcome("UNKNOWN", None, None, None, None, None, None)
    category = extra.get("error_category")
    if category in _INFRA_CATEGORIES or extra.get("transport_error") is True or extra.get("agent_status") == "InfraError":
        raw_category = "transport_error" if extra.get("transport_error") is True else "agent_status:InfraError" if extra.get("agent_status") == "InfraError" else str(category)
        try:
            return VerifierOutcome("INFRA_ERROR", None, None, raw_category, _returncode(extra.get("verifier_returncode")), extra.get("resolved"), "official_infrastructure_signal")
        except RewardContractError:
            return VerifierOutcome("UNKNOWN", None, None, raw_category, None, None, None)
    if category is not None:
        return VerifierOutcome("UNKNOWN", None, None, str(category) if isinstance(category, str) else None, _returncode(extra.get("verifier_returncode")), extra.get("resolved"), None)
    try:
        reward = _finite_number(raw_reward, "raw_reward")
        returncode = _returncode(extra.get("verifier_returncode"))
        resolved = extra.get("resolved")
        if returncode is None or not isinstance(resolved, bool):
            return VerifierOutcome("UNKNOWN", reward, None, None, returncode, resolved, None)
        status = "PASS" if reward == 1.0 and returncode == 0 and resolved else "TEST_FAIL" if reward == 0.0 and returncode != 0 and not resolved else "UNKNOWN"
        if status == "UNKNOWN":
            return VerifierOutcome("UNKNOWN", reward, None, None, returncode, resolved, None)
        return VerifierOutcome(status, reward, reward, None, returncode, resolved, "official_verifier_returncode")
    except RewardContractError:
        return VerifierOutcome("UNKNOWN", None, None, None, None, None, None)


__all__ = ["OUTCOME_SCHEMA_VERSION", "OUTCOME_STATUSES", "RewardContractError", "VerifierOutcome", "normalize_mimoagent_reward", "validate_verifier_outcome"]
