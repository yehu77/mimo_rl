"""Thin, Docker-only bridge to the fixed MiMo-Agent environment.

The default path builds a redacted plan and never imports mimoagent.  The
explicit execution function is intentionally guarded and only delegates task
commands to the official Docker environment's ``execute`` method.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping, Optional

from .inspection import inspect_dataset, validation_status
from .reward import VerifierOutcome, normalize_mimoagent_reward
from .task import TaskBundle, TaskContractError, load_image_mapping


class RuntimeBridgeError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_verified_task_bundle(parquet: Path, mapping: Path, manifest: Path, revision: str, task_id: str) -> TaskBundle:
    """Re-read the manifest and Parquet row instead of trusting a catalog."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeBridgeError("missing_data_dependency", "pyarrow is required to read the fixed Parquet input") from exc
    with TemporaryDirectory(prefix="mimo-b004-source-") as output_dir:
        result = inspect_dataset(parquet, mapping, Path(output_dir), source_revision=revision, manifest_path=manifest, strict_provenance=True)
        status = validation_status(result.summary)
        if status["status"] != "PASS":
            raise RuntimeBridgeError("source_validation_failed", "strict manifest/source validation did not pass")
    try:
        mapping_entries = load_image_mapping(mapping)
        rows = pq.read_table(parquet).to_pylist()
    except (OSError, ValueError, TaskContractError) as exc:
        raise RuntimeBridgeError("task_source_read_failed", str(exc)) from exc
    for row_index, row in enumerate(rows):
        extra = row.get("extra_info") if isinstance(row, Mapping) else None
        if isinstance(extra, Mapping) and extra.get("instance_id") == task_id:
            try:
                return TaskBundle.from_record(row, source_row_index=row_index, data_revision=revision, mapping=mapping_entries)
            except TaskContractError as exc:
                raise RuntimeBridgeError("task_contract_invalid", "; ".join(item.get("code", "invalid") for item in exc.errors)) from exc
    raise RuntimeBridgeError("task_id_not_found", f"debug task {task_id!r} was not found in the fixed Parquet input")


def build_execution_plan(bundle: TaskBundle) -> Dict[str, Any]:
    """Build a plan that contains no prompt, patch, command, or private path."""
    instance = bundle.to_mimoagent_instance()
    return {
        "status": "PLAN_ONLY",
        "task_id": bundle.task_id,
        "instance_id": instance["instance_id"],
        "dataset_type": instance["dataset_type"],
        "image": bundle.runtime.dockerhub_image,
        "image_digest": bundle.runtime.image_digest,
        "runtime_state": "PENDING_RUNTIME",
        "cwd": instance["cwd"],
        "test_command": {"length": len(instance["test_command"]), "sha256": _hash_text(instance["test_command"])},
        "test_patch": {"length": len(instance["test_patch"]), "sha256": _hash_text(instance["test_patch"])},
        "backend": "docker",
        "reward_mode": "programmatic",
        "model_enabled": False,
        "network_disabled": True,
        "privileged": False,
        "host_network": False,
        "host_mounts": [],
        "run_args": ["--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=4g", "--cpus=2"],
        "task_command_host_execution": False,
        "required_user_confirmation": "runtime location, Docker permission, and exact mapped image availability",
    }


def execute_official_smoke(bundle: TaskBundle, *, confirm_runtime: bool = False) -> Dict[str, Any]:
    """Run one no-model verifier smoke only after explicit confirmation.

    This function is never called by plan-only CLI mode.  It imports the fixed
    public factory lazily and lets the official Docker environment own command
    execution; no local subprocess is used for the task command.
    """
    if not confirm_runtime:
        raise RuntimeBridgeError("user_confirmation_required", "--execute requires explicit runtime confirmation")
    try:
        utils = importlib.import_module("mimoagent.environments.utils")
    except ImportError as exc:
        raise RuntimeBridgeError("missing_mimoagent", "fixed mimoagent runtime is not installed") from exc
    instance = bundle.to_mimoagent_instance()
    run_args = ["--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=4g", "--cpus=2"]
    env = None
    container_id = None
    try:
        env = utils.make_dataset_env(instance, environment_class="docker", reward_mode="programmatic", run_args=run_args)
        env.setup_environment()
        base_env = getattr(env, "env", None)
        container_id = getattr(base_env, "container_id", None)
        reward, test_output, extra = env.calculate_reward()
        outcome: VerifierOutcome = normalize_mimoagent_reward(reward, extra)
        return {
            "status": "SMOKE_COMPLETED",
            "task_id": bundle.task_id,
            "container_id_sha256": _hash_text(str(container_id)) if container_id else None,
            "base_ref": getattr(env, "_base_ref", None),
            "reward": outcome.to_dict(),
            "test_output_sha256": _hash_text(str(test_output)),
            "parameter_update": False,
        }
    finally:
        if env is not None:
            base_env = getattr(env, "env", None)
            cleanup = getattr(base_env, "cleanup", None)
            if callable(cleanup):
                cleanup()


__all__ = ["RuntimeBridgeError", "build_execution_plan", "execute_official_smoke", "load_verified_task_bundle"]
