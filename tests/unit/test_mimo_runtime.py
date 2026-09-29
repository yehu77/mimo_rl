from __future__ import annotations

import pytest

from mimo_rl.mimo_runtime import RuntimeBridgeError, build_execution_plan, execute_official_smoke
from mimo_rl.task import RuntimeSpec, SolverTask, TaskBundle, VerifierSpec


def _bundle() -> TaskBundle:
    return TaskBundle(
        solver=SolverTask(task_id="debug-task", messages=({"role": "user", "content": "Debug task"},), problem_statement="Debug task", cwd="/testbed"),
        runtime=RuntimeSpec(
            dataset_image="dataset:debug",
            dockerhub_image="docker.io/example/debug:1",
            cwd="/testbed",
            data_revision="a" * 40,
            source_row_index=0,
            source_extra_info_index=0,
            mapping_line_indices=(0,),
            repo_identity=None,
            base_ref=None,
            image_digest=None,
            pending_runtime=("repo_identity", "base_ref", "image_digest"),
        ),
        verifier=VerifierSpec(test_patch="diff --git a/a b/a\n", test_command="bash /testbed/check.sh", verifier_timeout_sec=30),
    )


def test_plan_is_redacted_and_docker_only() -> None:
    plan = build_execution_plan(_bundle())
    assert plan["status"] == "PLAN_ONLY"
    assert plan["backend"] == "docker"
    assert plan["host_mounts"] == []
    assert plan["task_command_host_execution"] is False
    assert isinstance(plan["test_command"], dict)
    assert isinstance(plan["test_patch"], dict)
    assert "bash /testbed/check.sh" not in str(plan)
    assert "diff --git" not in str(plan)


def test_execution_requires_explicit_runtime_confirmation_before_importing_upstream() -> None:
    with pytest.raises(RuntimeBridgeError) as exc_info:
        execute_official_smoke(_bundle())
    assert exc_info.value.code == "user_confirmation_required"
