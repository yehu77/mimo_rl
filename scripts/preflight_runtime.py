#!/usr/bin/env python3
"""Read-only checks for the official runtime prerequisites.

The command never installs packages, starts Docker, pulls images, creates
containers, imports model code, or executes task-provided commands.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


EXPECTED_MIMOAGENT_COMMIT = "467f0a19016f0ac4d63b8d17a1f0da9ba07f232c"
REQUIRED_PACKAGES = ("mimoagent", "uni_agent", "slime", "verl")


def _check(name: str, status: str, detail: str, *, required: bool = True, **evidence: Any) -> Dict[str, Any]:
    result = {"name": name, "status": status, "required": required, "detail": detail}
    result.update({key: value for key, value in evidence.items() if value is not None})
    return result


def _run_child(python: Path, code: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PATH": os.environ.get("PATH", "")},
    )


def _python_checks(runtime_python: Path) -> List[Dict[str, Any]]:
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        return [_check("runtime_interpreter", "FAIL", f"interpreter is missing or not executable: {runtime_python}", code="missing_interpreter")]
    try:
        version = _run_child(runtime_python, "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [_check("runtime_interpreter", "FAIL", f"interpreter probe failed: {exc}", code="interpreter_probe_failed")]
    if version.returncode != 0:
        return [_check("runtime_interpreter", "FAIL", "interpreter probe returned non-zero", code="interpreter_probe_failed")]
    version_text = version.stdout.strip()
    try:
        major, minor = (int(item) for item in version_text.split(".", 2)[:2])
    except (ValueError, TypeError):
        return [_check("runtime_interpreter", "FAIL", f"unreadable interpreter version: {version_text!r}", code="invalid_interpreter_version")]
    checks = [_check("runtime_interpreter", "PASS", f"Python {version_text}", version=version_text)]
    if (major, minor) != (3, 12):
        checks[0] = _check("runtime_interpreter", "FAIL", f"mimoagent requires Python 3.12.*, got {version_text}", code="python_version_mismatch", version=version_text)
    package_code = """import importlib.metadata as m, importlib.util, json
names = %r
out = {}
for name in names:
    spec = importlib.util.find_spec(name)
    try:
        version = m.version(name.replace('_', '-'))
    except m.PackageNotFoundError:
        version = None
    out[name] = {'origin': None if spec is None else spec.origin, 'version': version}
print(json.dumps(out, sort_keys=True))
""" % (REQUIRED_PACKAGES,)
    try:
        packages = _run_child(runtime_python, package_code)
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.append(_check("runtime_packages", "FAIL", f"package probe failed: {exc}", code="package_probe_failed"))
        return checks
    if packages.returncode != 0:
        checks.append(_check("runtime_packages", "FAIL", "package probe returned non-zero", code="package_probe_failed"))
        return checks
    try:
        found = json.loads(packages.stdout)
    except json.JSONDecodeError:
        checks.append(_check("runtime_packages", "FAIL", "package probe returned invalid JSON", code="package_probe_failed"))
        return checks
    missing = [name for name, info in found.items() if not info.get("origin") or not info.get("version")]
    if missing:
        checks.append(_check("runtime_packages", "FAIL", f"required packages missing or versionless: {', '.join(missing)}", code="missing_package", packages=found))
    else:
        checks.append(_check("runtime_packages", "PASS", "required package locations and versions resolved", packages=found))
    checks.append(_check("mimoagent_revision", "FAIL", f"installed mimoagent is not traceable to fixed source revision {EXPECTED_MIMOAGENT_COMMIT}", code="mimoagent_revision_unverified"))
    return checks


def _docker_checks() -> List[Dict[str, Any]]:
    docker = shutil.which("docker")
    if not docker:
        return [
            _check("docker_cli", "FAIL", "docker executable was not found on PATH", code="missing_docker_cli"),
            _check("docker_daemon", "NOT_RUN", "daemon probe skipped because Docker CLI is missing", code="missing_docker_cli"),
        ]
    try:
        result = subprocess.run([docker, "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=5, check=False, env={"PATH": os.environ.get("PATH", "")})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [_check("docker_cli", "PASS", f"Docker CLI found at {docker}"), _check("docker_daemon", "FAIL", f"Docker daemon probe failed: {exc}", code="docker_daemon_unavailable")]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Docker daemon did not answer").strip().splitlines()[-1][:300]
        code = "docker_permission_denied" if "permission" in detail.lower() or "denied" in detail.lower() else "docker_daemon_unavailable"
        return [_check("docker_cli", "PASS", f"Docker CLI found at {docker}"), _check("docker_daemon", "FAIL", detail, code=code)]
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [_check("docker_cli", "PASS", f"Docker CLI found at {docker}"), _check("docker_daemon", "FAIL", "Docker info was not valid JSON", code="docker_probe_invalid")]
    platform = info.get("OSType") or info.get("OperatingSystem")
    architecture = info.get("Architecture")
    return [
        _check("docker_cli", "PASS", f"Docker CLI found at {docker}"),
        _check("docker_daemon", "PASS", "Docker daemon answered a read-only info request", platform=platform, architecture=architecture),
    ]


def collect_preflight(runtime_python: Path) -> Dict[str, Any]:
    checks = _python_checks(runtime_python) + _docker_checks()
    blockers = [item for item in checks if item["required"] and item["status"] != "PASS"]
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "runtime_python": str(runtime_python),
        "checks": checks,
        "blockers": [{"name": item["name"], "code": item.get("code"), "detail": item["detail"]} for item in blockers],
        "actions_taken": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    result = collect_preflight(args.runtime_python.expanduser())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
