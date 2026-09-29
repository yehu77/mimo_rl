from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_preflight_reports_concrete_blockers_without_side_effects() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight_runtime.py"), "--runtime-python", sys.executable],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCKED"
    codes = {item["code"] for item in payload["blockers"]}
    assert "python_version_mismatch" in codes or "missing_package" in codes
    assert "actions_taken" in payload and payload["actions_taken"] == []
    assert "HOME" not in result.stdout
