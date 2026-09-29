#!/usr/bin/env python3
"""Build a redacted Docker-only official environment smoke plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.mimo_runtime import RuntimeBridgeError, build_execution_plan, execute_official_smoke, load_verified_task_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="run the official Docker environment after confirmation")
    parser.add_argument("--confirm-runtime", action="store_true", help="explicitly confirm Docker/runtime execution")
    args = parser.parse_args()
    try:
        bundle = load_verified_task_bundle(args.parquet, args.mapping, args.manifest, args.revision, args.task_id)
        if args.execute:
            result = execute_official_smoke(bundle, confirm_runtime=args.confirm_runtime)
        else:
            result = build_execution_plan(bundle)
    except RuntimeBridgeError as exc:
        result = {"status": "BLOCKED", "code": exc.code, "detail": exc.detail, "parameter_update": False}
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 1
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
