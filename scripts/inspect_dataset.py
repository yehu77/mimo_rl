#!/usr/bin/env python3
"""Inspect official code parquet and image mapping files and write four JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.inspection import inspect_dataset, write_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample-size", default=3, type=int)
    args = parser.parse_args()
    result = inspect_dataset(args.parquet, args.mapping, args.output_dir, args.revision, args.sample_size)
    write_result(result, args.output_dir)
    print(json.dumps({"rows": result.schema["rows"], "columns": result.schema["column_names"], "issue_count": result.summary["issue_count"], "mapping_unmatched": len(result.summary["mapping"]["unmatched_instance_images"])}, ensure_ascii=False, sort_keys=True))
    mapping = result.summary["mapping"]
    clean = (
        result.summary["issue_count"] == 0
        and not mapping["parse_errors"]
        and mapping["duplicate_key_count"] == 0
        and not mapping["unmatched_instance_images"]
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
