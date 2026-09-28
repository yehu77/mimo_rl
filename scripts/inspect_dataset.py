#!/usr/bin/env python3
"""Inspect official code parquet and image mapping files and write JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.inspection import inspect_dataset, validation_status, write_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", default=None, help="optional declared 40-character dataset commit SHA")
    parser.add_argument("--manifest", default=None, type=Path, help="download_manifest.json; otherwise auto-discover beside the inputs")
    parser.add_argument("--strict-provenance", action="store_true", help="fail when a matching download manifest cannot be verified")
    parser.add_argument("--write-samples", action="store_true", help="write complete sample_records.json locally; omitted by default")
    parser.add_argument("--sample-size", default=3, type=int)
    args = parser.parse_args()
    result = inspect_dataset(
        args.parquet,
        args.mapping,
        args.output_dir,
        args.revision,
        args.sample_size,
        args.manifest,
        args.strict_provenance,
    )
    write_result(result, args.output_dir, write_samples=args.write_samples)
    status = validation_status(result.summary)
    print(json.dumps({
        "status": status["status"],
        "provenance_status": result.provenance["status"],
        "rows": result.schema["rows"],
        "columns": result.schema["column_names"],
        "blocking_count": status["blocking_count"],
        "warning_count": status["warning_count"],
        "blocking_findings": status["blocking_findings"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
