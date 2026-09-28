#!/usr/bin/env python3
"""Fetch code.parquet and image-mapping.jsonl from one immutable HF revision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mimo_rl.fetch import DataFetchError, fetch_code_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="40-character Hugging Face dataset commit SHA")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = fetch_code_data(args.revision, args.output_dir)
    except DataFetchError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
