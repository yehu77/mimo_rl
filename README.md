# mimo_rl

`mimo_rl` is the portable data and validation bootstrap for a MiMo code-agent RL project. B001 establishes a reproducible Mac development environment, reads a fixed revision of the official code dataset, and records its actual schema. It does not implement rollout, verifier execution, slime/SGLang integration, optimizer updates, checkpoints, GAR, length penalties, or multi-harness training.

B001-R1 hardens the same data-only boundary after review: fatal findings share one status function between the report and CLI, every anomaly keeps source row or mapping line indices, and official-source checks bind both input files to a verified download manifest.

## B001 status

- Base source: `chatgpt/b001-handoff` at `cac2725629af889bc48c9348c88d98c1aa2481f6`.
- Implementation branch: `codex/b001-mac-bootstrap`; R1 is developed from `898b5f8` on `codex/b001-r1`.
- Dataset: `XiaomiMiMo/MiMo-V2.6-RL-oss` revision `639865fd3374018d6cb29b9fb82dd531406fcf5f`.
- The inspected `code.parquet` has 2,698 rows and columns `data_source`, `ability`, `agent_name`, `prompt`, `reward_model`, and `extra_info`. `prompt` is a list of `{content, role}` structs; `reward_model` and `extra_info` are structs; `extra_info.instance_json` is a JSON string. The real-data check found no parse errors, empty prompts, duplicate IDs, or unmatched image mappings.

The source revision, file sizes, SHA-256 values, schema, sample summaries, and validation counts are in [`artifacts/data`](artifacts/data). Raw data and complete samples are deliberately ignored by Git and can be recreated locally with explicit flags.

## Rebuild on a Mac

B001 was run on macOS 26.5.1 arm64. Python 3.12 was not installed on this host, so the repository-level environment used the available Python 3.9.6. A future Linux/GPU machine should create a fresh environment instead of copying `.venv`; the project does not install CUDA, slime, SGLang, Megatron, model weights, or task images here.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -c 'import mimo_rl; print(mimo_rl.__version__)'
.venv/bin/pytest -q
```

B002 also records the tested Python 3.9.6 dependency set in
[`requirements-dev.lock`](requirements-dev.lock). A clean core-data rebuild may
install that file first and then run `.venv/bin/python -m pip install -e .`.

The runtime dependencies are intentionally small: `pyarrow==14.0.2`, `numpy<2` (for the Python 3.9 arm64 wheel), and pytest. Importing `mimo_rl` only loads standard-library code and does not access the network or a GPU.

## Fetch and inspect the official code files

The downloader accepts an immutable 40-character dataset commit and fetches exactly `code.parquet` and `image-mapping.jsonl`. It writes a local manifest with each file's relative path, byte size, SHA-256, and revision. Repeating the command reuses only a matching, hash-verified manifest; a same-named file without a manifest is rejected.

```bash
.venv/bin/python scripts/fetch_code_data.py \
  --revision 639865fd3374018d6cb29b9fb82dd531406fcf5f \
  --output-dir artifacts/data/raw

.venv/bin/python scripts/inspect_dataset.py \
  --parquet artifacts/data/raw/code.parquet \
  --mapping artifacts/data/raw/image-mapping.jsonl \
  --output-dir artifacts/data \
  --revision 639865fd3374018d6cb29b9fb82dd531406fcf5f \
  --strict-provenance

# Optional local-only complete samples; sample_records.json is ignored by Git.
.venv/bin/python scripts/inspect_dataset.py \
  --parquet artifacts/data/raw/code.parquet \
  --mapping artifacts/data/raw/image-mapping.jsonl \
  --output-dir artifacts/data \
  --revision 639865fd3374018d6cb29b9fb82dd531406fcf5f \
  --strict-provenance --write-samples
```

`inspect_dataset.py` writes `schema.json`, `sample_records_summary.json`, `validation_summary.json`, and `provenance.json` by default. It preserves source row indices for malformed JSON, missing or duplicate identifiers, invalid prompt/reward/extra types, empty prompts or images, and mapping failures. `--write-samples` writes complete local samples only when explicitly requested. `--strict-provenance` requires the auto-discovered `download_manifest.json` beside the inputs (or a path supplied with `--manifest`) and fails on dataset, revision, filename, size, or SHA-256 mismatch. Without a manifest, structural checks remain available but provenance is reported as `UNVERIFIED`. The mapping report derives the observed `dataset_image` → `dockerhub_image` relationship from the file; it does not invent task image names or a verifier registry.

## Scope boundary

`configs/runtime/mac.yaml` is a path-and-capability record for local data work. `configs/runtime/linux_gpu.example.yaml` intentionally leaves model, service, and GPU paths unset. `manifests/versions.yaml` records the chosen upstream source references; slime, mimoagent, and the official recipe remain `null` until their exact commits are inspected. Docker was not available on the Mac used for B001. No model was loaded, no task container was started, and no RL parameter update occurred.
