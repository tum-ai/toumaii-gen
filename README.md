# Tool-calling data generation

Standalone package for generating canonical, DPO, and SFT tool-calling
datasets. 

## Layout

```text
config.yaml
tool_calling_data_generation/
tests/
pyproject.toml
README.md
```

The package supports Toucan, ToolACE, and existing canonical/DPO datasets as
inputs. To generate preference data from any other source, implement a lightweight adapter mirroring the Toucan/ToolACE loaders. Generation uses Transformers or vLLM, then writes canonical, DPO, and
SFT artifacts plus a reproducibility manifest.

## Install

From this directory, install the project with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Generate locally

Review `config.yaml`, especially the model, output path, and Toucan sample
count. For a quick local check, temporarily set `source.toucan.size` to a small
number such as `2`.

```bash
uv run tool-data-gen validate --config config.yaml
uv run tool-data-gen generate --config config.yaml
```

Each run creates a timestamped directory under the configured output root with
`canonical`, `dpo`, and `sft` Parquet files, optional JSONL/CSV previews,
and `manifest.json`.

## Upload to a private Hugging Face organization

Set `hub.repo_id` in `config.yaml` to `<organization>/<dataset>`. Authenticate
outside the config with `hf auth login` or `HF_TOKEN`, then upload one exact
local run:

```bash
uv run tool-data-gen upload \
  --config config.yaml \
  --manifest outputs/toucan/<run-id>/manifest.json \
  --confirm-private-upload
```

Upload is separate from generation. The command rejects credentials in YAML,
requires `hub.private: true`, checks organization access, refuses a public
target, and verifies privacy after upload.


## Post-process generated datasets

The ToolACE cleaner can run independently of the generation command.

No repository coordinates or commit revision are stored in the source tree.
Before each run, set all three required values in the environment:

```bash
export TOUCAN_CLEANING_SOURCE_DATASET_ID="<owner>/<source-dataset>"
export TOUCAN_CLEANING_SOURCE_REVISION="<40-character-commit-SHA>"
export TOUCAN_CLEANING_DESTINATION_DATASET_ID="<owner>/<destination-dataset>"
```

The command fails before any Hub access if one is absent. Then review the
dataset jobs, drop reasons, and long-value threshold in
`tool_calling_data_generation/post_processing/post_process_rejected.py`. Start
safely with:

```python
DROP_ROWS = False
PUSH_TO_HUB = False
DELETE_TEMPORARY_WORK_DATA = False
```

Then run it from the repository root:

```bash
uv run python -m tool_calling_data_generation.post_processing.post_process_rejected
```

The dry run still downloads and inspects every source Parquet shard, but writes
and uploads nothing. Allow enough temporary space; optionally set
`TOUCAN_CLEANING_TEMP_DIR` to a scratch directory. Only enable row dropping and
Hub upload after reviewing all warnings and counts, and authenticate first with
`hf auth login`.
