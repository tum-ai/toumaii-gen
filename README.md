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

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For vLLM on a supported Linux host, install the matching PyTorch/vLLM build and
then install `.[vllm,dev]`.

## Generate locally

Review `config.yaml`, especially the model, output path, and Toucan sample
count. For a quick local check, temporarily set `source.toucan.size` to a small
number such as `2`.

```powershell
tool-data-gen validate --config config.yaml
tool-data-gen generate --config config.yaml
```

Each run creates a timestamped directory under the configured output root with
`canonical`, `dpo`, and `sft` Parquet files, optional JSONL/CSV previews,
and `manifest.json`.

## Upload to a private Hugging Face organization

Set `hub.repo_id` in `config.yaml` to `<organization>/<dataset>`. Authenticate
outside the config with `hf auth login` or `HF_TOKEN`, then upload one exact
local run:

```powershell
tool-data-gen upload `
  --config config.yaml `
  --manifest outputs/toucan/<run-id>/manifest.json `
  --confirm-private-upload
```

Upload is separate from generation. The command rejects credentials in YAML,
requires `hub.private: true`, checks organization access, refuses a public
target, and verifies privacy after upload.


