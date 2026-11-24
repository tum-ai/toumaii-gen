# TOUMAII-Gen

**TOUMAII-Gen** converts tool-calling demonstrations into reference-scored preference pairs for direct
preference optimization (DPO). It supports Toucan, ToolACE, and existing canonical or DPO datasets.
A lightweight source adapter can add support for another dataset.

[Project page](https://tum-ai.github.io/toumaii-gen/) · [Paper on arXiv](https://arxiv.org/abs/2611.26722) · [Hugging Face collection](https://huggingface.co/collections/tum-ai/toumaii-generated-datasets-6ae48f51f5d7bb870b417cc0)

Accepted to the [NeurIPS 2026 AABA4ET Workshop](https://marscod.github.io/AABA4ET.NeurIPS.2026/accepted-papers.html).

![TOUMAII-Gen pipeline from tool-calling demonstrations through candidate generation, assessment, filtering, and training exports](docs/assets/pipeline.png)

*TOUMAII-Gen overview, Figure 1 in the paper. Source adapters convert demonstrations into a shared format. The generator produces alternatives, then scores and filters the resulting pairs for canonical, DPO, and SFT export.*

## Released datasets

[TOUMAII-Toucan](https://huggingface.co/collections/tum-ai/toumaii-toucan-6ae37226be696553bf45f1a5) contains **4,515,503** preference pairs from quality-filtered
[Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) trajectories. Applying the same pipeline to
[ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) yielded **100,846** additional pairs in TOUMAII-ACE.

| Release | Hugging Face dataset ID | Preference pairs |
| --- | --- | --- |
| [TOUMAII-Toucan · Kimi-K2](https://huggingface.co/datasets/tum-ai/toumaii-toucan-kimi) | `tum-ai/toumaii-toucan-kimi` | 1,369,232 |
| [TOUMAII-Toucan · OSS](https://huggingface.co/datasets/tum-ai/toumaii-toucan-oss) | `tum-ai/toumaii-toucan-oss` | 1,734,057 |
| [TOUMAII-Toucan · Qwen](https://huggingface.co/datasets/tum-ai/toumaii-toucan-qwen) | `tum-ai/toumaii-toucan-qwen` | 1,412,214 |
| [TOUMAII-ACE](https://huggingface.co/datasets/tum-ai/toumaii-toolace) | `tum-ai/toumaii-toolace` | 100,846 |

Each pair keeps the conversation context and tool definitions alongside a reference response and a
model-generated alternative. For example, a candidate may call the correct price lookup with
`vs_currency="USD"` where the tool schema expects `"usd"`. See the
[interactive examples](https://tum-ai.github.io/toumaii-gen/#examples) for more cases and their ToolRL rewards.

### Quickstart

Load any release directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("tum-ai/toumaii-toucan-qwen")

## Repository layout

- [`config.yaml`](config.yaml) defines the source, generation backend, and outputs.
- [`tool_calling_data_generation/`](tool_calling_data_generation/) contains source adapters,
  Transformers and vLLM generation backends, scoring, and export.
- [`tests/`](tests/) covers the generator and its integrations.

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

## Citation

If you use TOUMAII or TOUMAII-Gen, please cite the paper.

```bibtex
@misc{poetzsch2026toumaii,
  title = {{TOUMAII}: A 4.5M-Pair Tool-Calling Preference Dataset
           and Its Source-Agnostic Generator},
  author = {Moritz Pötzsch and Andrea Gentilini and Lisa Rossgoderer and
            Tom Osterfeld and Kesava Prasad Sanjivi Arul and Swanand Ravindra Kadhe},
  year = {2026},
  eprint = {2611.26722},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2611.26722},
  note = {Accepted to the 2nd Workshop on Agentic AI Benchmarks and Applications for Enterprise Tasks (AABA4ET) at NeurIPS 2026}
}
```
