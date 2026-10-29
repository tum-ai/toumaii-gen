"""
Clean the pinned ToolACE DPO generation dataset configured in ``DATASET_JOBS``.

Every run inspects every source Parquet shard.

Dry-run mode (both ``DROP_ROWS`` and ``PUSH_TO_HUB`` false) creates no output
and uploads nothing, but exact counts still require all source shards to be
downloaded into a Hugging Face cache. With ``DROP_ROWS=False`` and
``PUSH_TO_HUB=True``, every row is written and uploaded while the report still
shows which rows would have been dropped.

Workflow
1. Set ``TOUCAN_CLEANING_SOURCE_DATASET_ID``,
   ``TOUCAN_CLEANING_SOURCE_REVISION``, and
   ``TOUCAN_CLEANING_DESTINATION_DATASET_ID`` in the environment. No source,
   destination, or revision is embedded in this file. Then review the globals
   in ``User configuration``, especially ``DATASET_JOBS``, ``SOURCE_REVISIONS``,
   ``LONG_VALUE_WORD_THRESHOLD``, ``DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE``,
   ``DROP_REASONS``, ``DROP_ROWS``, ``WRITE_DROP_AUDIT``, ``PUSH_TO_HUB``, and
   ``DELETE_TEMPORARY_WORK_DATA``.
2. Check each job's output directory and subset name. The destination must not
   already exist and must not equal any configured source dataset ID.
3. From the repository root, run
   ``uv run python -m tool_calling_data_generation.post_processing.post_process_rejected``.
   Start with ``DROP_ROWS=False``, ``PUSH_TO_HUB=False``, and
   ``DELETE_TEMPORARY_WORK_DATA=False``, then review every
   ``DOUBLE-CHECK REQUIRED`` warning plus the per-file, per-model, and global
   counts. Set ``DROP_ROWS=True`` only when those results are acceptable. To
   upload complete, unfiltered copies instead, leave ``DROP_ROWS=False`` and
   set ``PUSH_TO_HUB=True``.

Prerequisites
-------------
Run the script in an environment containing ``pyarrow``, ``PyYAML``, and
``huggingface_hub``. The Hugging Face account used by the local environment
must be allowed to read every configured source dataset. ``HUGGING_FACE_TOKEN``
is set to ``True``, which tells ``huggingface_hub`` to use the locally stored
token without placing a credential in this file.

Allow enough temporary space and time before starting. Every mode downloads and
reads every Parquet shard. Set the task-specific
``TOUCAN_CLEANING_TEMP_DIR`` environment variable to cluster scratch when local
disk is insufficient. Upload-and-cleanup mode retains no cache or output from
this run after each successful upload or ordinary Python failure, although a
forced process kill can prevent cleanup.

How the code is organized
-------------------------
For a focused review, read the file in this order:

1. ``User configuration`` contains the source job, pinned revision, safe mode
   switches, and the long-value threshold.
2. ``Dataset contract`` lists expected folders, columns, file kinds, and drop
   reason names.
3. ``Parsing helpers`` defines strict metadata, response, message, tool-call,
   and word-count interpretation.
4. ``Row classification`` applies rules that can be decided for one row.
5. ``Cross-row planning`` joins shards within one model and file kind, groups
   interactions, verifies histories, and precomputes cross-row decisions.
6. ``Parquet processing and dataset output`` applies the combined decisions,
   preserves the rows selected by the current policy, and rebuilds the dataset
   card.
7. ``Reporting and orchestration`` prints the report and either finalizes all
   local outputs together or uploads and removes one temporary output at a time.

The warning catalogue below is an operational reference. It does not need to
be read before following the code path above.

``LONG_VALUE_WORD_THRESHOLD=30`` is the current provisional choice, not a
claim that 30 is the final production threshold. This should be discussed and
approved or changed before an official cleaning run.

The script refuses source/destination ID collisions, overlapping job output
paths, overwriting a final output, or reusing a previous ``.incomplete``
directory. Dedicated temporary mode removes its process-specific work tree on
ordinary success and failure.

How to read the report (dry-run)
----------------------
``unique_would_drop`` counts rows matching at least one drop rule;
``would_keep`` is the resulting row count. Individual drop-reason counts can
overlap and therefore need not sum to ``unique_would_drop``. ``warnings``
counts warning events, not unique rows.

The six known drop-reason names are ``turn_id_0_and_multicall``,
``zero_reward_diff_and_equivalent``, ``chosen_response_long_value``,
``rejected_response_long_value``, ``chosen_multiple_calls_in_singlecall``, and
``rejected_equals_chosen_interaction_suffix``. Only reasons enabled in
``DROP_REASONS`` authorize a drop or appear in the audit. The suffix rule combines every
Parquet shard of one model/file-kind partition before any shard is filtered. A
verified chosen response containing multiple tool calls in a ``*_singlecall``
file is dropped because its positive counterpart belongs in the corresponding
multicall data. ``rejected_long_observed`` counts rejected responses over the
word threshold even when rejected-only long values are not configured to cause
a drop.

Warnings requiring manual review
--------------------------------
Every warning includes ``reason``, the Parquet file, zero-based row index,
conversation identifiers, ``turn_id``, and relevant values. A warning never
causes a drop by itself, but a condition that was still established in the
same rule or another independently satisfied rule may drop that row.

- ``metadata_unavailable``: metadata could not be decoded unambiguously, for
  example because it contains a duplicate JSON key; metadata-dependent rules
  and reward comparison could not be completed.
- ``annotation_scores_unavailable``: a present ``annotation_scores`` value is
  malformed; usable top-level metadata is still evaluated.
- ``turn_id_unavailable``: the turn-zero rule could not be evaluated.
- ``singleturn_multicall_nonzero_turn_id``: a ``singleturn_multicall`` row has
  a nonzero turn even though this dataset kind is expected to contain one user
  turn and no follow-up row with the same conversation ID.
- ``turn_zero_multicall_mismatch``: for ``turn_id == 0``, exactly one of the
  filename and ``metadata.toolcall_mode`` says multicall.
- ``filename_toolcall_mode_mismatch`` and
  ``filename_subset_name_mismatch``: the filename kind disagrees with the
  corresponding metadata label. Either disagreement prevents a turn-zero
  multicall drop because all three labels must agree.
- ``zero_reward_equivalence_mismatch``: exactly one of
  ``toolrl_reward_diff == 0`` and an accepted true equivalence value holds.
  Accepted true values are the boolean ``True`` and the exact string
  ``"true"``. A common example is equal rewards, normally represented by a
  stored zero difference, with a missing or false equivalence flag. The row is
  retained by this rule and must be checked. This warning also applies when
  one condition is true and the other field is missing or invalid.
- ``zero_reward_strict_equivalence_mismatch``: the stored reward difference
  and equivalence flag both request a drop, but independently verified chosen
  and rejected call multisets are different. The row is retained.
- ``zero_reward_strict_equivalence_unavailable``: the stored reward difference
  and equivalence flag both request a drop, but independent call equality
  cannot be established. The row is retained.
- ``reward_equivalence_values_unavailable``: at least one field needed for the
  preceding agreement check is missing or invalid, and neither condition can
  currently be established as true.
- ``inverse_reward_values_unavailable``: the chosen and rejected rewards
  cannot be compared, so the rare inverse-reward condition is unknown.
- ``rejected_greater_than_chosen``: the rejected reward is greater than the
  chosen reward. This is not a drop reason, but it is expected to be rare and
  should be checked at the reported row locator.
- ``chosen_response_uninspectable`` / ``rejected_response_uninspectable``:
  the response could not be decoded unambiguously as a dictionary or list, so
  its long-value condition is unknown. Duplicate dictionary keys are one such
  ambiguity.
- ``chosen_response_nesting_limit`` / ``rejected_response_nesting_limit``:
  inspection reached ``MAX_NESTING_DEPTH``; inspect the nested response.
- ``interaction_*`` warnings: a conversation identifier, numeric turn,
  cumulative message history, or parsed tool-call representation needed by the
  cross-row rule is malformed or inconsistent. Ambiguous candidates are not
  used to authorize a cross-row drop. A malformed call representation blocks
  only suffixes that depend on it, so a fully valid later candidate can still
  apply. An unusable grouping field can conservatively disable a wider scope,
  as described below.
- ``interaction_equivalence_flag_disagrees_with_calls`` specifically means
  that the stored equivalence flag disagrees with strict order-independent,
  duplicate-preserving equality of the verified chosen/rejected calls.
- ``interaction_rejected_content_ambiguous`` means nonempty rejected assistant
  content accompanies best-effort or unavailable parser provenance. It might
  contain an unparsed malformed call, so that row cannot prove equality or
  start a cross-row match. The generator's explicit parser value ``"none"``
  is treated as unavailable provenance.
- ``interaction_rejected_calls_invalid_entries`` means one or more rejected
  entries are malformed. That rejected response cannot establish exact suffix
  equality and therefore cannot authorize a cross-row
  drop. The row may still be dropped by another independently satisfied rule.
- ``singlecall_file_contains_multiple_chosen_calls`` identifies a verified
  chosen response with more than one call in a ``*_singlecall`` file. The row
  is dropped under ``chosen_multiple_calls_in_singlecall`` and should be
  checked because it is expected to be represented in multicall data.

Cross-row interaction rule
--------------------------
Within one source dataset, model folder, full filename kind, conversation, and
latest-user interaction, rows are ordered by numeric ``turn_id``. For a row
``k`` that has at least one verified chosen tool call in a later row, flatten
chosen calls from ``k`` through the interaction's final row. If rejected ``k``
is exactly equal to that flattened multiset, only rows after ``k`` through the
final row receive the cross-row drop reason. Candidate row ``k`` is retained by
this rule; it can still be dropped by another independently satisfied rule,
notably verified zero-reward equivalence. Rows before ``k`` are unaffected.
If several rows in the same interaction are valid suffix candidates, every
candidate row is protected; only non-candidate rows after the earliest match
receive this cross-row reason.

Tool-call order is ignored, duplicate multiplicity is preserved, and extra
rejected calls prevent a match. Tool name and arguments are compared structurally
and type-strictly: dictionary-key order is ignored, list order is preserved,
and values such as ``True``, ``1``, and ``1.0`` are different. Non-semantic
response-envelope fields are ignored; the message must still be an assistant
message with the exact role ``"assistant"``, and every tool call must have
``type="function"``. Cumulative histories likewise require exact canonical
roles: padded or case-changed roles warn and cannot authorize a match. Both response
columns and metadata's parsed-call fields must be strictly readable and agree.
Malformed or disagreeing representations warn rather than establish a match.
A malformed rejected entry blocks that row from starting a match,
because the complete rejected response cannot be verified. A malformed chosen
entry blocks every suffix that depends on it because it is part of the required
suffix; a fully valid later candidate can still match.

Rejected reward greater than chosen
-----------------------------------
``rejected_toolrl_reward > chosen_toolrl_reward`` is not a drop condition.
``rejected_greater`` counts all matching input rows, while
``rejected_greater_retained`` counts the subset surviving every drop rule.
Every written match receives
``metadata["rejected_greater_than_chosen"] = True``.

Treat any nonzero ``rejected_greater`` as requiring manual review because the
condition is expected to be rare. Every input case emits a row-level warning,
and the report highlights aggregate counts. In write mode, written cases can
also be located through the metadata flag; cases dropped for another reason
have no output row.

Conservative policy choices
---------------------------
- Missing, malformed, nonfinite, or otherwise unusable metadata, labels, and
  numeric fields never authorize a drop; the affected rule retains the row
  and warns.
- For the cross-row rule, an unusable ``turn_id`` or message history disables
  that entire ``conv_id`` inside the current model/file-kind partition because
  the malformed row cannot be assigned safely to an interaction. Duplicate
  turns or conflicting/cumulative histories disable only the affected
  latest-user interaction. An unusable ``conv_id`` cannot be associated with
  other rows, so that row is retained by this rule and warned.
- ``chosen_rejected_equivalent`` accepts the boolean ``True`` and the exact
  string ``"true"`` as true. Other strings, including ``"True"``,
  ``" true "``, and ``"false"``, are currently treated as invalid.
- Reward values prefer ``metadata.annotation_scores`` with top-level fallback.
  Labels and the equivalence flag prefer top-level metadata with
  ``annotation_scores`` fallback. A present but invalid preferred value is not
  replaced by the fallback value.
- ``toolrl_reward_diff`` is trusted as stored; it is not recomputed from the
  chosen and rejected rewards. The zero-reward rule drops only when the stored
  difference is zero, the stored equivalence flag is accepted true, and strict
  independently verified call-multiset equality is true. Any disagreement or
  unverifiable representation retains the row under this rule and warns.
- A best-effort rejected parser with nonempty assistant content is
  unverifiable because that content may hide a malformed call. Best-effort
  parsing with empty content may still be verified when response and metadata
  agree strictly; because raw model output is not stored in these Parquets,
  this is the strongest available check rather than absolute parser proof.
- Numeric strings are accepted, comparisons are exact with no tolerance, and
  booleans, NaN, and infinity are not accepted as numeric values.
- A turn-zero row is dropped by the multicall rule only when the filename kind,
  ``metadata.toolcall_mode``, and ``metadata.subset_name`` identify the same
  multicall kind using the exact documented label strings. Missing, padded,
  case-changed, invalid, or disagreeing labels retain the row under this rule
  and warn.
- Long-value detection ignores dictionary keys, numbers, and operators. It
  measures each scalar textual dictionary value separately using
  whitespace-delimited tokens that contain at least one letter. An opaque ID,
  URL, or hash without whitespace therefore counts as one word, even when it
  mixes letters, digits, and punctuation. Words across separate fields or list
  elements are not added together.
  At threshold 100, a response such as ``{"a": <60 words>, "b": <60 words>}``
  is retained because neither individual value exceeds the threshold.
- Undecodable responses are retained by the long-value rule and warned. The
  accepted wrappers are complete JSON (including bounded repeated encoding),
  Python literals, code fences, and ``<tool_call>`` blocks; an arbitrary JSON
  fragment embedded in otherwise broken text does not authorize a drop. An
  over-nested response also warns; only shallower values that were actually
  inspected can still establish a long-value drop. Rejected long values cause
  drops only when their global toggle is on.
- A model may omit one or more of the four file kinds. The script processes the
  shards actually present and requires the dataset card to describe exactly
  those present model/kind pairs; it does not invent missing files.
- Only written inverse-reward rows receive a new ``True`` flag. Other rows do
  not receive ``False``; an existing flag is not removed from untouched rows.

Output notes
------------
For Parquet data, write mode preserves model folders, filenames, schema, the
selected rows' source order, and logical values except for the intentional
metadata flag. With ``DROP_ROWS=False``, every source row is selected. Flagged
JSON metadata is reserialized, so its JSON serialization may change while its
logical values stay the same.
Empty cleaned shards remain as valid zero-row Parquet files. Non-Parquet files
are copied except legacy ``dataset_infos.json``. The README prose and
``configs`` are preserved after validation, while ``dataset_info`` is
regenerated with cleaned row counts, logical Arrow sizes, and final Parquet
download sizes. If the card is missing required metadata or maps an output
shard to the wrong config/split, write mode aborts before finalizing rather
than publishing inaccurate metadata. Skipping a legacy ``dataset_infos.json``
is also logged as a warning.

When ``WRITE_DROP_AUDIT=True``, the output contains
``audit/would_drop.parquet``. It has one row per removed source row
with the pinned source dataset/revision, source file and zero-based row index,
conversation identifiers, and the complete list of enabled drop reasons.

With ``PUSH_TO_HUB=True`` and ``DELETE_TEMPORARY_WORK_DATA=True``, the private
destination is created first. The cleaned source is then completed, validated,
uploaded with its original model-config layout, and deleted from the dedicated
local work tree. Hugging Face cannot make these uploads transactional, so a
remote failure may leave a partial private dataset.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import warnings
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

# Keep every cache created by this process inside one uniquely named work tree.
# The token remains in the user's normal Hugging Face config location.
TEMPORARY_WORK_PARENT = Path(
    os.environ.get("TOUCAN_CLEANING_TEMP_DIR", tempfile.gettempdir())
).expanduser().resolve()
TEMPORARY_WORK_ROOT = (
    TEMPORARY_WORK_PARENT / f"toucan-cleaning-{os.getpid()}"
)
_source_cache_override = os.environ.get("TOUCAN_CLEANING_SOURCE_CACHE_DIR")
SOURCE_CACHE_DIR = Path(
    _source_cache_override
    if _source_cache_override
    else TEMPORARY_WORK_ROOT / "huggingface-cache"
).expanduser().resolve()
REUSING_SOURCE_CACHE = _source_cache_override is not None
XET_CACHE_DIR = TEMPORARY_WORK_ROOT / "xet-cache"
os.environ["HF_XET_CACHE"] = str(XET_CACHE_DIR)
# Kimi is large enough that Xet's optional chunk cache can exhaust local disk
# while the ordinary Hub cache and cleaned output coexist. The Hub cache still
# retains each downloaded file until the script explicitly removes it.
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ["HF_ASSETS_CACHE"] = str(TEMPORARY_WORK_ROOT / "assets-cache")

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetJob:
    source_dataset_id: str
    output_dataset_dir: Path
    subset_name: str


# Repository coordinates and the pinned revision are deliberately supplied at
# runtime. Keeping these defaults empty prevents private provenance from being
# disclosed when this source tree is shared.
SOURCE_DATASET_ID = os.environ.get("TOUCAN_CLEANING_SOURCE_DATASET_ID")
SOURCE_REVISION = os.environ.get("TOUCAN_CLEANING_SOURCE_REVISION")
DESTINATION_DATASET_ID = os.environ.get(
    "TOUCAN_CLEANING_DESTINATION_DATASET_ID"
)


ALL_DATASET_JOBS = (
    (
        DatasetJob(
            SOURCE_DATASET_ID,
            TEMPORARY_WORK_ROOT / "outputs" / "toolace",
            "toolace",
        ),
    )
    if SOURCE_DATASET_ID
    else ()
)

# A comma-separated subset filter supports safe partial retries, for example:
# TOUCAN_CLEANING_SUBSETS=kimi TOUCAN_CLEANING_RESUME_EXISTING=1 \
#   uv run python -m tool_calling_data_generation.post_processing.post_process_rejected
REQUESTED_SUBSETS = tuple(
    value.strip()
    for value in os.environ.get("TOUCAN_CLEANING_SUBSETS", "").split(",")
    if value.strip()
)
DATASET_JOBS = tuple(
    job
    for job in ALL_DATASET_JOBS
    if not REQUESTED_SUBSETS or job.subset_name in REQUESTED_SUBSETS
)

_resume_raw = os.environ.get("TOUCAN_CLEANING_RESUME_EXISTING", "0")
if _resume_raw not in {"0", "1"}:
    raise ValueError(
        "TOUCAN_CLEANING_RESUME_EXISTING must be exactly '0' or '1'"
    )
RESUME_EXISTING_DESTINATION = _resume_raw == "1"
# Pin exact commits for reproducible cleaning without storing the repository or
# commit in source control.
SOURCE_REVISIONS = (
    {SOURCE_DATASET_ID: SOURCE_REVISION}
    if SOURCE_DATASET_ID and SOURCE_REVISION
    else {}
)
INCOMPLETE_OUTPUT_SUFFIX = ".incomplete"

# Drop rows matching the documented rules, write the audit alongside each
# cleaned dataset, upload, and remove this run's dedicated temporary work tree.
DROP_ROWS = True
PUSH_TO_HUB = True
WRITE_DROP_AUDIT = True
DELETE_TEMPORARY_WORK_DATA = True
DROP_AUDIT_REPO_PATH = "audit/would_drop.parquet"

# The requested destination is a genuinely private organization dataset.
DESTINATION_DATASET_PRIVATE = True
MANUAL_GATE_DESTINATION_DATASETS = False

# ToolACE already has one useful Hugging Face config per model folder. Preserve
# that source layout at the destination instead of wrapping the whole source in
# an additional synthetic ``toolace`` config/prefix.
PRESERVE_SINGLE_SOURCE_LAYOUT = True

# A scalar textual dictionary value is long when it contains strictly more
# than this many lexical words. This working value is intentionally kept at 30.
# the final threshold is a configurable policy choice.
# Nested structures are traversed, but separate
# fields and list elements are not combined.  For example, at threshold 100,
# {"a": <60 words>, "b": <60 words>} is retained.
LONG_VALUE_WORD_THRESHOLD = 30

# Chosen long values always make a row a drop candidate.  Enable this to make
# rejected-only long values candidates too; they are measured either way.
DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE = False

REJECTED_GREATER_THAN_CHOSEN_FLAG = "rejected_greater_than_chosen"

PARQUET_BATCH_SIZE = 8_192
PARQUET_COMPRESSION = "zstd"
HUGGING_FACE_TOKEN: bool | str | None = True
MAX_NESTING_DEPTH = 30
MAX_STRING_DECODE_DEPTH = 4
LOG_LEVEL = logging.INFO


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------

EXPECTED_MODEL_FOLDERS = (
    "gemma4_E2B",
    "granite4_1b",
    "granite4_350m",
    "granite4_3b",
    "qwen3_0.6B",
    "qwen3_1.7B",
)

FILE_KIND_TO_METADATA = {
    "multiturn_singlecall": ("multiturn_singlecall", "single"),
    "multiturn_multicall": ("multiturn_multicall", "multi"),
    "singleturn_multicall": ("singleturn_multicall", "multi"),
    "singleturn_singlecall": ("singleturn_singlecall", "single"),
}

EXPECTED_COLUMNS = (
    "messages",
    "conv_id",
    "turn_id",
    "conv_tracking_id",
    "custom_system_prompt",
    "is_custom_system_prompt_used",
    "tool_catalog",
    "chosen_response",
    "rejected_response",
    "metadata",
    "prompt",
    "prompt_len",
)

CROSS_RULE_COLUMNS = (
    "messages",
    "conv_id",
    "turn_id",
    "conv_tracking_id",
    "metadata",
    "chosen_response",
    "rejected_response",
)

TURN_ZERO_MULTICALL = "turn_id_0_and_multicall"
ZERO_REWARD_EQUIVALENT = "zero_reward_diff_and_equivalent"
CHOSEN_LONG_VALUE = "chosen_response_long_value"
REJECTED_LONG_VALUE = "rejected_response_long_value"
CHOSEN_MULTIPLE_CALLS_IN_SINGLECALL = "chosen_multiple_calls_in_singlecall"
REJECTED_EQUALS_CHOSEN_SUFFIX = "rejected_equals_chosen_interaction_suffix"
KNOWN_DROP_REASONS = frozenset(
    (
        TURN_ZERO_MULTICALL,
        ZERO_REWARD_EQUIVALENT,
        CHOSEN_LONG_VALUE,
        REJECTED_LONG_VALUE,
        CHOSEN_MULTIPLE_CALLS_IN_SINGLECALL,
        REJECTED_EQUALS_CHOSEN_SUFFIX,
    )
)
DROP_REASONS = (
    # TURN_ZERO_MULTICALL,
    ZERO_REWARD_EQUIVALENT,
    CHOSEN_LONG_VALUE,
    REJECTED_LONG_VALUE,
    CHOSEN_MULTIPLE_CALLS_IN_SINGLECALL,
    REJECTED_EQUALS_CHOSEN_SUFFIX,
)
ENABLED_DROP_REASONS = frozenset(DROP_REASONS)

DROP_AUDIT_SCHEMA = pa.schema(
    (
        pa.field("source_dataset_id", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("model_folder", pa.string(), nullable=False),
        pa.field("file_kind", pa.string(), nullable=False),
        pa.field("row_index", pa.int64(), nullable=False),
        pa.field("conv_id", pa.string()),
        pa.field("conv_tracking_id", pa.string()),
        pa.field("turn_id", pa.string()),
        pa.field("would_drop_reasons", pa.list_(pa.string()), nullable=False),
    )
)

STALE_REPO_FILES = frozenset({"dataset_infos.json"})

FILE_KIND_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(kind) for kind in FILE_KIND_TO_METADATA) + r")(?=[._-]|$)"
)
# Count whitespace-delimited tokens containing a letter.  Numbers and
# operators do not count, so ``(3 * 5) + sqrt(16)`` is one word; an opaque
# alphanumeric ID or URL without whitespace also counts as one word.
WORD_PATTERN = re.compile(r"(?<!\S)(?=\S*[^\W\d_])\S+", re.UNICODE)
CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json|python|py)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
TOOL_CALL_TAG_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
CARD_FRONT_MATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)",
    re.DOTALL,
)

MISSING = object()

# One byte per row is stored during the cross-row pre-pass. Zero is deliberately
# the conservative default: an unverified row can never authorize a drop.
STRICT_EQ_UNVERIFIABLE = 0
STRICT_EQ_UNEQUAL = 1
STRICT_EQ_EQUAL = 2


# ---------------------------------------------------------------------------
# Metrics and decisions
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    input_rows: int = 0
    unique_would_drop: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    warning_reasons: Counter[str] = field(default_factory=Counter)
    rejected_long_observed: int = 0
    rejected_greater: int = 0
    rejected_greater_retained: int = 0
    retained_num_bytes: int = 0

    @property
    def would_keep(self) -> int:
        return self.input_rows - self.unique_would_drop

    @property
    def output_rows(self) -> int:
        """Rows written when output is enabled under the current policy."""

        return self.would_keep if DROP_ROWS else self.input_rows

    @property
    def warnings(self) -> int:
        return sum(self.warning_reasons.values())

    def merge(self, other: "Stats") -> None:
        self.input_rows += other.input_rows
        self.unique_would_drop += other.unique_would_drop
        self.drop_reasons.update(other.drop_reasons)
        self.warning_reasons.update(other.warning_reasons)
        self.rejected_long_observed += other.rejected_long_observed
        self.rejected_greater += other.rejected_greater
        self.rejected_greater_retained += other.rejected_greater_retained
        self.retained_num_bytes += other.retained_num_bytes


@dataclass(frozen=True)
class Decision:
    reasons: frozenset[str]
    metadata: dict[str, Any] | None
    rejected_greater: bool


@dataclass(frozen=True)
class ProcessedDataset:
    job: DatasetJob
    staged_output: Path
    combined_config: dict[str, Any]
    combined_info: dict[str, Any]


RowLocator = tuple[str, int]
CallCounter = Counter[bytes]


@dataclass(frozen=True)
class CrossWarning:
    reason: str
    detail: str


@dataclass(frozen=True)
class CrossRow:
    locator: RowLocator
    conv_id: str
    turn_id: int
    user_ordinal: int
    user_prefix_digest: bytes
    message_count: int
    messages_digest: bytes
    chosen_calls: CallCounter | None
    rejected_calls: CallCounter | None


@dataclass
class CrossRulePlan:
    drop_flags_by_file: dict[str, bytearray] = field(default_factory=dict)
    singlecall_multiple_chosen_flags_by_file: dict[str, bytearray] = field(
        default_factory=dict
    )
    strict_equivalence_states_by_file: dict[str, bytearray] = field(
        default_factory=dict
    )
    warnings_by_row: dict[RowLocator, list[CrossWarning]] = field(
        default_factory=dict
    )

    def warn(self, locator: RowLocator, reason: str, detail: str) -> None:
        warning = CrossWarning(reason, detail)
        warnings = self.warnings_by_row.setdefault(locator, [])
        if warning not in warnings:
            warnings.append(warning)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _file_kind(repo_path: str) -> str:
    match = FILE_KIND_PATTERN.match(PurePosixPath(repo_path).name)
    if match is None:
        raise ValueError(
            f"Cannot classify {repo_path!r}; expected a prefix in "
            f"{sorted(FILE_KIND_TO_METADATA)}"
        )
    return match.group(1)


def _parse_mapping(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Strictly parse a mapping stored directly or in JSON string layers."""

    if isinstance(raw, Mapping):
        return dict(raw), None

    value = raw
    for _ in range(MAX_STRING_DECODE_DEPTH):
        if not isinstance(value, str):
            break
        if not value.strip():
            return None, "value is an empty string"
        try:
            value = json.loads(
                value,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return None, f"invalid strict JSON ({type(error).__name__}: {error})"
        if isinstance(value, Mapping):
            return dict(value), None

    return None, f"expected an object, found {type(value).__name__}"


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number {value!r}")


def _strict_json_value(raw: Any) -> tuple[Any | None, str | None]:
    """Decode bounded JSON string layers without best-effort repair."""

    if not isinstance(raw, str):
        return raw, None

    value: Any = raw
    for _ in range(MAX_STRING_DECODE_DEPTH):
        if not isinstance(value, str):
            return value, None
        if not value.strip():
            return None, "value is an empty string"
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return None, f"invalid strict JSON ({type(error).__name__}: {error})"
        if decoded == value:
            return decoded, None
        value = decoded

    if isinstance(value, str):
        return None, f"exceeded {MAX_STRING_DECODE_DEPTH} JSON string layers"
    return value, None


def _canonical_value(value: Any) -> tuple[Any, ...]:
    """Return a hashable, type-strict representation of a JSON value."""

    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"nonfinite float {value!r}")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, Mapping):
        items = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"object key is {type(key).__name__}, not string")
            items.append((key, _canonical_value(child)))
        return ("object", tuple(sorted(items)))
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return ("array", tuple(_canonical_value(child) for child in value))
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def _digest_canonical(value: Any) -> bytes:
    canonical = repr(_canonical_value(value)).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def _canonical_call(call: Any) -> tuple[bytes | None, str | None]:
    if not isinstance(call, Mapping):
        return None, f"tool call is {type(call).__name__}, not an object"
    if "name" not in call:
        return None, "tool call has no name"
    name = call["name"]
    if not isinstance(name, str) or not name:
        return None, f"tool-call name is not a nonempty string: {name!r}"
    if "arguments" not in call:
        return None, "tool call has no arguments field"

    arguments, error = _strict_json_value(call["arguments"])
    if error:
        return None, f"arguments {error}"
    try:
        return _digest_canonical({"name": name, "arguments": arguments}), None
    except (TypeError, ValueError) as error:
        return None, str(error)


def _call_counter(
    raw_calls: Any,
    *,
    ignore_invalid: bool = False,
) -> tuple[CallCounter | None, str | None]:
    if not isinstance(raw_calls, Sequence) or isinstance(
        raw_calls, (str, bytes, bytearray)
    ):
        return None, f"expected a list, found {type(raw_calls).__name__}"

    calls: CallCounter = Counter()
    problems: list[str] = []
    for index, call in enumerate(raw_calls):
        key, error = _canonical_call(call)
        if error:
            problem = f"call[{index}] {error}"
            if not ignore_invalid:
                return None, problem
            problems.append(problem)
            continue
        calls[key] += 1
    return calls, "; ".join(problems) if problems else None


def _metadata_call_counter(
    metadata: Mapping[str, Any], side: str
) -> tuple[CallCounter | None, str | None]:
    key = f"parsed_{side}_tool_calls"
    if key not in metadata:
        return None, f"metadata has no {key}"
    return _call_counter(
        metadata[key],
        ignore_invalid=side == "rejected",
    )


def _strict_mapping_value(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _strict_json_value(raw)
    if error:
        return None, error
    if not isinstance(value, Mapping):
        return None, f"expected an object, found {type(value).__name__}"
    return dict(value), None


def _response_call_counter(
    raw: Any,
    side: str = "chosen",
) -> tuple[CallCounter | None, str | None, bool]:
    response, error = _strict_json_value(raw)
    if error:
        return None, error, False
    if not isinstance(response, Sequence) or isinstance(
        response, (str, bytes, bytearray)
    ):
        return (
            None,
            f"expected a message list, found {type(response).__name__}",
            False,
        )

    calls: list[Any] = []
    problems: list[str] = []
    has_nonempty_content = False
    for message_index, message in enumerate(response):
        if not isinstance(message, Mapping):
            return (
                None,
                f"message[{message_index}] is {type(message).__name__}, "
                "not an object",
                has_nonempty_content,
            )
        role = message.get("role")
        if role != "assistant":
            return (
                None,
                f"message[{message_index}] has role {role!r}, not 'assistant'",
                has_nonempty_content,
            )
        content = message.get("content")
        if (isinstance(content, str) and content.strip()) or (
            content is not None and not isinstance(content, str)
        ):
            has_nonempty_content = True
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, Sequence) or isinstance(
            raw_calls, (str, bytes, bytearray)
        ):
            return (
                None,
                f"message[{message_index}].tool_calls is not a list",
                has_nonempty_content,
            )
        for call_index, raw_call in enumerate(raw_calls):
            prefix = f"message[{message_index}].tool_calls[{call_index}]"
            if not isinstance(raw_call, Mapping):
                problem = (
                    f"{prefix} is {type(raw_call).__name__}, not an object"
                )
                if side != "rejected":
                    return None, problem, has_nonempty_content
                problems.append(problem)
                continue
            function = raw_call.get("function", MISSING)
            if raw_call.get("type") != "function":
                problem = (
                    f"{prefix}.type is {raw_call.get('type')!r}, not 'function'"
                )
                if side != "rejected":
                    return None, problem, has_nonempty_content
                problems.append(problem)
                continue
            if not isinstance(function, Mapping):
                problem = (
                    f"{prefix}.function is not an object"
                )
                if side != "rejected":
                    return None, problem, has_nonempty_content
                problems.append(problem)
                continue
            calls.append(function)
    call_counter, call_error = _call_counter(
        calls,
        ignore_invalid=side == "rejected",
    )
    if call_error:
        problems.append(call_error)
    return (
        call_counter,
        "; ".join(problems) if problems else None,
        has_nonempty_content,
    )


def _verified_call_counter(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    side: str,
) -> tuple[CallCounter | None, str | None, str | None, bool]:
    metadata_calls, metadata_error = _metadata_call_counter(metadata, side)
    response_calls, response_error, has_nonempty_content = _response_call_counter(
        row.get(f"{side}_response"),
        side,
    )
    if metadata_calls is None or response_calls is None:
        problems = []
        if metadata_error:
            problems.append(f"metadata: {metadata_error}")
        if response_error:
            problems.append(f"response: {response_error}")
        return (
            None,
            f"interaction_{side}_calls_unavailable",
            "; ".join(problems),
            has_nonempty_content,
        )
    if metadata_calls != response_calls:
        return (
            None,
            f"interaction_{side}_calls_disagree",
            "metadata and response call multisets disagree: "
            f"metadata_calls={sum(metadata_calls.values())}, "
            f"metadata_unique={len(metadata_calls)}, "
            f"response_calls={sum(response_calls.values())}, "
            f"response_unique={len(response_calls)}",
            has_nonempty_content,
        )
    if metadata_error or response_error:
        problems = []
        if metadata_error:
            problems.append(f"metadata: {metadata_error}")
        if response_error:
            problems.append(f"response: {response_error}")
        return (
            None,
            "interaction_rejected_calls_invalid_entries",
            "; ".join(problems),
            has_nonempty_content,
        )
    if side == "chosen" and not metadata_calls:
        return (
            metadata_calls,
            "interaction_chosen_calls_empty",
            "chosen metadata and response agree on an empty tool-call list",
            has_nonempty_content,
        )
    return metadata_calls, None, None, has_nonempty_content


def _rejected_content_ambiguity(
    metadata: Mapping[str, Any],
    has_nonempty_content: bool,
) -> str | None:
    """Explain when rejected content could hide an unparsed malformed call."""

    if not has_nonempty_content:
        return None

    raw_parser = metadata.get("parser_used_rejected", MISSING)
    raw_best_effort = metadata.get("is_best_effort_parser_used", MISSING)
    best_effort, best_effort_error = _canonical_bool(raw_best_effort)
    parser_is_best_effort = (
        isinstance(raw_parser, str)
        and raw_parser.lower().startswith("best_effort")
    )
    if parser_is_best_effort or best_effort is True:
        return (
            "rejected assistant content is nonempty while parser provenance "
            f"is parser_used_rejected={_shown(raw_parser)}, "
            f"is_best_effort_parser_used={_shown(raw_best_effort)}; content "
            "may contain an unparsed malformed call"
        )
    if raw_parser == "none" or not isinstance(raw_parser, str) or not raw_parser:
        detail = (
            f"parser_used_rejected={_shown(raw_parser)} indicates unavailable "
            "parser provenance while "
            "rejected assistant content is nonempty"
        )
        if best_effort_error:
            detail += (
                "; is_best_effort_parser_used "
                f"{best_effort_error} ({_shown(raw_best_effort)})"
            )
        return detail
    return None


def _strict_turn_id(raw: Any) -> tuple[int | None, str | None]:
    if type(raw) is int:
        turn_id = raw
    elif isinstance(raw, str):
        match = re.fullmatch(r"(?:(?:single|multi)_)?([0-9]+)", raw)
        if match is None:
            return None, (
                "expected a nonnegative integer or ToolACE turn label "
                f"('single_N'/'multi_N'), found {raw!r}"
            )
        turn_id = int(match.group(1))
    else:
        return None, (
            "expected a nonnegative integer or ToolACE turn label, "
            f"found {raw!r}"
        )
    if turn_id < 0:
        return None, f"turn_id is negative: {turn_id}"
    return turn_id, None


def _turn_id_mode(raw: Any) -> str | None:
    """Return the ToolACE call-mode prefix, if the turn ID carries one."""

    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"(single|multi)_[0-9]+", raw)
    return match.group(1) if match else None


def _messages_identity(
    raw: Any,
) -> tuple[tuple[int, bytes, int, bytes] | None, str | None]:
    messages, error = _strict_json_value(raw)
    if error:
        return None, error
    if not isinstance(messages, Sequence) or isinstance(
        messages, (str, bytes, bytearray)
    ):
        return None, f"expected a message list, found {type(messages).__name__}"

    normalized_messages: list[dict[str, Any]] = []
    user_messages: list[dict[str, Any]] = []
    accepted_roles = {"system", "user", "assistant", "tool", "function"}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return None, (
                f"message[{index}] is {type(message).__name__}, not an object"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None, f"message[{index}] has invalid role {role!r}"
        if role not in accepted_roles:
            return None, (
                f"message[{index}] has unknown or noncanonical role {role!r}"
            )
        normalized = dict(message)
        if "content" not in normalized:
            return None, f"message[{index}] has no content field"
        try:
            _canonical_value(normalized)
        except (TypeError, ValueError) as canonical_error:
            return None, f"message[{index}] {canonical_error}"
        normalized_messages.append(normalized)
        if normalized["role"] == "user":
            user_messages.append(normalized)

    if not user_messages:
        return None, "messages contain no user message"
    return (
        len(user_messages) - 1,
        _digest_canonical(user_messages),
        len(normalized_messages),
        _digest_canonical(normalized_messages),
    ), None


def _messages_prefix_digest(raw: Any, length: int) -> tuple[bytes | None, str | None]:
    messages, error = _strict_json_value(raw)
    if error:
        return None, error
    if not isinstance(messages, Sequence) or isinstance(
        messages, (str, bytes, bytearray)
    ):
        return None, f"expected a message list, found {type(messages).__name__}"
    if len(messages) < length:
        return None, f"history has {len(messages)} messages, expected at least {length}"

    normalized = []
    accepted_roles = {"system", "user", "assistant", "tool", "function"}
    for index, message in enumerate(messages[:length]):
        if not isinstance(message, Mapping):
            return None, f"message[{index}] is not an object"
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None, f"message[{index}] has invalid role {role!r}"
        if role not in accepted_roles:
            return None, (
                f"message[{index}] has unknown or noncanonical role {role!r}"
            )
        item = dict(message)
        if "content" not in item:
            return None, f"message[{index}] has no content field"
        normalized.append(item)
    try:
        return _digest_canonical(normalized), None
    except (TypeError, ValueError) as canonical_error:
        return None, str(canonical_error)


def _metadata_value(
    metadata: Mapping[str, Any],
    scores: Mapping[str, Any],
    key: str,
    *,
    reward: bool,
) -> Any:
    sources = (scores, metadata) if reward else (metadata, scores)
    for source in sources:
        if key in source:
            return source[key]
    return MISSING


def _number(value: Any) -> tuple[Decimal | None, str | None]:
    if value is MISSING:
        return None, "missing"
    if value is None:
        return None, "null"
    if isinstance(value, bool):
        return None, "boolean is not numeric"
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None, f"not numeric: {value!r}"
    if not number.is_finite():
        return None, f"not finite: {value!r}"
    return number, None


def _canonical_bool(value: Any) -> tuple[bool | None, str | None]:
    if value is MISSING:
        return None, "missing"
    if type(value) is bool:
        return value, None
    if value == "true":
        return True, None
    if isinstance(value, str):
        return None, "stored as a string, not a boolean"
    if value is None:
        return None, "null"
    return None, f"not boolean: {value!r}"


def _shown(value: Any) -> str:
    return "<missing>" if value is MISSING else repr(value)


def _audit_text(value: Any) -> str | None:
    """Preserve common identifiers as text without failing audit creation."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr(value)


def _is_structure(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )


def _strict_python_literal(text: str) -> Any:
    """Parse a Python literal while rejecting duplicate dictionary keys."""

    expression = ast.parse(text, mode="eval")
    for node in ast.walk(expression):
        if not isinstance(node, ast.Dict):
            continue
        keys: set[Any] = set()
        for raw_key in node.keys:
            if raw_key is None:
                raise ValueError("dictionary unpacking is not a literal value")
            key = ast.literal_eval(raw_key)
            try:
                if key in keys:
                    raise ValueError(f"duplicate Python dictionary key {key!r}")
                keys.add(key)
            except TypeError as error:
                raise ValueError(f"unhashable dictionary key {key!r}") from error
    return ast.literal_eval(expression)


def _parse_structure(text: str, decode_depth: int = 0) -> Any | None:
    """Parse JSON/Python structures, code fences, and tool-call wrappers."""

    if decode_depth >= MAX_STRING_DECODE_DEPTH:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced = CODE_FENCE_PATTERN.fullmatch(stripped)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        current: Any = candidate
        for _ in range(MAX_STRING_DECODE_DEPTH):
            if not isinstance(current, str):
                break
            try:
                parsed = json.loads(
                    current,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                break
            if _is_structure(parsed):
                return parsed
            if isinstance(parsed, str) and parsed != current:
                current = parsed.strip()
                continue
            break

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            try:
                parsed = _strict_python_literal(candidate)
            except (
                SyntaxError,
                ValueError,
                TypeError,
                MemoryError,
                RecursionError,
            ):
                parsed = None
        if _is_structure(parsed):
            return parsed

    tagged = TOOL_CALL_TAG_PATTERN.findall(stripped)
    if not tagged:
        return None
    parsed_tags = [
        _parse_structure(value, decode_depth + 1)
        for value in tagged
    ]
    return parsed_tags if all(value is not None for value in parsed_tags) else None


def _scan_response(raw: Any) -> tuple[bool, int, bool]:
    """Return ``(inspectable, maximum_leaf_word_count, depth_exceeded)``."""

    if _is_structure(raw):
        root = raw
    elif isinstance(raw, str):
        root = _parse_structure(raw)
    else:
        root = None
    if root is None:
        return False, 0, False

    def walk(
        value: Any,
        depth: int,
        inside_dictionary_value: bool,
        seen: set[int],
    ) -> tuple[int, bool]:
        if depth > MAX_NESTING_DEPTH:
            return 0, True
        if isinstance(value, str):
            nested = _parse_structure(value)
            if nested is not None:
                nested_maximum, exceeded = walk(
                    nested,
                    depth + 1,
                    inside_dictionary_value,
                    seen,
                )
                outside_tags = TOOL_CALL_TAG_PATTERN.sub(" ", value)
                outside_maximum = (
                    len(WORD_PATTERN.findall(outside_tags))
                    if outside_tags != value and inside_dictionary_value
                    else 0
                )
                return max(nested_maximum, outside_maximum), exceeded
            if not inside_dictionary_value:
                return 0, False
            return len(WORD_PATTERN.findall(value)), False

        if isinstance(value, Mapping):
            children = value.values()
            child_is_dictionary_value = True
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            children = value
            child_is_dictionary_value = inside_dictionary_value
        else:
            return 0, False

        object_id = id(value)
        if object_id in seen:
            return 0, False
        seen.add(object_id)
        try:
            maximum = 0
            exceeded = False
            for child in children:
                child_maximum, child_exceeded = walk(
                    child,
                    depth + 1,
                    child_is_dictionary_value,
                    seen,
                )
                maximum = max(maximum, child_maximum)
                exceeded = exceeded or child_exceeded
            return maximum, exceeded
        finally:
            seen.remove(object_id)

    maximum, exceeded = walk(root, 0, False, set())
    return True, maximum, exceeded


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------


def _warn(
    stats: Stats,
    dataset_id: str,
    reason: str,
    repo_path: str,
    row_index: int,
    row: Mapping[str, Any],
    detail: str,
) -> None:
    stats.warning_reasons[reason] += 1
    logging.warning(
        "DOUBLE-CHECK REQUIRED | reason=%s | dataset=%s | file=%s | row=%d | conv_id=%r | "
        "conv_tracking_id=%r | turn_id=%r | %s | This warning does not itself "
        "cause a drop; please double-check the annotation.",
        reason,
        dataset_id,
        repo_path,
        row_index,
        row.get("conv_id"),
        row.get("conv_tracking_id"),
        row.get("turn_id"),
        detail,
    )


def _inspect_row(
    row: Mapping[str, Any],
    *,
    dataset_id: str,
    file_kind: str,
    repo_path: str,
    row_index: int,
    stats: Stats,
    strict_equivalence_state: int,
    precomputed_reasons: Sequence[str] = (),
) -> Decision:
    """Apply per-row rules and return their combined keep or drop decision.

    Cross-row results and verified call equivalence arrive as precomputed
    inputs. Missing or ambiguous evidence warns and does not authorize the
    affected per-row rule.
    """

    if strict_equivalence_state not in (
        STRICT_EQ_UNVERIFIABLE,
        STRICT_EQ_UNEQUAL,
        STRICT_EQ_EQUAL,
    ):
        raise ValueError(
            f"Invalid strict equivalence state: {strict_equivalence_state!r}"
        )
    stats.input_rows += 1
    reasons = set(precomputed_reasons)
    metadata, metadata_error = _parse_mapping(row.get("metadata"))
    rejected_greater = False

    if metadata is None:
        turn_id, _ = _strict_turn_id(row.get("turn_id", MISSING))
        detail = f"metadata could not be inspected: {metadata_error}"
        if turn_id is not None and file_kind.endswith("_multicall"):
            detail += (
                "; filename is multicall, but "
                "metadata.toolcall_mode cannot be verified"
            )
        _warn(
            stats,
            dataset_id,
            "metadata_unavailable",
            repo_path,
            row_index,
            row,
            detail,
        )
    else:
        raw_scores = metadata.get("annotation_scores", MISSING)
        if raw_scores is MISSING:
            scores = {}
        else:
            scores, scores_error = _parse_mapping(raw_scores)
            if scores is None:
                _warn(
                    stats,
                    dataset_id,
                    "annotation_scores_unavailable",
                    repo_path,
                    row_index,
                    row,
                    f"annotation_scores could not be inspected: {scores_error}",
                )
                scores = {}

        expected_subset, expected_mode = FILE_KIND_TO_METADATA[file_kind]
        raw_subset = _metadata_value(
            metadata, scores, "subset_name", reward=False
        )
        raw_mode = _metadata_value(
            metadata, scores, "toolcall_mode", reward=False
        )
        actual_subset = raw_subset if isinstance(raw_subset, str) else None
        actual_mode = raw_mode if isinstance(raw_mode, str) else None
        filename_multicall = file_kind.endswith("_multicall")
        metadata_multicall = actual_mode == "multi"
        raw_turn_id = row.get("turn_id", MISSING)
        turn_id, turn_error = _strict_turn_id(raw_turn_id)
        turn_id_mode = _turn_id_mode(raw_turn_id)
        mode_warning_emitted = False

        if turn_error:
            _warn(
                stats,
                dataset_id,
                "turn_id_unavailable",
                repo_path,
                row_index,
                row,
                f"turn_id cannot be interpreted: {turn_error}",
            )
        elif turn_id_mode is not None and turn_id_mode != expected_mode:
            _warn(
                stats,
                dataset_id,
                "turn_id_toolcall_mode_mismatch",
                repo_path,
                row_index,
                row,
                f"turn_id mode {turn_id_mode!r} does not match expected "
                f"toolcall mode {expected_mode!r}",
            )
        elif turn_id == 0:
            if (
                filename_multicall
                and metadata_multicall
                and actual_subset == expected_subset
            ):
                reasons.add(TURN_ZERO_MULTICALL)
            elif filename_multicall != metadata_multicall:
                _warn(
                    stats,
                    dataset_id,
                    "turn_n_multicall_mismatch",
                    repo_path,
                    row_index,
                    row,
                    (
                        f"filename_multicall={filename_multicall}, "
                        f"metadata.toolcall_mode={_shown(raw_mode)} "
                        f"(metadata_multicall={metadata_multicall})"
                    ),
                )
                mode_warning_emitted = True
        # elif file_kind == "singleturn_multicall":
        #     _warn(
        #         stats,
        #         dataset_id,
        #         "singleturn_multicall_n_turn_id",
        #         repo_path,
        #         row_index,
        #         row,
        #         (
        #             f"singleturn_multicall expects turn_id=0, found "
        #             f"{_shown(row.get('turn_id', MISSING))}"
        #         ),
        #     )

        if actual_mode != expected_mode and not mode_warning_emitted:
            _warn(
                stats,
                dataset_id,
                "filename_toolcall_mode_mismatch",
                repo_path,
                row_index,
                row,
                (
                    f"filename kind {file_kind!r} expects toolcall_mode="
                    f"{expected_mode!r}, found {_shown(raw_mode)}"
                ),
            )
        if actual_subset != expected_subset:
            _warn(
                stats,
                dataset_id,
                "filename_subset_name_mismatch",
                repo_path,
                row_index,
                row,
                (
                    f"filename kind {file_kind!r} expects subset_name="
                    f"{expected_subset!r}, found {_shown(raw_subset)}"
                ),
            )

        raw_diff = _metadata_value(
            metadata, scores, "toolrl_reward_diff", reward=True
        )
        raw_equivalent = _metadata_value(
            metadata, scores, "chosen_rejected_equivalent", reward=False
        )
        reward_diff, diff_error = _number(raw_diff)
        equivalent, equivalent_error = _canonical_bool(raw_equivalent)
        diff_is_zero = diff_error is None and reward_diff == 0
        equivalent_is_true = equivalent is True and equivalent_error is None

        if (
            diff_is_zero
            and equivalent_is_true
            and strict_equivalence_state == STRICT_EQ_EQUAL
        ):
            reasons.add(ZERO_REWARD_EQUIVALENT)
        elif diff_is_zero and equivalent_is_true:
            verified_unequal = strict_equivalence_state == STRICT_EQ_UNEQUAL
            _warn(
                stats,
                dataset_id,
                (
                    "zero_reward_strict_equivalence_mismatch"
                    if verified_unequal
                    else "zero_reward_strict_equivalence_unavailable"
                ),
                repo_path,
                row_index,
                row,
                "stored toolrl_reward_diff is zero and stored equivalence is "
                "true, but independent strict call-multiset equality is "
                + ("false" if verified_unequal else "unverifiable"),
            )
        elif diff_is_zero != equivalent_is_true:
            _warn(
                stats,
                dataset_id,
                "zero_reward_equivalence_mismatch",
                repo_path,
                row_index,
                row,
                (
                    f"toolrl_reward_diff={_shown(raw_diff)} "
                    f"(is_zero={diff_is_zero}), chosen_rejected_equivalent="
                    f"{_shown(raw_equivalent)} (is_true={equivalent_is_true})"
                ),
            )
        elif diff_error or equivalent_error:
            problems = []
            if diff_error:
                problems.append(f"toolrl_reward_diff {diff_error}")
            if equivalent_error:
                problems.append(
                    f"chosen_rejected_equivalent {equivalent_error}"
                )
            _warn(
                stats,
                dataset_id,
                "reward_equivalence_values_unavailable",
                repo_path,
                row_index,
                row,
                "; ".join(problems),
            )

        raw_chosen = _metadata_value(
            metadata, scores, "chosen_toolrl_reward", reward=True
        )
        raw_rejected = _metadata_value(
            metadata, scores, "rejected_toolrl_reward", reward=True
        )
        chosen_reward, chosen_error = _number(raw_chosen)
        rejected_reward, rejected_error = _number(raw_rejected)
        if chosen_error or rejected_error:
            problems = []
            if chosen_error:
                problems.append(f"chosen_toolrl_reward {chosen_error}")
            if rejected_error:
                problems.append(f"rejected_toolrl_reward {rejected_error}")
            _warn(
                stats,
                dataset_id,
                "inverse_reward_values_unavailable",
                repo_path,
                row_index,
                row,
                "; ".join(problems),
            )
        else:
            rejected_greater = rejected_reward > chosen_reward
            if rejected_greater:
                _warn(
                    stats,
                    dataset_id,
                    "rejected_greater_than_chosen",
                    repo_path,
                    row_index,
                    row,
                    f"rejected_toolrl_reward={_shown(raw_rejected)} is greater "
                    f"than chosen_toolrl_reward={_shown(raw_chosen)}",
                )

    for side, reason in (
        ("chosen", CHOSEN_LONG_VALUE),
        ("rejected", REJECTED_LONG_VALUE),
    ):
        inspectable, max_words, depth_exceeded = _scan_response(
            row.get(f"{side}_response")
        )
        if not inspectable:
            _warn(
                stats,
                dataset_id,
                f"{side}_response_uninspectable",
                repo_path,
                row_index,
                row,
                f"{side}_response could not be decoded as a dictionary/list",
            )
        elif depth_exceeded:
            _warn(
                stats,
                dataset_id,
                f"{side}_response_nesting_limit",
                repo_path,
                row_index,
                row,
                f"{side}_response exceeded nesting limit {MAX_NESTING_DEPTH}",
            )

        if max_words <= LONG_VALUE_WORD_THRESHOLD:
            continue
        if side == "chosen":
            reasons.add(reason)
        else:
            stats.rejected_long_observed += 1
            if DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE:
                reasons.add(reason)

    reasons.intersection_update(ENABLED_DROP_REASONS)

    if rejected_greater:
        stats.rejected_greater += 1
        if not reasons:
            stats.rejected_greater_retained += 1
    for reason in reasons:
        stats.drop_reasons[reason] += 1
    if reasons:
        stats.unique_would_drop += 1

    return Decision(frozenset(reasons), metadata, rejected_greater)


def _flag_metadata(raw: Any, metadata: Mapping[str, Any]) -> Any:
    if metadata.get(REJECTED_GREATER_THAN_CHOSEN_FLAG) is True:
        return raw
    updated = dict(metadata)
    updated[REJECTED_GREATER_THAN_CHOSEN_FLAG] = True
    if isinstance(raw, str):
        return json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
    if isinstance(raw, Mapping):
        return updated
    raise TypeError(f"Unsupported metadata type: {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Validation and I/O
# ---------------------------------------------------------------------------


def _validate_configuration() -> None:
    required_external_configuration = {
        "TOUCAN_CLEANING_SOURCE_DATASET_ID": SOURCE_DATASET_ID,
        "TOUCAN_CLEANING_SOURCE_REVISION": SOURCE_REVISION,
        "TOUCAN_CLEANING_DESTINATION_DATASET_ID": DESTINATION_DATASET_ID,
    }
    missing = [
        name
        for name, value in required_external_configuration.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ValueError(
            "Required cleaning configuration is unset. Set these environment "
            f"variables manually before running: {', '.join(missing)}"
        )

    temporary_parent = TEMPORARY_WORK_PARENT.resolve()
    temporary_root = TEMPORARY_WORK_ROOT.resolve()
    if (
        temporary_root.parent != temporary_parent
        or not temporary_root.name.startswith("toucan-cleaning-")
    ):
        raise ValueError(
            f"Unsafe dedicated temporary work tree: {TEMPORARY_WORK_ROOT}"
        )
    source_cache = SOURCE_CACHE_DIR.resolve()
    if REUSING_SOURCE_CACHE:
        reused_root = source_cache.parent
        if (
            SOURCE_CACHE_DIR.is_symlink()
            or source_cache.name != "huggingface-cache"
            or reused_root.parent != temporary_parent
            or not reused_root.name.startswith("toucan-cleaning-")
        ):
            raise ValueError(
                f"Unsafe reusable source cache: {SOURCE_CACHE_DIR}"
            )
    elif temporary_root not in source_cache.parents:
        raise ValueError(
            f"Source cache is outside the temporary work tree: {SOURCE_CACHE_DIR}"
        )

    bools = {
        "DROP_ROWS": DROP_ROWS,
        "PUSH_TO_HUB": PUSH_TO_HUB,
        "DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE": (
            DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE
        ),
        "DESTINATION_DATASET_PRIVATE": DESTINATION_DATASET_PRIVATE,
        "MANUAL_GATE_DESTINATION_DATASETS": MANUAL_GATE_DESTINATION_DATASETS,
        "PRESERVE_SINGLE_SOURCE_LAYOUT": PRESERVE_SINGLE_SOURCE_LAYOUT,
        "WRITE_DROP_AUDIT": WRITE_DROP_AUDIT,
        "DELETE_TEMPORARY_WORK_DATA": DELETE_TEMPORARY_WORK_DATA,
        "RESUME_EXISTING_DESTINATION": RESUME_EXISTING_DESTINATION,
    }
    for name, value in bools.items():
        if type(value) is not bool:
            raise TypeError(f"{name} must be True or False, not {value!r}")
    for name, value, minimum in (
        ("LONG_VALUE_WORD_THRESHOLD", LONG_VALUE_WORD_THRESHOLD, 0),
        ("PARQUET_BATCH_SIZE", PARQUET_BATCH_SIZE, 1),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer, not {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if DESTINATION_DATASET_PRIVATE and MANUAL_GATE_DESTINATION_DATASETS:
        raise ValueError(
            "Manual gating for collaborator access requires a non-private "
            "personal repository"
        )
    if PRESERVE_SINGLE_SOURCE_LAYOUT and len(ALL_DATASET_JOBS) != 1:
        raise ValueError(
            "PRESERVE_SINGLE_SOURCE_LAYOUT requires exactly one source job"
        )
    if DELETE_TEMPORARY_WORK_DATA and not PUSH_TO_HUB:
        raise ValueError(
            "DELETE_TEMPORARY_WORK_DATA=True requires PUSH_TO_HUB=True so the "
            "only completed copy is not deleted"
        )
    if len(DROP_REASONS) != len(ENABLED_DROP_REASONS):
        raise ValueError("DROP_REASONS contains duplicates")
    unknown_drop_reasons = ENABLED_DROP_REASONS - KNOWN_DROP_REASONS
    if unknown_drop_reasons:
        raise ValueError(
            f"DROP_REASONS contains unknown values: {sorted(unknown_drop_reasons)!r}"
        )
    _safe_path(Path("."), DROP_AUDIT_REPO_PATH)

    all_subset_names = [job.subset_name for job in ALL_DATASET_JOBS]
    if len(all_subset_names) != len(set(all_subset_names)):
        raise ValueError("ALL_DATASET_JOBS contains repeated subset names")
    if len(REQUESTED_SUBSETS) != len(set(REQUESTED_SUBSETS)):
        raise ValueError("TOUCAN_CLEANING_SUBSETS contains duplicates")
    unknown_requested_subsets = set(REQUESTED_SUBSETS) - set(all_subset_names)
    if unknown_requested_subsets:
        raise ValueError(
            "TOUCAN_CLEANING_SUBSETS contains unknown subsets: "
            f"{sorted(unknown_requested_subsets)!r}"
        )
    if not DATASET_JOBS:
        raise ValueError("No dataset jobs were selected")
    if RESUME_EXISTING_DESTINATION and not PUSH_TO_HUB:
        raise ValueError(
            "TOUCAN_CLEANING_RESUME_EXISTING=1 requires PUSH_TO_HUB=True"
        )
    if REUSING_SOURCE_CACHE and not SOURCE_CACHE_DIR.is_dir():
        raise ValueError(
            "TOUCAN_CLEANING_SOURCE_CACHE_DIR is not an existing directory: "
            f"{SOURCE_CACHE_DIR}"
        )
    if len(DATASET_JOBS) != len(ALL_DATASET_JOBS) and not (
        RESUME_EXISTING_DESTINATION
    ):
        raise ValueError(
            "A partial subset selection requires "
            "TOUCAN_CLEANING_RESUME_EXISTING=1"
        )
    if not isinstance(DESTINATION_DATASET_ID, str) or not (
        DESTINATION_DATASET_ID.strip()
    ):
        raise ValueError(
            "DESTINATION_DATASET_ID must be a nonempty Hugging Face dataset "
            "ID in 'owner/dataset-name' form"
        )
    source_ids: set[str] = set()
    subset_names: set[str] = set()
    output_dirs: set[Path] = set()
    for index, job in enumerate(DATASET_JOBS):
        if not isinstance(job, DatasetJob):
            raise TypeError(f"DATASET_JOBS[{index}] is not a DatasetJob")
        if not job.source_dataset_id.strip():
            raise ValueError(f"DATASET_JOBS[{index}] has no source dataset ID")
        if not isinstance(job.subset_name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", job.subset_name
        ):
            raise ValueError(
                f"DATASET_JOBS[{index}] has invalid subset name "
                f"{job.subset_name!r}"
            )

        output_dir = job.output_dataset_dir.resolve()
        if output_dir == PROJECT_ROOT.resolve():
            raise ValueError("A dataset output directory cannot be PROJECT_ROOT")
        if job.source_dataset_id in source_ids:
            raise ValueError(
                f"Repeated source dataset ID: {job.source_dataset_id!r}"
            )
        if job.subset_name in subset_names:
            raise ValueError(
                f"Repeated destination subset name: {job.subset_name!r}"
            )
        if output_dir in output_dirs:
            raise ValueError(f"Repeated output directory: {output_dir}")
        if DELETE_TEMPORARY_WORK_DATA and TEMPORARY_WORK_ROOT.resolve() not in (
            output_dir,
            *output_dir.parents,
        ):
            raise ValueError(
                "Temporary cleanup is enabled but output is outside the "
                f"dedicated work tree: {output_dir}"
            )
        source_ids.add(job.source_dataset_id)
        subset_names.add(job.subset_name)
        output_dirs.add(output_dir)

    all_source_ids = {job.source_dataset_id for job in ALL_DATASET_JOBS}
    if set(SOURCE_REVISIONS) != all_source_ids:
        raise ValueError(
            "SOURCE_REVISIONS must contain exactly all configured source "
            f"dataset IDs: expected={sorted(all_source_ids)!r}, "
            f"found={sorted(SOURCE_REVISIONS)!r}"
        )
    for dataset_id, revision in SOURCE_REVISIONS.items():
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", revision
        ):
            raise ValueError(
                f"SOURCE_REVISIONS[{dataset_id!r}] must be a 40-character "
                "commit SHA"
            )

    if DESTINATION_DATASET_ID in all_source_ids:
        raise ValueError(
            "Refusing a destination ID that overwrites a configured source: "
            f"{DESTINATION_DATASET_ID!r}"
        )

    run_paths: list[tuple[str, Path]] = []
    for job in DATASET_JOBS:
        output_dir = job.output_dataset_dir.resolve()
        incomplete = output_dir.with_name(
            output_dir.name + INCOMPLETE_OUTPUT_SUFFIX
        )
        run_paths.extend(
            (("output", output_dir), ("incomplete output", incomplete))
        )
    for index, (left_label, left) in enumerate(run_paths):
        for right_label, right in run_paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "Dataset job paths overlap: "
                    f"{left_label} {left} and {right_label} {right}"
                )

    if DROP_ROWS or PUSH_TO_HUB:
        for job in DATASET_JOBS:
            output_dir = job.output_dataset_dir
            incomplete = output_dir.with_name(
                output_dir.name + INCOMPLETE_OUTPUT_SUFFIX
            )
            if output_dir.exists():
                raise FileExistsError(f"Output already exists: {output_dir}")
            if incomplete.exists():
                raise FileExistsError(
                    f"Incomplete prior run exists: {incomplete}; inspect and "
                    "remove it"
                )


def _safe_path(root: Path, repo_path: str) -> Path:
    relative = PurePosixPath(repo_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe repository path: {repo_path!r}")
    return root.joinpath(*relative.parts)


def _parquet_files(repo_files: Sequence[str]) -> list[str]:
    files = sorted(path for path in repo_files if path.endswith(".parquet"))
    if not files:
        raise RuntimeError("The source repository contains no Parquet files")

    for repo_path in files:
        parts = PurePosixPath(repo_path).parts
        if len(parts) < 2:
            raise RuntimeError(f"{repo_path!r} is not inside a model folder")
        if parts[0] not in EXPECTED_MODEL_FOLDERS:
            raise RuntimeError(f"Unexpected model folder in {repo_path!r}")
        _file_kind(repo_path)
    return files


def _validate_schema(repo_path: str, schema: pa.Schema) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in schema.names]
    if missing:
        raise RuntimeError(f"{repo_path} is missing columns: {missing}")
    metadata_type = schema.field("metadata").type
    if (DROP_ROWS or PUSH_TO_HUB) and not (
        pa.types.is_string(metadata_type) or pa.types.is_large_string(metadata_type)
    ):
        raise RuntimeError(
            f"{repo_path} stores metadata as {metadata_type}; writing is only "
            "supported for the known string-metadata schema. Dry-run is safe."
        )


def _download(dataset_id: str, repo_path: str, revision: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
            token=HUGGING_FACE_TOKEN,
            cache_dir=SOURCE_CACHE_DIR,
        )
    )


def _cleanup_source_download_cache() -> None:
    """Remove only this process's source/Xet caches, never staged output."""

    temporary_root = TEMPORARY_WORK_ROOT.resolve()
    for cache_dir in (SOURCE_CACHE_DIR, XET_CACHE_DIR):
        resolved = cache_dir.resolve()
        is_current_cache = temporary_root in resolved.parents
        is_safe_reused_source_cache = (
            cache_dir == SOURCE_CACHE_DIR
            and REUSING_SOURCE_CACHE
            and not cache_dir.is_symlink()
            and resolved.name == "huggingface-cache"
            and resolved.parent.parent == TEMPORARY_WORK_PARENT.resolve()
            and resolved.parent.name.startswith("toucan-cleaning-")
        )
        if not (is_current_cache or is_safe_reused_source_cache):
            raise RuntimeError(f"Refusing unsafe cache cleanup target: {cache_dir}")
        if resolved.exists():
            shutil.rmtree(resolved)
            logging.info("Removed temporary download cache: %s", resolved)


# ---------------------------------------------------------------------------
# Cross-row planning
# ---------------------------------------------------------------------------

CrossGroupKey = tuple[str, int, bytes]
PrefixExpectation = tuple[CrossGroupKey, RowLocator, int, bytes]


def _partition_parquet_files(
    parquet_files: Sequence[str],
) -> dict[tuple[str, str], list[str]]:
    partitions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for repo_path in parquet_files:
        folder = PurePosixPath(repo_path).parts[0]
        partitions[(folder, _file_kind(repo_path))].append(repo_path)
    return {
        key: sorted(paths)
        for key, paths in sorted(partitions.items())
    }


def _read_cross_rows(
    dataset_id: str,
    repo_paths: Sequence[str],
    revision: str,
    plan: CrossRulePlan,
) -> tuple[dict[str, list[CrossRow]], set[str]]:
    rows_by_conversation: dict[str, list[CrossRow]] = defaultdict(list)
    invalid_conversations: set[str] = set()

    for repo_path in repo_paths:
        source = pq.ParquetFile(_download(dataset_id, repo_path, revision))
        _validate_schema(repo_path, source.schema_arrow)
        if repo_path in plan.drop_flags_by_file:
            raise RuntimeError(f"Cross-rule plan repeats file {repo_path!r}")
        plan.drop_flags_by_file[repo_path] = bytearray(source.metadata.num_rows)
        plan.singlecall_multiple_chosen_flags_by_file[repo_path] = bytearray(
            source.metadata.num_rows
        )
        plan.strict_equivalence_states_by_file[repo_path] = bytearray(
            source.metadata.num_rows
        )
        file_kind = _file_kind(repo_path)
        expected_turn_mode = FILE_KIND_TO_METADATA[file_kind][1]
        singlecall_file = file_kind.endswith("_singlecall")

        row_index = 0
        for batch in source.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=CROSS_RULE_COLUMNS,
        ):
            for batch_index, row in enumerate(batch.to_pylist()):
                locator = (repo_path, row_index + batch_index)
                metadata, metadata_error = _strict_mapping_value(
                    row.get("metadata")
                )
                chosen_calls: CallCounter | None = None
                rejected_calls: CallCounter | None = None
                if metadata is None:
                    plan.warn(
                        locator,
                        "interaction_metadata_unavailable",
                        metadata_error or "metadata is unavailable",
                    )
                else:
                    rejected_has_nonempty_content = False
                    for side in ("chosen", "rejected"):
                        (
                            calls,
                            warning_reason,
                            detail,
                            has_nonempty_content,
                        ) = _verified_call_counter(
                            row,
                            metadata,
                            side,
                        )
                        if warning_reason:
                            plan.warn(locator, warning_reason, detail or "")
                        if side == "chosen":
                            chosen_calls = calls
                        else:
                            rejected_calls = calls
                            rejected_has_nonempty_content = has_nonempty_content

                    ambiguity = _rejected_content_ambiguity(
                        metadata,
                        rejected_has_nonempty_content,
                    )
                    if ambiguity:
                        plan.warn(
                            locator,
                            "interaction_rejected_content_ambiguous",
                            ambiguity,
                        )
                        rejected_calls = None

                    chosen_call_count = (
                        sum(chosen_calls.values())
                        if chosen_calls is not None
                        else 0
                    )
                    if singlecall_file and chosen_call_count > 1:
                        plan.singlecall_multiple_chosen_flags_by_file[repo_path][
                            row_index + batch_index
                        ] = 1
                        plan.warn(
                            locator,
                            "singlecall_file_contains_multiple_chosen_calls",
                            "verified chosen response contains "
                            f"{chosen_call_count} calls in a *_singlecall file",
                        )

                    if (
                        chosen_calls is not None
                        and rejected_calls is not None
                    ):
                        calls_equivalent = chosen_calls == rejected_calls
                        plan.strict_equivalence_states_by_file[repo_path][
                            row_index + batch_index
                        ] = (
                            STRICT_EQ_EQUAL
                            if calls_equivalent
                            else STRICT_EQ_UNEQUAL
                        )
                        raw_scores = metadata.get("annotation_scores", MISSING)
                        if raw_scores is MISSING:
                            scores: Mapping[str, Any] = {}
                        else:
                            parsed_scores, _ = _strict_mapping_value(raw_scores)
                            scores = parsed_scores or {}
                        raw_equivalent = _metadata_value(
                            metadata,
                            scores,
                            "chosen_rejected_equivalent",
                            reward=False,
                        )
                        equivalent, equivalent_error = _canonical_bool(
                            raw_equivalent
                        )
                        if (
                            equivalent_error is None
                            and equivalent != calls_equivalent
                        ):
                            plan.warn(
                                locator,
                                "interaction_equivalence_flag_disagrees_with_calls",
                                "stored chosen_rejected_equivalent="
                                f"{raw_equivalent!r}, but verified call-multiset "
                                f"equality is {calls_equivalent}",
                            )

                raw_conv_id = row.get("conv_id")
                if (
                    not isinstance(raw_conv_id, str)
                    or not raw_conv_id
                    or raw_conv_id != raw_conv_id.strip()
                ):
                    plan.warn(
                        locator,
                        "interaction_conv_id_unavailable",
                        f"conv_id must be a nonempty, unpadded string; found "
                        f"{raw_conv_id!r}",
                    )
                    continue
                conv_id = raw_conv_id

                raw_turn_id = row.get("turn_id", MISSING)
                turn_id, turn_error = _strict_turn_id(raw_turn_id)
                if turn_error:
                    plan.warn(
                        locator,
                        "interaction_turn_id_unavailable",
                        turn_error,
                    )
                    invalid_conversations.add(conv_id)
                    continue

                turn_id_mode = _turn_id_mode(raw_turn_id)
                if (
                    turn_id_mode is not None
                    and turn_id_mode != expected_turn_mode
                ):
                    plan.warn(
                        locator,
                        "interaction_turn_id_mode_mismatch",
                        f"turn_id mode {turn_id_mode!r} does not match file "
                        f"mode {expected_turn_mode!r}",
                    )
                    invalid_conversations.add(conv_id)
                    continue

                expected_tracking_id = f"{conv_id}_{raw_turn_id}"
                raw_tracking_id = row.get("conv_tracking_id")
                if raw_tracking_id != expected_tracking_id:
                    plan.warn(
                        locator,
                        "interaction_conv_tracking_id_mismatch",
                        f"expected {expected_tracking_id!r}, found "
                        f"{raw_tracking_id!r}",
                    )

                identity, identity_error = _messages_identity(row.get("messages"))
                if identity_error:
                    plan.warn(
                        locator,
                        "interaction_messages_unavailable",
                        identity_error,
                    )
                    invalid_conversations.add(conv_id)
                    continue
                (
                    user_ordinal,
                    user_prefix_digest,
                    message_count,
                    messages_digest,
                ) = identity

                rows_by_conversation[conv_id].append(
                    CrossRow(
                        locator=locator,
                        conv_id=conv_id,
                        turn_id=turn_id,
                        user_ordinal=user_ordinal,
                        user_prefix_digest=user_prefix_digest,
                        message_count=message_count,
                        messages_digest=messages_digest,
                        chosen_calls=chosen_calls,
                        rejected_calls=rejected_calls,
                    )
                )
            row_index += batch.num_rows

        if row_index != source.metadata.num_rows:
            raise RuntimeError(
                f"Cross-rule read-count mismatch for {repo_path}: {row_index} "
                f"!= {source.metadata.num_rows}"
            )

    return rows_by_conversation, invalid_conversations


def _group_cross_rows(
    rows_by_conversation: Mapping[str, Sequence[CrossRow]],
    invalid_conversations: set[str],
    plan: CrossRulePlan,
) -> tuple[
    dict[CrossGroupKey, list[CrossRow]],
    dict[RowLocator, PrefixExpectation],
    set[CrossGroupKey],
]:
    groups: dict[CrossGroupKey, list[CrossRow]] = defaultdict(list)
    expectations: dict[RowLocator, PrefixExpectation] = {}
    invalid_groups: set[CrossGroupKey] = set()

    for conv_id, conversation_rows in rows_by_conversation.items():
        if conv_id in invalid_conversations:
            for row in conversation_rows:
                plan.warn(
                    row.locator,
                    "interaction_conversation_invalid",
                    "another row with this conv_id has invalid grouping fields",
                )
            continue

        ordered = sorted(
            conversation_rows,
            key=lambda row: (row.turn_id, row.locator),
        )
        previous_ordinal = ordered[0].user_ordinal
        for row in ordered[1:]:
            if row.user_ordinal < previous_ordinal:
                plan.warn(
                    row.locator,
                    "interaction_user_order_mismatch",
                    f"latest-user ordinal decreased from {previous_ordinal} "
                    f"to {row.user_ordinal} in numeric turn order",
                )
            elif row.user_ordinal > previous_ordinal + 1:
                plan.warn(
                    row.locator,
                    "interaction_user_ordinal_gap",
                    f"latest-user ordinal jumped from {previous_ordinal} "
                    f"to {row.user_ordinal}",
                )
            previous_ordinal = row.user_ordinal

        digests_by_ordinal: dict[int, set[bytes]] = defaultdict(set)
        for row in ordered:
            digests_by_ordinal[row.user_ordinal].add(row.user_prefix_digest)
        conflicting_ordinals = {
            ordinal
            for ordinal, digests in digests_by_ordinal.items()
            if len(digests) > 1
        }
        for row in ordered:
            group_key = (
                row.conv_id,
                row.user_ordinal,
                row.user_prefix_digest,
            )
            groups[group_key].append(row)
            if row.user_ordinal in conflicting_ordinals:
                invalid_groups.add(group_key)
                plan.warn(
                    row.locator,
                    "interaction_user_history_mismatch",
                    f"conv_id {conv_id!r} has conflicting user histories "
                    f"for latest-user ordinal {row.user_ordinal}",
                )

    for group_key, group_rows in groups.items():
        group_rows.sort(key=lambda row: row.turn_id)
        if group_key in invalid_groups:
            continue
        by_turn: dict[int, list[CrossRow]] = defaultdict(list)
        for row in group_rows:
            by_turn[row.turn_id].append(row)
        duplicates = {
            turn_id: rows
            for turn_id, rows in by_turn.items()
            if len(rows) > 1
        }
        if duplicates:
            invalid_groups.add(group_key)
            for turn_id, rows in duplicates.items():
                locators = ", ".join(
                    f"{path}:{index}"
                    for path, index in (row.locator for row in rows)
                )
                for row in rows:
                    plan.warn(
                        row.locator,
                        "interaction_duplicate_turn_id",
                        f"turn_id {turn_id} repeats at {locators}",
                    )
            continue
        for previous, current in zip(group_rows, group_rows[1:]):
            if current.turn_id > previous.turn_id + 1:
                plan.warn(
                    current.locator,
                    "interaction_turn_id_gap",
                    f"turn_id jumped from {previous.turn_id} to "
                    f"{current.turn_id}; the available rows are still analyzed",
                )
            if current.message_count <= previous.message_count:
                plan.warn(
                    current.locator,
                    "interaction_message_history_not_growing",
                    f"message count changed from {previous.message_count} to "
                    f"{current.message_count}",
                )
                invalid_groups.add(group_key)
                continue
            expectations[current.locator] = (
                group_key,
                previous.locator,
                previous.message_count,
                previous.messages_digest,
            )

    return dict(groups), expectations, invalid_groups


def _validate_message_prefixes(
    dataset_id: str,
    repo_paths: Sequence[str],
    revision: str,
    expectations: Mapping[RowLocator, PrefixExpectation],
    invalid_groups: set[CrossGroupKey],
    plan: CrossRulePlan,
) -> None:
    unchecked = set(expectations)
    paths_to_check = {repo_path for repo_path, _ in expectations}
    for repo_path in repo_paths:
        if repo_path not in paths_to_check:
            continue
        source = pq.ParquetFile(_download(dataset_id, repo_path, revision))
        row_index = 0
        for batch in source.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=("messages",),
        ):
            for batch_index, row in enumerate(batch.to_pylist()):
                locator = (repo_path, row_index + batch_index)
                expectation = expectations.get(locator)
                if expectation is None:
                    continue
                unchecked.discard(locator)
                group_key, previous_locator, length, expected_digest = expectation
                actual_digest, error = _messages_prefix_digest(
                    row.get("messages"),
                    length,
                )
                if error or actual_digest != expected_digest:
                    previous_path, previous_index = previous_locator
                    detail = (
                        error
                        or "history does not start with the complete preceding "
                        "row history"
                    )
                    plan.warn(
                        locator,
                        "interaction_message_history_not_cumulative",
                        f"previous row={previous_path}:{previous_index}; {detail}",
                    )
                    invalid_groups.add(group_key)
            row_index += batch.num_rows

    if unchecked:
        raise RuntimeError(
            "Internal error: message-prefix expectations were not checked for "
            f"{sorted(unchecked)!r}"
        )


def _mark_cross_rule_matches(
    groups: Mapping[CrossGroupKey, Sequence[CrossRow]],
    invalid_groups: set[CrossGroupKey],
    plan: CrossRulePlan,
) -> int:
    marked_rows = 0
    for group_key, group_rows in groups.items():
        if group_key in invalid_groups:
            continue

        suffix_calls: CallCounter = Counter()
        suffix_complete = True
        later_chosen_calls = 0
        matching_positions: set[int] = set()
        for position in range(len(group_rows) - 1, -1, -1):
            row = group_rows[position]
            if row.chosen_calls is None:
                suffix_complete = False
            else:
                suffix_calls.update(row.chosen_calls)

            if (
                later_chosen_calls > 0
                and suffix_complete
                and suffix_calls
                and row.rejected_calls is not None
                and row.rejected_calls == suffix_calls
            ):
                matching_positions.add(position)
            if row.chosen_calls is not None:
                later_chosen_calls += sum(row.chosen_calls.values())

        if not matching_positions:
            continue
        earliest_match = min(matching_positions)
        for position, row in enumerate(
            group_rows[earliest_match + 1 :],
            start=earliest_match + 1,
        ):
            if position in matching_positions:
                continue
            repo_path, row_index = row.locator
            flags = plan.drop_flags_by_file[repo_path]
            if not flags[row_index]:
                flags[row_index] = 1
                marked_rows += 1
    return marked_rows


def _discover_cross_rule_partition(
    dataset_id: str,
    repo_paths: Sequence[str],
    revision: str,
    plan: CrossRulePlan,
) -> int:
    rows_by_conversation, invalid_conversations = _read_cross_rows(
        dataset_id,
        repo_paths,
        revision,
        plan,
    )
    groups, expectations, invalid_groups = _group_cross_rows(
        rows_by_conversation,
        invalid_conversations,
        plan,
    )
    _validate_message_prefixes(
        dataset_id,
        repo_paths,
        revision,
        expectations,
        invalid_groups,
        plan,
    )
    return _mark_cross_rule_matches(groups, invalid_groups, plan)


def _build_cross_rule_plan(
    dataset_id: str,
    parquet_files: Sequence[str],
    revision: str,
) -> CrossRulePlan:
    """Precompute cross-row drops, strict equality, and warnings for all shards.

    Shards are combined only within the same model folder and full file kind.
    Ambiguous evidence is recorded as a warning and cannot authorize a match.
    """

    plan = CrossRulePlan()
    partitions = _partition_parquet_files(parquet_files)
    total_marked = 0
    for number, ((folder, kind), repo_paths) in enumerate(partitions.items(), 1):
        logging.info(
            "Discovering cross-row rule %d/%d: dataset=%s, model=%s, "
            "kind=%s, shards=%d",
            number,
            len(partitions),
            dataset_id,
            folder,
            kind,
            len(repo_paths),
        )
        total_marked += _discover_cross_rule_partition(
            dataset_id,
            repo_paths,
            revision,
            plan,
        )
    if set(plan.drop_flags_by_file) != set(parquet_files):
        raise RuntimeError("Cross-rule plan does not cover every Parquet file")
    if set(plan.singlecall_multiple_chosen_flags_by_file) != set(parquet_files):
        raise RuntimeError(
            "Singlecall chosen-call plan does not cover every Parquet file"
        )
    if set(plan.strict_equivalence_states_by_file) != set(parquet_files):
        raise RuntimeError(
            "Strict-equivalence plan does not cover every Parquet file"
        )
    logging.info(
        "Cross-row discovery marked %d rows in %s",
        total_marked,
        dataset_id,
    )
    return plan


# ---------------------------------------------------------------------------
# Parquet processing and dataset output
# ---------------------------------------------------------------------------


def _write_batch(
    writer: pq.ParquetWriter,
    batch: pa.RecordBatch,
    keep: list[int],
    metadata: list[Any],
) -> None:
    if not keep:
        return
    table = pa.Table.from_batches([batch])
    if len(keep) != batch.num_rows:
        table = table.take(pa.array(keep, type=pa.int64()))
    column_index = table.schema.get_field_index("metadata")
    field = table.schema.field(column_index)
    table = table.set_column(
        column_index,
        field,
        pa.array(metadata, type=field.type),
    )
    writer.write_table(table)


def _canonical_parquet_num_bytes(path: Path) -> int:
    """Return batch-independent logical Arrow bytes for one Parquet file."""

    # Combining removes row-group/chunk boundary overhead. Reading the complete
    # table also preserves validity buffers exactly as Hugging Face's Arrow
    # representation does; reading columns separately can omit all-valid
    # bitmaps and undercount the logical size.
    return pq.ParquetFile(path).read().combine_chunks().nbytes


def _process_file(
    dataset_id: str,
    repo_path: str,
    *,
    revision: str,
    output_root: Path | None,
    cross_rule_plan: CrossRulePlan,
    audit_writer: pq.ParquetWriter | None,
) -> Stats:
    """Classify one shard and optionally write selected rows in source order.

    The source schema is reused unchanged. Written inverse-reward rows can
    receive the documented metadata flag. If ``DROP_ROWS`` is false, every
    source row is selected even when it has one or more drop reasons.
    """

    source = pq.ParquetFile(_download(dataset_id, repo_path, revision))
    schema = source.schema_arrow
    _validate_schema(repo_path, schema)
    kind = _file_kind(repo_path)
    stats = Stats()
    cross_flags = cross_rule_plan.drop_flags_by_file.get(repo_path)
    if cross_flags is None:
        raise RuntimeError(f"Cross-rule plan has no flags for {repo_path!r}")
    chosen_multiple_flags = (
        cross_rule_plan.singlecall_multiple_chosen_flags_by_file.get(repo_path)
    )
    if chosen_multiple_flags is None:
        raise RuntimeError(
            f"Singlecall chosen-call plan has no flags for {repo_path!r}"
        )
    strict_equivalence_states = (
        cross_rule_plan.strict_equivalence_states_by_file.get(repo_path)
    )
    if strict_equivalence_states is None:
        raise RuntimeError(
            f"Strict-equivalence plan has no states for {repo_path!r}"
        )
    if len(cross_flags) != source.metadata.num_rows:
        raise RuntimeError(
            f"Cross-rule flag count mismatch for {repo_path}: "
            f"{len(cross_flags)} != {source.metadata.num_rows}"
        )
    if len(chosen_multiple_flags) != source.metadata.num_rows:
        raise RuntimeError(
            f"Singlecall chosen-call flag count mismatch for {repo_path}: "
            f"{len(chosen_multiple_flags)} != {source.metadata.num_rows}"
        )
    if len(strict_equivalence_states) != source.metadata.num_rows:
        raise RuntimeError(
            f"Strict-equivalence state count mismatch for {repo_path}: "
            f"{len(strict_equivalence_states)} != {source.metadata.num_rows}"
        )

    output_path = _safe_path(output_root, repo_path) if output_root else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        writer_context = pq.ParquetWriter(
            output_path,
            schema=schema,
            compression=PARQUET_COMPRESSION,
        )
    else:
        writer_context = nullcontext(None)

    row_index = 0
    written_rows = 0
    try:
        with writer_context as writer:
            for batch in source.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                keep: list[int] = []
                output_metadata: list[Any] = []
                audit_records: list[dict[str, Any]] = []
                for batch_index, row in enumerate(batch.to_pylist()):
                    absolute_index = row_index + batch_index
                    locator = (repo_path, absolute_index)
                    for warning in cross_rule_plan.warnings_by_row.get(
                        locator, ()
                    ):
                        _warn(
                            stats,
                            dataset_id,
                            warning.reason,
                            repo_path,
                            absolute_index,
                            row,
                            warning.detail,
                        )
                    precomputed_reasons = []
                    if chosen_multiple_flags[absolute_index]:
                        precomputed_reasons.append(
                            CHOSEN_MULTIPLE_CALLS_IN_SINGLECALL
                        )
                    if cross_flags[absolute_index]:
                        precomputed_reasons.append(
                            REJECTED_EQUALS_CHOSEN_SUFFIX
                        )
                    decision = _inspect_row(
                        row,
                        dataset_id=dataset_id,
                        file_kind=kind,
                        repo_path=repo_path,
                        row_index=absolute_index,
                        stats=stats,
                        strict_equivalence_state=(
                            strict_equivalence_states[absolute_index]
                        ),
                        precomputed_reasons=precomputed_reasons,
                    )
                    if audit_writer is not None and decision.reasons:
                        audit_records.append(
                            {
                                "source_dataset_id": dataset_id,
                                "source_revision": revision,
                                "source_file": repo_path,
                                "model_folder": PurePosixPath(repo_path).parts[0],
                                "file_kind": kind,
                                "row_index": absolute_index,
                                "conv_id": _audit_text(row.get("conv_id")),
                                "conv_tracking_id": _audit_text(
                                    row.get("conv_tracking_id")
                                ),
                                "turn_id": _audit_text(row.get("turn_id")),
                                "would_drop_reasons": sorted(decision.reasons),
                            }
                        )
                    if writer is None:
                        continue
                    if DROP_ROWS and decision.reasons:
                        continue
                    keep.append(batch_index)
                    raw_metadata = row.get("metadata")
                    if decision.rejected_greater and decision.metadata is not None:
                        raw_metadata = _flag_metadata(
                            raw_metadata,
                            decision.metadata,
                        )
                    output_metadata.append(raw_metadata)

                if audit_writer is not None and audit_records:
                    audit_writer.write_table(
                        pa.Table.from_pylist(
                            audit_records,
                            schema=DROP_AUDIT_SCHEMA,
                        )
                    )
                if writer is not None:
                    _write_batch(
                        writer,
                        batch,
                        keep,
                        output_metadata,
                    )
                    written_rows += len(keep)
                row_index += batch.num_rows

            if stats.input_rows != source.metadata.num_rows:
                raise RuntimeError(
                    f"Read-count mismatch for {repo_path}: {stats.input_rows} "
                    f"!= {source.metadata.num_rows}"
                )
            if writer is not None and written_rows != stats.output_rows:
                raise RuntimeError(
                    f"Write-count mismatch for {repo_path}: {written_rows} "
                    f"!= {stats.output_rows}"
                )
            if writer is not None and written_rows == 0:
                writer.write_table(pa.Table.from_batches([], schema=schema))
    except Exception:
        if output_path:
            output_path.unlink(missing_ok=True)
        raise

    if output_path:
        stats.retained_num_bytes = _canonical_parquet_num_bytes(output_path)

    return stats


def _card_paths(value: Any, context: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        paths = list(value)
        if paths and all(isinstance(path, str) for path in paths):
            return paths
    raise RuntimeError(f"{context} must be a path or nonempty list of paths")


def _card_config_patterns(
    card_data: Mapping[str, Any],
    expected_folders: Sequence[str],
) -> dict[tuple[str, str], list[str]]:
    raw_configs = card_data.get("configs")
    if not isinstance(raw_configs, Sequence) or isinstance(
        raw_configs, (str, bytes, bytearray)
    ):
        raise RuntimeError("README.md must contain a configs list")

    patterns: dict[tuple[str, str], list[str]] = {}
    config_names: set[str] = set()
    for config_index, raw_config in enumerate(raw_configs):
        if not isinstance(raw_config, Mapping):
            raise RuntimeError(f"README configs[{config_index}] is not a mapping")
        config_name = raw_config.get("config_name")
        if not isinstance(config_name, str) or not config_name:
            raise RuntimeError(f"README configs[{config_index}] has no config_name")
        if config_name in config_names:
            raise RuntimeError(f"README repeats config {config_name!r}")
        config_names.add(config_name)

        data_files = raw_config.get("data_files")
        if not isinstance(data_files, Sequence) or isinstance(
            data_files, (str, bytes, bytearray)
        ):
            raise RuntimeError(
                f"README config {config_name!r} must contain a data_files list"
            )
        for file_index, declaration in enumerate(data_files):
            if not isinstance(declaration, Mapping):
                raise RuntimeError(
                    f"README config {config_name!r} data_files[{file_index}] "
                    "is not a mapping"
                )
            split = declaration.get("split")
            if split not in FILE_KIND_TO_METADATA:
                raise RuntimeError(
                    f"README config {config_name!r} has unknown split {split!r}"
                )
            key = (config_name, split)
            patterns.setdefault(key, []).extend(
                _card_paths(
                    declaration.get("path"),
                    f"README config {config_name!r} split {split!r} path",
                )
            )

    expected_configs = set(expected_folders)
    if config_names != expected_configs:
        raise RuntimeError(
            "README configs disagree with expected model folders: "
            f"expected={sorted(expected_configs)}, found={sorted(config_names)}"
        )
    return patterns


def _regenerate_dataset_card(
    output_root: Path,
    file_stats: Mapping[str, Stats],
    expected_folders: Sequence[str],
) -> None:
    card_path = output_root / "README.md"
    if not card_path.is_file():
        raise RuntimeError("The source dataset has no README.md dataset card")

    text = card_path.read_text(encoding="utf-8")
    match = CARD_FRONT_MATTER_PATTERN.match(text)
    if match is None:
        raise RuntimeError("README.md has no YAML front matter")
    loaded = yaml.safe_load(match.group(1)) or {}
    if not isinstance(loaded, Mapping):
        raise RuntimeError("README.md YAML front matter is not a mapping")
    card_data = dict(loaded)

    config_patterns = _card_config_patterns(card_data, expected_folders)
    repo_paths = tuple(file_stats)
    actual_pairs: set[tuple[str, str]] = set()
    for repo_path in repo_paths:
        folder = PurePosixPath(repo_path).parts[0]
        expected_pair = (folder, _file_kind(repo_path))
        actual_pairs.add(expected_pair)
        matches = {
            pair
            for pair, patterns in config_patterns.items()
            if any(fnmatchcase(repo_path, pattern) for pattern in patterns)
        }
        if matches != {expected_pair}:
            raise RuntimeError(
                f"README configs map {repo_path!r} to {sorted(matches)}, "
                f"expected only {expected_pair}"
            )
    if set(config_patterns) != actual_pairs:
        raise RuntimeError(
            "README declared config/split pairs disagree with output files: "
            f"declared={sorted(config_patterns)}, actual={sorted(actual_pairs)}"
        )
    for pair, patterns in config_patterns.items():
        for pattern in patterns:
            if not any(fnmatchcase(repo_path, pattern) for repo_path in repo_paths):
                raise RuntimeError(
                    f"README config/split {pair} has a path pattern that matches no "
                    f"output file: {pattern!r}"
                )

    raw_infos = card_data.get("dataset_info")
    if isinstance(raw_infos, Sequence) and not isinstance(
        raw_infos, (str, bytes, bytearray)
    ):
        info_items = list(raw_infos)
    else:
        raise RuntimeError("README.md must contain a dataset_info list")

    info_by_config: dict[str, dict[str, Any]] = {}
    info_order: list[str] = []
    for info_index, raw_info in enumerate(info_items):
        if not isinstance(raw_info, Mapping):
            raise RuntimeError(f"README dataset_info[{info_index}] is not a mapping")
        info = dict(raw_info)
        config_name = info.get("config_name", "default")
        if not isinstance(config_name, str) or config_name in info_by_config:
            raise RuntimeError(f"Invalid or repeated dataset_info config {config_name!r}")
        if "features" not in info:
            raise RuntimeError(f"dataset_info config {config_name!r} has no features")
        info_by_config[config_name] = info
        info_order.append(config_name)

    expected_configs = set(expected_folders)
    if set(info_by_config) != expected_configs:
        raise RuntimeError(
            "README dataset_info disagrees with expected model folders: "
            f"expected={sorted(expected_configs)}, "
            f"found={sorted(info_by_config)}"
        )

    metrics: dict[str, dict[str, Any]] = {
        folder: {"download_size": 0, "splits": {}}
        for folder in expected_folders
    }
    for repo_path, stats in file_stats.items():
        folder = PurePosixPath(repo_path).parts[0]
        split = _file_kind(repo_path)
        output_path = _safe_path(output_root, repo_path)
        if not output_path.is_file():
            raise RuntimeError(f"Missing cleaned Parquet file: {output_path}")

        config_metrics = metrics[folder]
        config_metrics["download_size"] += output_path.stat().st_size
        split_metrics = config_metrics["splits"].setdefault(
            split,
            {"num_bytes": 0, "num_examples": 0},
        )
        split_metrics["num_bytes"] += stats.retained_num_bytes
        split_metrics["num_examples"] += stats.output_rows

    updated_infos: list[dict[str, Any]] = []
    for config_name in info_order:
        info = info_by_config[config_name]
        config_metrics = metrics[config_name]
        split_metrics = config_metrics["splits"]
        raw_splits = info.get("splits")
        if not isinstance(raw_splits, Sequence) or isinstance(
            raw_splits, (str, bytes, bytearray)
        ):
            raise RuntimeError(
                f"dataset_info config {config_name!r} must contain a splits list"
            )

        splits: list[dict[str, Any]] = []
        split_names: list[str] = []
        for split_index, raw_split in enumerate(raw_splits):
            if not isinstance(raw_split, Mapping):
                raise RuntimeError(
                    f"dataset_info config {config_name!r} "
                    f"splits[{split_index}] is not a mapping"
                )
            split = raw_split.get("name")
            if not isinstance(split, str) or split in split_names:
                raise RuntimeError(
                    f"Invalid or repeated split {split!r} in dataset_info "
                    f"config {config_name!r}"
                )
            split_names.append(split)
            if split not in split_metrics:
                continue
            updated_split = dict(raw_split)
            updated_split["num_bytes"] = split_metrics[split]["num_bytes"]
            updated_split["num_examples"] = split_metrics[split]["num_examples"]
            splits.append(updated_split)
        if set(split_names) != set(split_metrics):
            raise RuntimeError(
                f"dataset_info splits disagree with output files for {config_name!r}: "
                f"declared={sorted(split_names)}, actual={sorted(split_metrics)}"
            )

        updated = dict(info)
        updated["splits"] = splits
        updated["download_size"] = config_metrics["download_size"]
        updated["dataset_size"] = sum(
            split_info["num_bytes"] for split_info in splits
        )
        updated_infos.append(updated)

    card_data["dataset_info"] = updated_infos
    front_matter = yaml.safe_dump(
        card_data,
        allow_unicode=True,
        sort_keys=False,
        width=1_000,
    ).rstrip()
    card_path.write_text(
        f"---\n{front_matter}\n---\n{text[match.end():]}",
        encoding="utf-8",
    )
    logging.info("Regenerated cleaned dataset_info metadata in README.md")


def _combined_subset_metadata(
    job: DatasetJob,
    output_root: Path,
    file_stats: Mapping[str, Stats],
    expected_folders: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collapse one source's model configs into one destination config."""

    card_path = output_root / "README.md"
    text = card_path.read_text(encoding="utf-8")
    match = CARD_FRONT_MATTER_PATTERN.match(text)
    if match is None:
        raise RuntimeError("Cleaned README.md has no YAML front matter")
    loaded = yaml.safe_load(match.group(1)) or {}
    if not isinstance(loaded, Mapping):
        raise RuntimeError("Cleaned README.md front matter is not a mapping")
    raw_infos = loaded.get("dataset_info")
    if not isinstance(raw_infos, Sequence) or isinstance(
        raw_infos, (str, bytes, bytearray)
    ):
        raise RuntimeError("Cleaned README.md has no dataset_info list")

    infos_by_model: dict[str, Mapping[str, Any]] = {}
    for raw_info in raw_infos:
        if not isinstance(raw_info, Mapping):
            raise RuntimeError("Cleaned dataset_info contains a non-mapping item")
        config_name = raw_info.get("config_name", "default")
        if not isinstance(config_name, str) or config_name in infos_by_model:
            raise RuntimeError(
                f"Invalid or repeated cleaned config name {config_name!r}"
            )
        infos_by_model[config_name] = raw_info
    if set(infos_by_model) != set(expected_folders):
        raise RuntimeError(
            "Cleaned dataset_info model configs disagree with processed folders"
        )

    first_info = infos_by_model[expected_folders[0]]
    features = first_info.get("features")
    if features is None:
        raise RuntimeError("Cleaned dataset_info has no features")
    for folder in expected_folders[1:]:
        if infos_by_model[folder].get("features") != features:
            raise RuntimeError(
                f"Cannot combine {job.subset_name!r}: model folder {folder!r} "
                "has a different feature schema"
            )

    data_files: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, int]] = {}
    download_size = 0
    for split in FILE_KIND_TO_METADATA:
        repo_paths = sorted(
            repo_path
            for repo_path in file_stats
            if _file_kind(repo_path) == split
        )
        if not repo_paths:
            continue
        data_files.append(
            {
                "split": split,
                "path": [
                    f"{job.subset_name}/{repo_path}"
                    for repo_path in repo_paths
                ],
            }
        )
        metrics = {"num_bytes": 0, "num_examples": 0}
        for repo_path in repo_paths:
            stats = file_stats[repo_path]
            output_path = _safe_path(output_root, repo_path)
            metrics["num_bytes"] += stats.retained_num_bytes
            metrics["num_examples"] += stats.output_rows
            download_size += output_path.stat().st_size
        split_metrics[split] = metrics

    config = {
        "config_name": job.subset_name,
        "data_files": data_files,
    }
    splits = [
        {
            "name": split,
            "num_bytes": split_metrics[split]["num_bytes"],
            "num_examples": split_metrics[split]["num_examples"],
        }
        for split in FILE_KIND_TO_METADATA
        if split in split_metrics
    ]
    info = {
        "config_name": job.subset_name,
        "features": features,
        "splits": splits,
        "download_size": download_size,
        "dataset_size": sum(item["num_bytes"] for item in splits),
    }
    return config, info


def _combined_dataset_card(processed: Sequence[ProcessedDataset]) -> bytes:
    expected_subsets = [job.subset_name for job in ALL_DATASET_JOBS]
    actual_subsets = [item.job.subset_name for item in processed]
    if actual_subsets != expected_subsets:
        raise RuntimeError(
            "Cannot publish the combined card until every subset is ready: "
            f"expected={expected_subsets!r}, actual={actual_subsets!r}"
        )

    card_data = {
        "pretty_name": "Toucan full-generation cleaned",
        "configs": [item.combined_config for item in processed],
        "dataset_info": [item.combined_info for item in processed],
    }
    front_matter = yaml.safe_dump(
        card_data,
        allow_unicode=True,
        sort_keys=False,
        width=1_000,
    ).rstrip()
    sources = "\n".join(
        f"- `{item.job.subset_name}`: `{item.job.source_dataset_id}` at "
        f"`{SOURCE_REVISIONS[item.job.source_dataset_id]}`"
        for item in processed
    )
    body = f"""# Toucan full-generation cleaned

This manually gated dataset combines three cleaned generation sources as
Hugging Face configs: `qwen`, `oss`, and `kimi`. Each config combines all of
the model folders present in its pinned source while preserving the original
four task splits.

Pinned sources:

{sources}

Rows matching the enabled cleaning policy were removed. The corresponding
row-level audit is available to approved users at
`<config>/audit/would_drop.parquet`.
"""
    return f"---\n{front_matter}\n---\n{body}".encode("utf-8")


def _copy_support_files(
    dataset_id: str,
    repo_files: Sequence[str],
    *,
    revision: str,
    output_root: Path,
) -> None:
    for repo_path in sorted(path for path in repo_files if not path.endswith(".parquet")):
        if repo_path in STALE_REPO_FILES:
            logging.warning("Skipping legacy generated file: %s", repo_path)
            continue
        source = _download(dataset_id, repo_path, revision)
        destination = _safe_path(output_root, repo_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copy2(source, destination)


def _prepare_output(output_dataset_dir: Path) -> Path:
    incomplete = output_dataset_dir.with_name(
        output_dataset_dir.name + INCOMPLETE_OUTPUT_SUFFIX
    )
    if output_dataset_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dataset_dir}")
    if incomplete.exists():
        raise FileExistsError(
            f"Incomplete prior run exists: {incomplete}; inspect and remove it"
        )
    incomplete.mkdir(parents=True)
    return incomplete


def _create_gated_destination(api: HfApi) -> None:
    """Create and gate the empty shared repository before any data upload."""

    api.create_repo(
        repo_id=DESTINATION_DATASET_ID,
        repo_type="dataset",
        private=DESTINATION_DATASET_PRIVATE,
        exist_ok=False,
    )
    if MANUAL_GATE_DESTINATION_DATASETS:
        # Gate the empty repository before uploading any dataset files. This
        # keeps the data inaccessible until access is explicitly granted.
        api.update_repo_settings(
            repo_id=DESTINATION_DATASET_ID,
            repo_type="dataset",
            gated="manual",
        )


def _validate_resumable_destination(api: HfApi) -> tuple[str, dict[str, int]]:
    """Validate a partial gated destination and refuse subset overwrites."""

    info = api.dataset_info(DESTINATION_DATASET_ID, files_metadata=True)
    if not info.sha:
        raise RuntimeError("Existing destination has no commit SHA")
    if info.private != DESTINATION_DATASET_PRIVATE:
        raise RuntimeError(
            "Existing destination visibility changed: "
            f"private={info.private}, expected={DESTINATION_DATASET_PRIVATE}"
        )
    expected_gating: str | bool = (
        "manual" if MANUAL_GATE_DESTINATION_DATASETS else False
    )
    if info.gated != expected_gating:
        raise RuntimeError(
            "Existing destination gating changed: "
            f"gated={info.gated!r}, expected={expected_gating!r}"
        )

    files = {
        sibling.rfilename: sibling.size or 0
        for sibling in info.siblings or ()
    }
    active_subsets = {job.subset_name for job in DATASET_JOBS}
    for subset in active_subsets:
        prefix = f"{subset}/"
        existing = sorted(path for path in files if path.startswith(prefix))
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing destination subset {subset!r}; "
                f"found {len(existing)} remote files"
            )

    for job in ALL_DATASET_JOBS:
        if job.subset_name in active_subsets:
            continue
        prefix = f"{job.subset_name}/"
        data_files = [
            path
            for path in files
            if path.startswith(prefix)
            and path.endswith(".parquet")
            and not path.startswith(f"{prefix}audit/")
        ]
        audit_path = f"{prefix}{DROP_AUDIT_REPO_PATH}"
        if not data_files or audit_path not in files:
            raise RuntimeError(
                f"Existing subset {job.subset_name!r} is incomplete: "
                f"data_files={len(data_files)}, audit_present={audit_path in files}"
            )

    return info.sha, files


def _source_card_summary(
    job: DatasetJob,
) -> tuple[Any, dict[str, int]]:
    """Read a pinned source card's common features and input split counts."""

    revision = SOURCE_REVISIONS[job.source_dataset_id]
    card_path = _download(job.source_dataset_id, "README.md", revision)
    text = card_path.read_text(encoding="utf-8")
    match = CARD_FRONT_MATTER_PATTERN.match(text)
    if match is None:
        raise RuntimeError(f"{job.source_dataset_id} README has no front matter")
    loaded = yaml.safe_load(match.group(1)) or {}
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"{job.source_dataset_id} README metadata is invalid")
    raw_infos = loaded.get("dataset_info")
    if not isinstance(raw_infos, Sequence) or isinstance(
        raw_infos, (str, bytes, bytearray)
    ):
        raise RuntimeError(
            f"{job.source_dataset_id} README has no dataset_info list"
        )

    features: Any = None
    input_rows: Counter[str] = Counter()
    for raw_info in raw_infos:
        if not isinstance(raw_info, Mapping):
            raise RuntimeError("Source dataset_info contains a non-mapping item")
        raw_features = raw_info.get("features")
        if raw_features is None:
            raise RuntimeError("Source dataset_info has no features")
        current_features = raw_features
        if features is None:
            features = current_features
        elif current_features != features:
            raise RuntimeError(
                f"{job.source_dataset_id} model configs have different features"
            )
        raw_splits = raw_info.get("splits")
        if not isinstance(raw_splits, Sequence) or isinstance(
            raw_splits, (str, bytes, bytearray)
        ):
            raise RuntimeError("Source dataset_info has no splits list")
        for raw_split in raw_splits:
            if not isinstance(raw_split, Mapping):
                raise RuntimeError("Source dataset_info contains an invalid split")
            name = raw_split.get("name")
            num_examples = raw_split.get("num_examples")
            if name not in FILE_KIND_TO_METADATA or type(num_examples) is not int:
                raise RuntimeError(
                    f"Invalid source split metadata: name={name!r}, "
                    f"num_examples={num_examples!r}"
                )
            input_rows[name] += num_examples
    if features is None:
        raise RuntimeError(f"{job.source_dataset_id} dataset_info is empty")
    return features, dict(input_rows)


def _existing_subset_metadata(
    job: DatasetJob,
    *,
    destination_revision: str,
    destination_files: Mapping[str, int],
) -> ProcessedDataset:
    """Reconstruct card metadata without rewriting an existing subset."""

    prefix = f"{job.subset_name}/"
    relative_data_files = sorted(
        path[len(prefix) :]
        for path in destination_files
        if path.startswith(prefix)
        and path.endswith(".parquet")
        and not path.startswith(f"{prefix}audit/")
    )
    _parquet_files(relative_data_files)

    source_files = HfApi(token=HUGGING_FACE_TOKEN).list_repo_files(
        repo_id=job.source_dataset_id,
        repo_type="dataset",
        revision=SOURCE_REVISIONS[job.source_dataset_id],
    )
    expected_data_files = _parquet_files(source_files)
    if relative_data_files != expected_data_files:
        raise RuntimeError(
            f"Existing {job.subset_name!r} file set differs from its pinned "
            f"source: remote={len(relative_data_files)}, "
            f"expected={len(expected_data_files)}"
        )

    features, input_rows = _source_card_summary(job)
    audit_repo_path = f"{prefix}{DROP_AUDIT_REPO_PATH}"
    audit_path = _download(
        DESTINATION_DATASET_ID,
        audit_repo_path,
        destination_revision,
    )
    audit_table = pq.read_table(audit_path, columns=["file_kind"])
    dropped_rows = Counter(audit_table.column("file_kind").to_pylist())
    unknown_audit_kinds = set(dropped_rows) - set(FILE_KIND_TO_METADATA)
    if unknown_audit_kinds:
        raise RuntimeError(
            f"Existing {job.subset_name!r} audit has unknown file kinds: "
            f"{sorted(unknown_audit_kinds)!r}"
        )

    data_files: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for split in FILE_KIND_TO_METADATA:
        paths = [
            f"{prefix}{repo_path}"
            for repo_path in relative_data_files
            if _file_kind(repo_path) == split
        ]
        if not paths:
            continue
        output_rows = input_rows.get(split, 0) - dropped_rows.get(split, 0)
        if output_rows < 0:
            raise RuntimeError(
                f"Existing {job.subset_name!r} audit drops more {split!r} "
                "rows than the pinned source contains"
            )
        data_files.append({"split": split, "path": paths})
        splits.append({"name": split, "num_examples": output_rows})

    download_size = sum(
        destination_files[f"{prefix}{repo_path}"]
        for repo_path in relative_data_files
    )
    return ProcessedDataset(
        job=job,
        staged_output=job.output_dataset_dir,
        combined_config={
            "config_name": job.subset_name,
            "data_files": data_files,
        },
        combined_info={
            "config_name": job.subset_name,
            "features": features,
            "splits": splits,
            "download_size": download_size,
        },
    )


def _upload_subset(api: HfApi, processed: ProcessedDataset) -> None:
    """Upload one processed source, optionally preserving its root layout."""

    if PRESERVE_SINGLE_SOURCE_LAYOUT:
        api.upload_folder(
            repo_id=DESTINATION_DATASET_ID,
            repo_type="dataset",
            folder_path=processed.job.output_dataset_dir,
            path_in_repo="",
            commit_message="Upload cleaned ToolACE dataset",
        )
        return

    api.upload_folder(
        repo_id=DESTINATION_DATASET_ID,
        repo_type="dataset",
        folder_path=processed.job.output_dataset_dir,
        path_in_repo=processed.job.subset_name,
        allow_patterns=("*.parquet", "**/*.parquet"),
        commit_message=f"Upload cleaned {processed.job.subset_name} config",
    )


def _upload_combined_card(
    api: HfApi,
    processed: Sequence[ProcessedDataset],
) -> None:
    api.upload_file(
        path_or_fileobj=_combined_dataset_card(processed),
        path_in_repo="README.md",
        repo_id=DESTINATION_DATASET_ID,
        repo_type="dataset",
        commit_message="Add combined dataset configs and metadata",
    )


# ---------------------------------------------------------------------------
# Reporting and orchestration
# ---------------------------------------------------------------------------


def _format_stats(stats: Stats) -> str:
    reasons = ", ".join(
        f"{reason}={stats.drop_reasons.get(reason, 0)}"
        for reason in DROP_REASONS
    )
    return (
        f"rows={stats.input_rows}, unique_would_drop={stats.unique_would_drop}, "
        f"would_keep={stats.would_keep}, {reasons}, "
        f"rejected_long_observed={stats.rejected_long_observed}, "
        f"warnings={stats.warnings}, rejected_greater={stats.rejected_greater}, "
        f"rejected_greater_retained={stats.rejected_greater_retained}"
    )


def _print_report(
    job: DatasetJob,
    file_stats: Mapping[str, Stats],
    folder_stats: Mapping[str, Stats],
    total: Stats,
    revision: str,
    staged_output: Path | None,
) -> None:
    output_enabled = DROP_ROWS or PUSH_TO_HUB
    if DROP_ROWS and PUSH_TO_HUB:
        mode = "DROP MATCHING ROWS, WRITE, AND UPLOAD"
    elif DROP_ROWS:
        mode = "DROP MATCHING ROWS AND WRITE LOCALLY"
    elif PUSH_TO_HUB:
        mode = "WRITE ALL ROWS AND UPLOAD"
    else:
        mode = "DRY RUN ONLY"

    print("\n=== Cleaning configuration ===")
    print(f"Mode: {mode}")
    print(f"Source: {job.source_dataset_id}@{revision}")
    print(f"Long-value threshold: > {LONG_VALUE_WORD_THRESHOLD} words")
    print(
        "Rejected-response long values cause drops: "
        f"{DROP_IF_REJECTED_RESPONSE_HAS_LONG_VALUE}"
    )
    print(f"Enabled drop reasons: {sorted(ENABLED_DROP_REASONS)}")

    print("\n=== Per Parquet file ===")
    for repo_path in sorted(file_stats):
        print(f"{repo_path}: {_format_stats(file_stats[repo_path])}")

    print("\n=== Per model folder ===")
    for folder in folder_stats:
        print(f"{folder}: {_format_stats(folder_stats[folder])}")

    print("\n=== Global ===")
    print(_format_stats(total))
    if total.warning_reasons:
        warnings = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(total.warning_reasons.items())
        )
        print(f"Warning breakdown: {warnings}")

    if output_enabled:
        if staged_output is None:
            raise RuntimeError("Write mode completed without a staged output")
        print(f"Validated staged local dataset: {staged_output}")
        print(f"Rows written under current policy: {total.output_rows}")
        if WRITE_DROP_AUDIT:
            print(
                f"Drop audit: {DROP_AUDIT_REPO_PATH} "
                f"({total.unique_would_drop} rows)"
            )
        if DELETE_TEMPORARY_WORK_DATA:
            print(
                "This staged dataset and its dedicated source cache will be "
                "removed after the gated upload attempt."
            )
        else:
            print(
                "The final output path is created only after every configured "
                "dataset job succeeds."
            )
    else:
        print(
            "No output was written because DROP_ROWS=False and "
            "PUSH_TO_HUB=False. Source shards may have been added to the "
            "Hugging Face cache."
        )

    print("\n=== Rejected reward greater than chosen reward ===")
    print(
        f"Input rows meeting the condition: {total.rejected_greater} / "
        f"{total.input_rows}"
    )
    if total.rejected_greater:
        print(
            "DOUBLE-CHECK REQUIRED: rejected reward is greater than chosen "
            "reward in one or more rows; this is expected to be rare."
        )
    if output_enabled:
        written_flags = (
            total.rejected_greater_retained
            if DROP_ROWS
            else total.rejected_greater
        )
        print(
            "Flags written to output dataset: "
            f"{written_flags} / {total.output_rows}"
        )
    else:
        print(
            "Rows that would receive the flag after cleaning: "
            f"{total.rejected_greater_retained} / {total.would_keep}"
        )


def _process_dataset(api: HfApi, job: DatasetJob) -> ProcessedDataset | None:
    """Validate one pinned source, process all shards, and stage local output.

    Dry-run mode returns no output path. Either row dropping or upload mode
    enables writing and leaves the completed dataset in its ``.incomplete``
    staging directory until every configured job passes.
    """

    requested_revision = SOURCE_REVISIONS[job.source_dataset_id]
    source_info = api.dataset_info(
        job.source_dataset_id,
        revision=requested_revision,
    )
    revision = source_info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a source commit SHA")
    if revision.lower() != requested_revision.lower():
        raise RuntimeError(
            f"Source revision drift for {job.source_dataset_id}: requested "
            f"{requested_revision}, resolved {revision}"
        )
    repo_files = api.list_repo_files(
        repo_id=job.source_dataset_id,
        repo_type="dataset",
        revision=revision,
    )
    parquet_files = _parquet_files(repo_files)
    discovered_folders = {
        PurePosixPath(repo_path).parts[0]
        for repo_path in parquet_files
    }
    present_folders = tuple(
        folder
        for folder in EXPECTED_MODEL_FOLDERS
        if folder in discovered_folders
    )
    logging.info(
        "Pinned %s@%s with %d Parquet files",
        job.source_dataset_id,
        revision,
        len(parquet_files),
    )
    logging.info(
        "Mode: %s",
        (
            "drop matching rows, write, and upload"
            if DROP_ROWS and PUSH_TO_HUB
            else "drop matching rows and write locally"
            if DROP_ROWS
            else "write all rows and upload"
            if PUSH_TO_HUB
            else "dry-run; no output"
        ),
    )

    cross_rule_plan = _build_cross_rule_plan(
        job.source_dataset_id,
        parquet_files,
        revision,
    )
    if DELETE_TEMPORARY_WORK_DATA:
        # The cross-row plan is now fully resident in memory. Reclaim the first
        # pass's source files before output starts, then download and discard
        # one source shard at a time during the write pass.
        _cleanup_source_download_cache()
    output_root = (
        _prepare_output(job.output_dataset_dir)
        if DROP_ROWS or PUSH_TO_HUB
        else None
    )
    file_stats: dict[str, Stats] = {}
    folder_stats = {folder: Stats() for folder in present_folders}
    total = Stats()
    audit_path = (
        _safe_path(output_root, DROP_AUDIT_REPO_PATH)
        if output_root is not None and WRITE_DROP_AUDIT
        else None
    )
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_context = pq.ParquetWriter(
            audit_path,
            schema=DROP_AUDIT_SCHEMA,
            compression=PARQUET_COMPRESSION,
        )
    else:
        audit_context = nullcontext(None)

    with audit_context as audit_writer:
        for number, repo_path in enumerate(parquet_files, 1):
            logging.info(
                "Processing %d/%d: %s",
                number,
                len(parquet_files),
                repo_path,
            )
            try:
                stats = _process_file(
                    job.source_dataset_id,
                    repo_path,
                    revision=revision,
                    output_root=output_root,
                    cross_rule_plan=cross_rule_plan,
                    audit_writer=audit_writer,
                )
            finally:
                if DELETE_TEMPORARY_WORK_DATA:
                    _cleanup_source_download_cache()
            file_stats[repo_path] = stats
            folder = PurePosixPath(repo_path).parts[0]
            folder_stats[folder].merge(stats)
            total.merge(stats)

    if audit_path is not None:
        audit_rows = pq.ParquetFile(audit_path).metadata.num_rows
        if audit_rows != total.unique_would_drop:
            raise RuntimeError(
                f"Audit row-count mismatch for {job.source_dataset_id}: "
                f"{audit_rows} != {total.unique_would_drop}"
            )
        logging.info(
            "Wrote %d audit rows to %s",
            audit_rows,
            audit_path,
        )

    combined_metadata: tuple[dict[str, Any], dict[str, Any]] | None = None
    if output_root:
        _copy_support_files(
            job.source_dataset_id,
            repo_files,
            revision=revision,
            output_root=output_root,
        )
        _regenerate_dataset_card(output_root, file_stats, present_folders)
        combined_metadata = _combined_subset_metadata(
            job,
            output_root,
            file_stats,
            present_folders,
        )

    _print_report(
        job,
        file_stats,
        folder_stats,
        total,
        revision,
        output_root,
    )
    if output_root is None:
        return None
    if combined_metadata is None:
        raise RuntimeError("Write mode completed without combined metadata")
    combined_config, combined_info = combined_metadata
    return ProcessedDataset(
        job=job,
        staged_output=output_root,
        combined_config=combined_config,
        combined_info=combined_info,
    )


def _finalize_outputs(staged_outputs: Sequence[tuple[DatasetJob, Path]]) -> None:
    finalized: list[tuple[DatasetJob, Path]] = []
    try:
        for job, staged_output in staged_outputs:
            if job.output_dataset_dir.exists():
                raise FileExistsError(
                    f"Output appeared during the run: {job.output_dataset_dir}"
                )
            staged_output.rename(job.output_dataset_dir)
            finalized.append((job, staged_output))
    except Exception as finalize_error:
        rollback_errors = []
        for job, staged_output in reversed(finalized):
            try:
                job.output_dataset_dir.rename(staged_output)
            except Exception as rollback_error:
                rollback_errors.append(f"{job.output_dataset_dir}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Local output finalization failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from finalize_error
        raise


def _cleanup_temporary_work_data() -> None:
    """Delete only this process's dedicated temporary work tree."""

    work_root = TEMPORARY_WORK_ROOT
    resolved_parent = TEMPORARY_WORK_PARENT.resolve()
    resolved_root = work_root.resolve()
    if (
        work_root.is_symlink()
        or resolved_root.parent != resolved_parent
        or not resolved_root.name.startswith("toucan-cleaning-")
    ):
        raise RuntimeError(
            f"Refusing unsafe temporary cleanup target: {work_root}"
        )
    if resolved_root.exists():
        shutil.rmtree(resolved_root)
        logging.info("Removed temporary cleaning data: %s", resolved_root)


def main() -> None:
    _validate_configuration()
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

    api = HfApi(token=HUGGING_FACE_TOKEN)
    existing_by_subset: dict[str, ProcessedDataset] = {}
    if RESUME_EXISTING_DESTINATION:
        destination_revision, destination_files = (
            _validate_resumable_destination(api)
        )
        active_subsets = {job.subset_name for job in DATASET_JOBS}
        try:
            for job in ALL_DATASET_JOBS:
                if job.subset_name not in active_subsets:
                    existing_by_subset[job.subset_name] = (
                        _existing_subset_metadata(
                            job,
                            destination_revision=destination_revision,
                            destination_files=destination_files,
                        )
                    )
        finally:
            if not REUSING_SOURCE_CACHE:
                _cleanup_source_download_cache()
        print(
            "Resume mode: validated the existing manually gated destination; "
            f"selected subsets={sorted(active_subsets)!r}."
        )
    elif PUSH_TO_HUB:
        _create_gated_destination(api)

    if PUSH_TO_HUB and DELETE_TEMPORARY_WORK_DATA:
        print(
            "Temporary mode: each selected config is processed, uploaded, "
            "and removed locally before the next config starts."
        )
        processed_datasets: list[ProcessedDataset] = []
        for job in DATASET_JOBS:
            try:
                processed = _process_dataset(api, job)
                if processed is None:
                    raise RuntimeError(
                        f"{job.source_dataset_id} produced no staged output"
                    )
                _finalize_outputs(((job, processed.staged_output),))
                _upload_subset(api, processed)
                processed_datasets.append(processed)
                print(
                    f"Uploaded cleaned source {job.subset_name!r}: "
                    f"https://huggingface.co/datasets/{DESTINATION_DATASET_ID}"
                )
            finally:
                _cleanup_temporary_work_data()
        processed_by_subset = {
            **existing_by_subset,
            **{item.job.subset_name: item for item in processed_datasets},
        }
        complete_processed = [
            processed_by_subset[job.subset_name]
            for job in ALL_DATASET_JOBS
        ]
        if PRESERVE_SINGLE_SOURCE_LAYOUT:
            print(
                "Published cleaned dataset with its source model-config "
                f"layout: https://huggingface.co/datasets/{DESTINATION_DATASET_ID}"
            )
        else:
            _upload_combined_card(api, complete_processed)
            print(
                "Published combined dataset card: "
                f"https://huggingface.co/datasets/{DESTINATION_DATASET_ID}"
            )
        return

    processed_datasets = []
    for job in DATASET_JOBS:
        processed = _process_dataset(api, job)
        if processed is not None:
            processed_datasets.append(processed)

    if not (DROP_ROWS or PUSH_TO_HUB):
        return
    if len(processed_datasets) != len(DATASET_JOBS):
        raise RuntimeError("Not every dataset job produced a staged output")

    staged_outputs = [
        (processed.job, processed.staged_output)
        for processed in processed_datasets
    ]
    _finalize_outputs(staged_outputs)
    print("\n=== Finalized local datasets ===")
    for job, _ in staged_outputs:
        print(job.output_dataset_dir)

    if PUSH_TO_HUB:
        for processed in processed_datasets:
            _upload_subset(api, processed)
            print(
                f"Uploaded config {processed.job.subset_name!r}: "
                f"https://huggingface.co/datasets/{DESTINATION_DATASET_ID}"
            )
        processed_by_subset = {
            **existing_by_subset,
            **{item.job.subset_name: item for item in processed_datasets},
        }
        complete_processed = [
            processed_by_subset[job.subset_name]
            for job in ALL_DATASET_JOBS
        ]
        if PRESERVE_SINGLE_SOURCE_LAYOUT:
            print("Published cleaned dataset with its source model-config layout.")
        else:
            _upload_combined_card(api, complete_processed)
            print("Published combined dataset card.")


if __name__ == "__main__":
    main()
