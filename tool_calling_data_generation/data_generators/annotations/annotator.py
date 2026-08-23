"""
Annotation Pipeline
===================

Takes a DataFrame produced by the DPO data generator (containing raw
completions and pre-parsed tool calls) and runs **all** registered rewards
on both the ``chosen`` and ``rejected`` completions, adding score columns.

Input format adaptation
-----------------------
The ``annotate()`` method auto-adapts to whatever format the DataFrame uses:

* Fresh from ``dpo_generator.to_analysis_dataframe()`` — has
  ``chosen``, ``rejected``, ``tool_calls_chosen``
* Reloaded from ``CanonicalDataset.to_df()`` — has
  ``chosen_response``, ``rejected_response``, ``metadata``

Missing columns are derived automatically.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from tool_calling_data_generation.rewards import RewardRegistry, RewardResult

logger = logging.getLogger(__name__)

# Sentinel used when a reward crashes on a malformed completion.
_ZERO_RESULT = RewardResult(score=0.0)


class Annotator:
    """Run every registered reward on chosen & rejected completions.

    Usage::

        annotator = Annotator()
        annotated_df = annotator.annotate(final_df)
    """

    def __init__(self, registry: Optional[RewardRegistry] = None):
        self.registry = registry or RewardRegistry()
        logger.info("Annotator initialised with rewards: %s", self.registry.list_all())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reward columns to *df* and return the augmented DataFrame.

        Automatically adapts to whatever format the DataFrame uses:

        * ``chosen`` / ``rejected`` — raw completion text.  If missing,
          derived from ``chosen_response`` / ``rejected_response``.
        * ``tool_calls_chosen`` — parsed ground-truth tool calls.  If
          missing, extracted from ``metadata.parsed_chosen_tool_calls``
          or derived from ``chosen_response``.
        * ``tool_catalog`` — optional; read from column if available.

        Returns:
            A **new** DataFrame with reward columns appended.
        """
        df = self._adapt_df(df.copy())
        reward_names = self.registry.list_all()
        if not reward_names:
            logger.warning("No rewards registered — annotation is a no-op.")
            return df

        logger.info("Annotating %d rows with %d rewards …", len(df), len(reward_names))

        ground_truths = [
            {
                "expected_calls": _parse_tool_calls(gtc),
                "tool_catalog": _parse_tool_catalog(tc),
            }
            for gtc, tc in zip(
                df["tool_calls_chosen"],
                df.get("tool_catalog", [None] * len(df)),
            )
        ]

        for rname in reward_names:
            reward = self.registry.get(rname)

            chosen_col = f"chosen_{rname}_reward"
            rej_col = f"rejected_{rname}_reward"
            diff_col = f"{rname}_reward_diff"

            chosen_scores: List[Any] = []
            rej_scores: List[Any] = []
            diffs: List[Any] = []

            comp_chosen: Dict[str, List] = {}
            comp_rej: Dict[str, List] = {}
            extra_chosen: Dict[str, List] = {}
            extra_rej: Dict[str, List] = {}

            # Track all keys seen so far so crash rows still contribute a
            # value (None / []) — prevents length mismatches when a reward
            # crashes on some rows.
            _comp_keys: set = set()
            _extra_keys: set = set()

            for idx, (chosen_text, rej_text, gt) in enumerate(
                zip(df["chosen"], df["rejected"], ground_truths)
            ):
                c_res = _safe_compute(reward, chosen_text, gt, idx, "chosen")
                r_res = _safe_compute(reward, rej_text, gt, idx, "rejected")

                cs = c_res.score
                rs = r_res.score
                chosen_scores.append(cs)
                rej_scores.append(rs)
                diffs.append(cs - rs if cs is not None and rs is not None else None)

                # Track all known keys
                _comp_keys.update(c_res.components)
                _comp_keys.update(r_res.components)
                _extra_keys.update(c_res.extra)
                _extra_keys.update(r_res.extra)

                # Collect components (pad with None for crash rows)
                for k in _comp_keys:
                    comp_chosen.setdefault(f"chosen_{rname}_{k}", []).append(
                        c_res.components.get(k)
                    )
                    comp_rej.setdefault(f"rejected_{rname}_{k}", []).append(
                        r_res.components.get(k)
                    )

                # Collect extra (pad with [] for crash rows)
                for k in _extra_keys:
                    extra_chosen.setdefault(f"chosen_{rname}_{k}", []).append(
                        c_res.extra.get(k, [])
                    )
                    extra_rej.setdefault(f"rejected_{rname}_{k}", []).append(
                        r_res.extra.get(k, [])
                    )

            df[chosen_col] = chosen_scores
            df[rej_col] = rej_scores
            df[diff_col] = diffs
            nrows = len(chosen_scores)
            for vals in comp_chosen.values():
                vals.extend([None] * (nrows - len(vals)))
            for vals in comp_rej.values():
                vals.extend([None] * (nrows - len(vals)))
            for vals in extra_chosen.values():
                vals.extend([[] for _ in range(nrows - len(vals))])
            for vals in extra_rej.values():
                vals.extend([[] for _ in range(nrows - len(vals))])

            for col, vals in {**comp_chosen, **comp_rej}.items():
                df[col] = vals
            for col, vals in {**extra_chosen, **extra_rej}.items():
                df[col] = vals

        return df

    # ------------------------------------------------------------------
    # annotate_row — for per-row use (dpo_generator inline path)
    # ------------------------------------------------------------------

    def annotate_row(
        self, chosen_text: str, rejected_text: str,
        expected_calls: List[Dict],
        tool_catalog: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Return a dict of reward scores for one row."""
        gt = {"expected_calls": expected_calls}
        if tool_catalog is not None:
            gt["tool_catalog"] = tool_catalog
        row: Dict[str, Any] = {}
        for rname in self.registry.list_all():
            reward = self.registry.get(rname)
            c_res = _safe_compute(reward, chosen_text, gt)
            r_res = _safe_compute(reward, rejected_text, gt)
            row[f"chosen_{rname}_reward"] = c_res.score
            row[f"rejected_{rname}_reward"] = r_res.score
            row[f"{rname}_reward_diff"] = (
                c_res.score - r_res.score
                if c_res.score is not None and r_res.score is not None
                else None
            )
            for k, v in c_res.components.items():
                row[f"chosen_{rname}_{k}"] = v
            for k, v in r_res.components.items():
                row[f"rejected_{rname}_{k}"] = v
            for k, v in c_res.extra.items():
                row[f"chosen_{rname}_{k}"] = v
            for k, v in r_res.extra.items():
                row[f"rejected_{rname}_{k}"] = v
        return row

    # ------------------------------------------------------------------
    # DataFrame adaptation — normalise any input format
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_df(df: "pd.DataFrame") -> "pd.DataFrame":
        """Ensure *df* has ``chosen``, ``rejected``, and
        ``tool_calls_chosen`` columns, deriving them from canonical
        columns (*chosen_response*, *rejected_response*, *metadata*)
        when the shorthand columns are absent.
        """
        if "chosen" not in df.columns:
            df = Annotator._derive_from_response(df, "chosen_response", "chosen")
        if "rejected" not in df.columns:
            df = Annotator._derive_from_response(df, "rejected_response", "rejected")
        if "tool_calls_chosen" not in df.columns:
            df = Annotator._derive_tool_calls_chosen(df)
        return df

    @staticmethod
    def _derive_from_response(
        df: "pd.DataFrame", src_col: str, dst_col: str,
    ) -> "pd.DataFrame":
        """Derive a ``chosen`` / ``rejected`` text column from a
        structured response column (*chosen_response* /
        *rejected_response*).
        """
        if src_col in df.columns:
            df[dst_col] = df[src_col].apply(_response_to_tool_calls_text)
        else:
            df[dst_col] = "[]"
        return df

    @staticmethod
    def _derive_tool_calls_chosen(df: "pd.DataFrame") -> "pd.DataFrame":
        """Derive ``tool_calls_chosen`` from the most specific source
        available.

        Priority: ``metadata.parsed_chosen_tool_calls`` > extraction
        from ``chosen_response`` > empty-list fallback.
        """
        # 1. Try metadata.parsed_chosen_tool_calls
        if "metadata" in df.columns:
            try:
                meta_sample = _parse_metadata(df.iloc[0].get("metadata"))
                if "parsed_chosen_tool_calls" in meta_sample:
                    df["tool_calls_chosen"] = df["metadata"].apply(
                        lambda m: _parse_metadata(m).get("parsed_chosen_tool_calls", [])
                    )
                    return df
            except Exception:
                pass

        # 2. Extract from chosen_response
        if "chosen_response" in df.columns:
            df["tool_calls_chosen"] = df["chosen_response"].apply(
                _extract_calls_from_response
            )
            return df

        # 3. Fallback
        df["tool_calls_chosen"] = [[] for _ in range(len(df))]
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _response_to_tool_calls_text(val: Any) -> str:
    """Convert a structured response to a JSON string of flat tool calls."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return "[]"
    calls = []
    for msg in (val or []):
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            calls.append({"name": fn.get("name", ""), "arguments": args})
    return json.dumps(calls)


def _extract_calls_from_response(val: Any) -> List[Dict[str, Any]]:
    """Extract ``[{name, arguments}, ...]`` from a structured response."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    calls = []
    for msg in (val or []):
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            calls.append({"name": fn.get("name", ""), "arguments": args})
    return calls


def _parse_metadata(val: Any) -> Dict[str, Any]:
    """Accept a dict or JSON-string metadata value."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _parse_tool_calls(val: Any) -> List[Dict[str, Any]]:
    """Accept parsed list, JSON string, or None -> always return a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _parse_tool_catalog(val: Any) -> List[Dict[str, Any]]:
    """Accept parsed list, JSON string, or None -> always return a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _safe_compute(
    reward,
    completion: str,
    ground_truth: Dict,
    row_idx: int = -1,
    side: str = "?",
) -> RewardResult:
    """Call ``reward.compute()`` with a broad safety net."""
    try:
        return reward.compute(completion, ground_truth)
    except Exception as exc:
        logger.debug(
            "[reward crash] %s on row %d (%s): %s",
            reward.name or reward.__class__.__name__,
            row_idx,
            side,
            exc,
        )
        return _ZERO_RESULT


