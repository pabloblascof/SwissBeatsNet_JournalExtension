"""S0.2 -- the CODE conflicting-label policies, applied as columns.

What. CODE-15% is a repeat-visit corpus: one patient can have up to 38 exams,
each carrying its own Chagas label. 1,825 patients have at least one positive
and one negative exam, and those patients own 45% of every positive exam in
the corpus (``docs/journal/S0.2_LABEL_POLICY.md``). Because Chagas is a
lifelong infection, a within-patient disagreement is a labelling error, not a
clinical change, and some rule has to resolve it. Five rules are defined here
and applied *side by side* as ``label_*`` columns of the metadata table, so
that every downstream run picks one with :func:`select` and S6.2 can sweep
the rest without rebuilding anything.

Why side by side rather than one column. The choice of policy is a human
decision (project rules, "What requires a human") that was still open when this
module was written. Putting all five in the table makes the frozen splits and
every cache independent of that decision: splits stratify on
``ever_positive``, which is the same under every policy.

The policies, on the per-patient label vector ``y`` (CODE only; SaMi-Trop and
PTB-XL have one label per patient and every policy reduces to the raw label):

``any_positive``           patient positive iff any exam positive (D-012 default)
``majority_tie_positive``  patient positive iff mean(y) >= 0.5
``majority_tie_negative``  patient positive iff mean(y) >  0.5
``drop_conflicting``       patients with both labels are dropped (NaN)
``exam_level``             each exam keeps its own label

Costs on the real join, asserted by ``tests/test_labels.py``:

=======================  ==============  ==========
policy                   positive exams  exams kept
=======================  ==============  ==========
any_positive                     10,231     343,424
majority_tie_positive             7,183     343,424
majority_tie_negative             5,347     343,424
drop_conflicting                  3,600     336,793
exam_level                        6,561     343,424
=======================  ==============  ==========
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

#: The CODE source string, exactly as ``prepare_code15_data.py`` writes it.
CODE_SOURCE = "CODE-15%"

#: Columns a frame must carry before a policy can be applied.
REQUIRED_COLUMNS = ("patient_key", "chagas_raw", "source")


class LabelPolicy(str, Enum):
    """The five S0.2 policies. Values are the config-file spellings."""

    any_positive = "any_positive"
    majority_tie_positive = "majority_tie_positive"
    majority_tie_negative = "majority_tie_negative"
    drop_conflicting = "drop_conflicting"
    exam_level = "exam_level"

    @property
    def column(self) -> str:
        """The metadata column that holds this policy's labels."""
        return POLICY_COLUMNS[self]


#: Policy -> metadata column name, spelled exactly as ARCHITECTURE.md lists them.
POLICY_COLUMNS: dict[LabelPolicy, str] = {
    LabelPolicy.any_positive: "label_any_positive",
    LabelPolicy.majority_tie_positive: "label_majority_pos",
    LabelPolicy.majority_tie_negative: "label_majority_neg",
    LabelPolicy.drop_conflicting: "label_drop_conflicting",
    LabelPolicy.exam_level: "label_exam_level",
}

#: Default for every run until the human decision lands (D-012).
DEFAULT_POLICY = LabelPolicy.any_positive


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"frame lacks required columns {missing}; have {list(df.columns)}")


def patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-``patient_key`` label statistics: ``mean``, ``n_labels``, ``n_exams``.

    ``mean`` is the fraction of the patient's exams labelled positive;
    ``n_labels`` is the number of distinct labels (2 means conflicting).
    Index is ``patient_key``.
    """
    _check_columns(df)
    y = df["chagas_raw"].astype(float)
    g = y.groupby(df["patient_key"])
    return pd.DataFrame(
        {"mean": g.mean(), "n_labels": g.nunique(), "n_exams": g.size()}
    )


def apply_all_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the five ``label_*`` columns added.

    The frame is per record and must carry ``patient_key``, ``chagas_raw``
    (0/1) and ``source``. For non-CODE sources every policy column equals
    ``chagas_raw``: SaMi-Trop and PTB-XL have one exam per patient and a
    corpus-wide constant label, so there is nothing to resolve. Rows with a
    missing ``chagas_raw`` are left NaN under every policy.

    Columns are float so that NaN can mean "dropped under this policy"
    (ARCHITECTURE.md). :func:`select` casts back to int after dropping.
    """
    _check_columns(df)
    out = df.copy()
    raw = out["chagas_raw"].astype(float)
    is_code = out["source"] == CODE_SOURCE

    # Per-patient statistics broadcast back to each record. Computed on the
    # whole frame; non-CODE patients have one record each so they are inert.
    g = raw.groupby(out["patient_key"])
    mean = g.transform("mean")
    n_labels = g.transform("nunique")

    any_pos = (mean > 0).astype(float)
    tie_pos = (mean >= 0.5).astype(float)
    tie_neg = (mean > 0.5).astype(float)
    drop = raw.where(n_labels <= 1)  # NaN where the patient conflicts

    policy_values = {
        LabelPolicy.any_positive: any_pos,
        LabelPolicy.majority_tie_positive: tie_pos,
        LabelPolicy.majority_tie_negative: tie_neg,
        LabelPolicy.drop_conflicting: drop,
        LabelPolicy.exam_level: raw,
    }
    for policy, values in policy_values.items():
        col = np.where(is_code, values, raw)
        col = pd.Series(col, index=out.index, dtype=float)
        col[raw.isna()] = np.nan
        out[policy.column] = col
    return out


def select(df: pd.DataFrame, policy: LabelPolicy | str) -> pd.DataFrame:
    """Rows kept under ``policy``, with an int ``label`` column.

    Applies the policies first if the frame does not already carry the
    ``label_*`` columns, so it works on a raw join as well as on
    ``metadata.parquet``. Rows whose policy label is NaN are dropped; that is
    the whole content of ``drop_conflicting`` and, for every policy, of a
    missing raw label.
    """
    policy = LabelPolicy(policy)
    col = policy.column
    if col not in df.columns:
        df = apply_all_policies(df)
    kept = df[df[col].notna()].copy()
    kept["label"] = kept[col].astype(int)
    return kept


def policy_costs(df: pd.DataFrame) -> pd.DataFrame:
    """The S0.2 table: per policy, positive exams, exams kept, prevalence.

    Computed over CODE rows only, since the table in S0.2_LABEL_POLICY.md is.
    Index is the policy value; columns ``positive_exams``, ``exams_kept``,
    ``positive_patients``, ``prevalence``.
    """
    if not all(p.column in df.columns for p in LabelPolicy):
        df = apply_all_policies(df)
    code = df[df["source"] == CODE_SOURCE]
    rows = {}
    for policy in LabelPolicy:
        kept = code[code[policy.column].notna()]
        pos = kept[policy.column] == 1
        rows[policy.value] = {
            "positive_exams": int(pos.sum()),
            "exams_kept": int(len(kept)),
            "positive_patients": int(kept.loc[pos, "patient_key"].nunique()),
            "prevalence": float(pos.mean()) if len(kept) else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
