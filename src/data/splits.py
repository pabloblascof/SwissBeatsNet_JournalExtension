"""S0.2/S0.7 -- the frozen, patient-disjoint nested CV split.

What. ``configs/splits/frozen_splits.json`` assigns every ``patient_key`` in
``metadata.parquet`` to one of 5 outer folds and one of 4 inner folds::

    {"version": 1, "seed": 20260923, "n_outer": 5, "n_inner": 4,
     "stratify": ["source", "ever_positive", "age_band", "sex"],
     "sha256_of_patient_table": "...",
     "outer": {"<patient_key>": k}, "inner": {"<patient_key>": j}}

Outer fold ``k`` is a reporting fold. Its training pool is every patient with
``outer != k``; ``inner`` partitions that pool into 4 selection folds. The
inner assignment is a single stratified 4-fold over *all* patients, drawn
with a seed derived from the outer seed, so that for every ``k`` the pool's
inner folds are the restriction of one global assignment. That keeps the file
small and makes "inner fold j of outer fold k" well defined without storing
20 lists.

Why patients, why these strata. The unit is the patient, never the record:
CODE-15% has up to 38 exams per patient and a record-level split would put
the same heart on both sides of the line (the anchor did exactly this,
HANDOFF.md section 2). Stratification is on variables that do not depend on
the S0.2 label-policy decision -- ``ever_positive`` is the same under every
policy -- so the file can be frozen before that decision and never has to
change afterwards.

Small strata. A stratum with fewer patients than folds cannot be spread over
every fold. Any full stratum ``(source, ever_positive, age_band, sex)`` with
fewer than ``max(n_outer, n_inner)`` patients is merged into the fallback
stratum ``(source, ever_positive)``; if that is still too small, into
``(source,)``; and finally into a single global stratum. Which strata were
merged is recorded in the JSON under ``merged_strata``.

Immutability (project rule 3). The builder never overwrites an existing
split file, and writing one at all requires ``--force-new-file-i-understand``.
Changing the file invalidates every result computed on it. The CLI has no
``--seed``: the contract fixes it at ``SEED`` and a file drawn with another
seed would be a different split wearing the same name. ``build_splits`` keeps
a ``seed`` parameter for tests only.

File size. Two maps over every patient key (~254k at full scale) are written
with ``indent=None`` and ``sort_keys=True``; that is roughly 6-7 MB under
``configs/`` rather than ~13 MB with indentation, and still a stable diff.

Usage::

    python -m src.data.splits --metadata data/cache/metadata.parquet \\
        --out configs/splits/frozen_splits.json --force-new-file-i-understand
    python -m src.data.splits --check --metadata data/cache/metadata.parquet \\
        --splits configs/splits/frozen_splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.labels import LabelPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SPLITS = REPO_ROOT / "configs" / "splits" / "frozen_splits.json"

SEED = 20260923
N_OUTER = 5
N_INNER = 4
STRATIFY = ["source", "ever_positive", "age_band", "sex"]
VERSION = 1

#: Columns of the per-patient table; the sha256 in the JSON is over these.
PATIENT_COLUMNS = ["patient_key", *STRATIFY]


class SplitFileExists(FileExistsError):
    """Raised on any attempt to write over an existing split file."""


def derived_seed(seed: int, tag: str) -> int:
    """A second seed deterministically derived from ``seed``; used for inner."""
    h = hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()
    return int(h[:8], 16)


def patient_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per ``patient_key`` with the stratification variables.

    ``ever_positive`` is the patient-level max of ``chagas_raw``; ``age_band``
    and ``sex`` are taken from the patient's first record, defined as the one
    with the lowest ``exam_id`` (the only ordering available for every
    corpus; CODE exam_ids are not guaranteed chronological). Sorted by
    ``patient_key``.
    """
    needed = {"patient_key", "source", "chagas_raw", "age_band", "sex", "exam_id"}
    missing = needed - set(meta.columns)
    if missing:
        raise KeyError(f"metadata lacks {sorted(missing)}")
    first = (
        meta.sort_values(["patient_key", "exam_id"], kind="mergesort")
        .drop_duplicates("patient_key", keep="first")
        .set_index("patient_key")
    )
    ever = meta.groupby("patient_key")["chagas_raw"].max().astype(int)
    multi_source = meta.groupby("patient_key")["source"].nunique()
    if (multi_source > 1).any():
        raise ValueError(
            f"{int((multi_source > 1).sum())} patient_key(s) span more than one source"
        )
    out = pd.DataFrame(
        {
            "patient_key": first.index,
            "source": first["source"].values,
            "ever_positive": ever.loc[first.index].values,
            "age_band": first["age_band"].fillna("unknown").astype(str).values,
            "sex": first["sex"].fillna("unknown").astype(str).values,
        }
    )
    return out.sort_values("patient_key", kind="mergesort").reset_index(drop=True)


def sha256_of_patient_table(patients: pd.DataFrame) -> str:
    """sha256 of the sorted patient table as CSV (no index), the freeze fingerprint."""
    p = patients[PATIENT_COLUMNS].sort_values("patient_key", kind="mergesort")
    return hashlib.sha256(p.to_csv(index=False).encode("utf-8")).hexdigest()


def strata(patients: pd.DataFrame, min_count: int) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Stratum label per patient, with small strata merged (see module docstring).

    Returns the label series and a log of the merges performed.
    """
    levels = [STRATIFY, STRATIFY[:2], STRATIFY[:1], []]

    def key(cols: list[str]) -> pd.Series:
        if not cols:
            return pd.Series("all", index=patients.index)
        return patients[cols].astype(str).agg("|".join, axis=1)

    label = key(levels[0])
    merged: list[dict[str, Any]] = []
    for coarser in levels[1:]:
        counts = label.value_counts()
        small = counts[counts < min_count]
        if small.empty:
            break
        fallback = key(coarser)
        # Collapse the *whole* fallback group that each small stratum belongs
        # to, not just the small members: the fallback stratum is defined as
        # every patient sharing the coarser key, otherwise it could stay small.
        affected = set(fallback[label.isin(small.index)])
        for s, n in small.items():
            merged.append({"stratum": s, "n": int(n),
                           "merged_into": str(fallback[label == s].iloc[0])})
        label = label.where(~fallback.isin(affected), fallback)
    return label, merged


def build_splits(
    patients: pd.DataFrame,
    seed: int = SEED,
    n_outer: int = N_OUTER,
    n_inner: int = N_INNER,
) -> dict[str, Any]:
    """Assign outer and inner folds to every patient; return the JSON payload.

    Deterministic in ``patients`` (sorted by ``patient_key`` first) and the
    seeds. ``StratifiedKFold(shuffle=True)`` is used for both; the inner
    assignment uses :func:`derived_seed`.
    """
    patients = patients.sort_values("patient_key", kind="mergesort").reset_index(drop=True)
    if not patients["patient_key"].is_unique:
        raise ValueError("patient table has duplicate patient_key")
    labels, merged = strata(patients, min_count=max(n_outer, n_inner))
    dummy = np.zeros(len(patients))

    outer = np.full(len(patients), -1, dtype=int)
    for k, (_, test_idx) in enumerate(
        StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(dummy, labels)
    ):
        outer[test_idx] = k

    inner_seed = derived_seed(seed, "inner")
    inner = np.full(len(patients), -1, dtype=int)
    for j, (_, test_idx) in enumerate(
        StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed).split(dummy, labels)
    ):
        inner[test_idx] = j

    assert (outer >= 0).all() and (inner >= 0).all()
    keys = patients["patient_key"].tolist()
    return {
        "version": VERSION,
        "seed": seed,
        "inner_seed": inner_seed,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "stratify": list(STRATIFY),
        "n_patients": len(keys),
        "merged_strata": merged,
        "sha256_of_patient_table": sha256_of_patient_table(patients),
        "outer": dict(zip(keys, outer.tolist())),
        "inner": dict(zip(keys, inner.tolist())),
    }


def write_splits(splits: dict[str, Any], out: str | Path, force_new_file: bool = False) -> Path:
    """Write the split file. Refuses if ``out`` exists or ``force_new_file`` is False.

    Both conditions are deliberate: there is no code path that replaces a
    split file, and creating one is a freeze that has to be asked for by name.
    """
    out = Path(out)
    if out.exists():
        raise SplitFileExists(
            f"{out} already exists and split files are immutable (project rule 3). "
            "If you believe it is wrong, stop and write to DECISIONS.md."
        )
    if not force_new_file:
        raise PermissionError(
            f"refusing to create {out}: pass --force-new-file-i-understand to freeze a split file"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=None, sort_keys=True), encoding="utf-8")
    return out


def load_splits(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_disjoint(meta: pd.DataFrame) -> None:
    """Assert the fold structure on a metadata frame with folds attached.

    Every patient_key has exactly one outer and one inner fold across all of
    its records; no patient_key appears in two outer folds; every assigned
    fold index is within range. Records with ``outer_fold == -1`` are
    unassigned and ignored, but a patient may not be part-assigned.
    """
    for col in ("outer_fold", "inner_fold"):
        if col not in meta.columns:
            raise KeyError(f"metadata lacks {col}; call attach_splits first")
    per_patient = meta.groupby("patient_key")[["outer_fold", "inner_fold"]].nunique()
    bad = per_patient[(per_patient > 1).any(axis=1)]
    if not bad.empty:
        raise AssertionError(
            f"{len(bad)} patient(s) have records in more than one fold, e.g. "
            f"{bad.head(5).index.tolist()}"
        )
    assigned = meta[meta["outer_fold"] >= 0]
    if assigned.empty:
        return
    if (assigned["inner_fold"] < 0).any():
        raise AssertionError("records with an outer fold but no inner fold")
    for k in assigned["outer_fold"].unique():
        others = assigned.loc[assigned["outer_fold"] != k, "patient_key"]
        overlap = set(others) & set(assigned.loc[assigned["outer_fold"] == k, "patient_key"])
        if overlap:
            raise AssertionError(f"patient_key(s) in outer fold {k} and another: {sorted(overlap)[:5]}")


def fold_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-outer-fold counts by source: patients, records, raw positives and
    positive records under each label policy (NaN-dropped rows excluded)."""
    rows = []
    for (k, source), sub in meta.groupby(["outer_fold", "source"]):
        row = {
            "outer_fold": int(k),
            "source": source,
            "patients": int(sub["patient_key"].nunique()),
            "records": int(len(sub)),
            "raw_pos": int(sub["chagas_raw"].sum()),
        }
        for policy in LabelPolicy:
            col = policy.column
            if col in sub.columns:
                row[f"pos_{policy.value}"] = int((sub[col] == 1).sum())
                row[f"kept_{policy.value}"] = int(sub[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=FROZEN_SPLITS)
    ap.add_argument("--splits", type=Path, default=FROZEN_SPLITS, help="(--check) the split file")
    ap.add_argument("--check", action="store_true", help="verify and report instead of building")
    ap.add_argument("--force-new-file-i-understand", action="store_true",
                    help="required to create a split file; never overwrites")
    args = ap.parse_args(argv)

    from src.data.metadata import attach_splits, load_metadata

    meta = load_metadata(args.metadata)

    if args.check:
        splits = load_splits(args.splits)
        patients = patient_table(meta)
        digest = sha256_of_patient_table(patients)
        same = digest == splits.get("sha256_of_patient_table")
        print(f"patient table sha256 {'matches' if same else 'DIFFERS FROM'} {args.splits}")
        with_folds = attach_splits(meta, splits)
        check_disjoint(with_folds)
        n_unassigned = int((with_folds["outer_fold"] < 0).sum())
        print(f"disjointness OK; {n_unassigned:,} record(s) unassigned (-1)")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(fold_report(with_folds).to_string(index=False))
        return 0 if same else 2

    patients = patient_table(meta)
    splits = build_splits(patients, seed=SEED)
    try:
        write_splits(splits, args.out, force_new_file=args.force_new_file_i_understand)
    except (SplitFileExists, PermissionError) as e:
        print(f"REFUSED: {e}")
        return 1
    with_folds = attach_splits(meta, splits)
    check_disjoint(with_folds)
    print(f"wrote {args.out}: {splits['n_patients']:,} patients, "
          f"{len(splits['merged_strata'])} small strata merged")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(fold_report(with_folds).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
