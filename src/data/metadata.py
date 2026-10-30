"""S0.7 -- the per-record metadata table, ``data/cache/metadata.parquet``.

What. One row per converted WFDB record across the three training corpora,
with everything a run needs to know about a record without opening its
signal: identity (``record``, ``source``, ``exam_id``, ``patient_key``),
demographics, the raw and policy-resolved Chagas labels, the six conduction
abnormalities and ``normal_ecg`` for the P2 near-normal proxy, acquisition
parameters from the header, and the outer/inner fold. Columns are exactly the
ARCHITECTURE.md contract; row order is sorted by ``(source, record)`` and is
the row order of every signal and feature cache written afterwards.

Why a separate table. The organisers' converters write only age, sex, Chagas
label and source into the header. ``patient_id`` -- without which there is no
patient-disjoint split -- and the six conduction labels live in the source
CSVs. This module is the one place that join happens; consumers read the
parquet and never touch the CSVs.

Header parsing uses the official ``helper_code.py`` functions
(``get_sampling_frequency``, ``get_num_samples``, ``get_age``, ``get_sex``,
``get_source``, ``get_label``), loaded by path so the patched copy (whose
``compute_auc`` returns zeros) can never be picked up by name. Parsing ~370k
headers is parallelised with a process pool. A record is included only if its
header *and* its signal file(s) exist, so the table can be built from a tree
that is still being converted without admitting half-written records.

Label provenance. CODE's six abnormalities come from ``exams.csv`` and are
text-mined from reports plus automatic annotation, not cardiologist
adjudication. PTB-XL's come from ``scp_codes`` in ``ptbxl_database.csv``,
which are cardiologist-annotated SCP statements. The ``label_provenance``
column records which. The mapping is 1AVB->1dAVb, CRBBB->RBBB, CLBBB->LBBB,
SBRAD->SB, STACH->ST, AFIB->AF, NORM->normal_ecg. A code counts as present
when it is *listed* in ``scp_codes`` (``code_present``, the default). PTB-XL
records a likelihood of 0 when *no likelihood was given*, which is how rhythm
statements are usually stored (SBRAD: 637 listed, 0 above 0; STACH: 826
listed, 4 above 0; AFIB: 1,514 listed, 48 above 0), so the older
``likelihood_gt_0`` rule would set SB/ST/AF to 0 for ~2,900 records and call
them near-normal, corrupting P2 and P5 for the whole negative class. It stays
available as an explicit opt-in (``--ptbxl-presence likelihood_gt_0``) and
which rule was used is recorded in ``label_provenance``.

Header hygiene. ``get_label`` returns NaN for a present-but-unrecognised
``# Chagas label:`` value (``Unknown``, ``nan``, a typo) and ``bool(nan)`` is
True, so the label is checked to be exactly 0 or 1 and anything else is an
error naming the header. Every parse failure is re-raised with the header
path so a 370k-file run says *which* file broke. ``record`` must start with
the folder its source lives in (``code_15``, ``samitrop``, ``ptbxl``), which
is only true when ``--wfdb`` is the ``data/wfdb`` root; a sub-folder root
would write records that ``src/data/signals.py`` resolves to the wrong files.

Usage::

    python -m src.data.metadata --wfdb data/wfdb \\
        --source /data/rbg/shared/datasets/Chagas \\
        --ptbxl-db data/raw/ptbxl/ptbxl_database.csv \\
        --out data/cache/metadata.parquet --jobs 40
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.labels import CODE_SOURCE, LabelPolicy, apply_all_policies
from src.eval.metrics import OFFICIAL_HELPER, _load_isolated

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMITROP_SOURCE = "SaMi-Trop"
PTBXL_SOURCE = "PTB-XL"

#: Source string -> patient_key prefix (ARCHITECTURE.md).
KEY_PREFIX = {CODE_SOURCE: "code15", SAMITROP_SOURCE: "samitrop", PTBXL_SOURCE: "ptbxl"}

#: The six conduction abnormalities, ST before AF (D-006).
SIX = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")

#: PTB-XL SCP statement -> CODE column.
PTBXL_SCP_MAP = {
    "1AVB": "1dAVb", "CRBBB": "RBBB", "CLBBB": "LBBB",
    "SBRAD": "SB", "STACH": "ST", "AFIB": "AF",
}
PTBXL_NORMAL_CODE = "NORM"

#: How a PTB-XL SCP code counts as present. ``code_present`` (default) treats
#: any listed code as present; ``likelihood_gt_0`` is an explicit opt-in.
PTBXL_PRESENCE_RULES = ("code_present", "likelihood_gt_0")
PTBXL_PRESENCE_DEFAULT = "code_present"

#: Source string -> the folder under data/wfdb its records must live in.
SOURCE_FOLDER = {CODE_SOURCE: "code_15", SAMITROP_SOURCE: "samitrop", PTBXL_SOURCE: "ptbxl"}

#: The contract's column order. Nothing else is written.
COLUMNS = [
    "record", "source", "exam_id", "patient_key", "age", "sex", "chagas_raw",
    "ever_positive",
    "label_any_positive", "label_majority_pos", "label_majority_neg",
    "label_drop_conflicting", "label_exam_level",
    *SIX, "normal_ecg", "n_abnormal", "near_normal_six", "label_provenance",
    "fs", "n_samples", "duration_s", "age_band", "outer_fold", "inner_fold",
]

AGE_BANDS = ("<40", "40-60", ">60")

# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------

_helper = None


def official_helper():
    """The official helper module, loaded by path (never ``helper_code_patched``)."""
    global _helper
    if _helper is None:
        _helper = _load_isolated(OFFICIAL_HELPER, "_sbn_official_helper_metadata")
    return _helper


def parse_header(hea_path: str | Path, wfdb_root: str | Path) -> dict[str, Any] | None:
    """Parse one ``.hea`` with the official helper functions.

    Returns None when any signal file the header names is missing, which is
    what a partially converted tree looks like. ``n_samples`` is whatever the
    header says: the CODE and SaMi-Trop converters trim leading/trailing
    all-zero rows before writing, so the header count is the post-trim count.

    Raises ``ValueError`` naming the header when the Chagas label is absent
    or not exactly 0/1 (``sanitize_boolean_value`` returns NaN for an
    unrecognised value and ``bool(nan)`` would have made it a positive), and
    re-raises any other parse failure with the header path prepended.
    """
    hea_path = Path(hea_path)
    try:
        return _parse_header_unchecked(hea_path, wfdb_root)
    except Exception as e:  # noqa: BLE001 - the point is to name the file
        raise ValueError(f"{hea_path}: {e}") from e


def _parse_header_unchecked(hea_path: Path, wfdb_root: str | Path) -> dict[str, Any] | None:
    h = official_helper()
    stem = str(hea_path.with_suffix(""))
    for sig in h.get_signal_files(stem):
        if not os.path.exists(sig):
            return None
    text = h.load_text(str(hea_path))
    record = os.path.relpath(stem, wfdb_root)
    age = h.get_age(text)
    label = h.get_label(text)
    if label not in (0, 1):
        raise ValueError(f"unparseable '# Chagas label:' value -> {label!r}; expected True/False")
    return {
        "record": record,
        "source": h.get_source(text),
        "age": float(age) if isinstance(age, (int, float)) else np.nan,
        "sex": h.get_sex(text),
        "chagas_raw": int(label),
        "fs": h.get_sampling_frequency(text),
        "n_samples": h.get_num_samples(text),
    }


def _parse_one(args: tuple[str, str]) -> dict[str, Any] | None:
    return parse_header(*args)


def find_headers(wfdb_root: str | Path) -> list[str]:
    """All ``.hea`` paths under ``wfdb_root``, sorted."""
    out = []
    for root, _dirs, files in os.walk(wfdb_root):
        out.extend(os.path.join(root, f) for f in files if f.endswith(".hea"))
    return sorted(out)


def scan_wfdb(wfdb_root: str | Path, jobs: int = 1, chunksize: int = 256) -> pd.DataFrame:
    """Parse every complete record under ``wfdb_root`` into a frame.

    ``jobs`` > 1 uses a process pool; header parsing is I/O bound on a
    network filesystem and 40 workers cover ~370k files in minutes.
    """
    wfdb_root = str(wfdb_root)
    headers = find_headers(wfdb_root)
    work = [(p, wfdb_root) for p in headers]
    if jobs > 1 and len(work) > chunksize:
        with Pool(jobs) as pool:
            rows = list(pool.imap_unordered(_parse_one, work, chunksize=chunksize))
    else:
        rows = [_parse_one(w) for w in work]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows, columns=["record", "source", "age", "sex", "chagas_raw", "fs", "n_samples"])
    n_skipped = len(headers) - len(df)
    if n_skipped:
        print(f"  skipped {n_skipped:,} header(s) whose signal file is not (yet) on disk")
    return df


# --------------------------------------------------------------------------
# Per-source joins
# --------------------------------------------------------------------------


def exam_id_from_record(record: str, source: str) -> int:
    """CODE/SaMi: the filename is the exam_id. PTB-XL: ``00001_hr`` -> ecg_id 1."""
    name = os.path.basename(record)
    if source == PTBXL_SOURCE:
        name = name.split("_")[0]
    return int(name)


def load_code_source(source_dir: str | Path) -> pd.DataFrame:
    """``exams.csv`` joined to the Chagas labels: patient_id, six, normal_ecg."""
    source_dir = Path(source_dir)
    exams = pd.read_csv(source_dir / "exams.csv")
    keep = ["exam_id", "patient_id", *SIX, "normal_ecg"]
    df = exams[keep].copy()
    for c in (*SIX, "normal_ecg"):
        df[c] = df[c].astype(float)
    return df


def load_samitrop_source(source_dir: str | Path) -> pd.DataFrame:
    """``exams_samitrop.csv``: normal_ecg only; SaMi-Trop has no abnormality labels."""
    df = pd.read_csv(Path(source_dir) / "exams_samitrop.csv")
    out = df[["exam_id", "normal_ecg"]].copy()
    out["normal_ecg"] = out["normal_ecg"].astype(float)
    return out


def parse_scp_codes(scp: str, presence: str = PTBXL_PRESENCE_DEFAULT) -> dict[str, float]:
    """Derive the six abnormalities and ``normal_ecg`` from one ``scp_codes`` string.

    ``scp_codes`` is a python-dict literal ``{'NORM': 100.0, 'SR': 0.0}``
    mapping SCP statement to likelihood. Under ``code_present`` (the default)
    a code is present iff listed at all; under ``likelihood_gt_0`` iff listed
    with likelihood > 0. See the module docstring for why the difference
    matters for the rhythm codes.
    """
    if presence not in PTBXL_PRESENCE_RULES:
        raise ValueError(f"presence must be one of {PTBXL_PRESENCE_RULES}, got {presence!r}")
    codes = ast.literal_eval(scp) if isinstance(scp, str) else {}
    if not isinstance(codes, dict):
        raise ValueError(f"scp_codes did not parse to a dict: {scp!r}")

    def present(code: str) -> float:
        if code not in codes:
            return 0.0
        if presence == "code_present":
            return 1.0
        return float(float(codes[code]) > 0)

    out = {col: present(code) for code, col in PTBXL_SCP_MAP.items()}
    out["normal_ecg"] = present(PTBXL_NORMAL_CODE)
    return out


def load_ptbxl_source(db_csv: str | Path, presence: str = PTBXL_PRESENCE_DEFAULT) -> pd.DataFrame:
    """``ptbxl_database.csv``: ecg_id -> patient_id, six abnormalities, normal_ecg."""
    db = pd.read_csv(db_csv, usecols=["ecg_id", "patient_id", "scp_codes"])
    derived = pd.DataFrame([parse_scp_codes(s, presence) for s in db["scp_codes"]])
    out = pd.concat([db[["ecg_id", "patient_id"]].reset_index(drop=True), derived], axis=1)
    out = out.rename(columns={"ecg_id": "exam_id"})
    out["patient_id"] = out["patient_id"].astype(int)
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def check_record_prefixes(df: pd.DataFrame) -> None:
    """Every ``record`` must start with its source's folder (``SOURCE_FOLDER``).

    ``record`` is relative to the ``--wfdb`` root, and ``src/data/signals.py``
    resolves ``wfdb_root / record``. A scan rooted at ``data/wfdb/samitrop``
    would write ``100726`` instead of ``samitrop/100726`` and every later
    cache would point at the wrong files, so this is an error, not a warning.
    """
    first = df["record"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    expected = df["source"].map(SOURCE_FOLDER)
    bad = df[first != expected]
    if not bad.empty:
        raise ValueError(
            f"{len(bad):,} record(s) are not under their source folder "
            f"{SOURCE_FOLDER}, e.g. {bad['record'].head(3).tolist()} for "
            f"{bad['source'].head(3).tolist()}. --wfdb must be the data/wfdb root, "
            "not a per-source sub-folder."
        )


def age_band(age: pd.Series) -> pd.Series:
    """``<40``, ``40-60`` (inclusive both ends), ``>60``; NaN age -> NaN."""
    a = age.astype(float)
    band = pd.Series(np.nan, index=a.index, dtype=object)
    band[a < 40] = "<40"
    band[(a >= 40) & (a <= 60)] = "40-60"
    band[a > 60] = ">60"
    return band


def assemble(
    headers: pd.DataFrame,
    code: pd.DataFrame | None = None,
    samitrop: pd.DataFrame | None = None,
    ptbxl: pd.DataFrame | None = None,
    ptbxl_presence: str = PTBXL_PRESENCE_DEFAULT,
) -> pd.DataFrame:
    """Join parsed headers to the per-source tables and derive every column.

    Each source table is required only if the header frame has records from
    that source. A CODE or PTB-XL record whose exam_id is absent from its
    source table has no patient_id and therefore no patient_key; that is an
    error, not something to paper over, because an orphan record would leak
    across folds.
    """
    df = headers.copy()
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    unknown = set(df["source"]) - set(KEY_PREFIX)
    if unknown:
        raise ValueError(f"unknown # Source: values in headers: {sorted(unknown)}")
    check_record_prefixes(df)

    df["exam_id"] = [exam_id_from_record(r, s) for r, s in zip(df["record"], df["source"])]

    parts = []
    for source, table, name in (
        (CODE_SOURCE, code, "--source (exams.csv)"),
        (SAMITROP_SOURCE, samitrop, "--source (exams_samitrop.csv)"),
        (PTBXL_SOURCE, ptbxl, "--ptbxl-db"),
    ):
        sub = df[df["source"] == source].copy()
        if sub.empty:
            continue
        if table is None:
            raise ValueError(f"{len(sub):,} {source} records on disk but {name} was not given")
        if not table["exam_id"].is_unique:
            raise ValueError(f"{source} source table has duplicate exam_id")
        sub = sub.merge(table, on="exam_id", how="left", indicator=True)
        missing = sub["_merge"] != "both"
        if missing.any():
            raise ValueError(
                f"{int(missing.sum()):,} {source} records have no row in the source "
                f"table, e.g. {sub.loc[missing, 'record'].head(3).tolist()}"
            )
        sub = sub.drop(columns="_merge")

        if source == CODE_SOURCE:
            sub["patient_key"] = "code15:" + sub["patient_id"].astype(int).astype(str)
            sub["label_provenance"] = "code15_text_mined"
        elif source == SAMITROP_SOURCE:
            sub["patient_key"] = "samitrop:" + sub["exam_id"].astype(int).astype(str)
            sub["label_provenance"] = None
        else:
            sub["patient_key"] = "ptbxl:" + sub["patient_id"].astype(int).astype(str)
            sub["label_provenance"] = f"ptbxl_cardiologist_scp:{ptbxl_presence}"
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    for c in SIX + ("normal_ecg",):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype(float)

    df["ever_positive"] = df.groupby("patient_key")["chagas_raw"].transform("max").astype(int)
    df = apply_all_policies(df)

    six = df[list(SIX)]
    df["n_abnormal"] = six.sum(axis=1, min_count=len(SIX)).astype(float)
    df["near_normal_six"] = (df["n_abnormal"] == 0).astype(float).where(df["n_abnormal"].notna())

    df["fs"] = df["fs"].astype(int)
    df["n_samples"] = df["n_samples"].astype(int)
    df["duration_s"] = df["n_samples"] / df["fs"]
    df["age_band"] = age_band(df["age"])
    df["outer_fold"] = -1
    df["inner_fold"] = -1
    df["exam_id"] = df["exam_id"].astype(int)
    df["chagas_raw"] = df["chagas_raw"].astype(int)

    df = df.sort_values(["source", "record"], kind="mergesort").reset_index(drop=True)
    return df[COLUMNS]


def build_metadata(
    wfdb_root: str | Path,
    source_dir: str | Path | None,
    ptbxl_db: str | Path | None,
    jobs: int = 1,
    ptbxl_presence: str = PTBXL_PRESENCE_DEFAULT,
) -> pd.DataFrame:
    """Scan the tree, load whichever source tables exist, assemble."""
    headers = scan_wfdb(wfdb_root, jobs=jobs)
    present = set(headers["source"]) if not headers.empty else set()
    code = samitrop = ptbxl = None
    if source_dir is not None:
        source_dir = Path(source_dir)
        if CODE_SOURCE in present and (source_dir / "exams.csv").exists():
            code = load_code_source(source_dir)
        if SAMITROP_SOURCE in present and (source_dir / "exams_samitrop.csv").exists():
            samitrop = load_samitrop_source(source_dir)
    if ptbxl_db is not None and PTBXL_SOURCE in present and Path(ptbxl_db).exists():
        ptbxl = load_ptbxl_source(ptbxl_db, ptbxl_presence)
    return assemble(headers, code, samitrop, ptbxl, ptbxl_presence)


# --------------------------------------------------------------------------
# I/O and splits
# --------------------------------------------------------------------------


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read the parquet and verify the contract columns are all present, in order."""
    df = pd.read_parquet(path)
    if list(df.columns) != COLUMNS:
        raise ValueError(
            f"{path} does not match the metadata contract; "
            f"missing {sorted(set(COLUMNS) - set(df.columns))}, "
            f"extra {sorted(set(df.columns) - set(COLUMNS))}"
        )
    return df


def attach_splits(df: pd.DataFrame, splits_json: str | Path | dict) -> pd.DataFrame:
    """Fill ``outer_fold`` / ``inner_fold`` from the frozen split file.

    A ``patient_key`` absent from the file gets -1 in both, which is how a
    record converted after the freeze is kept out of every fold rather than
    silently landing in one.
    """
    if not isinstance(splits_json, dict):
        with open(splits_json, encoding="utf-8") as fh:
            splits_json = json.load(fh)
    out = df.copy()
    outer = splits_json.get("outer", {})
    inner = splits_json.get("inner", {})
    out["outer_fold"] = out["patient_key"].map(outer).fillna(-1).astype(int)
    out["inner_fold"] = out["patient_key"].map(inner).fillna(-1).astype(int)
    return out


def summarise(df: pd.DataFrame) -> str:
    lines = [f"  records        {len(df):,}", f"  patients       {df['patient_key'].nunique():,}"]
    for source, sub in df.groupby("source"):
        lines.append(
            f"  {source:<10} {len(sub):>8,} records  {sub['patient_key'].nunique():>8,} patients  "
            f"{int(sub['chagas_raw'].sum()):>6,} raw-positive  fs={sorted(int(v) for v in sub['fs'].unique())}  "
            f"dur={sub['duration_s'].min():.2f}-{sub['duration_s'].max():.2f}s"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wfdb", type=Path, required=True, help="root of the converted WFDB tree")
    ap.add_argument("--source", type=Path, default=None,
                    help="directory with exams.csv, code15_chagas_labels.zip, exams_samitrop.csv")
    ap.add_argument("--ptbxl-db", type=Path, default=None, help="ptbxl_database.csv")
    ap.add_argument("--out", type=Path, required=True, help="output parquet")
    ap.add_argument("--jobs", type=int, default=1, help="header-parsing worker processes")
    ap.add_argument("--splits", type=Path, default=None,
                    help="frozen_splits.json to attach (else folds are -1)")
    ap.add_argument("--ptbxl-presence", choices=PTBXL_PRESENCE_RULES, default=PTBXL_PRESENCE_DEFAULT,
                    help="how a PTB-XL scp code counts as present (see module docstring)")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = build_metadata(args.wfdb, args.source, args.ptbxl_db, jobs=args.jobs,
                        ptbxl_presence=args.ptbxl_presence)
    t1 = time.time()
    if args.splits is not None:
        df = attach_splits(df, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(df):,} rows, {len(COLUMNS)} columns) in {t1 - t0:.1f}s")
    print(summarise(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
