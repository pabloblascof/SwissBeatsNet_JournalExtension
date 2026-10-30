"""S0.7 -- data inventory, the metadata join, and version manifests.

Two jobs, and the second is the one that matters.

*Inventory.* Hash and record every source file so a result can be traced to the
bytes it came from.

*The join.* The WFDB headers written by ``prepare_*_data.py`` carry only age,
sex, Chagas label and source. ``patient_id`` and the six conduction labels live
in ``exams.csv`` and nowhere else. Without the first there are no
patient-disjoint splits (S0.2); without the second there is no near-normal
proxy (S1.3). This module builds and checks that join, and refuses to report a
partial one silently -- a join that quietly drops records would shrink and bias
P2, the paper's most load-bearing instrument.

Usage::

    python -m src.data.inventory --root /data/rbg/shared/datasets/Chagas
    python -m src.data.inventory --root ... --full-hash   # hash the HDF5 parts too
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

#: The six conduction abnormalities carried by CODE's exams.csv. Derived by
#: automatic annotation and report text-mining, NOT cardiologist adjudication
#: -- see PLAN.md S1.3 on why that provenance compounds with the self-reported
#: Chagas labels.
SIX_ABNORMALITIES = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")

#: Files small enough to hash in full without thinking about it.
SMALL_FILE_BYTES = 256 * 1024 * 1024


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    sha256: str | None
    hash_scope: str  # "full" | "head+tail" -- never leave this ambiguous

    @property
    def size_human(self) -> str:
        n = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}TB"


def hash_file(path: Path, *, full: bool, chunk: int = 8 << 20) -> FileRecord:
    """Hash ``path``.

    Large files are fingerprinted by first and last 64 MiB plus exact size
    unless ``full`` is set. The scope is recorded alongside the digest so a
    ``head+tail`` fingerprint is never mistaken for a content hash later.
    """
    size = path.stat().st_size
    h = hashlib.sha256()

    if full or size <= SMALL_FILE_BYTES:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        scope = "full"
    else:
        window = 64 << 20
        with open(path, "rb") as fh:
            h.update(fh.read(window))
            fh.seek(max(0, size - window))
            h.update(fh.read(window))
        h.update(str(size).encode())
        scope = "head+tail"

    return FileRecord(str(path), size, h.hexdigest(), scope)


def _read_labels(root: Path) -> pd.DataFrame:
    """Read code15_chagas_labels.csv, from the zip if it has not been extracted."""
    csv = root / "code15_chagas_labels.csv"
    if csv.exists():
        return pd.read_csv(csv)
    zipped = root / "code15_chagas_labels.zip"
    if zipped.exists():
        return pd.read_csv(zipped, compression="zip")
    raise FileNotFoundError(
        f"no code15_chagas_labels.csv or .zip under {root}; it is a required "
        "-l argument to prepare_code15_data.py"
    )


def build_join(root: Path) -> dict[str, Any]:
    """Join ``exams.csv`` to the Chagas labels on ``exam_id`` and characterise it.

    Returns the report; raises nothing on a partial join but records the
    shortfall explicitly, because the number is itself a result (S0.7).
    """
    exams = pd.read_csv(root / "exams.csv")
    labels = _read_labels(root)

    report: dict[str, Any] = {}

    # --- integrity of the key ------------------------------------------------
    report["exams_rows"] = int(len(exams))
    report["exams_unique_exam_id"] = int(exams.exam_id.nunique())
    report["exam_id_is_unique"] = bool(exams.exam_id.is_unique)
    report["labels_rows"] = int(len(labels))
    report["labels_unique_exam_id"] = int(labels.exam_id.nunique())

    merged = exams.merge(
        labels[["exam_id", "chagas"]], on="exam_id", how="outer", indicator=True
    )
    counts = merged["_merge"].value_counts()
    report["join_both"] = int(counts.get("both", 0))
    report["join_exams_only"] = int(counts.get("left_only", 0))
    report["join_labels_only"] = int(counts.get("right_only", 0))
    report["join_coverage"] = report["join_both"] / max(1, report["exams_rows"])

    # patient_id appears in both files; disagreement would mean the key is
    # not what we think it is.
    if "patient_id_x" in merged or "patient_id" in labels:
        chk = exams.merge(labels, on="exam_id", how="inner", suffixes=("_ex", "_lb"))
        if "patient_id_lb" in chk:
            report["patient_id_agrees_across_files"] = bool(
                (chk.patient_id_ex == chk.patient_id_lb).all()
            )

    df = exams.merge(labels[["exam_id", "chagas"]], on="exam_id", how="inner")

    # --- patients (S0.2) -----------------------------------------------------
    report["unique_patients_all_exams"] = int(exams.patient_id.nunique())
    report["unique_patients_labelled"] = int(df.patient_id.nunique())
    report["max_exams_per_patient"] = int(exams.patient_id.value_counts().max())
    report["mean_exams_per_patient"] = float(
        len(exams) / exams.patient_id.nunique()
    )

    report["positive_exams"] = int(df.chagas.sum())
    report["exam_level_prevalence"] = float(df.chagas.mean())

    grouped = df.groupby("patient_id")["chagas"]
    per_patient = grouped.agg(["nunique", "size", "sum"])
    conflicting = per_patient[per_patient["nunique"] > 1]

    report["conflicting_label_patients"] = int(len(conflicting))
    report["conflicting_label_exams"] = int(conflicting["size"].sum())

    consistent = per_patient[per_patient["nunique"] == 1]
    report["all_positive_patients"] = int((consistent["sum"] > 0).sum())
    report["all_negative_patients"] = int((consistent["sum"] == 0).sum())

    # The three S0.2 options, costed. A human picks; this only frames it.
    report["label_policy_options"] = {
        "majority": int((grouped.mean() > 0.5).sum()),
        "any_positive": int((grouped.mean() > 0).sum()),
        "drop_conflicting": int((consistent["sum"] > 0).sum()),
    }

    # --- the P2 partition (S1.3) --------------------------------------------
    n_abn = df[list(SIX_ABNORMALITIES)].sum(axis=1)
    near_normal_six = n_abn == 0
    near_normal_flag = df["normal_ecg"].astype(bool)

    pos = df.chagas.astype(bool)
    report["p2"] = {
        "six_label": {
            "positives_near_normal": int(near_normal_six[pos].sum()),
            "positives_abnormal": int((~near_normal_six[pos]).sum()),
            "negatives_near_normal": int(near_normal_six[~pos].sum()),
        },
        "normal_ecg": {
            "positives_near_normal": int(near_normal_flag[pos].sum()),
            "positives_abnormal": int((~near_normal_flag[pos]).sum()),
            "negatives_near_normal": int(near_normal_flag[~pos].sum()),
        },
        "definitions_agree_overall": float((near_normal_six == near_normal_flag).mean()),
        "definitions_agree_positives": float(
            (near_normal_six[pos] == near_normal_flag[pos]).mean()
        ),
        # normal_ecg is expected to imply "none of the six". If this is False
        # the two definitions are not nested and S1.3's robustness check means
        # something different -- so it is asserted, not assumed.
        "normal_ecg_implies_no_six_abnormalities": bool(
            (~near_normal_six & near_normal_flag).sum() == 0
        ),
    }

    report["abnormality_prevalence"] = {
        c: {
            "positives": float(df.loc[pos, c].mean()),
            "negatives": float(df.loc[~pos, c].mean()),
        }
        for c in SIX_ABNORMALITIES
    }

    report["demographics"] = {
        "age_mean_positive": float(df.loc[pos, "age"].mean()),
        "age_mean_negative": float(df.loc[~pos, "age"].mean()),
        "male_frac_positive": float(df.loc[pos, "is_male"].mean()),
        "male_frac_negative": float(df.loc[~pos, "is_male"].mean()),
    }

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="directory holding exams.csv, the exams_part*.zip/hdf5 and the labels")
    ap.add_argument("--out", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--full-hash", action="store_true",
                    help="hash large files in full instead of head+tail fingerprinting")
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip hashing entirely; build the join report only")
    args = ap.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    if not args.skip_hash:
        for p in sorted(root.iterdir()):
            if p.is_file():
                rec = hash_file(p, full=args.full_hash)
                files.append(asdict(rec))
                print(f"  {rec.size_human:>9}  {rec.hash_scope:<9}  {p.name}")

    print("\nbuilding the exam_id join ...")
    report = build_join(root)

    manifest = {
        "source_root": str(root),
        "files": files,
        "join": report,
        "notes": [
            "prepare_*_data.py writes only age, sex, Chagas label and source into "
            "the WFDB header; patient_id and the six conduction labels survive "
            "only in exams.csv. Do not delete the source CSVs (project rule 10).",
        ],
    }

    out = args.out / "code15.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    j = report
    print(f"\nwrote {out}")
    print(f"  exams                     {j['exams_rows']:,}")
    print(f"  labelled (join coverage)  {j['join_both']:,} "
          f"({100 * j['join_coverage']:.2f}%)")
    print(f"  UNLABELLED                {j['join_exams_only']:,}")
    print(f"  patients                  {j['unique_patients_all_exams']:,}")
    print(f"  exam-level prevalence     {100 * j['exam_level_prevalence']:.3f}%")
    print(f"  conflicting-label patients {j['conflicting_label_patients']:,}  "
          f"<- S0.2 needs a human decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
