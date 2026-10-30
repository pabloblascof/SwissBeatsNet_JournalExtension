"""MIMIC-IV-ECG v1.0 -> the anchor's 250 Hz signal cache (S4 own encoder, arm M; D-048).

MIMIC-IV-ECG (PhysioNet, Open Access, ODbL v1.0, DOI 10.13026/4nqg-sb35) is
admitted **for self-supervised pretraining only** (D-048): no label, note or
machine measurement is read here, and the corpus is never an evaluation set.

Two steps, both resumable and read straight from the PhysioNet zip (no
1.6-million-file extraction onto NFS):

``index``
    Reads ``record_list.csv`` from the zip and writes
    ``data/cache/mimic/metadata.parquet``: one row per WFDB record whose
    ``.hea`` and ``.dat`` are both in the zip, sorted by path, with
    ``record`` (path inside the zip, no extension), ``subject_id`` and
    ``study_id``. Row *i* of the cache is row *i* of this parquet.

``build``
    A process pool; each worker opens the zip once, copies one record's
    ``.hea``/``.dat`` to a private directory under ``/dev/shm`` and runs
    **exactly the anchor preprocessing** on it,
    :func:`src.data.signals.preprocess_record` (``team.data.load_and_process_signal``
    with ``ANCHOR_PREPROC``: 500 Hz input resampled to 250 Hz, the anchor's
    cleaning, per-lead standardisation, first 2,250 samples / symmetric
    zero padding). The 12 columns are then put in the order every cache of
    this branch uses (I, II, III, aVR, aVL, aVF, V1-V6) by the header's
    signal names -- MIMIC-IV-ECG stores aVF before aVL. Every step of the
    preprocessing acts on each lead independently, so reordering after it is
    identical to reordering before. A record whose header does not name the
    twelve standard leads, or whose preprocessed signal is not finite (MIMIC
    marks missing samples as NaN), is written as zeros and marked failed
    (``2`` in the ``.done`` bitmap, ``failed_rows`` in the sidecar), exactly
    the contract of ``src/data/signals.py``, so pretraining never samples it.

The cache's sidecars are those of :mod:`src.data.signals` (``.json``,
``.done``, ``.failed.jsonl``), so ``src.objectives.data.SignalCacheDataset``
reads it unchanged (``data_source: external``).

CLI::

    python -m src.data.mimic index --zip /data/rbg/shared/datasets/MIMIC-IV-ECG/mimic-iv-ecg-1.0/mimic-iv-ecg-1.0.zip
    python -m src.data.mimic build --zip ... --jobs 24
    python -m src.data.mimic manifest --zip ...      # data/manifests/mimic_iv_ecg.json
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data import signals as S

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Datasets live in shared storage, never in $HOME (project rules). The repo path
#: ``data/raw/mimic-iv-ecg`` is kept only as a fallback symlink.
SHARED_ZIP = Path("/data/rbg/shared/datasets/MIMIC-IV-ECG/mimic-iv-ecg-1.0/mimic-iv-ecg-1.0.zip")
_REPO_ZIP = REPO_ROOT / "data" / "raw" / "mimic-iv-ecg" / "mimic-iv-ecg-1.0.zip"
DEFAULT_ZIP = SHARED_ZIP if SHARED_ZIP.exists() else _REPO_ZIP
CACHE_DIR = REPO_ROOT / "data" / "cache" / "mimic"
DEFAULT_METADATA = CACHE_DIR / "metadata.parquet"
DEFAULT_OUT = CACHE_DIR / "signals_250hz.f32"
MANIFEST = REPO_ROOT / "data" / "manifests" / "mimic_iv_ecg.json"

#: The column order of every signal cache on this branch.
CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

DATASET = {
    "name": "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset",
    "version": "1.0",
    "doi": "10.13026/4nqg-sb35",
    "url": "https://physionet.org/content/mimic-iv-ecg/1.0/",
    "access": "Open Access",
    "license": "Open Data Commons Open Database License v1.0 (ODbL)",
    "use": "self-supervised pretraining only (DECISIONS.md D-048); labels, notes and machine measurements unused",
}


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def _find_member(names: Sequence[str], suffix: str) -> str:
    hits = [n for n in names if n.endswith(suffix)]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one zip member ending in {suffix!r}, found {hits[:5]}")
    return hits[0]


SUMS_NAME = "SHA256SUMS.txt"


def load_sha256sums(zf: zipfile.ZipFile, root: str) -> dict[str, str]:
    """PhysioNet's ``SHA256SUMS.txt`` from the zip: ``{path inside the zip: sha256}``."""
    out = {}
    for line in zf.read(root + SUMS_NAME).decode().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out[root + rel.strip()] = h.lower()
    return out


def verify_member(name: str, data: bytes, sums: dict[str, str]) -> None:
    import hashlib

    want = sums.get(name)
    if want is None:
        raise ValueError(f"sha256 missing: {name} not listed in {SUMS_NAME}")
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"sha256 mismatch: {name} {got} != {want}")


def build_index(zip_path: Path | str = DEFAULT_ZIP, out: Path | str = DEFAULT_METADATA) -> dict[str, Any]:
    """``record_list.csv`` -> metadata parquet (records with both WFDB files only).

    Every non-waveform file used here (``record_list.csv``) is verified against
    PhysioNet's ``SHA256SUMS.txt``, and every file the checksum list names is
    checked to be present in the zip; the waveform files are verified by
    :func:`build_cache` as they are read.
    """
    import pandas as pd

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        rl = _find_member(names, "record_list.csv")
        root = rl[: -len("record_list.csv")]
        sums = load_sha256sums(zf, root)
        raw = zf.read(rl)
        verify_member(rl, raw, sums)
        df = pd.read_csv(io.BytesIO(raw))
    members = set(names)
    missing_listed = sorted(set(sums) - members)
    unlisted = sorted(n for n in members if not n.endswith("/") and n not in sums and not n.endswith(SUMS_NAME))
    df["record"] = [root + str(p).strip("/") for p in df["path"]]
    has = np.array([(r + ".hea") in members and (r + ".dat") in members for r in df["record"]])
    df = df.loc[has, ["record", "subject_id", "study_id"]].sort_values("record").reset_index(drop=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    info = {"n_record_list": int(len(has)), "n_indexed": int(len(df)), "n_missing_files": int((~has).sum()),
            "n_sha256sums": len(sums), "n_listed_but_missing_from_zip": len(missing_listed),
            "n_in_zip_but_unlisted": len(unlisted), "unlisted_examples": unlisted[:5],
            "record_list_sha256_verified": True,
            "n_subjects": int(df["subject_id"].nunique()), "zip_root": root, "metadata": str(out),
            "records_sha256": S.records_sha256(df["record"].tolist())}
    print(json.dumps(info, indent=1), flush=True)
    return info


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #

_W: dict[str, Any] = {}


def lead_permutation(sig_names: Sequence[str]) -> np.ndarray:
    """Column indices that put ``sig_names`` in :data:`CANONICAL_LEADS` order."""
    lookup = {str(n).strip().lower(): i for i, n in enumerate(sig_names)}
    if len(lookup) != len(sig_names) or len(sig_names) != len(CANONICAL_LEADS):
        raise ValueError(f"expected 12 distinct leads, got {list(sig_names)}")
    try:
        return np.array([lookup[c.lower()] for c in CANONICAL_LEADS], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"lead {e} missing from {list(sig_names)}") from None


def preprocess_wfdb(tmp_root: Path | str, name: str) -> np.ndarray:
    """The anchor preprocessing of ``tmp_root/name`` with canonical lead order; raises if not finite."""
    import wfdb

    hdr = wfdb.rdheader(os.path.join(str(tmp_root), name))
    perm = lead_permutation(hdr.sig_name)
    x = S.preprocess_record(tmp_root, name)[:, perm]
    if not np.isfinite(x).all():
        raise ValueError("non-finite samples after preprocessing")
    return np.ascontiguousarray(x, dtype=S.DTYPE)


#: ``SHA256SUMS.txt`` of the zip, set in the parent before the pool forks.
_SUMS: dict[str, str] = {}


def _init_worker(zip_path: str, out_path: str, n: int, shm_root: str) -> None:
    from threadpoolctl import threadpool_limits

    for v in S.THREAD_ENV:
        os.environ[v] = "1"
    _W["threads"] = threadpool_limits(limits=1)
    _W["zip"] = zipfile.ZipFile(zip_path)
    _W["mm"] = np.memmap(out_path, dtype=S.DTYPE, mode="r+", shape=(n, S.SEQ_LEN, S.N_LEADS))
    tmp = Path(shm_root) / f"w{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    _W["tmp"] = tmp
    S._team()


def _process_chunk(chunk: list[tuple[int, str]]) -> tuple[list[int], list[tuple[int, str, str]]]:
    zf: zipfile.ZipFile = _W["zip"]
    mm: np.memmap = _W["mm"]
    tmp: Path = _W["tmp"]
    ok, failed = [], []
    for row, record in chunk:
        name = record.rsplit("/", 1)[-1]
        try:
            for ext in (".hea", ".dat"):
                data = zf.read(record + ext)
                verify_member(record + ext, data, _SUMS)  # PhysioNet's checksum, every file
                (tmp / (name + ext)).write_bytes(data)
            mm[row] = preprocess_wfdb(tmp, name)
            ok.append(row)
        except Exception as e:  # noqa: BLE001 -- logged as a failed row, never silently
            mm[row] = 0.0
            failed.append((row, record, f"{type(e).__name__}: {e}"))
        finally:
            for ext in (".hea", ".dat"):
                (tmp / (name + ext)).unlink(missing_ok=True)
    mm.flush()
    return ok, failed


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_cache(zip_path: Path | str = DEFAULT_ZIP, metadata: Path | str = DEFAULT_METADATA,
                out_path: Path | str = DEFAULT_OUT, jobs: int = 16, chunk_size: int = 64, limit: int | None = None,
                log_every: int = 100, shm_root: Path | str | None = None) -> dict[str, Any]:
    """Resumable; same resume rules and sidecars as :func:`src.data.signals.build_signal_cache`."""
    out_path = Path(out_path)
    records = S.load_records(metadata, limit)
    n = len(records)
    rec_sha = S.records_sha256(records)
    meta_sha = S.sha256_of_file(metadata)
    meta_path = S._sidecar(out_path, ".json")
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior.get("records_sha256") != rec_sha or prior.get("n") != n:
            raise RuntimeError(f"{meta_path} was built for a different row order/size; refusing to resume")
    elif out_path.exists() or S._sidecar(out_path, ".done").exists():
        raise RuntimeError(f"{out_path} exists without {meta_path.name}; delete .f32/.done to rebuild")
    S._ensure_memmap(out_path, n)
    done = S.open_done_bitmap(out_path, n, mode="r+")
    todo = [(i, r) for i, r in enumerate(records) if done[i] == 0]
    kw = dict(n=n, rec_sha=rec_sha, meta_sha=meta_sha, metadata_parquet=metadata, wfdb_root=Path(str(zip_path)),
              limit=limit)
    S._write_meta(out_path, done, **kw)
    with zipfile.ZipFile(zip_path) as zf:
        root = _find_member(zf.namelist(), "record_list.csv")[: -len("record_list.csv")]
        _SUMS.clear()
        _SUMS.update(load_sha256sums(zf, root))
    shm = Path(shm_root) if shm_root else Path("/dev/shm") / f"mimic_build_{os.getpid()}"
    shm.mkdir(parents=True, exist_ok=True)
    failed_log = S._sidecar(out_path, ".failed.jsonl")
    t0 = time.time()
    n_ok = n_fail = 0
    print(f"[mimic] n={n} todo={len(todo)} jobs={jobs} out={out_path}", flush=True)
    try:
        if todo:
            ctx = mp.get_context("fork")
            jobs = max(1, min(int(jobs), len(todo)))
            with ctx.Pool(jobs, initializer=_init_worker, initargs=(str(zip_path), str(out_path), n, str(shm))) as pool, \
                    open(failed_log, "a") as flog:
                for k, (ok, failed) in enumerate(pool.imap(_process_chunk, S._chunks(todo, chunk_size)), start=1):
                    for row in ok:
                        done[row] = 1
                    for row, record, err in failed:
                        done[row] = 2
                        flog.write(json.dumps({"row": row, "record": record, "error": err}) + "\n")
                    n_ok += len(ok)
                    n_fail += len(failed)
                    if k % log_every == 0 or k * chunk_size >= len(todo):
                        done.flush()
                        flog.flush()
                        S._write_meta(out_path, done, **kw)
                        dt = time.time() - t0
                        print(f"[mimic] {n_ok + n_fail}/{len(todo)} ({(n_ok + n_fail) / max(dt, 1e-9):.1f} rec/s, "
                              f"failed={n_fail}, {dt / 60:.1f} min)", flush=True)
    finally:
        shutil.rmtree(shm, ignore_errors=True)
    done.flush()
    meta = S._write_meta(out_path, done, **kw)
    meta["source"] = "mimic-iv-ecg-1.0"
    meta["lead_order"] = list(CANONICAL_LEADS)
    S._sidecar(out_path, ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[mimic] done={meta['n_done']}/{n} failed={meta['n_failed']}", flush=True)
    return meta


def _count_failed(failed_log: Path, needle: str) -> int:
    if not failed_log.exists():
        return 0
    return sum(needle in json.loads(l)["error"] for l in failed_log.read_text().splitlines() if l.strip())


def _rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def write_manifest(zip_path: Path | str = DEFAULT_ZIP, metadata: Path | str = DEFAULT_METADATA,
                   out_path: Path | str = DEFAULT_OUT, manifest: Path | str = MANIFEST,
                   zip_sha256: str | None = None, cache_sha256: bool = True) -> dict[str, Any]:
    """``data/manifests/mimic_iv_ecg.json``: provenance of the zip, the index and the cache."""
    import pandas as pd

    meta = S.read_cache_meta(out_path)
    df = pd.read_parquet(metadata)
    failed_reasons: dict[str, int] = {}
    fl = S._sidecar(out_path, ".failed.jsonl")
    if fl.exists():
        for line in fl.read_text().splitlines():
            err = json.loads(line)["error"].split(":")[0]
            failed_reasons[err] = failed_reasons.get(err, 0) + 1
    m = {
        "dataset": DATASET,
        "zip": {"path": _rel(zip_path),
                "bytes": Path(zip_path).stat().st_size,
                "sha256": zip_sha256 or S.sha256_of_file(zip_path)},
        "index": {"path": _rel(metadata), "sha256": S.sha256_of_file(metadata), "n_records": int(len(df)),
                  "n_subjects": int(df["subject_id"].nunique()), "records_sha256": meta["records_sha256"],
                  "columns": list(df.columns)},
        "cache": {"path": _rel(out_path), "shape": meta["shape"], "dtype": meta["dtype"],
                  "preproc": meta["preproc"], "preproc_config": meta["preproc_config"],
                  "lead_order": meta.get("lead_order", list(CANONICAL_LEADS)),
                  "n_done": meta["n_done"], "n_failed": meta["n_failed"], "failed_by_reason": failed_reasons,
                  "complete": meta["complete"],
                  "sha256": S.sha256_of_file(out_path) if cache_sha256 else None},
        "transfer_route": ("Hugging Face mirror PhurinutR/MIMIC-IV-ECG of the PhysioNet zip (DECISIONS.md D-049): "
                           "size equal to PhysioNet's content-length, first 9,283,763,804 bytes identical to the "
                           "PhysioNet download; provenance is PhysioNet"),
        "sha256sums_verification": {
            "reference": f"{SUMS_NAME} shipped by PhysioNet inside the zip",
            "waveform_files_verified_ok": 2 * int(meta["n_done"] - meta["n_failed"]),
            "records_failed_checksum": _count_failed(fl, "sha256"),
        },
        "overlap_with_training_corpora": "none (US Beth Israel Deaconess records; CODE-15%, SaMi-Trop, PTB-XL are "
                                         "Brazilian / German)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(m, indent=2) + "\n")
    return m


def main(argv: list[str] | None = None) -> int:
    for v in S.THREAD_ENV:
        os.environ.setdefault(v, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["index", "build", "manifest"])
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--zip-sha256", default=None, help="skip re-hashing the zip when already known")
    a = ap.parse_args(argv)
    if a.step == "index":
        build_index(a.zip, a.metadata)
    elif a.step == "build":
        build_cache(a.zip, a.metadata, a.out, a.jobs, a.chunk_size, a.limit)
    else:
        print(json.dumps(write_manifest(a.zip, a.metadata, a.out, a.manifest, a.zip_sha256), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
