"""S3.10 -- the Jidling et al. (2023) native input, cached as a memmap.

The reference implementation (``carji475/ecg-chagas``, README "model input")
consumes ``(N, 12, 4096)`` float32 at 400 Hz, leads ordered
``DI, DII, DIII, AVR, AVL, AVF, V1..V6``, zero-padded to 4096 samples, in the
units of the CODE/SaMi-Trop HDF5 tracings ("if the signal is in V it should be
multiplied by 1000", i.e. millivolts). Our WFDB records were written by the
organisers' converters straight from those HDF5 tracings with ``units=mV``
(``prepare_code15_data.py`` / ``prepare_samitrop_data.py``), so the physical
WFDB signal *is* the reference's input scale and no further gain is applied.

What this module does per record, and nothing more (no filtering, no
normalisation -- the reference has none):

1. ``wfdb.rdsamp`` -> physical ``[T, 12]`` float, leads re-ordered by name to
   :data:`LEAD_ORDER`; ``uV`` channels are divided by 1000; NaN -> 0.
2. ``fs != 400`` (PTB-XL, 500 Hz) -> ``scipy.signal.resample_poly`` to 400 Hz
   (up/down by the gcd: 4/5). The reference never saw a non-400 Hz record;
   this is a documented deviation (docs/journal/S3_10_JIDLING_NOTES.md).
3. The converters strip the HDF5's zero padding; it is restored here,
   *centred* (``left = (4096 - L) // 2``) as in the CODE-15% HDF5. A record
   longer than 4096 samples (none in the current corpora) is centre-cropped.

Cache layout is ``[N, 4096, 12]`` float32 (time-major, like
``signals_250hz.f32``) with the same sidecars as :mod:`src.data.signals`
(``.json`` with ``shape``/``records_sha256``/``complete``/``failed_rows``, and
a ``.done`` bitmap), so :func:`src.objectives.data.cache_status` and
:func:`src.data.signals.read_cache_meta` read it unchanged. The consumer
transposes to ``[B, 12, 4096]``.

CLI::

    python -m src.data.signals_jidling --out data/cache/signals_400hz_4096.f32 --jobs 24 [--limit N]
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data.signals import (
    DEFAULT_METADATA,
    DEFAULT_WFDB_ROOT,
    THREAD_ENV,
    _sidecar,
    load_records,
    open_done_bitmap,
    records_sha256,
    sha256_of_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FS = 400
SEQ_LEN = 4096
N_LEADS = 12
DTYPE = np.float32
ROW_BYTES = SEQ_LEN * N_LEADS * np.dtype(DTYPE).itemsize
PREPROC_ID = "jidling_v1"
LEAD_ORDER = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_OUT = REPO_ROOT / "data" / "cache" / "signals_400hz_4096.f32"
PREPROC_CONFIG = {
    "fs": FS, "seq_len": SEQ_LEN, "lead_order": list(LEAD_ORDER), "units": "mV (= V x 1000)",
    "pad": "zeros, centred, left=(4096-L)//2", "crop": "centre (if L > 4096)",
    "resample": "scipy.signal.resample_poly(up, down) by gcd when fs != 400", "nan": "-> 0",
    "filter": None, "normalisation": None,
    "reference": "https://github.com/carji475/ecg-chagas README 'model input'",
}


def to_reference_input(x: np.ndarray, fs: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Physical ``[T, 12]`` mV at ``fs`` -> float32 ``[seq_len, 12]`` at 400 Hz, zero-padded centred."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != N_LEADS:
        raise ValueError(f"expected [T, {N_LEADS}], got {x.shape}")
    x = np.where(np.isfinite(x), x, 0.0)
    fs = int(fs)
    if fs != FS:
        from scipy.signal import resample_poly

        g = math.gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g, axis=0)
    n = x.shape[0]
    out = np.zeros((seq_len, N_LEADS), dtype=DTYPE)
    if n >= seq_len:
        s = (n - seq_len) // 2
        out[:] = x[s:s + seq_len]
    else:
        left = (seq_len - n) // 2
        out[left:left + n] = x
    return out


def load_record(wfdb_root: Path | str, record: str) -> tuple[np.ndarray, int]:
    """WFDB record -> (physical ``[T, 12]`` in mV, lead order :data:`LEAD_ORDER`, fs)."""
    import wfdb

    sig, fields = wfdb.rdsamp(os.path.join(str(wfdb_root), record))
    names = [str(n).upper() for n in fields["sig_name"]]
    try:
        idx = [names.index(l) for l in LEAD_ORDER]
    except ValueError as e:
        raise ValueError(f"{record}: leads {names} lack one of {LEAD_ORDER}") from e
    sig = np.asarray(sig, dtype=np.float64)[:, idx]
    units = [str(fields["units"][i]) for i in idx]
    for j, u in enumerate(units):
        if u.lower() in ("uv", "µv"):
            sig[:, j] /= 1000.0
        elif u.lower() != "mv":
            raise ValueError(f"{record}: unsupported unit {u!r}")
    return sig, int(round(float(fields["fs"])))


def preprocess_record(wfdb_root: Path | str, record: str) -> np.ndarray:
    x, fs = load_record(wfdb_root, record)
    return to_reference_input(x, fs)


# --------------------------------------------------------------------------- #
# Build (same sidecar contract as src.data.signals)
# --------------------------------------------------------------------------- #

_W: dict[str, Any] = {}


def _init_worker(out_path: str, n: int, wfdb_root: str) -> None:
    from threadpoolctl import threadpool_limits

    for v in THREAD_ENV:
        os.environ[v] = "1"
    _W["threads"] = threadpool_limits(limits=1)
    _W["mm"] = np.memmap(out_path, dtype=DTYPE, mode="r+", shape=(n, SEQ_LEN, N_LEADS))
    _W["wfdb_root"] = wfdb_root


def _process_chunk(chunk: list[tuple[int, str]]):
    mm = _W["mm"]
    ok, failed = [], []
    for row, record in chunk:
        try:
            mm[row] = preprocess_record(_W["wfdb_root"], record)
            ok.append(row)
        except Exception as e:  # noqa: BLE001 -- logged by the parent
            mm[row] = 0.0
            failed.append((row, record, f"{type(e).__name__}: {e}"))
    mm.flush()
    return ok, failed


def _write_meta(out_path: Path, done: np.ndarray, **kw) -> dict[str, Any]:
    d = np.asarray(done)
    meta = {"n": kw["n"], "shape": [kw["n"], SEQ_LEN, N_LEADS], "dtype": "float32", "preproc": PREPROC_ID,
            "preproc_config": PREPROC_CONFIG, "metadata_parquet": str(kw["metadata_parquet"]),
            "metadata_sha256": kw["meta_sha"], "records_sha256": kw["rec_sha"], "wfdb_root": str(kw["wfdb_root"]),
            "limit": kw["limit"], "n_done": int((d > 0).sum()), "n_failed": int((d == 2).sum()),
            "failed_rows": [int(i) for i in np.flatnonzero(d == 2)], "complete": bool((d > 0).all())}
    tmp = _sidecar(out_path, ".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, _sidecar(out_path, ".json"))
    return meta


def build_cache(metadata_parquet: Path | str = DEFAULT_METADATA, wfdb_root: Path | str = DEFAULT_WFDB_ROOT,
                out_path: Path | str = DEFAULT_OUT, jobs: int = 16, limit: Optional[int] = None,
                chunk_size: int = 64) -> dict[str, Any]:
    """Resumable build; refuses to resume into a cache built for another row order."""
    out_path = Path(out_path)
    records = load_records(metadata_parquet, limit)
    n = len(records)
    kw = dict(n=n, rec_sha=records_sha256(records), meta_sha=sha256_of_file(metadata_parquet),
              metadata_parquet=metadata_parquet, wfdb_root=Path(wfdb_root), limit=limit)
    meta_path = _sidecar(out_path, ".json")
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior.get("records_sha256") != kw["rec_sha"] or prior.get("n") != n or prior.get("preproc") != PREPROC_ID:
            raise RuntimeError(f"{meta_path} was built for another row order/size/preproc; refusing to resume")
    elif out_path.exists() or _sidecar(out_path, ".done").exists():
        raise RuntimeError(f"{out_path} exists without {meta_path.name}; delete it to rebuild")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with open(out_path, "wb") as fh:
            fh.truncate(n * ROW_BYTES)
    elif out_path.stat().st_size != n * ROW_BYTES:
        raise RuntimeError(f"{out_path} has the wrong size for n={n}")
    done = open_done_bitmap(out_path, n, mode="r+")
    todo = [(i, r) for i, r in enumerate(records) if done[i] == 0]
    _write_meta(out_path, done, **kw)
    print(f"[jidling-cache] n={n} todo={len(todo)} jobs={jobs} out={out_path}", flush=True)
    t0, n_done = time.time(), 0
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(max(1, min(jobs, len(todo))), initializer=_init_worker,
                      initargs=(str(out_path), n, str(wfdb_root))) as pool, \
                open(_sidecar(out_path, ".failed.jsonl"), "a") as flog:
            chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
            for k, (ok, failed) in enumerate(pool.imap(_process_chunk, chunks), start=1):
                for r in ok:
                    done[r] = 1
                for r, rec, err in failed:
                    done[r] = 2
                    flog.write(json.dumps({"row": r, "record": rec, "error": err}) + "\n")
                n_done += len(ok) + len(failed)
                if k % 100 == 0 or n_done == len(todo):
                    done.flush()
                    dt = time.time() - t0
                    print(f"[jidling-cache] {n_done}/{len(todo)} ({n_done / max(dt, 1e-9):.0f} rec/s, "
                          f"{dt / 60:.1f} min)", flush=True)
    done.flush()
    meta = _write_meta(out_path, done, **kw)
    print(f"[jidling-cache] done={meta['n_done']}/{n} failed={meta['n_failed']}", flush=True)
    return meta


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    for v in THREAD_ENV:
        os.environ.setdefault(v, "1")
    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0])
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--wfdb", default=str(DEFAULT_WFDB_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    meta = build_cache(a.metadata, a.wfdb, a.out, a.jobs, a.limit)
    return 0 if meta["complete"] else 1


__all__ = ["FS", "LEAD_ORDER", "PREPROC_ID", "SEQ_LEN", "build_cache", "load_record", "preprocess_record",
           "to_reference_input"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
