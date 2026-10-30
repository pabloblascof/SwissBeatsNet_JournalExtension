"""The anchor's preprocessing, applied once and cached as a memmap.

What
----
``preprocess_record`` turns one WFDB record into the float32 ``[2250, 12]``
array the anchor encoder consumes, by calling ``team.data.load_and_process_signal``
with the exact configuration of ``mm_mae/config/chagas_config_data.py``
(``target_sr=250``, ``preprocessing='team'``,
``intensity_normalization='standardize_regular_leadwise'``,
``standard_len=2250``, ``pad_mode='symmetric'``, ``heartbeat_mode='multiple'``).
``build_signal_cache`` runs it over every row of ``metadata.parquet`` with a
process pool and writes ``data/cache/signals_250hz.f32``, row *i* being
metadata row *i* (ARCHITECTURE.md, "Data on disk").

Why
---
The anchor's loader re-runs neurokit cleaning, resampling and normalisation
on every epoch of every run. The journal branch trains dozens of heads and
evaluates five proxy batteries over the same ~550k records; preprocessing
once and reading a memmap makes every downstream step I/O-bound instead of
CPU-bound, and -- more importantly -- guarantees every run sees bit-identical
inputs. The preprocessing is *called*, not reimplemented, so what the cache
holds is what submission 2813 saw (project rule 12).

Two facts about the anchor's preprocessing worth knowing before reading the
cache: standardisation happens on the *resampled* signal, and the symmetric
zero-padding to 2250 samples comes afterwards. A 7.3 s record therefore has
per-lead std ~0.90 over the padded window (sqrt(1833/2250)), not 1.0, and
the padding is exact zeros. Nothing here changes that; it is the anchor.

Resumability
------------
A run over the full corpus takes hours. A byte-per-row ``.done`` bitmap next
to the memmap records finished rows; a restart skips them. The sidecar
``.json`` (row count, ``records_sha256`` of the metadata order, ``complete``)
is written *before* the pool starts and rewritten at the end, so a cache that
crashed half-way still carries the row order it was built for; a resume into
a ``.f32``/``.done`` pair with no ``.json`` is refused, and one whose
``records_sha256`` differs from the metadata is refused too. Records that fail
to load are written as zeros, marked ``2`` in the bitmap, listed under
``failed_rows`` in the ``.json`` and logged to ``.failed.jsonl`` so the job
never stalls on one bad file. :func:`valid_rows` turns that list into a
boolean mask; consumers must exclude those rows (they are not signals).

Threads
-------
Nothing here mutates ``os.environ`` at import time. The pool workers pin
BLAS/OpenMP to one thread via ``threadpoolctl`` in their initialiser, and the
CLI entry point additionally sets ``*_NUM_THREADS=1`` for libraries the
workers import lazily (torch inside ``team.data``). Importing this module
from a head fit or the compass battery leaves their thread pools alone.

CLI::

    python -m src.data.signals --metadata data/cache/metadata.parquet \\
        --wfdb data/wfdb --out data/cache/signals_250hz.f32 --jobs 48 [--limit N] \\
        [--crop tail --copy-short-from data/cache/signals_250hz.f32]
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MM_MAE_ROOT = REPO_ROOT / "mm_mae"
# Appended, not inserted, exactly as team/utils.py does it: mm_mae's top-level
# ``config``/``utils``/``datasets`` packages must not shadow site-packages.
for _p in (str(REPO_ROOT), str(MM_MAE_ROOT)):
    if _p not in sys.path:
        sys.path.append(_p)

#: Environment variables that cap BLAS/OpenMP threads in the pool workers.
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

#: Cache geometry. Matches ``src.models.anchor.SEQ_LEN`` / ``N_LEADS``.
SEQ_LEN = 2250
N_LEADS = 12
DTYPE = np.float32
ROW_BYTES = SEQ_LEN * N_LEADS * np.dtype(DTYPE).itemsize

#: Name of this preprocessing in every cache's meta. Bump only if the anchor
#: call below changes, which it must not.
PREPROC_ID = "anchor_v1"

#: Where a record longer than 2250 samples (after resampling to 250 Hz) is
#: cut (S5.5, test-time augmentation). ``head`` is the anchor's rule -- the
#: first 2250 samples -- and the default everywhere; ``tail`` keeps the last
#: 2250, ``center`` the middle 2250 (start ``(L - 2250) // 2``). A record of at
#: most 2250 samples is zero-padded symmetrically whatever the crop, so its
#: row is identical in every crop's cache.
CROPS = ("head", "tail", "center")

DEFAULT_WFDB_ROOT = REPO_ROOT / "data" / "wfdb"
DEFAULT_METADATA = REPO_ROOT / "data" / "cache" / "metadata.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "cache" / "signals_250hz.f32"

def _team():
    """Import the anchor's loader lazily (it pulls wandb/hydra/matplotlib, ~30 s).

    ``team.utils`` imports matplotlib, whose default cache dir is not writable
    on this cluster; ``.mpl_cache/`` is gitignored for exactly this. Set here,
    at first use, so importing this module has no environment side effect.
    """
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(REPO_ROOT / ".mpl_cache")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import team.data as td
    from team.utils import HeartbeatMode, NormalizationMode, PaddingMode, PreprocessingMode

    return td, PreprocessingMode, NormalizationMode, PaddingMode, HeartbeatMode


#: The anchor configuration, spelled as the enum *values* so that the test
#: can compare against ``chagas_config_data.py`` without importing enums.
ANCHOR_PREPROC = dict(
    target_sr=250,
    preprocessing="team",
    intensity_normalization="standardize_regular_leadwise",
    standard_len=SEQ_LEN,
    pad_mode="symmetric",
    heartbeat_mode="multiple",
)


def read_sampling_rate(wfdb_root: Path | str, record: str) -> int:
    """Sampling rate from the WFDB header, via the anchor's own parser."""
    td, *_ = _team()
    header = td.load_header(os.path.join(str(wfdb_root), record))
    return td.load_sampling_rate(header)


def _check_crop(crop: str) -> str:
    if crop not in CROPS:
        raise ValueError(f"crop must be one of {CROPS}, got {crop!r}")
    return crop


def crop_window(length: int, crop: str = "head", standard_len: int = SEQ_LEN) -> tuple[int, int]:
    """``[start, stop)`` of the kept window in a processed signal of ``length`` samples.

    Only meaningful when ``length > standard_len``; shorter signals are padded.
    """
    _check_crop(crop)
    if length <= standard_len:
        return 0, length
    if crop == "head":
        start = 0
    elif crop == "tail":
        start = length - standard_len
    else:
        start = (length - standard_len) // 2
    return start, start + standard_len


def load_raw_signal(wfdb_root: Path | str, record: str) -> np.ndarray:
    """The raw ``[T, 12]`` signal exactly as ``load_and_process_signal`` loads it."""
    td, *_ = _team()
    signal, _ = td.load_signals(os.path.join(str(wfdb_root), record))
    return signal


def preprocess_record(wfdb_root: Path | str, record: str, crop: str = "head") -> np.ndarray:
    """Anchor preprocessing of one record -> float32 ``[2250, 12]``.

    ``record`` is the path relative to ``wfdb_root`` without extension
    (``samitrop/100726``, ``code_15/1234567``, ``ptbxl/00000/00001_hr``).

    ``crop='head'`` (default) is the anchor itself: ``team.data.load_and_process_signal``
    is called unchanged. ``'tail'``/``'center'`` (S5.5) load the record with the
    same loader and run :func:`preprocess_array` with that crop -- identical
    resampling, cleaning and standardisation (over the whole record), only the
    2250-sample window differs.
    """
    _check_crop(crop)
    if crop != "head":
        return preprocess_array(load_raw_signal(wfdb_root, record),
                                read_sampling_rate(wfdb_root, record), crop=crop)
    td, PreprocessingMode, NormalizationMode, PaddingMode, HeartbeatMode = _team()
    sr = read_sampling_rate(wfdb_root, record)
    sig = td.load_and_process_signal(
        str(wfdb_root),
        record,
        sr,
        ANCHOR_PREPROC["target_sr"],
        PreprocessingMode(ANCHOR_PREPROC["preprocessing"]),
        NormalizationMode(ANCHOR_PREPROC["intensity_normalization"]),
        ANCHOR_PREPROC["standard_len"],
        PaddingMode(ANCHOR_PREPROC["pad_mode"]),
        st_mem_transforms=None,
        heartbeat_mode=HeartbeatMode(ANCHOR_PREPROC["heartbeat_mode"]),
        numpy=True,
    )
    sig = np.ascontiguousarray(sig, dtype=DTYPE)
    if sig.shape != (SEQ_LEN, N_LEADS):
        raise ValueError(f"{record}: preprocessed shape {sig.shape} != {(SEQ_LEN, N_LEADS)}")
    return sig


def preprocess_record_crops(wfdb_root: Path | str, record: str,
                            crops: Sequence[str] = CROPS) -> dict[str, np.ndarray]:
    """``{crop: [2250, 12]}`` for one record with one preprocessing pass (S5.5)."""
    return preprocess_array_crops(load_raw_signal(wfdb_root, record),
                                  read_sampling_rate(wfdb_root, record), crops)


def preprocess_array(x: np.ndarray, fs: int, crop: str = "head") -> np.ndarray:
    """Anchor preprocessing of an in-memory signal -> float32 ``[2250, 12]``.

    What
    ----
    ``x`` is a ``[T, 12]`` float array in physical units at ``fs`` Hz -- what
    the official ``load_signals`` returns for a record, or that array after a
    P1 corruption (``src.compass.p1_corruptions``). The steps are the ones
    ``team.data.load_and_process_signal`` applies *after* loading, called
    through the same ``team.data`` functions with the anchor enums:

    1. ``scipy.signal.resample`` to ``num_target_samples = int(duration * 250)``
       with ``duration = len(x) / fs`` (``team.data.resample`` is that function);
    2. ``team.data.team_process`` (neurokit clean + lead-wise standardisation)
       when the resampled signal is longer than 50 samples, as the anchor guards;
    3. symmetric zero trim/pad to 2250 samples -- the anchor's rule is inline in
       ``load_and_process_signal`` (not a function), so those lines are
       repeated here verbatim; ``tests/test_p1_runner.py`` asserts equality
       with :func:`preprocess_record` on real records;
    4. no heartbeat segmentation (``heartbeat_mode='multiple'``).

    Why
    ---
    P1 corrupts records at native ``fs`` *before* the anchor pipeline (that is
    where its SNRs are defined) and then needs exactly the pipeline the cached
    clean signals went through. Loading is the only step that differs, so the
    corrupted copy is preprocessed by this function and the clean one by
    :func:`preprocess_record`; both must agree bit-for-bit on an uncorrupted
    array or the P1 deltas would measure the preprocessing, not the corruption.

    The input is cast to float64 because ``wfdb.rdsamp`` hands the anchor
    float64; ``x`` itself is never modified.

    ``crop`` (S5.5) picks the 2250-sample window of a longer processed signal
    (:data:`CROPS`, :func:`crop_window`); ``'head'`` is the anchor's rule and
    the default. Padding of shorter signals does not depend on it.
    """
    return preprocess_array_crops(x, fs, (crop,))[crop]


def preprocess_array_crops(x: np.ndarray, fs: int, crops: Sequence[str] = CROPS) -> dict[str, np.ndarray]:
    """:func:`preprocess_array` for several crops at once -> ``{crop: [2250, 12]}``.

    Resampling, cleaning and standardisation run once (they act on the whole
    record); only the final 2250-sample window differs per crop, so each
    entry is bit-identical to ``preprocess_array(x, fs, crop)``. This is what
    the container's test-time augmentation calls (S5.5): three crops cost one
    preprocessing pass.
    """
    crops = tuple(crops)
    if not crops:
        raise ValueError("crops must be non-empty")
    for c in crops:
        _check_crop(c)
    td, PreprocessingMode, NormalizationMode, PaddingMode, HeartbeatMode = _team()
    assert ANCHOR_PREPROC["preprocessing"] == "team", ANCHOR_PREPROC
    assert HeartbeatMode(ANCHOR_PREPROC["heartbeat_mode"]) == HeartbeatMode.MULTIPLE
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != N_LEADS:
        raise ValueError(f"expected [T, {N_LEADS}], got {signal.shape}")
    sr = int(fs)
    if sr <= 0:
        raise ValueError(f"fs must be positive, got {fs!r}")
    target_sr = int(ANCHOR_PREPROC["target_sr"])
    standard_len = int(ANCHOR_PREPROC["standard_len"])
    pad_mode = PaddingMode(ANCHOR_PREPROC["pad_mode"])

    # RESAMPLE -- as load_and_process_signal, same function object.
    duration = len(signal) / sr
    num_target_samples = int(duration * target_sr)
    signal = td.resample(signal, num_target_samples)

    # PREPROCESSING -- the anchor skips it for very short signals.
    if signal.shape[0] > 50:
        signal = td.team_process(
            signal, target_sr,
            NormalizationMode(ANCHOR_PREPROC["intensity_normalization"]),
            standard_len, pad_mode,
        )

    return {c: _trim_pad(signal, c, standard_len, pad_mode, PaddingMode) for c in crops}


def _trim_pad(signal: np.ndarray, crop: str, standard_len: int, pad_mode, PaddingMode) -> np.ndarray:
    """The anchor's trim/pad step, with the trim window chosen by ``crop``."""
    # TRIMMING / PADDING -- inline in load_and_process_signal; repeated verbatim
    # for crop='head'; 'tail'/'center' move the window (S5.5).
    if signal.shape[0] > standard_len:
        if crop == "head":
            signal = signal[:standard_len, :]
        else:
            start, stop = crop_window(signal.shape[0], crop, standard_len)
            signal = signal[start:stop, :]
    elif signal.shape[0] < standard_len:
        missing = standard_len - signal.shape[0]
        if pad_mode == PaddingMode.SYMMETRIC:
            left_pad = missing // 2
            right_pad = missing - left_pad
            signal = np.concatenate([np.zeros((left_pad, 12)), signal, np.zeros((right_pad, 12))], axis=0)
        elif pad_mode == PaddingMode.LEFT:
            signal = np.concatenate([np.zeros((missing, 12)), signal], axis=0)
        elif pad_mode == PaddingMode.RIGHT:
            signal = np.concatenate([signal, np.zeros((missing, 12))], axis=0)

    sig = np.ascontiguousarray(signal, dtype=DTYPE)
    if sig.shape != (SEQ_LEN, N_LEADS):
        raise ValueError(f"preprocessed shape {sig.shape} != {(SEQ_LEN, N_LEADS)}")
    return sig


# --------------------------------------------------------------------------- #
# Cache files
# --------------------------------------------------------------------------- #


def _sidecar(out_path: Path | str, suffix: str) -> Path:
    return Path(str(out_path) + suffix)


def records_sha256(records: Sequence[str]) -> str:
    """Hash of the row order -- the invariant every cache actually depends on."""
    h = hashlib.sha256()
    for r in records:
        h.update(r.encode())
        h.update(b"\n")
    return h.hexdigest()


def sha256_of_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_signal_cache(path: Path | str = DEFAULT_OUT, mode: str = "r") -> np.memmap:
    """Open ``signals_250hz.f32`` read-only as ``[N, 2250, 12]`` float32.

    ``N`` comes from the sidecar ``.json`` written by :func:`build_signal_cache`.
    """
    meta = read_cache_meta(path)
    return np.memmap(path, dtype=DTYPE, mode=mode, shape=tuple(meta["shape"]))


def read_cache_meta(path: Path | str = DEFAULT_OUT) -> dict[str, Any]:
    p = _sidecar(path, ".json")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; build the cache first")
    with open(p) as fh:
        return json.load(fh)


def open_done_bitmap(path: Path | str, n: int, mode: str = "r+") -> np.memmap:
    """One byte per row: 0 not done, 1 done, 2 done-but-failed (zeros written).

    Created (all zeros) only when absent and ``mode`` is writable. An existing
    file of the wrong size is never silently reset: that would erase the
    progress of a build for a different ``n`` and is always a caller bug.
    """
    p = _sidecar(path, ".done")
    if p.exists():
        if p.stat().st_size != n:
            raise RuntimeError(
                f"{p} has {p.stat().st_size} bytes, expected {n}; refusing to reset it -- "
                "delete the cache (.f32/.done/.json) to rebuild"
            )
    elif mode == "r":
        raise FileNotFoundError(f"{p} missing")
    else:
        with open(p, "wb") as fh:
            fh.truncate(n)
    return np.memmap(p, dtype=np.uint8, mode=mode, shape=(n,))


def valid_rows(path: Path | str = DEFAULT_OUT) -> np.ndarray:
    """Boolean ``[N]`` mask: ``False`` where the record failed to preprocess.

    Failed rows hold zeros in the memmap and must not be used as samples. The
    mask comes from ``failed_rows`` in the sidecar ``.json``; a cache written
    before that key existed falls back to the ``.done`` bitmap (``== 2``).
    """
    meta = read_cache_meta(path)
    n = int(meta["n"])
    mask = np.ones(n, dtype=bool)
    if "failed_rows" in meta:
        mask[np.asarray(meta["failed_rows"], dtype=np.int64)] = False
        return mask
    done_path = _sidecar(path, ".done")
    if done_path.exists():
        mask &= np.fromfile(done_path, dtype=np.uint8, count=n) != 2
    return mask


def _ensure_memmap(path: Path, n: int) -> None:
    """Create (sparse) or validate the memmap file for ``n`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    want = n * ROW_BYTES
    if path.exists() and path.stat().st_size == want:
        return
    if path.exists() and path.stat().st_size != want:
        raise RuntimeError(
            f"{path} exists with {path.stat().st_size} bytes, expected {want} for n={n}; "
            "refusing to overwrite -- delete it (and its .done/.json) to rebuild"
        )
    with open(path, "wb") as fh:
        fh.truncate(want)


# --------------------------------------------------------------------------- #
# Pool workers
# --------------------------------------------------------------------------- #

_W: dict[str, Any] = {}


def _init_worker(out_path: str, n: int, wfdb_root: str, crop: str = "head",
                 copy_short_from: str | None = None) -> None:
    """Pool initialiser: open the memmap and pin this worker to one BLAS thread.

    With 48 workers each spinning up 80 BLAS/OpenMP threads the job is slower
    than single-process. ``threadpoolctl`` limits the pools numpy/scipy have
    already opened (forked workers inherit them), and the env vars cover
    libraries the worker imports lazily (torch, via ``team.data``). Both are
    process-local to the worker; the parent's environment is untouched.
    """
    from threadpoolctl import threadpool_limits

    for v in THREAD_ENV:
        os.environ[v] = "1"
    _W["threads"] = threadpool_limits(limits=1)  # kept alive for the worker's lifetime
    _W["mm"] = np.memmap(out_path, dtype=DTYPE, mode="r+", shape=(n, SEQ_LEN, N_LEADS))
    _W["wfdb_root"] = wfdb_root
    _W["crop"] = crop
    _W["src"] = (np.memmap(copy_short_from, dtype=DTYPE, mode="r", shape=(n, SEQ_LEN, N_LEADS))
                 if copy_short_from else None)
    _team()  # import once per worker (forked workers inherit it anyway)


def _process_chunk(chunk: list[tuple[int, str]]) -> tuple[list[int], list[tuple[int, str, str]]]:
    """Preprocess ``(row, record)`` pairs into the shared memmap.

    Returns ``(rows_ok, failures)`` with failures as ``(row, record, error)``.
    Failed rows are zero-filled so the cache never holds uninitialised bytes.
    """
    mm: np.memmap = _W["mm"]
    ok: list[int] = []
    failed: list[tuple[int, str, str]] = []
    crop = _W.get("crop", "head")
    src = _W.get("src")
    for row, record in chunk:
        try:
            if crop == "head":
                mm[row] = preprocess_record(_W["wfdb_root"], record)
            else:
                x = load_raw_signal(_W["wfdb_root"], record)
                fs = read_sampling_rate(_W["wfdb_root"], record)
                if src is not None and _resampled_len(len(x), fs) <= SEQ_LEN:
                    # Padded, not cropped: the row is the head cache's row (S5.5).
                    mm[row] = src[row]
                else:
                    mm[row] = preprocess_array(x, fs, crop=crop)
            ok.append(row)
        except Exception as e:  # noqa: BLE001 -- logged, never swallowed silently
            mm[row] = 0.0
            failed.append((row, record, f"{type(e).__name__}: {e}"))
    mm.flush()
    return ok, failed


def _resampled_len(n_samples: int, fs: int) -> int:
    """Length after the anchor's resampling to 250 Hz (``int(n / fs * 250)``)."""
    return int(n_samples / int(fs) * int(ANCHOR_PREPROC["target_sr"]))


def _chunks(items: list[tuple[int, str]], size: int) -> Iterable[list[tuple[int, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def load_records(metadata_parquet: Path | str, limit: int | None = None) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(metadata_parquet, columns=["record"])
    recs = df["record"].astype(str).tolist()
    if limit is not None:
        recs = recs[: int(limit)]
    if not recs:
        raise ValueError(f"{metadata_parquet}: no records")
    return recs


def _write_meta(
    out_path: Path,
    done: np.ndarray,
    *,
    n: int,
    rec_sha: str,
    meta_sha: str,
    metadata_parquet: Path | str,
    wfdb_root: Path,
    limit: int | None,
    crop: str = "head",
    copy_short_from: str | None = None,
) -> dict[str, Any]:
    """Write the sidecar ``.json`` from the current bitmap and return it."""
    done_arr = np.asarray(done)
    meta = {
        "n": n,
        "shape": [n, SEQ_LEN, N_LEADS],
        "dtype": "float32",
        "preproc": PREPROC_ID,
        "preproc_config": ANCHOR_PREPROC,
        "metadata_parquet": str(metadata_parquet),
        "metadata_sha256": meta_sha,
        "records_sha256": rec_sha,
        "wfdb_root": str(wfdb_root),
        "limit": limit,
        "n_done": int((done_arr > 0).sum()),
        "n_failed": int((done_arr == 2).sum()),
        "failed_rows": [int(i) for i in np.flatnonzero(done_arr == 2)],
        "complete": bool((done_arr > 0).all()),
    }
    if crop != "head":  # the head cache's sidecar keeps its original key set
        meta["crop"] = crop
        meta["copy_short_from"] = copy_short_from
    meta_path = _sidecar(out_path, ".json")
    tmp = _sidecar(out_path, ".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, meta_path)  # atomic: a crash never leaves a half-written sidecar
    return meta


def build_signal_cache(
    metadata_parquet: Path | str = DEFAULT_METADATA,
    wfdb_root: Path | str = DEFAULT_WFDB_ROOT,
    out_path: Path | str = DEFAULT_OUT,
    jobs: int = 8,
    limit: int | None = None,
    chunk_size: int = 64,
    log_every: int = 50,
    crop: str = "head",
    copy_short_from: Path | str | None = None,
) -> dict[str, Any]:
    """Preprocess every metadata row into ``out_path`` (resumable).

    ``crop`` (S5.5) selects the window of records longer than 2250 samples
    (:data:`CROPS`). With ``crop != 'head'`` and ``copy_short_from`` pointing at
    a complete head cache of the same row order, rows that are padded rather
    than cropped are copied from it instead of recomputed (they are identical
    by construction; ``tests/test_tta.py`` checks it).

    Returns the sidecar meta dict. ``limit`` restricts the cache to the first
    ``limit`` metadata rows (the cache then has ``n = limit``); it is for smoke
    tests, and a limited cache is refused as the base of a full one.

    Resume rules (all raise ``RuntimeError`` rather than guess): an existing
    ``.json`` must carry this metadata's ``records_sha256`` and ``n``; an
    existing ``.f32`` or ``.done`` without a ``.json`` is refused, because the
    row order it was built for is unknown.
    """
    _check_crop(crop)
    out_path = Path(out_path)
    wfdb_root = Path(wfdb_root)
    if copy_short_from is not None:
        if crop == "head":
            raise ValueError("copy_short_from only applies to crop != 'head'")
        copy_short_from = str(Path(copy_short_from))
    records = load_records(metadata_parquet, limit)
    n = len(records)
    rec_sha = records_sha256(records)
    meta_sha = sha256_of_file(metadata_parquet)

    meta_path = _sidecar(out_path, ".json")
    if meta_path.exists():
        with open(meta_path) as fh:
            prior = json.load(fh)
        if prior.get("records_sha256") != rec_sha or prior.get("n") != n:
            raise RuntimeError(
                f"{meta_path} was built for a different row order/size "
                f"(n={prior.get('n')}); refusing to resume into it"
            )
        if prior.get("crop", "head") != crop:
            raise RuntimeError(f"{meta_path} was built with crop={prior.get('crop', 'head')!r}, not {crop!r}")
    elif out_path.exists() or _sidecar(out_path, ".done").exists():
        raise RuntimeError(
            f"{out_path} (or its .done) exists without {meta_path.name}: the row order it "
            "was built for is unknown; delete .f32/.done/.failed.jsonl to rebuild"
        )

    if copy_short_from is not None:
        src_meta = read_cache_meta(copy_short_from)
        if (src_meta.get("records_sha256") != rec_sha or int(src_meta.get("n", -1)) != n
                or not src_meta.get("complete") or src_meta.get("crop", "head") != "head"):
            raise RuntimeError(f"{copy_short_from} is not a complete head cache of this row order")
        if src_meta.get("failed_rows"):
            raise RuntimeError(f"{copy_short_from} has failed rows; copying would propagate zeros")

    _ensure_memmap(out_path, n)
    done = open_done_bitmap(out_path, n, mode="r+")
    todo = [(i, r) for i, r in enumerate(records) if done[i] == 0]
    failed_log = _sidecar(out_path, ".failed.jsonl")
    meta_kw = dict(
        n=n, rec_sha=rec_sha, meta_sha=meta_sha, metadata_parquet=metadata_parquet,
        wfdb_root=wfdb_root, limit=limit, crop=crop, copy_short_from=copy_short_from,
    )
    # Written before any row is touched: pins the row order for every later resume.
    _write_meta(out_path, done, **meta_kw)

    t0 = time.time()
    n_ok = n_fail = 0
    print(f"[signals] n={n} todo={len(todo)} jobs={jobs} crop={crop} out={out_path}", flush=True)
    if todo:
        ctx = mp.get_context("fork")
        jobs = max(1, min(int(jobs), len(todo)))
        with ctx.Pool(jobs, initializer=_init_worker, initargs=(str(out_path), n, str(wfdb_root), crop, copy_short_from)) as pool, open(
            failed_log, "a"
        ) as flog:
            for k, (ok, failed) in enumerate(
                pool.imap(_process_chunk, _chunks(todo, chunk_size)), start=1
            ):
                for row in ok:
                    done[row] = 1
                for row, record, err in failed:
                    done[row] = 2
                    flog.write(json.dumps({"row": row, "record": record, "error": err}) + "\n")
                n_ok += len(ok)
                n_fail += len(failed)
                done.flush()
                flog.flush()
                if k % log_every == 0 or k * chunk_size >= len(todo):
                    dt = time.time() - t0
                    rate = (n_ok + n_fail) / max(dt, 1e-9)
                    print(
                        f"[signals] {n_ok + n_fail}/{len(todo)} ({rate:.1f} rec/s, "
                        f"failed={n_fail}, {dt/60:.1f} min)",
                        flush=True,
                    )

    done.flush()
    meta = _write_meta(out_path, done, **meta_kw)
    print(f"[signals] wrote {meta_path}: done={meta['n_done']}/{n} failed={meta['n_failed']}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    import argparse

    # CLI only: cap threads for anything the (forked) workers import lazily.
    # Library calls never see this; the workers pin themselves in _init_worker.
    for v in THREAD_ENV:
        os.environ.setdefault(v, "1")

    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0])
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--wfdb", default=str(DEFAULT_WFDB_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke tests)")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--crop", default="head", choices=CROPS,
                    help="window of records longer than 2250 samples (S5.5); head = the anchor")
    ap.add_argument("--copy-short-from", default=None,
                    help="complete head cache to copy padded (not cropped) rows from; crop != head only")
    args = ap.parse_args(argv)
    meta = build_signal_cache(args.metadata, args.wfdb, args.out, args.jobs, args.limit, args.chunk_size,
                              crop=args.crop, copy_short_from=args.copy_short_from)
    return 0 if meta["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
