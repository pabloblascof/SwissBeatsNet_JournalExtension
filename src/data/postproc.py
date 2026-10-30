"""Post-processing of the anchor's cached signals: S4X wave-0 arms L7 and N60.

What
----
Two transforms of one preprocessed row ``x`` (float32 ``[2250, 12]``: the
anchor's output, standardised per lead and then zero-padded symmetrically or
head-cropped to 2250 samples at 250 Hz; see :mod:`src.data.signals`):

* :func:`edge_window` (L7; DECISIONS.md D-035, consequence 3) zeroes every
  sample outside one fixed window ``[KEEP_START, KEEP_START + KEEP_LEN)``.
  After it every record of at least ``KEEP_LEN`` samples carries the *same*
  zero pattern, so the padding no longer tells a 7.3 s record from a 10 s one.
* :func:`notch60` (N60; D-035 addendum) applies ``scipy.signal.iirnotch(60,
  Q=30, fs=250)`` through ``filtfilt`` to each lead's valid (non-padded) span
  only. The anchor's cleaning attenuates 60 Hz by only ~31 dB; the notch is
  the cheapest test of whether that residue is what P1's powerline_60 cells
  measure.

and :func:`build_postproc_cache`, which streams the base cache
``data/cache/signals_250hz.f32`` through one of them into
``data/cache/signals_250hz_<kind>.f32`` with the same row order, a ``.done``
bitmap (resumable) and a ``.json`` sidecar that mirrors the base cache's plus
a ``postproc`` block. :func:`postprocess` is the per-record entry point the
P1 wrapper (``scripts/experiments/p1_postproc.py``) applies after
``src.data.signals.preprocess_array``.

Why the window is [208, 2041), not [207, 2042)
----------------------------------------------
The 7.3 s format is 2934 samples at 400 Hz. The anchor resamples it to
``int(2934 / 400 * 250) = 1833`` samples, not 1835, and pads 208 left and
209 right, so its valid span is ``[208, 2041)``. This holds for 199,601
CODE-15% and 609 SaMi-Trop records (checked on the cache: first and last
non-zero samples at 208 and 2040). A 1835-sample window starting at 207 would
leave a one-sample zero rim at each end of exactly those records and not of
the cropped 10 s ones, and the padding would still encode the format. The
window is therefore the 7.3 s format's own span. Records of 1834-2249
samples (4,124) lose their extra samples and become indistinguishable by
padding. Records shorter than 1833 samples (4,463, mostly truncated CODE-15%
exams) keep their wider padding: they cannot be lengthened. That residue is
reported, not hidden.

Two cues survive by construction and are not claimed removed. A 7.3 s
record's window edges are its true edges (with the cleaning filter's edge
transients) whereas a cropped record's are mid-record. And the per-lead
standardisation was computed over the whole resampled record before
windowing. P4's duration probe measures what is left.

Why the notch pads its own edges
--------------------------------
``filtfilt`` with scipy's default odd extension (9 samples) leaves a start-up
transient at both ends of the span. The notch's poles sit at radius 0.975
(time constant ~40 samples), and the odd reflection of a sinusoid does not
continue it in phase. Measured on a pure 60 Hz tone over a 1833-sample span,
that caps the whole-span attenuation at ~24-26 dB; longer odd padding makes
it worse (~20 dB) and Gustafsson's initial conditions worse still (~17 dB).
So each edge is extended here by ``NOTCH_PAD`` samples before ``filtfilt``.
The 60 Hz component is fitted by least squares on the ``NOTCH_FIT`` samples
next to that edge (250 samples = exactly 60 cycles, so the fit is orthogonal
to DC). The extension is the fitted tone continued analytically, plus the odd
reflection of the tone-free remainder (what scipy's odd padding does to the
signal). The filter's start-up transient then decays inside the extension,
which is cropped. On a pure tone the whole-span attenuation exceeds 60 dB,
and a 10 Hz tone passes within 0.01 dB (``tests/test_wave0.py``). A span
shorter than ``NOTCH_MIN_SPAN`` samples (7 records of < 35 samples) is left
unfiltered.

Harmonics are out of scope. P1's powerline corruption adds the 2nd and 3rd
harmonics too. At 250 Hz the 180 Hz one is removed by resampling, but 120 Hz
survives the cleaning (about -29 dB) and this notch alike.

CLI::

    python -m src.data.postproc --kind len7 [--src data/cache/signals_250hz.f32] [--jobs 16]
    python -m src.data.postproc --kind n60
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy import signal as sps

from src.data.signals import (
    ANCHOR_PREPROC,
    DEFAULT_METADATA,
    DEFAULT_OUT,
    DTYPE,
    N_LEADS,
    ROW_BYTES,
    SEQ_LEN,
    THREAD_ENV,
    open_done_bitmap,
    read_cache_meta,
    records_sha256,
    sha256_of_file,
)

#: The transforms this module builds caches for.
KINDS = ("len7", "n60")

#: L7 window: the 7.3 s format's own span in the anchor's padded row (see module doc).
KEEP_START = 208
KEEP_LEN = 1833

#: N60 filter (D-035 addendum): scipy.signal.iirnotch(60, Q=30, fs=250).
NOTCH_F0 = 60.0
NOTCH_Q = 30.0
FS = int(ANCHOR_PREPROC["target_sr"])
#: Samples of tone-continued padding per edge (> 6 notch time constants).
NOTCH_PAD = 250
#: Samples next to each edge used to fit the 60 Hz tone (60 cycles exactly at 250 Hz).
NOTCH_FIT = 250
#: Spans shorter than this are left unfiltered.
NOTCH_MIN_SPAN = 35


# --------------------------------------------------------------------------- #
# Geometry of the anchor's padding
# --------------------------------------------------------------------------- #


def resampled_length(n_samples: int, fs: int, target_sr: int = FS) -> int:
    """Length of a record after the anchor's resampling: ``int(n / fs * 250)``.

    The same arithmetic, in the same order, as ``preprocess_array``
    (``duration = len / sr``; ``int(duration * target_sr)``).
    """
    return int(int(n_samples) / int(fs) * int(target_sr))


def valid_span(length: int, standard_len: int = SEQ_LEN) -> tuple[int, int]:
    """``[start, stop)`` of the record's samples inside the anchor's padded row.

    A record of ``length`` resampled samples is head-cropped when longer than
    ``standard_len`` (valid everywhere). When it is shorter it is padded
    ``(standard_len - length) // 2`` samples on the left, the anchor's
    symmetric rule.
    """
    length = int(length)
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if length >= standard_len:
        return 0, standard_len
    left = (standard_len - length) // 2
    return left, left + length


def valid_mask(length: int, standard_len: int = SEQ_LEN) -> np.ndarray:
    """Boolean ``[standard_len]``: ``True`` on the record's samples, ``False`` on the padding."""
    s, e = valid_span(length, standard_len)
    m = np.zeros(standard_len, dtype=bool)
    m[s:e] = True
    return m


# --------------------------------------------------------------------------- #
# L7
# --------------------------------------------------------------------------- #


def edge_window(x: np.ndarray, keep_start: int = KEEP_START, keep_len: int = KEEP_LEN) -> np.ndarray:
    """Zero every sample of ``x [T, 12]`` outside ``[keep_start, keep_start + keep_len)``.

    Returns a float32 copy and never modifies ``x``. The window is the same
    for every record, so every record whose valid span covers it ends up with
    one zero pattern (see the module docstring for the choice of window).
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected [T, leads], got {x.shape}")
    keep_start, keep_len = int(keep_start), int(keep_len)
    if keep_start < 0 or keep_len <= 0 or keep_start + keep_len > x.shape[0]:
        raise ValueError(f"window [{keep_start}, {keep_start + keep_len}) outside [0, {x.shape[0]})")
    y = np.zeros(x.shape, dtype=DTYPE)
    y[keep_start:keep_start + keep_len] = x[keep_start:keep_start + keep_len]
    return y


# --------------------------------------------------------------------------- #
# N60
# --------------------------------------------------------------------------- #


def notch_coefficients(f0: float = NOTCH_F0, q: float = NOTCH_Q, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """``(b, a)`` of ``scipy.signal.iirnotch(f0, Q=q, fs=fs)``."""
    return sps.iirnotch(float(f0), float(q), fs=float(fs))


def _tone_fit(seg: np.ndarray, n: np.ndarray, omega: float) -> np.ndarray:
    """Least-squares ``[cos, sin]`` coefficients ``[2, C]`` of the ``omega`` tone in ``seg [len(n), C]``."""
    basis = np.stack([np.cos(omega * n), np.sin(omega * n)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, seg, rcond=None)
    return coef


def _tone(coef: np.ndarray, n: np.ndarray, omega: float) -> np.ndarray:
    return np.cos(omega * n)[:, None] * coef[0] + np.sin(omega * n)[:, None] * coef[1]


def _notch_span(seg: np.ndarray, b: np.ndarray, a: np.ndarray, *, f0: float, fs: int,
                pad: int = NOTCH_PAD, fit: int = NOTCH_FIT) -> np.ndarray:
    """``filtfilt`` of ``seg [m, C]`` (float64) with tone-continued odd padding at both edges.

    Returns ``[m, C]``. See the module docstring for why the padding is not
    scipy's default.
    """
    m = seg.shape[0]
    p = min(int(pad), m - 1)
    w = min(int(fit), m)
    omega = 2.0 * np.pi * float(f0) / float(fs)
    k = np.arange(1, p + 1)

    # Left edge: tone fitted on seg[:w], continued to n = -1..-p; remainder reflected oddly about n = 0.
    n_l = np.arange(w)
    c_l = _tone_fit(seg[:w], n_l, omega)
    r_l = seg[: p + 1] - _tone(c_l, np.arange(p + 1), omega)
    left = _tone(c_l, -k, omega) + (2.0 * r_l[0] - r_l[k])  # index j <-> n = -k[j]
    left = left[::-1]  # n = -p .. -1

    # Right edge: the same about n = m - 1.
    n_r = np.arange(m - w, m)
    c_r = _tone_fit(seg[m - w:], n_r, omega)
    tail_n = np.arange(m - 1 - p, m)
    r_r = seg[m - 1 - p:] - _tone(c_r, tail_n, omega)  # r_r[-1] <-> n = m - 1
    right = _tone(c_r, m - 1 + k, omega) + (2.0 * r_r[-1] - r_r[-1 - k])  # n = m .. m + p - 1

    ext = np.concatenate([left, seg, right], axis=0)
    y = sps.filtfilt(b, a, ext, axis=0, padtype=None)
    return y[p: p + m]


def notch60(x: np.ndarray, valid_mask: np.ndarray, *, f0: float = NOTCH_F0, q: float = NOTCH_Q,
            fs: int = FS) -> np.ndarray:
    """60 Hz notch over each lead's valid span of ``x [T, 12]``; padding left untouched.

    ``valid_mask`` is boolean ``[T]`` (one span shared by every lead, which
    is what the anchor's symmetric padding produces) or ``[T, leads]``. Each
    contiguous run of ``True`` is filtered on its own with
    ``scipy.signal.iirnotch(f0, Q=q, fs=fs)`` through ``filtfilt`` (edges
    padded as described in the module docstring). Samples outside the mask
    are copied unchanged, so zero padding stays exactly zero. A run shorter
    than :data:`NOTCH_MIN_SPAN` is copied unfiltered. Returns a float32 copy;
    the arithmetic is float64.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected [T, leads], got {x.shape}")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape not in ((x.shape[0],), x.shape):
        raise ValueError(f"valid_mask shape {mask.shape} matches neither [T] nor {x.shape}")
    b, a = notch_coefficients(f0, q, fs)
    y = np.array(x, dtype=np.float64, copy=True)
    if mask.ndim == 1:
        for s, e in _runs(mask):
            if e - s >= NOTCH_MIN_SPAN:
                y[s:e] = _notch_span(y[s:e], b, a, f0=f0, fs=fs)
    else:
        for lead in range(x.shape[1]):
            for s, e in _runs(mask[:, lead]):
                if e - s >= NOTCH_MIN_SPAN:
                    y[s:e, lead: lead + 1] = _notch_span(y[s:e, lead: lead + 1], b, a, f0=f0, fs=fs)
    return y.astype(DTYPE)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of every contiguous run of ``True`` in a 1-D mask."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


# --------------------------------------------------------------------------- #
# One entry point per record
# --------------------------------------------------------------------------- #


def postprocess(x: np.ndarray, kind: str, length: int) -> np.ndarray:
    """Apply ``kind`` to one anchor-preprocessed row ``x [2250, 12]`` of a record of ``length`` resampled samples."""
    if kind == "len7":
        return edge_window(x)
    if kind == "n60":
        return notch60(x, valid_mask(length, x.shape[0]))
    raise ValueError(f"unknown postproc kind {kind!r}; one of {KINDS}")


def describe(kind: str) -> dict[str, Any]:
    """The parameters of ``kind``, as written into every sidecar."""
    if kind == "len7":
        return {"kind": kind, "transform": "src.data.postproc.edge_window", "keep_start": KEEP_START,
                "keep_len": KEEP_LEN, "window": [KEEP_START, KEEP_START + KEEP_LEN],
                "note": "zero outside the 7.3 s format's own span [208, 2041) (1833 samples at 250 Hz)"}
    if kind == "n60":
        b, a = notch_coefficients()
        return {"kind": kind, "transform": "src.data.postproc.notch60", "f0_hz": NOTCH_F0, "q": NOTCH_Q,
                "fs": FS, "b": [float(v) for v in b], "a": [float(v) for v in a],
                "filtfilt_padding": f"tone-continued odd extension, {NOTCH_PAD} samples per edge "
                                    f"(tone fitted on {NOTCH_FIT} edge samples)",
                "span": "each lead's valid (non-padded) span; padding untouched",
                "min_span": NOTCH_MIN_SPAN}
    raise ValueError(f"unknown postproc kind {kind!r}; one of {KINDS}")


def default_out(kind: str, src: Path | str = DEFAULT_OUT) -> Path:
    """``signals_250hz.f32`` -> ``signals_250hz_<kind>.f32`` next to it."""
    src = Path(src)
    return src.with_name(f"{src.stem}_{kind}{src.suffix}")


# --------------------------------------------------------------------------- #
# Cache builder
# --------------------------------------------------------------------------- #


def _sidecar(path: Path | str, suffix: str) -> Path:
    return Path(str(path) + suffix)


def _ensure_memmap(path: Path, n: int) -> None:
    want = n * ROW_BYTES
    if path.exists():
        if path.stat().st_size != want:
            raise RuntimeError(f"{path} has {path.stat().st_size} bytes, expected {want} for n={n}; "
                               "refusing to overwrite -- delete .f32/.done/.json to rebuild")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(want)


def record_lengths(metadata_parquet: Path | str) -> tuple[list[str], np.ndarray]:
    """``(records, resampled lengths)`` in metadata row order (the cache's row order)."""
    import pandas as pd

    df = pd.read_parquet(metadata_parquet, columns=["record", "n_samples", "fs"])
    lengths = np.array([resampled_length(n, f) for n, f in zip(df["n_samples"], df["fs"])], dtype=np.int64)
    return df["record"].astype(str).tolist(), lengths


_W: dict[str, Any] = {}


def _init_worker(src: str, out: str, n: int, kind: str, lengths: np.ndarray) -> None:
    from threadpoolctl import threadpool_limits

    for v in THREAD_ENV:
        os.environ[v] = "1"
    _W["threads"] = threadpool_limits(limits=1)
    _W["src"] = np.memmap(src, dtype=DTYPE, mode="r", shape=(n, SEQ_LEN, N_LEADS))
    _W["out"] = np.memmap(out, dtype=DTYPE, mode="r+", shape=(n, SEQ_LEN, N_LEADS))
    _W["kind"] = kind
    _W["lengths"] = lengths


def _check_padding(x: np.ndarray, length: int, row: int) -> None:
    """The source row must be zero exactly where the anchor pads a record of ``length`` samples."""
    s, e = valid_span(length)
    if (s > 0 and np.any(x[:s] != 0)) or (e < SEQ_LEN and np.any(x[e:] != 0)):
        raise RuntimeError(f"row {row}: non-zero samples outside the valid span [{s}, {e}) implied by "
                           f"length {length}; the metadata and the signal cache disagree")


def _process_block(block: tuple[int, int]) -> tuple[int, int]:
    s, e = block
    src, out, kind, lengths = _W["src"], _W["out"], _W["kind"], _W["lengths"]
    x = np.array(src[s:e], dtype=DTYPE)
    y = np.empty_like(x)
    for i in range(e - s):
        _check_padding(x[i], int(lengths[s + i]), s + i)
        y[i] = postprocess(x[i], kind, int(lengths[s + i]))
    out[s:e] = y
    out.flush()
    return s, e


def _pending_blocks(done: np.ndarray, block_rows: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(done) == 0)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    out = []
    for run in np.split(idx, breaks):
        s, e = int(run[0]), int(run[-1]) + 1
        out += [(a, min(a + block_rows, e)) for a in range(s, e, block_rows)]
    return out


def _write_meta(out: Path, done: np.ndarray, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    d = np.asarray(done)
    meta = dict(base)
    meta.update(extra)
    meta.update(n_done=int((d > 0).sum()), n_failed=int((d == 2).sum()),
                failed_rows=[int(i) for i in np.flatnonzero(d == 2)], complete=bool((d > 0).all()))
    tmp = _sidecar(out, ".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, _sidecar(out, ".json"))
    return meta


def build_postproc_cache(
    kind: str,
    src: Path | str = DEFAULT_OUT,
    out: Optional[Path | str] = None,
    metadata_parquet: Path | str = DEFAULT_METADATA,
    jobs: int = 16,
    block_rows: int = 512,
    limit: Optional[int] = None,
    log_every: int = 50,
) -> dict[str, Any]:
    """Stream the complete base cache ``src`` through ``kind`` into ``out`` (resumable).

    ``out`` defaults to ``signals_250hz_<kind>.f32`` next to ``src``. Its
    ``.json`` is the base sidecar with ``preproc`` set to
    ``<base preproc>+<kind>`` and a ``postproc`` block (parameters, source
    path, the source sidecar's hash) added. It is written before any row is
    touched and rewritten at the end, so a resume always knows the row
    order. A resume into a cache built for another kind, source or row order
    is refused. Rows the base cache failed are copied as zeros and stay
    failed. Every source row is checked to be zero exactly where the
    metadata's record length says the anchor padded it, which ties the row
    order to the metadata a second time. ``limit`` builds only the first
    ``limit`` rows (smoke tests; ``complete`` stays false).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown postproc kind {kind!r}; one of {KINDS}")
    src = Path(src)
    out = default_out(kind, src) if out is None else Path(out)
    if out.resolve() == src.resolve():
        raise ValueError("out must differ from src")
    base = read_cache_meta(src)
    if not base.get("complete"):
        raise RuntimeError(f"{src} is not a complete signal cache")
    if base.get("crop", "head") != "head" or base.get("postproc"):
        raise RuntimeError(f"{src} is not the anchor's head cache (crop={base.get('crop')}, "
                           f"postproc={base.get('postproc')})")
    n = int(base["n"])
    records, lengths = record_lengths(metadata_parquet)
    rec_sha = records_sha256(records)
    if len(records) != n or rec_sha != base.get("records_sha256"):
        raise RuntimeError(f"{metadata_parquet} (n={len(records)}) is not the row order of {src} (n={n})")

    src_json_sha = hashlib.sha256(_sidecar(src, ".json").read_bytes()).hexdigest()
    extra = {
        "preproc": f"{base.get('preproc')}+{kind}",
        "postproc": {**describe(kind), "source_cache": str(src), "source_cache_json_sha256": src_json_sha,
                     "source_preproc": base.get("preproc"), "metadata_parquet": str(metadata_parquet),
                     "metadata_sha256": sha256_of_file(metadata_parquet)},
        "limit": limit,
    }
    meta_path = _sidecar(out, ".json")
    if meta_path.exists():
        with open(meta_path) as fh:
            prior = json.load(fh)
        same = (prior.get("records_sha256") == rec_sha and int(prior.get("n", -1)) == n
                and (prior.get("postproc") or {}).get("kind") == kind
                and (prior.get("postproc") or {}).get("source_cache_json_sha256") == src_json_sha)
        if not same:
            raise RuntimeError(f"{meta_path} was built for another kind, source or row order; "
                               "refusing to resume into it")
    elif out.exists() or _sidecar(out, ".done").exists():
        raise RuntimeError(f"{out} (or its .done) exists without {meta_path.name}; delete .f32/.done to rebuild")

    _ensure_memmap(out, n)
    done = open_done_bitmap(out, n, mode="r+")
    base_failed = np.asarray(base.get("failed_rows", []), dtype=np.int64)
    if base_failed.size:
        mm = np.memmap(out, dtype=DTYPE, mode="r+", shape=(n, SEQ_LEN, N_LEADS))
        mm[base_failed] = 0.0
        mm.flush()
        del mm
        done[base_failed] = 2
        done.flush()
    scope = np.asarray(done).copy()
    if limit is not None:
        scope[int(limit):] = 1
    blocks = _pending_blocks(scope, block_rows)
    _write_meta(out, done, base, extra)

    n_todo = sum(e - s for s, e in blocks)
    print(f"[postproc] kind={kind} n={n} todo={n_todo} blocks={len(blocks)} jobs={jobs} out={out}", flush=True)
    t0 = time.time()
    n_ok = 0
    if blocks:
        ctx = mp.get_context("fork")
        with ctx.Pool(max(1, min(int(jobs), len(blocks))), initializer=_init_worker,
                      initargs=(str(src), str(out), n, kind, lengths)) as pool:
            for k, (s, e) in enumerate(pool.imap_unordered(_process_block, blocks), start=1):
                done[s:e] = 1
                n_ok += e - s
                if k % log_every == 0 or k == len(blocks):
                    done.flush()
                    dt = time.time() - t0
                    print(f"[postproc] {n_ok}/{n_todo} rows ({n_ok / max(dt, 1e-9):.0f} rows/s, "
                          f"{dt / 60:.1f} min)", flush=True)
    done.flush()
    meta = _write_meta(out, done, base, extra)
    print(f"[postproc] wrote {meta_path}: done={meta['n_done']}/{n} failed={meta['n_failed']} "
          f"complete={meta['complete']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    for v in THREAD_ENV:
        os.environ.setdefault(v, "1")
    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--src", default=str(DEFAULT_OUT))
    ap.add_argument("--out", default=None, help="default: signals_250hz_<kind>.f32 next to --src")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--block-rows", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke tests)")
    a = ap.parse_args(argv)
    meta = build_postproc_cache(a.kind, a.src, a.out, a.metadata, a.jobs, a.block_rows, a.limit)
    return 0 if (meta["complete"] or a.limit is not None) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
