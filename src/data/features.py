"""Frozen-encoder features for every cached signal, as memmaps.

What
----
``build_features`` streams ``data/cache/signals_250hz.f32`` through an
``ST_MEM`` encoder on the GPU and writes, per ARCHITECTURE.md::

    data/cache/features/<encoder_id>/cls.f32    float32 [N, 12, 768]
    data/cache/features/<encoder_id>/mean.f32   float32 [N, 12, 768]
    data/cache/features/<encoder_id>/meta.json  {"n", "encoder_id", "sha256", "preproc", ...}

``cls`` is what the anchor's classifier heads read; ``mean`` is what the
alignment objective acts on (see ``src.models.anchor.encode``). Row *i* is
metadata row *i*, exactly as in the signal cache.

Why
---
The anchor freezes its encoder (``freeze_encoder: True``). Every head fit,
every proxy in the compass battery and every seed therefore consumes the same
encoder outputs; computing them once per encoder turns the S4 sweep and the
S1 battery from GPU jobs into CPU jobs on a memmap. ``encoder_id`` keys the
directory: ``anchor_good_disco`` for the shipped checkpoint, otherwise the
pretraining run_id (``--encoder-ckpt``).

Resumability and failed rows
----------------------------
A byte-per-row ``rows.done`` bitmap in the directory records finished rows; a
restart skips them. ``meta.json`` is written at the end and re-written on
every (partial) run with ``complete`` reflecting the bitmap.

Rows the signal cache could not preprocess (``failed_rows`` in its ``.json``,
``2`` in its ``.done`` bitmap) hold zeros there and are *never encoded*: an
all-zero input still yields plausible, finite features (SEP and positional
embeddings), which would turn a missing record into a silent sample. They are
marked ``2`` here too, left as zeros in ``cls.f32``/``mean.f32``, listed under
``failed_rows`` in ``meta.json``, and reported as ``False`` in the ``valid``
mask :func:`open_features` returns. Rows the signal cache has not built yet
are left pending (``0``), so a feature run on a half-built cache resumes
correctly once the cache is complete.

CLI::

    python -m src.data.features --encoder-id anchor_good_disco \\
        --signals data/cache/signals_250hz.f32 --device cuda:0 [--limit N] \\
        [--encoder-ckpt results/runs/<run_id>/encoder.pt]

    # S5.5 crop caches: the anchor encoder over another crop's signal cache
    python -m src.data.features --encoder-id anchor_good_disco \
        --signals data/cache/signals_250hz_tail.f32 --out-dir data/cache/features/anchor_good_disco_tail
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.signals import (
    N_LEADS,
    SEQ_LEN,
    _sidecar,
    open_done_bitmap,
    open_signal_cache,
    read_cache_meta,
)
from src.models import anchor as A

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = REPO_ROOT / "data" / "cache" / "features"
EMB_DIM = A.EMB_DIM
DTYPE = np.float32
ROW_BYTES = N_LEADS * EMB_DIM * np.dtype(DTYPE).itemsize
FILES = ("cls", "mean")


def features_dir(encoder_id: str, root: Path | str = FEATURES_ROOT) -> Path:
    return Path(root) / encoder_id


def _ensure_memmap(path: Path, n: int) -> None:
    want = n * ROW_BYTES
    if path.exists() and path.stat().st_size == want:
        return
    if path.exists():
        raise RuntimeError(
            f"{path} has {path.stat().st_size} bytes, expected {want} for n={n}; "
            "refusing to overwrite -- delete the directory to rebuild"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(want)


def _source_state(signal_cache: Path, sig_meta: dict[str, Any], n: int) -> np.ndarray:
    """Per-row state of the signal cache: 0 not built, 1 usable, 2 failed.

    Prefers the ``.done`` bitmap (exact, includes rows still pending in an
    unfinished build); falls back to ``failed_rows``/``complete`` in the
    ``.json`` when the bitmap is absent (a cache copied without it).
    """
    done_path = _sidecar(signal_cache, ".done")
    if done_path.exists():
        state = np.fromfile(done_path, dtype=np.uint8, count=n)
        if state.shape[0] != n:
            raise RuntimeError(f"{done_path} has {state.shape[0]} rows, expected {n}")
        return state
    state = np.full(n, 1 if sig_meta.get("complete", True) else 0, dtype=np.uint8)
    state[np.asarray(sig_meta.get("failed_rows", []), dtype=np.int64)] = 2
    return state


def _pending_ranges(done: np.ndarray, batch_size: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` ranges of undone rows, each <= batch_size."""
    idx = np.flatnonzero(done == 0)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    out: list[tuple[int, int]] = []
    for run in np.split(idx, breaks):
        s, e = int(run[0]), int(run[-1]) + 1
        for a in range(s, e, batch_size):
            out.append((a, min(a + batch_size, e)))
    return out


def build_features(
    encoder_id: str,
    signal_cache: Path | str,
    out_dir: Path | str | None = None,
    device: str | torch.device = "cuda:0",
    batch_size: int = 256,
    encoder: torch.nn.Module | None = None,
    limit: int | None = None,
    encoder_sha256: str | None = None,
    log_every: int = 20,
) -> dict[str, Any]:
    """Encode every row of ``signal_cache`` into ``out_dir`` (resumable).

    Args:
        encoder_id: directory name under ``data/cache/features``.
        signal_cache: path to ``signals_250hz.f32`` (its ``.json`` must exist).
        out_dir: defaults to ``data/cache/features/<encoder_id>``.
        device: where the encoder runs. A caller-supplied ``encoder`` is
            moved there **in place** (``encoder.to(device)``) and put in eval
            mode; it is not copied, so a 330 MB ViT-B is never duplicated.
            Move it back yourself if you need it elsewhere afterwards.
        batch_size: rows per forward; each row is 12 leads, so the encoder
            sees ``12 * batch_size`` sequences of 47 tokens.
        encoder: an ``ST_MEM`` with ``num_leads`` 1 or 12 and width
            ``EMB_DIM``; ``None`` loads the verified anchor.
        limit: encode only the first ``limit`` rows (smoke tests). The
            memmaps are still allocated for the full cache so a later full
            run can resume into them.
        encoder_sha256: provenance written to ``meta.json``. Taken from
            ``encoder.anchor_sha256`` when present (set by
            ``load_anchor_encoder``), else ``None``.

    Returns the meta dict.
    """
    signal_cache = Path(signal_cache)
    out_dir = features_dir(encoder_id) if out_dir is None else Path(out_dir)
    sig_meta = read_cache_meta(signal_cache)
    n = int(sig_meta["n"])
    signals = open_signal_cache(signal_cache)
    assert signals.shape == (n, SEQ_LEN, N_LEADS), signals.shape

    if encoder is None:
        encoder = A.load_anchor_encoder(device=device, verify=True)
    else:
        encoder = encoder.to(device).eval()
    width = int(encoder.encoder.width)
    if width != EMB_DIM:
        raise ValueError(f"encoder width {width} != {EMB_DIM}; the feature cache layout is [N, 12, {EMB_DIM}]")
    if encoder_sha256 is None:
        encoder_sha256 = getattr(encoder, "anchor_sha256", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    mms: dict[str, np.memmap] = {}
    for name in FILES:
        p = out_dir / f"{name}.f32"
        _ensure_memmap(p, n)
        mms[name] = np.memmap(p, dtype=DTYPE, mode="r+", shape=(n, N_LEADS, EMB_DIM))
    done = open_done_bitmap(out_dir / "rows", n, mode="r+")  # -> out_dir/rows.done

    # Rows whose signal failed are marked 2 here and never encoded; rows the
    # signal cache has not built yet stay pending for a later run.
    source = _source_state(signal_cache, sig_meta, n)
    newly_failed = (source == 2) & (np.asarray(done) == 0)
    done[newly_failed] = 2
    scope = np.asarray(done).copy()
    scope[source == 0] = 1  # not buildable yet: out of scope, but stays 0 in `done`
    if limit is not None:
        scope[int(limit):] = 1  # out of scope for this run: treat as done
    ranges = _pending_ranges(scope, batch_size)
    n_todo = sum(e - s for s, e in ranges)
    n_unbuilt = int(((source == 0) & (np.asarray(done) == 0)).sum())
    print(
        f"[features] {encoder_id}: n={n} todo={n_todo} skipped_failed={int((source == 2).sum())} "
        f"unbuilt={n_unbuilt} device={device} bs={batch_size}",
        flush=True,
    )

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for k, (s, e) in enumerate(ranges, start=1):
            x = torch.from_numpy(np.array(signals[s:e], dtype=np.float32))  # copy: memmap is read-only
            out = A.encode(encoder, x.to(device, non_blocking=True), batch_leads=True)
            for name in FILES:
                mms[name][s:e] = out[name].float().cpu().numpy()
            done[s:e] = 1
            n_done += e - s
            if k % log_every == 0 or k == len(ranges):
                for name in FILES:
                    mms[name].flush()
                done.flush()
                dt = time.time() - t0
                print(f"[features] {n_done}/{n_todo} ({n_done / max(dt, 1e-9):.1f} rec/s, {dt/60:.1f} min)", flush=True)
    for name in FILES:
        mms[name].flush()
    done.flush()

    done_arr = np.asarray(done)
    meta = {
        "n": n,
        "encoder_id": encoder_id,
        "sha256": encoder_sha256,
        "preproc": sig_meta.get("preproc"),
        "shape": [n, N_LEADS, EMB_DIM],
        "dtype": "float32",
        "files": {name: f"{name}.f32" for name in FILES},
        "signal_cache": str(signal_cache),
        "signal_records_sha256": sig_meta.get("records_sha256"),
        "n_done": int((done_arr > 0).sum()),
        "n_failed": int((done_arr == 2).sum()),
        "failed_rows": [int(i) for i in np.flatnonzero(done_arr == 2)],
        "complete": bool((done_arr > 0).all()),
        "batch_size": batch_size,
        "device": str(device),
    }
    if sig_meta.get("crop", "head") != "head":  # S5.5: a non-anchor crop of the same records
        meta["crop"] = sig_meta["crop"]
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"[features] wrote {out_dir / 'meta.json'}: done={meta['n_done']}/{n}", flush=True)
    return meta


def open_features(encoder_id: str, root: Path | str = FEATURES_ROOT) -> dict[str, Any]:
    """Read-only view of a feature cache.

    Returns ``{"cls": memmap[N,12,768], "mean": memmap[N,12,768],
    "valid": bool[N], "meta": dict}``. ``valid`` is ``False`` for rows whose
    signal failed to preprocess (zeros in both memmaps); every consumer must
    drop those rows before fitting or scoring anything.
    """
    d = features_dir(encoder_id, root)
    mp = d / "meta.json"
    if not mp.exists():
        raise FileNotFoundError(f"{mp} missing; run src.data.features --encoder-id {encoder_id}")
    with open(mp) as fh:
        meta = json.load(fh)
    n = int(meta["n"])
    valid = np.ones(n, dtype=bool)
    if "failed_rows" in meta:
        valid[np.asarray(meta["failed_rows"], dtype=np.int64)] = False
    elif (d / "rows.done").exists():  # cache written before failed_rows existed
        valid &= np.fromfile(d / "rows.done", dtype=np.uint8, count=n) != 2
    out: dict[str, Any] = {"meta": meta, "valid": valid}
    for name in FILES:
        out[name] = np.memmap(d / f"{name}.f32", dtype=DTYPE, mode="r", shape=(n, N_LEADS, EMB_DIM))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0])
    ap.add_argument("--encoder-id", required=True)
    ap.add_argument("--signals", default=str(REPO_ROOT / "data" / "cache" / "signals_250hz.f32"))
    ap.add_argument("--out-dir", default=None, help="default data/cache/features/<encoder-id>")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--encoder-ckpt",
        default=None,
        help="ST_MEM checkpoint (.pt/.ckpt; plain state_dict or src.objectives.model.save_encoder "
        "payload, per-lead or joint) to encode with instead of good_disco",
    )
    args = ap.parse_args(argv)

    encoder = None
    sha = None
    if args.encoder_ckpt:
        if args.encoder_id == A.ANCHOR_ENCODER_ID:
            ap.error(f"--encoder-ckpt must not be keyed as {A.ANCHOR_ENCODER_ID}")
        encoder = A.load_encoder_from_file(args.encoder_ckpt, device=args.device)
        sha = A.sha256_of_file(args.encoder_ckpt)
    elif args.encoder_id != A.ANCHOR_ENCODER_ID:
        ap.error(f"without --encoder-ckpt the encoder id must be {A.ANCHOR_ENCODER_ID}")

    meta = build_features(
        args.encoder_id,
        args.signals,
        args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        encoder=encoder,
        limit=args.limit,
        encoder_sha256=sha,
    )
    return 0 if meta["complete"] or args.limit is not None else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
