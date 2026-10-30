"""Frozen-encoder features of SynthAcq-augmented views, for S4C wave B variant H5 ``aug``.

Specification: ``docs/journal/S4C_IMPROVE.md`` amendment B2 item 2 (D-078).

What
----
For one encoder, this module encodes K augmented views of the cached 250 Hz
signals of a set of rows, plus optional fresh views for a stability
diagnostic. It writes into ``data/cache/features_aug/<encoder_id>/``, which
:func:`scripts.experiments.own_encoder_queue.link_bulk` places in shared
derived storage::

    rows.npy             int64 [M]   cache rows with training views (sorted)
    view<v>.f32          float32 [M, 12, 768]  CLS features of view v = 1..K
    diag_rows.npy        int64 [Md]  cache rows with diagnostic views (optional)
    diag<v>.f32          float32 [Md, 12, 768]  fresh views v = 1..Kd (another salt)
    <file>.done          uint8 per row: 1 encoded, 2 the signal failed (zeros, never encoded)
    meta.json            provenance; ``complete`` once every bitmap is done

Only CLS features are written, because the anchor heads read CLS.

How a view is made
------------------
The augmentation is ``src.objectives.synth_acq.SynthAcq`` with ``aug_prob = 1``,
so every view is perturbed. The class (uniform over the seven non-identity
members) and the severity (``U[0, 1]``) of row ``r``, view ``v`` come from
``SeedSequence([salt, v, r])``. The transformation's own noise comes from a
generator seeded with ``(salt, v, block)``.

Rows are always augmented in canonical blocks of the full cache
(``[b * BLOCK, (b + 1) * BLOCK)``), whichever subset is encoded. The augmented
signal of a (row, view) is therefore a fixed function of (salt, row, view). It
is identical for every encoder and every requested subset, as B2 requires.
SynthAcq confines every transformation to the record's support and restores the
padding to exact zeros, so the augmentation cannot leak duration.

The cache holds ``[N, 2250, 12]`` (time first). SynthAcq works on
``[B, 12, T]``, so blocks are transposed in and out. Encoding is float32, as
for the clean cache (``src.data.features``).

CLI::

    python -m src.data.features_aug --encoder-id <rid> --encoder-ckpt results/runs/<rid>/encoder.pt \\
        --inner-folds 2 [--views 2] [--diag-views 1] [--device cuda:0]

``--inner-folds k[,k2,...]`` selects the inner-protocol rows. Training views
cover ``outer_fold not in {k}, outer_fold >= 0, inner_fold != 3`` (the union
over the listed folds). Diagnostic views cover the scored inner fold 3, which
is ``inner_fold == 3`` within the same pools.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from src.data.signals import N_LEADS, SEQ_LEN, open_done_bitmap, open_signal_cache, read_cache_meta
from src.models import anchor as A
from src.objectives import synth_acq as SA

REPO_ROOT = Path(__file__).resolve().parents[2]
AUG_ROOT = REPO_ROOT / "data" / "cache" / "features_aug"
EMB_DIM = A.EMB_DIM
DTYPE = np.float32
ROW_BYTES = N_LEADS * EMB_DIM * np.dtype(DTYPE).itemsize
#: Canonical augmentation block (rows of the full cache). Changing it changes every view.
BLOCK = 256
#: Seed salts: training views and the stability diagnostic's fresh views never share a draw.
TRAIN_SALT = 20261012
DIAG_SALT = 20261013
INNER_EVAL = 3
VERSION = 1


def aug_dir(encoder_id: str, root: Path | str = AUG_ROOT) -> Path:
    return Path(root) / encoder_id


def view_draws(rows: np.ndarray, view: int, salt: int, n_classes: int = SA.N_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """``(class ids, severities)`` for ``rows`` of view ``view``: class uniform over ``1..n_classes-1``
    (``aug_prob = 1``), severity uniform on ``[0, 1)``. Each comes from ``SeedSequence([salt, view, row])``."""
    rows = np.asarray(rows, dtype=np.int64)
    ids = np.empty(len(rows), dtype=np.int64)
    sev = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        a, b = np.random.SeedSequence([int(salt), int(view), int(r)]).generate_state(2)
        ids[i] = 1 + int(a) % (n_classes - 1)
        sev[i] = float(b) / 2.0**32
    return ids, sev


def augment_block(x: torch.Tensor, block: int, rows: np.ndarray, view: int, salt: int,
                  synth: SA.SynthAcq) -> torch.Tensor:
    """The view ``view`` of canonical block ``block``. ``x`` holds its rows ``[B, 2250, 12]`` (time first,
    on ``synth``'s device); ``rows`` are their cache indices. Returns the same layout."""
    ids, sev = view_draws(rows, view, salt, synth.n_classes)
    synth.gen.manual_seed(int(np.random.SeedSequence([int(salt), int(view), int(block)]).generate_state(1)[0]))
    xt = x.permute(0, 2, 1).contiguous()
    y = synth.transform(xt, torch.as_tensor(ids, device=x.device), torch.as_tensor(sev, device=x.device))
    return y.permute(0, 2, 1).contiguous()


def inner_rows(outer: np.ndarray, inner: np.ndarray, folds: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(training rows, scored rows)`` of the inner protocol (``inner_eval_fold = 3``), united over ``folds``."""
    outer, inner = np.asarray(outer), np.asarray(inner)
    train = np.zeros(len(outer), dtype=bool)
    scored = np.zeros(len(outer), dtype=bool)
    for k in folds:
        pool = (outer >= 0) & (outer != int(k))
        train |= pool & (inner != INNER_EVAL)
        scored |= pool & (inner == INNER_EVAL)
    return np.flatnonzero(train), np.flatnonzero(scored)


def _memmap(path: Path, n: int) -> np.memmap:
    want = n * ROW_BYTES
    if path.exists() and path.stat().st_size != want:
        raise RuntimeError(f"{path} has {path.stat().st_size} bytes, expected {want}; delete the directory to rebuild")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.truncate(want)
    return np.memmap(path, dtype=DTYPE, mode="r+", shape=(n, N_LEADS, EMB_DIM))


def _rows_file(path: Path, rows: np.ndarray) -> None:
    if path.exists():
        have = np.load(path)
        if not np.array_equal(have, rows):
            raise RuntimeError(f"{path} covers other rows; delete the directory to rebuild")
    else:
        np.save(path, rows)


def build_aug_features(
    encoder_id: str,
    signal_cache: Path | str,
    train_rows: np.ndarray,
    out_dir: Path | str | None = None,
    *,
    views: int = 2,
    diag_rows: Optional[np.ndarray] = None,
    diag_views: int = 1,
    device: str | torch.device = "cuda:0",
    encoder: torch.nn.Module | None = None,
    encoder_sha256: str | None = None,
    log_every: int = 50,
) -> dict[str, Any]:
    """Encode ``views`` training views of ``train_rows`` (and ``diag_views`` fresh views of ``diag_rows``).
    The build is resumable."""
    signal_cache = Path(signal_cache)
    out_dir = aug_dir(encoder_id) if out_dir is None else Path(out_dir)
    sig_meta = read_cache_meta(signal_cache)
    n = int(sig_meta["n"])
    signals = open_signal_cache(signal_cache)
    source = np.fromfile(str(signal_cache) + ".done", dtype=np.uint8, count=n) \
        if Path(str(signal_cache) + ".done").exists() else np.ones(n, dtype=np.uint8)
    if encoder is None:
        raise ValueError("an encoder is required (the clean cache's good_disco default does not apply here)")
    encoder = encoder.to(device).eval()
    if int(encoder.encoder.width) != EMB_DIM:
        raise ValueError(f"encoder width {encoder.encoder.width} != {EMB_DIM}")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[str, np.ndarray, int, int]] = []  # (prefix, rows, view, salt)
    train_rows = np.unique(np.asarray(train_rows, dtype=np.int64))
    _rows_file(out_dir / "rows.npy", train_rows)
    parts += [("view", train_rows, v, TRAIN_SALT) for v in range(1, views + 1)]
    if diag_rows is not None and diag_views > 0:
        diag_rows = np.unique(np.asarray(diag_rows, dtype=np.int64))
        _rows_file(out_dir / "diag_rows.npy", diag_rows)
        parts += [("diag", diag_rows, v, DIAG_SALT) for v in range(1, diag_views + 1)]
    mms, dones, pos = {}, {}, {}
    for prefix, rows, v, _ in parts:
        key = f"{prefix}{v}"
        mms[key] = _memmap(out_dir / f"{key}.f32", len(rows))
        dones[key] = open_done_bitmap(out_dir / key, len(rows), mode="r+")  # -> <key>.done
        failed = source[rows] == 2
        d = np.asarray(dones[key])
        dones[key][(d == 0) & failed] = 2
        pos[key] = {int(r): i for i, r in enumerate(rows)}
    if (source[np.unique(np.concatenate([p[1] for p in parts]))] == 0).any():
        raise RuntimeError("the signal cache has unbuilt rows among the requested ones")
    synth = SA.SynthAcq(seed=0, device=device, aug_prob=1.0)
    t0, n_enc = time.time(), 0
    todo = {key: np.flatnonzero(np.asarray(dones[key]) == 0) for key in dones}
    n_todo = sum(len(t) for t in todo.values())
    print(f"[features_aug] {encoder_id}: {', '.join(f'{k}={len(t)}' for k, t in todo.items())} rows to encode "
          f"on {device}", flush=True)
    need_blocks = sorted({int(r) // BLOCK for (prefix, rows, v, _) in parts
                          for r in rows[todo[f'{prefix}{v}']]})
    with torch.no_grad():
        for bi, b in enumerate(need_blocks, start=1):
            s, e = b * BLOCK, min((b + 1) * BLOCK, n)
            block_rows = np.arange(s, e)
            x = torch.from_numpy(np.array(signals[s:e], dtype=np.float32)).to(device, non_blocking=True)
            for prefix, rows, v, salt in parts:
                key = f"{prefix}{v}"
                want = [r for r in block_rows if r in pos[key] and dones[key][pos[key][r]] == 0]
                if not want:
                    continue
                xa = augment_block(x, b, block_rows, v, salt, synth)
                sel = torch.as_tensor(np.asarray(want) - s, device=device)
                out = A.encode(encoder, xa.index_select(0, sel), batch_leads=True)["cls"].float().cpu().numpy()
                p = np.array([pos[key][r] for r in want])
                mms[key][p] = out
                dones[key][p] = 1
                n_enc += len(want)
            if bi % log_every == 0 or bi == len(need_blocks):
                for key in mms:
                    mms[key].flush()
                    dones[key].flush()
                dt = time.time() - t0
                print(f"[features_aug] block {bi}/{len(need_blocks)}: {n_enc}/{n_todo} encoded "
                      f"({n_enc / max(dt, 1e-9):.1f} rec/s, {dt / 60:.1f} min)", flush=True)
    for key in mms:
        mms[key].flush()
        dones[key].flush()
    status = {key: {"n": int(len(d)), "done": int((np.asarray(d) > 0).sum()), "failed": int((np.asarray(d) == 2).sum())}
              for key, d in dones.items()}
    meta = {"version": VERSION, "encoder_id": encoder_id, "encoder_sha256": encoder_sha256,
            "signal_cache": str(signal_cache), "signal_records_sha256": sig_meta.get("records_sha256"),
            "preproc": sig_meta.get("preproc"), "n_cache": n, "views": views, "diag_views": diag_views if parts and
            parts[-1][0] == "diag" else 0, "block": BLOCK, "train_salt": TRAIN_SALT, "diag_salt": DIAG_SALT,
            "aug_prob": 1.0, "classes": list(SA.CLASSES), "feature": "cls", "dtype": "float32",
            "files": {key: f"{key}.f32" for key in mms}, "status": status,
            "complete": all(s["done"] == s["n"] for s in status.values()), "device": str(device)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[features_aug] wrote {out_dir / 'meta.json'}: complete={meta['complete']}", flush=True)
    return meta


def open_aug(encoder_id: str, root: Path | str = AUG_ROOT, prefix: str = "view") -> dict[str, Any]:
    """Read-only views: ``{"rows": int64 [M], "views": [memmap [M, 12, 768], ...], "failed": bool [M], "meta"}``.

    ``prefix`` is ``view`` for training views or ``diag`` for the diagnostic's fresh views. A row whose
    signal failed is zeros and marked ``failed``.
    """
    d = aug_dir(encoder_id, root)
    meta = json.loads((d / "meta.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError(f"{d}: augmented feature cache incomplete")
    rows = np.load(d / ("rows.npy" if prefix == "view" else "diag_rows.npy"))
    k = meta["views"] if prefix == "view" else meta["diag_views"]
    views = [np.memmap(d / f"{prefix}{v}.f32", dtype=DTYPE, mode="r", shape=(len(rows), N_LEADS, EMB_DIM))
             for v in range(1, k + 1)]
    failed = np.zeros(len(rows), dtype=bool)
    for v in range(1, k + 1):
        failed |= np.fromfile(d / f"{prefix}{v}.done", dtype=np.uint8, count=len(rows)) == 2
    return {"rows": rows, "views": views, "failed": failed, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0])
    ap.add_argument("--encoder-id", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--inner-folds", required=True, help="outer folds whose inner-protocol rows are covered")
    ap.add_argument("--signals", default=str(REPO_ROOT / "data" / "cache" / "signals_250hz.f32"))
    ap.add_argument("--metadata", default=str(REPO_ROOT / "data" / "cache" / "metadata.parquet"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "splits" / "frozen_splits.json"))
    ap.add_argument("--out-dir", default=None, help="default data/cache/features_aug/<encoder-id>")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--diag-views", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)

    from src import run as runmod

    meta = runmod.load_world({"metadata": a.metadata, "splits": a.splits})
    folds = [int(x) for x in a.inner_folds.split(",")]
    train, scored = inner_rows(meta["outer_fold"].to_numpy(), meta["inner_fold"].to_numpy(), folds)
    encoder = A.load_encoder_from_file(a.encoder_ckpt, device=a.device)
    out = build_aug_features(a.encoder_id, a.signals, train, a.out_dir, views=a.views, diag_rows=scored,
                             diag_views=a.diag_views, device=a.device, encoder=encoder,
                             encoder_sha256=A.sha256_of_file(a.encoder_ckpt))
    del pd
    return 0 if out["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
