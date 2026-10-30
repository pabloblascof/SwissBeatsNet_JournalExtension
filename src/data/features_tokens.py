"""Per-lead encoder output tokens of a frozen encoder, for S4C wave B variant H6 ``attnpool``.

Spec: ``docs/journal/S4C_IMPROVE.md`` amendment B3 (D-081).

What
----
For one encoder and a set of cache rows, this writes into
``data/cache/features_tokens/<encoder_id>/`` (linked to shared derived storage)::

    rows.npy      int64 [M]        cache rows covered (sorted)
    tokens.f16    float16 [M, 12, 47, 768]  every output token of every lead
    mask.u8       uint8 [M, 47]    token validity (1 valid, 0 masked)
    tokens.done   uint8 per row: 1 encoded, 2 the signal failed (zeros, never encoded)
    meta.json     provenance; ``complete`` once every row is done

The caches are regenerable. Delete them when the programme ends (amendment B3).

Token order and validity (checked against ``ST_MEM.forward_encoder``)
-------------------------------------------------------------------
The encoder output of one lead is ``[left SEP, patch 0, ..., patch 44, right
SEP]``: 47 tokens.

* ``to_patch_embedding`` cuts the 2250 samples with ``'b c (n p) -> b c n p'``
  and ``p = 50``. Patch ``i`` (token ``i + 1``) is therefore samples
  ``[50 i, 50 i + 50)``.
* The SEP tokens are always valid.
* A patch token is valid iff its 50 samples lie entirely inside the record's
  support: the span between the first and last non-zero sample across leads
  (``src.objectives.synth_acq.support_mask``).
* Patches that touch the anchor's symmetric zero padding are masked, because
  the padding is label-informative (project constraint 3, D-035).

``token_validity`` computes the mask from the cached signal. The encoder runs
in float32, as for the CLS cache (``src.data.features``); the tokens are stored
in float16. Token 0 is the CLS feature, equal to the CLS cache up to float16
rounding (tested).

CLI::

    python -m src.data.features_tokens --encoder-id <rid> --encoder-ckpt results/runs/<rid>/encoder.pt \\
        (--pool-folds k[,k2...] | --all) [--device cuda:0] [--batch-size 64]

``--pool-folds k`` covers fold k's inner-protocol pool: every row with
``outer_fold >= 0`` and ``outer_fold != k`` (training rows plus inner fold 3).
``--all`` covers every row of the cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.data.signals import N_LEADS, SEQ_LEN, open_done_bitmap, open_signal_cache, read_cache_meta
from src.models import anchor as A

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS_ROOT = REPO_ROOT / "data" / "cache" / "features_tokens"
PATCH = 50
N_PATCHES = SEQ_LEN // PATCH  # 45
N_TOKENS = N_PATCHES + 2  # 47: [SEP, 45 patches, SEP]
EMB_DIM = A.EMB_DIM
DTYPE = np.float16
TOKEN_BYTES = N_LEADS * N_TOKENS * EMB_DIM * np.dtype(DTYPE).itemsize
VERSION = 1


def tokens_dir(encoder_id: str, root: Path | str = TOKENS_ROOT) -> Path:
    return Path(root) / encoder_id


def token_validity(x: np.ndarray) -> np.ndarray:
    """``x [B, 2250, 12]`` (time first, as cached) -> bool ``[B, 47]``. The SEP tokens are always valid; a patch
    token is valid iff its 50 samples lie inside the support (first to last non-zero sample across leads). An
    all-zero record gets a full support, as in ``synth_acq.support_mask``."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] != SEQ_LEN:
        raise ValueError(f"expected [B, {SEQ_LEN}, leads], got {x.shape}")
    nz = (x != 0).any(axis=2)  # [B, T]
    t = np.arange(SEQ_LEN)
    first = np.where(nz.any(axis=1), np.where(nz, t, SEQ_LEN).min(axis=1), 0)
    last = np.where(nz.any(axis=1), np.where(nz, t, -1).max(axis=1), SEQ_LEN - 1)
    start = np.arange(N_PATCHES) * PATCH
    patch_ok = (start[None, :] >= first[:, None]) & (start[None, :] + PATCH - 1 <= last[:, None])
    out = np.ones((x.shape[0], N_TOKENS), dtype=bool)
    out[:, 1:-1] = patch_ok
    return out


@torch.no_grad()
def encode_tokens(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``x [B, 2250, 12]`` -> every output token ``[B, 12, 47, D]`` (float32, the encoder's device)."""
    if x.ndim != 3 or x.shape[1] != SEQ_LEN or x.shape[2] != N_LEADS:
        raise ValueError(f"expected [B, {SEQ_LEN}, {N_LEADS}], got {tuple(x.shape)}")
    if int(encoder.num_leads) != 1:
        raise ValueError("token caches are defined for the anchor's shared per-lead encoder (num_leads=1)")
    dev = next(encoder.parameters()).device
    b = x.shape[0]
    leads = x.to(dev, dtype=torch.float32).permute(0, 2, 1).reshape(b * N_LEADS, 1, SEQ_LEN)
    encoder.eval()
    tok, _, _ = encoder.forward_encoder(leads, 0)  # [B*12, 47, D]
    if tok.shape[1] != N_TOKENS:
        raise ValueError(f"encoder emitted {tok.shape[1]} tokens per lead, expected {N_TOKENS}")
    return tok.reshape(b, N_LEADS, N_TOKENS, tok.shape[-1])


def pool_rows(outer: np.ndarray, folds) -> np.ndarray:
    """Rows of the inner-protocol pools of ``folds``: ``outer_fold >= 0`` and ``!= k`` (union over ``folds``)."""
    outer = np.asarray(outer)
    keep = np.zeros(len(outer), dtype=bool)
    for k in folds:
        keep |= (outer >= 0) & (outer != int(k))
    return np.flatnonzero(keep)


def _memmap(path: Path, shape: tuple, dtype) -> np.memmap:
    want = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.exists() and path.stat().st_size != want:
        raise RuntimeError(f"{path} has {path.stat().st_size} bytes, expected {want}; delete the directory to rebuild")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.truncate(want)
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def build_token_cache(encoder_id: str, signal_cache: Path | str, rows: np.ndarray, out_dir: Path | str | None = None,
                      *, device: str | torch.device = "cuda:0", encoder: torch.nn.Module | None = None,
                      encoder_sha256: Optional[str] = None, batch_size: int = 64, log_every: int = 100) -> dict[str, Any]:
    """Encode ``rows`` of ``signal_cache`` into a token cache (resumable)."""
    if encoder is None:
        raise ValueError("an encoder is required")
    signal_cache = Path(signal_cache)
    out_dir = tokens_dir(encoder_id) if out_dir is None else Path(out_dir)
    sig_meta = read_cache_meta(signal_cache)
    n = int(sig_meta["n"])
    signals = open_signal_cache(signal_cache)
    assert signals.shape == (n, SEQ_LEN, N_LEADS), signals.shape
    done_src = Path(str(signal_cache) + ".done")
    source = np.fromfile(done_src, dtype=np.uint8, count=n) if done_src.exists() else np.ones(n, dtype=np.uint8)
    encoder = encoder.to(device).eval()
    if int(encoder.encoder.width) != EMB_DIM:
        raise ValueError(f"encoder width {encoder.encoder.width} != {EMB_DIM}")
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "rows.npy").exists():
        if not np.array_equal(np.load(out_dir / "rows.npy"), rows):
            raise RuntimeError(f"{out_dir} covers other rows; delete it to rebuild")
    else:
        np.save(out_dir / "rows.npy", rows)
    m = len(rows)
    toks = _memmap(out_dir / "tokens.f16", (m, N_LEADS, N_TOKENS, EMB_DIM), DTYPE)
    mask = _memmap(out_dir / "mask.u8", (m, N_TOKENS), np.uint8)
    done = open_done_bitmap(out_dir / "tokens", m, mode="r+")  # -> tokens.done
    failed = source[rows] == 2
    d = np.asarray(done)
    done[(d == 0) & failed] = 2
    if (source[rows] == 0).any():
        raise RuntimeError("the signal cache has unbuilt rows among the requested ones")
    todo = np.flatnonzero(np.asarray(done) == 0)
    print(f"[features_tokens] {encoder_id}: {len(todo)}/{m} rows to encode on {device}", flush=True)
    t0, n_done = time.time(), 0
    for bi, s in enumerate(range(0, len(todo), batch_size), start=1):
        pos = todo[s:s + batch_size]
        r = rows[pos]
        x = np.array(signals[r], dtype=np.float32)  # rows sorted: a monotone read
        out = encode_tokens(encoder, torch.from_numpy(x))
        toks[pos] = out.to(torch.float16).cpu().numpy()
        mask[pos] = token_validity(x).astype(np.uint8)
        done[pos] = 1
        n_done += len(pos)
        if bi % log_every == 0 or s + batch_size >= len(todo):
            toks.flush()
            mask.flush()
            done.flush()
            dt = time.time() - t0
            print(f"[features_tokens] {n_done}/{len(todo)} ({n_done / max(dt, 1e-9):.1f} rec/s, {dt / 60:.1f} min)",
                  flush=True)
    toks.flush()
    mask.flush()
    done.flush()
    d = np.asarray(done)
    meta = {"version": VERSION, "encoder_id": encoder_id, "encoder_sha256": encoder_sha256,
            "signal_cache": str(signal_cache), "signal_records_sha256": sig_meta.get("records_sha256"),
            "preproc": sig_meta.get("preproc"), "n_cache": n, "n_rows": m, "shape": [m, N_LEADS, N_TOKENS, EMB_DIM],
            "dtype": "float16", "patch": PATCH, "token_order": "SEP, patch 0..44 (samples [50i, 50i+50)), SEP",
            "validity": "SEP always; a patch iff inside the support (first..last non-zero sample across leads)",
            "n_failed": int((d == 2).sum()), "complete": bool((d > 0).all()), "device": str(device),
            "regenerable": True}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[features_tokens] wrote {out_dir / 'meta.json'}: complete={meta['complete']}", flush=True)
    return meta


class TokenView:
    """Cache-row-indexed view of a token cache: ``view[rows]`` -> float32 ``[b, 12, 47, 768 + 1]``.

    The last channel holds the token's validity (1 / 0, the same for every lead): this is how the attention
    head receives its mask through the ordinary feature paths. Reading a row the cache does not cover
    raises."""

    def __init__(self, cache: dict[str, Any], n_cache: int) -> None:
        self.rows, self.tokens, self.mask = cache["rows"], cache["tokens"], cache["mask"]
        self.pos_of = np.full(int(n_cache), -1, dtype=np.int64)
        self.pos_of[self.rows] = np.arange(len(self.rows))
        self.shape = (int(n_cache), N_LEADS, N_TOKENS, EMB_DIM + 1)
        self.dtype = np.float32
        self.ndim = 4

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> np.ndarray:
        rows = np.arange(self.shape[0])[idx]
        scalar = np.ndim(rows) == 0
        rows = np.atleast_1d(rows)
        pos = self.pos_of[rows]
        if (pos < 0).any():
            raise KeyError(f"{int((pos < 0).sum())} row(s) are not in this token cache")
        order = np.argsort(pos, kind="stable")
        out = np.empty((len(pos), N_LEADS, N_TOKENS, EMB_DIM + 1), dtype=np.float32)
        out[order, ..., :EMB_DIM] = np.asarray(self.tokens[pos[order]], dtype=np.float32)
        out[..., EMB_DIM] = np.asarray(self.mask[pos], dtype=np.float32)[:, None, :]
        return out[0] if scalar else out


def open_tokens(encoder_id: str, root: Path | str = TOKENS_ROOT) -> dict[str, Any]:
    d = tokens_dir(encoder_id, root)
    meta = json.loads((d / "meta.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError(f"{d}: token cache incomplete")
    rows = np.load(d / "rows.npy")
    m = len(rows)
    return {"rows": rows, "meta": meta,
            "tokens": np.memmap(d / "tokens.f16", dtype=DTYPE, mode="r", shape=(m, N_LEADS, N_TOKENS, EMB_DIM)),
            "mask": np.memmap(d / "mask.u8", dtype=np.uint8, mode="r", shape=(m, N_TOKENS))}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("CLI::")[0])
    ap.add_argument("--encoder-id", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pool-folds", help="outer folds whose inner-protocol pool is covered")
    g.add_argument("--all", action="store_true", help="every row of the cache")
    ap.add_argument("--signals", default=str(REPO_ROOT / "data" / "cache" / "signals_250hz.f32"))
    ap.add_argument("--metadata", default=str(REPO_ROOT / "data" / "cache" / "metadata.parquet"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "splits" / "frozen_splits.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args(argv)
    from src import run as runmod

    meta = runmod.load_world({"metadata": a.metadata, "splits": a.splits})
    rows = np.arange(len(meta)) if a.all else pool_rows(meta["outer_fold"].to_numpy(),
                                                        [int(x) for x in a.pool_folds.split(",")])
    encoder = A.load_encoder_from_file(a.encoder_ckpt, device=a.device)
    out = build_token_cache(a.encoder_id, a.signals, rows, a.out_dir, device=a.device, encoder=encoder,
                            encoder_sha256=A.sha256_of_file(a.encoder_ckpt), batch_size=a.batch_size)
    return 0 if out["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
