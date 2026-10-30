"""The anchor encoder: the shipped ``good_disco`` ST-MEM, loaded verifiably.

What this module does
---------------------
Submission 2813 never pretrains anything. ``mm_mae/main_mm_mae_stripped.py``
byte-concatenates the 66 chunks under ``mm_mae/pretrained_enc/good_disco``,
``torch.load``s the result, and fits twelve MLP heads on top of a frozen
encoder (HANDOFF.md section 2, D-009). Everything the paper says about "our
model" therefore rests on one binary blob that this repository cannot
regenerate. This module is the single place that blob is turned into a
``torch.nn.Module``, and it does three things the anchor code does not:

1. It records a content hash. ``data/manifests/good_disco.json`` holds the
   sha256 of the concatenated chunks together with the checkpoint's own
   ``epoch``/``global_step`` and a subset of its recorded hyperparameters, so
   that a feature cache or a registry row can be traced to the exact weights.
   ``load_anchor_encoder(verify=True)`` refuses to return a model whose bytes
   disagree with the manifest.

2. It loads exactly one ``ST_MEM``. The Lightning checkpoint carries twelve
   copies of the shared module under ``encoders.m0.`` .. ``encoders.m11.``
   (``share_weights: True`` registers the same object twelve times). We load
   the ``m0`` subtree with ``strict=True`` and assert every other prefix is a
   tensor-for-tensor identical copy, so a checkpoint that was *not* produced
   with shared weights cannot slip through looking like the anchor.

3. It exposes :func:`encode`, the feature contract in ARCHITECTURE.md: for a
   batch ``[B, 2250, 12]`` of anchor-preprocessed signals, ``cls`` is token 0
   of ``forward_encoder(lead, mask_ratio=0)`` per lead (what the anchor's
   ``EnsembleClassifier`` reads), and ``mean`` is the mean over all 47 tokens
   (what the MSE-alignment term acts on). The twelve leads are pushed through
   the encoder in one forward as a ``[B*12, 1, 2250]`` batch; since the model
   is constructed with ``num_leads=1`` this is exactly the anchor's per-lead
   loop, only faster. ``tests/test_anchor.py`` asserts that equivalence.
   A *joint* encoder (``num_leads=12``, half of the S4 sweep, written by
   ``src.objectives.model.save_encoder``) takes the same input as one
   ``[B, 12, 2250]`` sequence; its tokens come back lead-major
   (``'b (c n) p'``) and are unfolded to the same ``[B, 12, 768]`` outputs, so
   the feature cache has one layout for both modes.

Why it lives here and not in ``team/``
--------------------------------------
Project rule 12: the anchor is not refactored. ``team/utils.py`` holds a
global config cache and imports wandb, hydra and matplotlib on import; none of
that is needed to load weights and run an encoder.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
MM_MAE_ROOT = REPO_ROOT / "mm_mae"

# ``team.*`` and ``networks.*`` are imported by path, exactly as
# ``team/utils.py`` does it (``sys.path.append``), so that no anchor file has
# to change and mm_mae's top-level ``config``/``utils``/``datasets``/``eval``
# packages never shadow site-packages.
for _p in (str(REPO_ROOT), str(MM_MAE_ROOT)):
    if _p not in sys.path:
        sys.path.append(_p)

from networks.st_mem import ST_MEM  # noqa: E402  (path-dependent import)

#: The shipped checkpoint, as 66 byte-concatenated chunks.
GOOD_DISCO_DIR = MM_MAE_ROOT / "pretrained_enc" / "good_disco"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "good_disco.json"

#: ``encoder_id`` of the shipped checkpoint everywhere in ``data/cache``.
ANCHOR_ENCODER_ID = "anchor_good_disco"

#: The state_dict prefix of the shared module (``ModuleDict`` key ``m0``).
SHARED_PREFIX = "encoders.m0."

#: Architecture constants of the anchor (``main_mm_mae_stripped.init_model``
#: with ``vit_base_{encoder,decoder}_module`` and ``model.patch_size: 50``).
SEQ_LEN = 2250
PATCH_SIZE = 50
N_LEADS = 12
EMB_DIM = 768
N_TOKENS = SEQ_LEN // PATCH_SIZE + 2  # 45 patches + 2 SEP = 47

ANCHOR_ARCH: dict[str, Any] = dict(
    seq_len=SEQ_LEN,
    patch_size=PATCH_SIZE,
    num_leads=1,
    embed_dim=EMB_DIM,
    depth=12,
    num_heads=12,
    decoder_embed_dim=768,
    decoder_depth=4,
    decoder_num_heads=12,
    mlp_ratio=4,
)

#: Hyperparameters copied verbatim from the checkpoint into the manifest. The
#: full ``hyper_parameters`` dict also holds ``repr``s of live objects
#: (``ModuleDict``, an evaluator instance) that are neither stable nor useful.
HPARAMS_SUBSET = (
    "learning_rate",
    "learning_rate_weight_decay",
    "batch_size",
    "num_nodes",
    "num_gpus",
    "n_epochs",
    "n_epochs_warmup",
    "patch_size",
    "mask_ratio",
    "regularization_tokens",
    "regularization_metric",
    "use_latent_heads",
    "compute_rec",
    "temperature",
    "use_bce_loss",
    "modality_names",
)

_CHUNK_READ = 16 << 20


# --------------------------------------------------------------------------- #
# Bytes: hashing and concatenation
# --------------------------------------------------------------------------- #


def chunk_paths(directory: Path | str = GOOD_DISCO_DIR) -> list[Path]:
    """The chunk files in the order they were split (sorted by name)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no chunk files under {directory}")
    return paths


def _iter_chunk_bytes(directory: Path | str) -> Iterator[bytes]:
    for p in chunk_paths(directory):
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(_CHUNK_READ), b""):
                yield block


def sha256_of_chunks(directory: Path | str = GOOD_DISCO_DIR) -> str:
    """sha256 of the byte-concatenated chunks, streamed (never all in RAM twice)."""
    h = hashlib.sha256()
    for block in _iter_chunk_bytes(directory):
        h.update(block)
    return h.hexdigest()


def _read_checkpoint(directory: Path | str) -> tuple[dict[str, Any], str, int]:
    """One pass over the chunks: returns ``(checkpoint, sha256, total_bytes)``.

    Hashing and concatenating in the same pass halves the NFS traffic; on this
    filesystem the read is ~5 s and the ``torch.load`` ~25 s.
    """
    h = hashlib.sha256()
    buf = io.BytesIO()
    for block in _iter_chunk_bytes(directory):
        h.update(block)
        buf.write(block)
    total = buf.tell()
    buf.seek(0)
    # weights_only=False: the Lightning checkpoint pickles omegaconf/hparams
    # objects; we take only ``state_dict`` and scalars out of it.
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    return ckpt, h.hexdigest(), total


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _manifest_from_checkpoint(
    ckpt: dict[str, Any], sha: str, total: int, n_chunks: int
) -> dict[str, Any]:
    hp = ckpt.get("hyper_parameters") or {}
    hp_sub = {k: _jsonable(hp[k]) for k in HPARAMS_SUBSET if k in hp}
    return {
        "encoder_id": ANCHOR_ENCODER_ID,
        "path": str(GOOD_DISCO_DIR.relative_to(REPO_ROOT)),
        "sha256": sha,
        "n_chunks": n_chunks,
        "total_bytes": total,
        "epoch": _jsonable(ckpt.get("epoch")),
        "global_step": _jsonable(ckpt.get("global_step")),
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
        "n_state_dict_keys": len(ckpt["state_dict"]),
        "arch": ANCHOR_ARCH,
        "hparams": hp_sub,
    }


def _jsonable(v: Any) -> Any:
    """JSON-safe copy of a checkpoint hyperparameter.

    Lightning stores some hparams as ``repr`` strings (``modality_names`` is
    ``"['m0', ..., 'm11']"``); a string that parses as a Python literal
    container is turned back into that container so the manifest holds a list,
    not a string that happens to look like one.
    """
    if isinstance(v, str):
        if v[:1] in "[({":
            try:
                return _jsonable(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                return v
        return v
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, torch.Tensor):
        return v.tolist()
    return str(v)


def read_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def write_manifest(
    directory: Path | str = GOOD_DISCO_DIR, path: Path | str = MANIFEST_PATH
) -> dict[str, Any]:
    """Hash the chunks, read the checkpoint and write ``good_disco.json``.

    Also returns the manifest. Idempotent: rewriting produces identical bytes
    for identical chunks.
    """
    ckpt, sha, total = _read_checkpoint(directory)
    manifest = _manifest_from_checkpoint(ckpt, sha, total, len(chunk_paths(directory)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


# --------------------------------------------------------------------------- #
# Model construction and loading
# --------------------------------------------------------------------------- #


def build_anchor_st_mem() -> ST_MEM:
    """A fresh, randomly initialised ``ST_MEM`` with the anchor's architecture.

    Needed wherever the architecture is required without the shipped weights:
    the objective sweep (S4) trains from this, and tests use it on CPU.
    """
    return ST_MEM(**ANCHOR_ARCH)


def _split_by_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group ``encoders.<mod>.<rest>`` keys by ``encoders.<mod>.``."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        parts = k.split(".", 2)
        if len(parts) < 3 or parts[0] != "encoders":
            raise KeyError(f"unexpected state_dict key (not under 'encoders.'): {k}")
        prefix = f"{parts[0]}.{parts[1]}."
        groups.setdefault(prefix, {})[parts[2]] = v
    return groups


def shared_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the shared module's weights from a Lightning ``state_dict``.

    Returns the ``encoders.m0.`` subtree with the prefix stripped. Every other
    ``encoders.m<i>.`` subtree present must be identical tensor for tensor;
    a difference means the checkpoint is not a weight-shared model and must
    not be loaded as the anchor. A plain (already stripped) ``ST_MEM``
    state_dict is passed through unchanged.
    """
    if not any(k.startswith("encoders.") for k in state_dict):
        return dict(state_dict)
    groups = _split_by_prefix(state_dict)
    if SHARED_PREFIX not in groups:
        raise KeyError(f"no '{SHARED_PREFIX}' subtree; prefixes: {sorted(groups)}")
    base = groups[SHARED_PREFIX]
    for prefix, other in groups.items():
        if prefix == SHARED_PREFIX:
            continue
        if set(other) != set(base):
            raise ValueError(f"{prefix} keys differ from {SHARED_PREFIX}")
        for name, t in base.items():
            if t.dtype != other[name].dtype or not torch.equal(t, other[name]):
                raise ValueError(
                    f"{prefix}{name} differs from {SHARED_PREFIX}{name}: "
                    "checkpoint is not weight-shared"
                )
    return base


def load_encoder_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
    arch: dict[str, Any] | None = None,
    num_leads: int | None = None,
) -> ST_MEM:
    """Strict-load a (Lightning-prefixed or plain) state_dict into an ``ST_MEM``.

    Defaults to the anchor architecture (``num_leads=1``). ``arch`` and
    ``num_leads`` override it for encoders written by
    ``src.objectives.model.save_encoder``; ``arch`` may or may not carry its
    own ``num_leads`` key, and an explicit ``num_leads`` wins.
    """
    kw = dict(ANCHOR_ARCH if arch is None else arch)
    if num_leads is not None:
        kw["num_leads"] = int(num_leads)
    kw.setdefault("num_leads", 1)
    model = ST_MEM(**kw)
    model.load_state_dict(shared_encoder_state_dict(state_dict), strict=True)
    model.arch = {k: v for k, v in kw.items() if k != "num_leads"}  # type: ignore[attr-defined]
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_encoder_from_file(path: Path | str, device: str | torch.device = "cpu") -> ST_MEM:
    """Load an ``ST_MEM`` from a single ``.pt``/``.ckpt`` file (a run's own encoder).

    Accepts a bare state_dict, ``{"state_dict": ...}`` (Lightning), or
    ``{"model": ...}``. A payload from ``src.objectives.model.save_encoder``
    (``{"arch", "num_leads", "mode", "state_dict"}``) is rebuilt with *its*
    architecture, so joint (``num_leads=12``) encoders load too; anything
    without those keys is assumed to be the anchor architecture. Used by
    ``src.data.features --encoder-ckpt``.
    """
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = num_leads = mode = None
    if isinstance(obj, dict):
        arch = obj.get("arch") if isinstance(obj.get("arch"), dict) else None
        num_leads = obj.get("num_leads")
        mode = obj.get("mode")
        for key in ("state_dict", "model", "encoder"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected a state_dict, got {type(obj).__name__}")
    model = load_encoder_from_state_dict(obj, device, arch=arch, num_leads=num_leads)
    if mode is not None:
        model.mode = mode  # type: ignore[attr-defined]
    return model


def sha256_of_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK_READ), b""):
            h.update(block)
    return h.hexdigest()


def load_anchor_encoder(
    device: str | torch.device = "cpu",
    verify: bool = True,
    directory: Path | str = GOOD_DISCO_DIR,
    manifest_path: Path | str = MANIFEST_PATH,
) -> ST_MEM:
    """The shipped encoder, in eval mode, on ``device``, with its bytes verified.

    On the first call (no manifest yet) the manifest is written. Afterwards
    ``verify=True`` recomputes the sha256 of the chunks and raises
    ``RuntimeError`` on mismatch, so an overwritten or partially copied
    checkpoint can never silently become "the anchor". ``verify=False`` skips
    only the comparison, not the load.
    """
    ckpt, sha, total = _read_checkpoint(directory)
    n_chunks = len(chunk_paths(directory))
    manifest = read_manifest(manifest_path)
    if manifest is None:
        manifest = _manifest_from_checkpoint(ckpt, sha, total, n_chunks)
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
    elif verify:
        if (
            manifest["sha256"] != sha
            or manifest["total_bytes"] != total
            or manifest.get("n_chunks") != n_chunks
        ):
            raise RuntimeError(
                f"good_disco checkpoint does not match {manifest_path}: "
                f"sha256 {sha} ({total} bytes, {n_chunks} chunks) vs manifest "
                f"{manifest['sha256']} ({manifest['total_bytes']} bytes, "
                f"{manifest.get('n_chunks')} chunks)"
            )
    model = load_encoder_from_state_dict(ckpt["state_dict"], device)
    model.anchor_sha256 = sha  # type: ignore[attr-defined]  # provenance tag read by features.py
    return model


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def _forward_tokens(encoder: ST_MEM, leads: torch.Tensor) -> torch.Tensor:
    """``leads`` ``[M, L, 2250]`` -> all tokens ``[M, L*47, D]`` (mask_ratio=0).

    For ``L=1`` (the anchor) that is ``[M, 47, D]``; for a joint encoder the
    twelve leads' tokens are concatenated lead-major (``'b (c n) p'`` in
    ``ST_MEM.forward_encoder``).
    """
    latent, _mask, _ids = encoder.forward_encoder(leads, 0)
    return latent


@torch.no_grad()
def encode(
    encoder: ST_MEM, x: torch.Tensor, batch_leads: bool = True
) -> dict[str, torch.Tensor]:
    """Anchor features for a batch of preprocessed signals.

    Args:
        encoder: an ``ST_MEM`` with ``num_leads=1`` (the anchor: leads are
           folded into the batch) or ``num_leads=12`` (a joint encoder from
           the S4 sweep: leads share one sequence).
        x: float32 ``[B, 2250, 12]`` -- the output of
           ``src.data.signals.preprocess_record`` (time first, leads last,
           header lead order I, II, III, aVR, aVL, aVF, V1..V6).
        batch_leads: push all ``B*12`` leads through one forward (default).
           ``False`` runs the anchor's literal per-lead loop; it exists so a
           test can assert the two agree. Ignored for a joint encoder, which
           has no per-lead form.

    Returns:
        ``{"cls": [B, 12, D], "mean": [B, 12, D]}`` float32 on the encoder's
        device (``D = 768`` for the anchor). ``cls`` is token 0 of each lead
        (the left SEP token, which the anchor's ``feature_policy: cls``
        reads); ``mean`` is the mean over all 47 tokens of the lead.
    """
    if x.ndim != 3 or x.shape[1] != SEQ_LEN or x.shape[2] != N_LEADS:
        raise ValueError(f"expected [B, {SEQ_LEN}, {N_LEADS}], got {tuple(x.shape)}")
    if encoder.num_leads not in (1, N_LEADS):
        raise ValueError(
            f"encode() takes a num_leads=1 (anchor) or num_leads={N_LEADS} (joint) "
            f"encoder, got num_leads={encoder.num_leads}"
        )
    device = next(encoder.parameters()).device
    x = x.to(device=device, dtype=torch.float32)
    b = x.shape[0]
    was_training = encoder.training
    encoder.eval()
    try:
        if encoder.num_leads == N_LEADS:
            tok = _forward_tokens(encoder, x.permute(0, 2, 1))  # [B, 12*47, D], lead-major
            tok = tok.reshape(b, N_LEADS, -1, tok.shape[-1])  # [B, 12, 47, D]
            cls = tok[:, :, 0, :]
            mean = tok.mean(dim=2)
        elif batch_leads:
            leads = x.permute(0, 2, 1).reshape(b * N_LEADS, 1, SEQ_LEN)
            tok = _forward_tokens(encoder, leads)  # [B*12, 47, D]
            d = tok.shape[-1]
            cls = tok[:, 0, :].reshape(b, N_LEADS, d)
            mean = tok.mean(dim=1).reshape(b, N_LEADS, d)
        else:
            cls_l, mean_l = [], []
            for i in range(N_LEADS):
                tok = _forward_tokens(encoder, x[:, :, i].unsqueeze(1))  # [B, 47, 768]
                cls_l.append(tok[:, 0, :])
                mean_l.append(tok.mean(dim=1))
            cls = torch.stack(cls_l, dim=1)
            mean = torch.stack(mean_l, dim=1)
    finally:
        if was_training:
            encoder.train()
    return {"cls": cls.contiguous(), "mean": mean.contiguous()}


# --------------------------------------------------------------------------- #
# CLI: write / check the manifest
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Write or verify data/manifests/good_disco.json for the anchor checkpoint."
    )
    ap.add_argument("--dir", default=str(GOOD_DISCO_DIR))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--check", action="store_true", help="verify instead of (re)writing")
    args = ap.parse_args(argv)
    if args.check:
        load_anchor_encoder("cpu", verify=True, directory=args.dir, manifest_path=args.manifest)
        print(f"OK: {args.dir} matches {args.manifest}")
        return 0
    m = write_manifest(args.dir, args.manifest)
    print(json.dumps({k: m[k] for k in ("sha256", "n_chunks", "total_bytes", "epoch", "global_step")}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
