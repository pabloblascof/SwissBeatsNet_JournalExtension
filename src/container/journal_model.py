"""The journal model as the organisers' harness sees it (PLAN.md S7.2 / S7.4).

What this module is
-------------------
``run_model.py`` -- the organisers' entry point, unmodifiable -- calls
``team_code.load_model(model_folder)`` once and then
``team_code.run_model(record_path, model)`` per record, expecting
``(binary_output: int, probability_output: float)``. ``team_code`` dispatches to
this module whenever the model folder contains ``journal_manifest.json``; the
anchor path (CinC submission 2813) is untouched otherwise.

A *journal model folder* is::

    model/
      journal_manifest.json      the layout below
      encoder.pt                 optional; {"arch": {...}, "state_dict": {...}}
      head_<slot>.pt             one EnsembleHead state_dict per slot

    journal_manifest.json:
    {"format": "swissbeatsnet-journal-v1",
     "slots": [{"name": "final", "encoder": "encoder.pt" | "good_disco",
                "encoder_arch": {...}, "head": "head_final.pt",
                "head_kind": "non_linear", "feature": "cls",
                "preproc": "anchor_v1"}, ...],
     "output_slot": "final",
     "binarization": {"kind": "top_fraction", "fraction": 0.05,
                      "threshold_logit": 1.23}
                  |  {"kind": "logit_threshold", "value": 0.0},
     "tta": ["head", "tail", "center"]}        # optional (S5.5)

Test-time augmentation (optional, S5.5)
---------------------------------------
The anchor keeps the first 2250 samples at 250 Hz (9.0 s) of a longer record.
With ``"tta"`` present, every slot's logit is the mean of its logits over
those crops of the same preprocessed record (``src.data.signals.CROPS``:
``head`` = the anchor's window, ``tail`` = the last 2250 samples, ``center``
= the middle 2250). Preprocessing runs once per record; the encoder runs once
per crop. Records of at most 2250 samples are padded, so their crops coincide
and TTA leaves their logit unchanged up to float rounding. Absent ``"tta"``
the code path is exactly the one without it.

Why several slots but one output
--------------------------------
The harness scores exactly one probability and one label per record. Whether a
single container run can score several configurations is still an open
organiser question (``docs/journal/ORGANISER_EMAIL.md``), so the manifest can
carry several slots -- every slot is evaluated on every record -- but only
``output_slot`` reaches the harness. The other slots' logits are kept in memory
(``JournalModel.extra_scores``) and, when the environment variable
``SBN_EXTRA_OUTPUT_DIR`` is set (only ever on our side, for dry runs), appended
to ``<dir>/extra_slots.jsonl``. The container filesystem is not persistent on
the organisers' side, so nothing is written there unless that variable is set.

k-slot output combination (optional, S4C amendment B4)
-----------------------------------------------------
With an ``"output"`` key the harness's score is not one slot's logit but a
fixed combination of several slots' logits:

* ``{"kind": "mean_logit", "slots": [...], "zmap": {slot: {"mean", "sd"}}}`` --
  the mean of the listed slots' logits, each z-mapped first if ``zmap`` names
  it. The z-map constants come from the slot's own training pool, as
  ``scripts/experiments/ensemble_runs`` uses them. This covers the raw X1 /
  ``cmens_e1`` form and the z-scored X2 form.
* ``{"kind": "affine", "terms": [{"slot", "weight", "shift", "scale"}, ...],
  "offset": c}`` -- ``c + sum w_i (logit_i - shift_i) / scale_i``. This covers
  forms such as E3's: arm M's seeds z-mapped and averaged, then put back on
  seed 0's logit scale.

``output_slot`` stays required and is used only for logging. The binarization
threshold is in the combination's units (``export_model`` computes it from the
combined training-pool scores). A record with any non-finite member logit
scores ``(0, 0.0)``, as a failed record does. Without ``"output"``, the code path
is exactly the single-slot one.

Why the probability is a sigmoid
--------------------------------
D-010: the anchor wrote the raw ensemble logit into the "probability" column and
thresholded it at 0.5. Here ``probability = sigmoid(logit)`` and the binary label
comes from the manifest's binarization rule, applied in logit space so that
``top_fraction`` and ``logit_threshold`` are both a single comparison.

Failure policy
--------------
``run_model.py`` without ``-f`` aborts the whole run on the first exception, and
the run cannot be restarted. A record that cannot be read or preprocessed, or
whose features are non-finite, therefore yields ``(0, 0.0)`` and a logged
warning instead of an exception. Short signals (<= 50 samples) go through the
same code path as the anchor (preprocessing skipped, zero-padded) because the
preprocessing *is* the anchor's ``team.data.load_and_process_signal``.

That per-record tolerance must not hide a *systematic* failure (a head whose
width disagrees with the encoder, a CUDA fault, a broken preprocessing
dependency): the same policy applied 81,000 times would hand the organisers a
complete, plausible-looking, all-zero output set. Two guards:

* ``JournalModel.load`` runs a self-test -- a zero signal and a fixed
  pseudo-random signal of the contract shape through every slot -- and raises
  unless every logit is finite. Load-time failure is visible; record-time
  failure is not.
* A circuit breaker in ``predict``: if the first ``BREAKER_FIRST_N`` records
  all fail, or more than ``BREAKER_MAX_FRACTION`` of at least
  ``BREAKER_MIN_SEEN`` records have failed, ``predict`` raises so the run
  stops early rather than finishing wrong.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "mm_mae")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Matplotlib is imported transitively by team.utils; its default config dir is
# not writable in the container, and the warning it prints otherwise is noise.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sbn_mpl")

from team.data import load_and_process_signal, load_sampling_rate  # noqa: E402
from team.utils import (  # noqa: E402
    HeartbeatMode,
    NormalizationMode,
    PaddingMode,
    PreprocessingMode,
)
from team.utils_challenge import load_header  # noqa: E402
from networks.st_mem import ST_MEM  # noqa: E402

log = logging.getLogger("sbn.container")

MANIFEST_NAME = "journal_manifest.json"
MANIFEST_FORMAT = "swissbeatsnet-journal-v1"
GOOD_DISCO_DIR = REPO_ROOT / "mm_mae" / "pretrained_enc" / "good_disco"
EXTRA_OUTPUT_ENV = "SBN_EXTRA_OUTPUT_DIR"
EXTRA_OUTPUT_FILE = "extra_slots.jsonl"

#: The anchor's preprocessing (mm_mae/config/chagas_config_data.py), frozen.
ANCHOR_PREPROC = {
    "name": "anchor_v1",
    "target_sr": 250,
    "preprocessing": PreprocessingMode.TEAM,
    "intensity_normalization": NormalizationMode.STANDARDIZE_REGULAR_LEADWISE,
    "standard_len": 2250,
    "pad_mode": PaddingMode.SYMMETRIC,
    "heartbeat_mode": HeartbeatMode.MULTIPLE,
}
N_LEADS = 12

#: The anchor encoder as instantiated by ``main_mm_mae_stripped.init_model``
#: with the ViT-Base config and ``patch_size 50``.
ANCHOR_ENCODER_ARCH = {
    "seq_len": 2250,
    "patch_size": 50,
    "num_leads": 1,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "decoder_embed_dim": 768,
    "decoder_depth": 4,
    "decoder_num_heads": 12,
    "mlp_ratio": 4,
}

HEAD_KINDS = ("non_linear", "linear", "lead_attention", "concat_mlp", "pooled_ensemble")  # = src.models.heads.HEAD_KINDS
#: ``both`` = CLS and mean concatenated per lead (CLS first), width 2 * embed_dim --
#: the order of ``src.run.ConcatFeatures``, which the heads were fitted on.
FEATURES = ("cls", "mean", "both")
FEATURE_WIDTH_FACTOR = {"cls": 1, "mean": 1, "both": 2}
PREPROCS = ("anchor_v1",)
ERASES = ("none", "leace_source")  # = src.models.heads.ERASURES
TTA_CROPS = ("head", "tail", "center")  # = src.data.signals.CROPS
#: ``"output"`` combination kinds (S4C amendment B4; module docstring).
OUTPUT_KINDS = ("mean_logit", "affine")
ENCODE_CHUNK = 32  # records per encoder forward (32 * 12 lead sequences)

#: Circuit breaker (module docstring, "Failure policy"). Class attributes on
#: JournalModel so a test can tighten them without touching the env.
BREAKER_FIRST_N = 20        # every one of the first N records failed -> abort
BREAKER_MIN_SEEN = 100      # fraction rule applies only after this many records
BREAKER_MAX_FRACTION = 0.05  # > this fraction failed -> abort

#: Keys a file encoder's ``encoder_arch`` must state explicitly; the rest have
#: ST_MEM defaults that match the anchor.
REQUIRED_FILE_ARCH_KEYS = ("embed_dim", "depth", "num_heads")


class ManifestError(ValueError):
    """The manifest is malformed. Raised at load time, never per record."""


class SystematicFailure(RuntimeError):
    """The model cannot score records at all (self-test or circuit breaker).

    Deliberately *not* caught by ``predict``: the harness must abort visibly.
    """


def _is_number(v: Any) -> bool:
    """A JSON number: int or float but not bool (``isinstance(True, int)`` holds)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def normalize_arch(arch: dict[str, Any]) -> dict[str, int]:
    """Fill the ``ST_MEM`` constructor defaults so two arch dicts compare by meaning.

    ``save_encoder`` (src/objectives/model.py) stores ``num_leads`` beside
    ``arch`` rather than inside it, and a hand-written manifest may omit keys;
    both must compare equal to the fully spelled-out dict. ``num_leads``
    defaults to 1, as ``validate_manifest`` assumes, not to ST_MEM's 12.
    """
    import inspect

    full = {k: p.default for k, p in inspect.signature(ST_MEM).parameters.items() if k in ANCHOR_ENCODER_ARCH}
    full["num_leads"] = 1  # the branch's encoders are the anchor's shared per-lead ST_MEM
    full.update({k: v for k, v in arch.items() if v is not None})
    return {k: int(v) for k, v in full.items()}


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------

try:  # the journal branch's head module, once it exists (ARCHITECTURE.md)
    from src.models.heads import EnsembleHead  # type: ignore
except Exception:  # pragma: no cover - exercised only before src/models/heads.py lands

    class EnsembleHead(nn.Module):
        """Twelve per-lead MLPs whose logits are averaged -- the anchor's head.

        Identical to ``mm_mae/classifiers/Classifiers.py::EnsembleClassifier``
        (768 -> 384 -> ReLU -> Dropout(0.3) -> 1 per lead, unweighted mean)
        and to the contract in ``docs/journal/ARCHITECTURE.md``. Defined here
        so the container never depends on a file that may not exist yet.
        """

        def __init__(self, n_leads: int = N_LEADS, emb_dim: int = 768, kind: str = "non_linear",
                     dropout: float = 0.3):
            super().__init__()
            if kind not in HEAD_KINDS:
                raise ValueError(f"unknown head kind {kind!r}")
            self.n_leads, self.emb_dim, self.kind, self.dropout = int(n_leads), int(emb_dim), kind, float(dropout)
            # Attribute name and per-lead layout match src/models/heads.py so a
            # head.pt saved by either class loads into the other unchanged.
            self.leads = nn.ModuleList([self._make(self.emb_dim, kind, self.dropout) for _ in range(self.n_leads)])

        @staticmethod
        def _make(emb_dim: int, kind: str, dropout: float) -> nn.Module:
            if kind == "linear":
                return nn.Linear(emb_dim, 1)
            return nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim // 2, 1),
            )

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            """``feats [B, n_leads, emb_dim] -> logits [B]``."""
            cols = [h(feats[:, i, :]) for i, h in enumerate(self.leads)]
            return torch.cat(cols, dim=1).mean(dim=1)


_LEAD_PREFIXES = ("classifiers.m", "leads.", "heads.")


def _remap_head_keys(state: dict[str, torch.Tensor], target_keys: Iterable[str]) -> dict[str, torch.Tensor]:
    """Accept the anchor's ``classifiers.m<i>.*`` layout as well as ``leads.<i>.*``.

    The anchor's ``clf.pth`` stores each lead head under ``classifiers.m<i>``;
    ``src/models/heads.py::EnsembleHead`` under ``leads.<i>``. Same tensors,
    so a head trained either way loads into the container. The target prefix
    is read off the target keys rather than assumed.
    """
    target = set(target_keys)
    if set(state) == target:
        return state
    tgt_prefix = next((p for p in _LEAD_PREFIXES if any(k.startswith(p) for k in target)), None)
    if tgt_prefix is None:
        raise ManifestError("cannot determine the per-lead key prefix of EnsembleHead")
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for p in _LEAD_PREFIXES:
            if k.startswith(p):
                idx, _, tail = k[len(p):].partition(".")
                nk = f"{tgt_prefix}{idx}.{tail}"
                break
        remapped[nk] = v
    if set(remapped) != target:
        missing = sorted(target - set(remapped))[:5]
        unexpected = sorted(set(remapped) - target)[:5]
        raise ManifestError(f"head state_dict does not fit EnsembleHead; missing={missing} unexpected={unexpected}")
    return remapped


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    name: str
    encoder: str  # "good_disco" or a file name relative to the model folder
    encoder_arch: dict[str, Any]
    head: str
    head_kind: str = "non_linear"
    feature: str = "cls"
    preproc: str = "anchor_v1"
    #: The head reads age and sex from the WFDB header (official
    #: ``helper_code.get_age`` / ``get_sex``); must agree with the head file.
    demographics: bool = False
    #: Hybrid rhythm view (exploratory): the head adds the S3.3 rhythm/interval
    #: descriptors, recomputed from the preprocessed record
    #: (``src.data.rhythm_view.from_signals``); must agree with the head file.
    rhythm: bool = False
    #: S4C H3: the head starts with a fixed linear concept eraser (``leace_source``); manifest key ``erase``,
    #: absent = ``none`` (every earlier folder). Must agree with the head file's spec.
    erase: str = "none"


@dataclass(frozen=True)
class Binarization:
    kind: str
    threshold_logit: float
    fraction: float | None = None

    def label(self, logit: float) -> int:
        return int(logit > self.threshold_logit)


@dataclass(frozen=True)
class OutputTerm:
    slot: str
    weight: float
    shift: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class OutputCombination:
    """The k-slot output (module docstring): ``offset + sum_i weight_i * (logit_i - shift_i) / scale_i``."""

    kind: str
    terms: tuple[OutputTerm, ...]
    offset: float = 0.0

    def combine(self, scores: dict[str, float]) -> float:
        total = float(self.offset)
        for t in self.terms:
            v = scores.get(t.slot, float("nan"))
            if not math.isfinite(v):
                return float("nan")
            total += t.weight * (v - t.shift) / t.scale
        return float(total)


@dataclass(frozen=True)
class Manifest:
    slots: tuple[Slot, ...]
    output_slot: str
    binarization: Binarization
    smoke: bool = False
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    #: Crops whose logits are averaged (S5.5); empty = no TTA, the anchor's single head crop.
    tta: tuple[str, ...] = ()
    #: k-slot output combination (S4C B4); ``None`` = the output slot's logit (every earlier folder).
    output: OutputCombination | None = None

    @property
    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ManifestError(msg)


def validate_manifest(manifest: dict[str, Any], model_folder: Path | None = None) -> Manifest:
    """Turn the JSON dict into a :class:`Manifest`, or raise :class:`ManifestError`.

    Every check here runs once at load time. The organisers' harness gives no
    second chance, so a manifest that cannot work must fail before the first
    record, loudly, rather than on record 40,000.
    """
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    _require(manifest.get("format") == MANIFEST_FORMAT,
             f"manifest format must be {MANIFEST_FORMAT!r}, got {manifest.get('format')!r}")
    slots_raw = manifest.get("slots")
    _require(isinstance(slots_raw, list) and len(slots_raw) > 0, "manifest needs a non-empty 'slots' list")

    slots: list[Slot] = []
    seen: set[str] = set()
    for i, s in enumerate(slots_raw):
        _require(isinstance(s, dict), f"slot {i} must be an object")
        name = s.get("name")
        _require(isinstance(name, str) and name and name not in seen, f"slot {i}: missing or duplicate name")
        seen.add(name)
        enc = s.get("encoder")
        _require(isinstance(enc, str) and enc, f"slot {name!r}: 'encoder' must be 'good_disco' or a file name")
        arch = dict(s.get("encoder_arch") or {})
        if enc == "good_disco":
            _require(not arch or arch == ANCHOR_ENCODER_ARCH,
                     f"slot {name!r}: encoder_arch of good_disco must be the anchor architecture")
            arch = dict(ANCHOR_ENCODER_ARCH)
        else:
            _require(bool(arch), f"slot {name!r}: encoder_arch is required for a file encoder")
            missing = [k for k in REQUIRED_FILE_ARCH_KEYS if not _is_number(arch.get(k))]
            _require(not missing, f"slot {name!r}: encoder_arch lacks numeric {missing} (required for a file encoder)")
            bad = [k for k, v in arch.items() if not _is_number(v)]
            _require(not bad, f"slot {name!r}: encoder_arch values must be numbers; bad {bad}")
        _require(int(arch.get("num_leads", 1)) == 1,
                 f"slot {name!r}: only num_leads=1 encoders are supported (the anchor's shared per-lead ST_MEM)")
        _require(int(arch.get("seq_len", 2250)) == ANCHOR_PREPROC["standard_len"],
                 f"slot {name!r}: encoder seq_len must equal the preprocessing length {ANCHOR_PREPROC['standard_len']}")
        head = s.get("head")
        _require(isinstance(head, str) and head, f"slot {name!r}: 'head' file name required")
        head_kind = s.get("head_kind", "non_linear")
        _require(head_kind in HEAD_KINDS, f"slot {name!r}: head_kind must be one of {HEAD_KINDS}")
        feature = s.get("feature", "cls")
        _require(feature in FEATURES, f"slot {name!r}: feature must be one of {FEATURES}")
        preproc = s.get("preproc", "anchor_v1")
        _require(preproc in PREPROCS, f"slot {name!r}: preproc must be one of {PREPROCS}")
        demographics = s.get("demographics", False)
        _require(isinstance(demographics, bool), f"slot {name!r}: 'demographics' must be a JSON boolean")
        rhythm = s.get("rhythm", False)
        _require(isinstance(rhythm, bool), f"slot {name!r}: 'rhythm' must be a JSON boolean")
        erase = s.get("erase", "none")
        _require(erase in ERASES, f"slot {name!r}: 'erase' must be one of {ERASES}")
        if model_folder is not None:
            _require((model_folder / head).is_file(), f"slot {name!r}: head file {head!r} not found in {model_folder}")
            if enc != "good_disco":
                ep = model_folder / enc
                _require(ep.is_file() or (ep.is_dir() and any(p.is_file() for p in ep.iterdir())),
                         f"slot {name!r}: encoder file or chunk directory {enc!r} not found in {model_folder}")
        slots.append(Slot(name, enc, arch, head, head_kind, feature, preproc, demographics, rhythm, erase))

    out = manifest.get("output_slot")
    _require(out in seen, f"output_slot {out!r} is not one of the slots {sorted(seen)}")

    b = manifest.get("binarization")
    _require(isinstance(b, dict) and "kind" in b, "manifest needs a 'binarization' object with a 'kind'")
    kind = b["kind"]
    if kind == "top_fraction":
        frac = b.get("fraction")
        thr = b.get("threshold_logit")
        _require(_is_number(frac) and 0 < frac < 1, "top_fraction needs 0 < fraction < 1")
        _require(_is_number(thr) and math.isfinite(thr),
                 "top_fraction needs a finite 'threshold_logit' (records arrive one at a time; "
                 "the fraction cannot be applied at run time)")
        binz = Binarization("top_fraction", float(thr), float(frac))
    elif kind == "logit_threshold":
        val = b.get("value")
        _require(_is_number(val) and math.isfinite(val), "logit_threshold needs a finite 'value'")
        binz = Binarization("logit_threshold", float(val))
    else:
        raise ManifestError(f"unknown binarization kind {kind!r}")

    tta: tuple[str, ...] = ()
    if "tta" in manifest:
        t = manifest["tta"]
        _require(isinstance(t, list) and len(t) > 0 and all(isinstance(c, str) for c in t),
                 "'tta' must be a non-empty list of crop names")
        _require(all(c in TTA_CROPS for c in t) and len(set(t)) == len(t),
                 f"'tta' crops must be distinct names from {TTA_CROPS}, got {t}")
        tta = tuple(t)

    output = parse_output(manifest["output"], seen) if "output" in manifest else None
    return Manifest(tuple(slots), out, binz, bool(manifest.get("smoke", False)), manifest, tta, output)


def parse_output(o: Any, slot_names: set[str]) -> OutputCombination:
    """The ``"output"`` object of a manifest (module docstring); raises :class:`ManifestError`."""
    _require(isinstance(o, dict) and o.get("kind") in OUTPUT_KINDS,
             f"'output' must be an object with 'kind' one of {OUTPUT_KINDS}")

    def number(v: Any, what: str, positive: bool = False) -> float:
        _require(_is_number(v) and math.isfinite(v) and (v > 0 or not positive),
                 f"'output': {what} must be a finite{' positive' if positive else ''} number, got {v!r}")
        return float(v)

    if o["kind"] == "mean_logit":
        names = o.get("slots")
        _require(isinstance(names, list) and len(names) >= 2 and len(set(names)) == len(names)
                 and all(isinstance(n, str) and n in slot_names for n in names),
                 f"'output' mean_logit needs >= 2 distinct slot names from {sorted(slot_names)}")
        zmap = o.get("zmap", {})
        _require(isinstance(zmap, dict) and set(zmap) <= set(names), "'output' zmap must map listed slots")
        terms = []
        for n in names:
            z = zmap.get(n)
            if z is None:
                terms.append(OutputTerm(n, 1.0 / len(names)))
            else:
                _require(isinstance(z, dict), f"'output' zmap[{n!r}] must be an object with 'mean' and 'sd'")
                terms.append(OutputTerm(n, 1.0 / len(names), number(z.get("mean"), f"zmap[{n!r}].mean"),
                                        number(z.get("sd"), f"zmap[{n!r}].sd", positive=True)))
        return OutputCombination("mean_logit", tuple(terms), 0.0)
    raw = o.get("terms")
    _require(isinstance(raw, list) and len(raw) >= 1, "'output' affine needs a non-empty 'terms' list")
    terms = []
    for i, t in enumerate(raw):
        _require(isinstance(t, dict) and t.get("slot") in slot_names, f"'output' term {i}: unknown or missing slot")
        terms.append(OutputTerm(t["slot"], number(t.get("weight"), f"term {i} weight"),
                                number(t.get("shift", 0.0), f"term {i} shift"),
                                number(t.get("scale", 1.0), f"term {i} scale", positive=True)))
    return OutputCombination("affine", tuple(terms), number(o.get("offset", 0.0), "offset"))


def load_manifest(model_folder: str | Path) -> Manifest:
    model_folder = Path(model_folder)
    path = model_folder / MANIFEST_NAME
    _require(path.is_file(), f"no {MANIFEST_NAME} in {model_folder}")
    with open(path) as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ManifestError(f"{path} is not valid JSON: {e}") from e
    return validate_manifest(raw, model_folder)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


def build_st_mem(arch: dict[str, Any]) -> ST_MEM:
    """A bare ``ST_MEM`` from an arch dict (keys = the constructor's arguments)."""
    allowed = set(ANCHOR_ENCODER_ARCH)
    unknown = set(arch) - allowed
    if unknown:
        raise ManifestError(f"encoder_arch has unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")
    return ST_MEM(**{k: int(v) for k, v in arch.items()})


def strip_anchor_prefix(state: dict[str, torch.Tensor], lead: str = "m0") -> dict[str, torch.Tensor]:
    """``encoders.m0.<param>`` -> ``<param>``.

    The good_disco checkpoint is a Lightning state_dict of the MMMAE wrapper
    whose ``encoders`` ModuleDict holds the *same* ST_MEM under all twelve
    keys (``share_weights: True``), so any one prefix carries the full model.
    """
    prefix = f"encoders.{lead}."
    out = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise ManifestError(f"no keys with prefix {prefix!r} in checkpoint")
    return out


def read_good_disco_checkpoint(directory: str | Path = GOOD_DISCO_DIR) -> dict[str, Any]:
    """The good_disco Lightning checkpoint, deserialised onto the CPU.

    Same bytes, same order as ``team.utils.load_split_pth`` (every regular
    file of ``directory``, sorted by name, concatenated), but with
    ``map_location="cpu"``. The checkpoint's tensors were saved on CUDA, so
    ``load_split_pth``'s bare ``torch.load`` raises on a host without a usable
    GPU (audit 2026-10-06, container finding 3). On a GPU host the result is
    the same: the weights are copied into a CPU-built ``ST_MEM`` either way and
    :class:`JournalModel` moves every module to its device afterwards, so the
    parameters -- and therefore the outputs -- are bit-identical; only the
    transient 1.4 GB copy no longer lands on the GPU. ``team.utils`` (the CinC
    anchor) is left untouched. ``weights_only=False``: the pickle carries the
    Lightning ``hyper_parameters`` objects (torch 2.5.1's default, made explicit).
    """
    import io

    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if (directory / f).is_file())
    _require(bool(files), f"no checkpoint chunks in {directory}")
    buf = io.BytesIO()
    for f in files:
        with open(directory / f, "rb") as fh:
            buf.write(fh.read())
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def load_good_disco(directory: str | Path = GOOD_DISCO_DIR) -> ST_MEM:
    """The shipped anchor encoder, reassembled from its 66 chunks in memory (CPU; see
    :func:`read_good_disco_checkpoint`)."""
    directory = Path(directory)
    _require(directory.is_dir(), f"good_disco encoder directory not found: {directory}")
    ckpt = read_good_disco_checkpoint(directory)
    state = strip_anchor_prefix(ckpt["state_dict"])
    enc = build_st_mem(ANCHOR_ENCODER_ARCH)
    enc.load_state_dict(state, strict=True)
    return enc


def encoder_chunk_files(directory: str | Path) -> list[Path]:
    """The chunk files of a chunked encoder directory, in reassembly order (sorted names)."""
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    _require(bool(files), f"no encoder chunks in {directory}")
    return files


def _encoder_source(path: str | Path):
    """A file path, or the in-memory concatenation of a chunk directory (``export_model
    --encoder-chunk-mb``: a git host refuses single files over 100 MB, D-067)."""
    import io

    path = Path(path)
    if not path.is_dir():
        return path
    buf = io.BytesIO()
    for f in encoder_chunk_files(path):
        buf.write(f.read_bytes())
    buf.seek(0)
    return buf


def load_encoder_file(path: str | Path, expected_arch: dict[str, Any] | None = None) -> ST_MEM:
    """``encoder.pt`` = ``{"arch": {...}, "state_dict": {...}[, "num_leads": 1]}``.

    ``path`` may also be a directory of byte chunks of such a file (reassembled
    in sorted-name order, as the good_disco chunks are).

    Written by ``export_model`` or by ``src/objectives/model.py::save_encoder``
    (which keeps ``num_leads`` beside ``arch``). With ``expected_arch`` (the
    manifest slot's), the file's architecture must agree with it after
    :func:`normalize_arch`; a manifest that claims one width while the file
    holds another fails here, at load, not as a shape error on record one.
    """
    blob = torch.load(_encoder_source(path), map_location="cpu", weights_only=True)
    _require(isinstance(blob, dict) and "arch" in blob and "state_dict" in blob,
             f"{path}: expected a dict with 'arch' and 'state_dict'")
    arch = dict(blob["arch"])
    if "num_leads" not in arch and "num_leads" in blob:
        arch["num_leads"] = blob["num_leads"]
    if expected_arch is not None:
        got, want = normalize_arch(arch), normalize_arch(expected_arch)
        diff = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
        _require(not diff, f"{path}: encoder architecture disagrees with the manifest "
                           f"(manifest, file): {diff}")
    enc = build_st_mem(normalize_arch(arch))
    state = blob["state_dict"]
    if any(k.startswith("encoders.") for k in state):
        state = strip_anchor_prefix(state)
    enc.load_state_dict(state, strict=True)
    return enc


@torch.no_grad()
def encode(encoder: ST_MEM, x: torch.Tensor, feature: str) -> torch.Tensor:
    """``x [B, 2250, 12]`` -> per-lead features ``[B, 12, D]``.

    One batched forward over ``B * 12`` single-lead sequences with
    ``mask_ratio = 0``. This equals the anchor's twelve per-lead forwards
    because the anchor shares one ``num_leads=1`` encoder across leads, so every
    lead sees lead-embedding 0 either way. ``cls`` is token 0 (the left SEP
    token the anchor's classifier reads); ``mean`` averages all 47 tokens.
    """
    b, t, l = x.shape
    seqs = x.permute(0, 2, 1).reshape(b * l, 1, t)
    tokens, _, _ = encoder.forward_encoder(seqs, 0)  # [b*l, n_tokens, D]
    if feature == "cls":
        feats = tokens[:, 0, :]
    elif feature == "mean":
        feats = tokens.mean(dim=1)
    elif feature == "both":
        feats = torch.cat([tokens[:, 0, :], tokens.mean(dim=1)], dim=-1)
    else:  # pragma: no cover - validated at manifest load
        raise ValueError(feature)
    return feats.reshape(b, l, -1)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def preprocess_record(record: str) -> torch.Tensor:
    """The anchor's ``team_code.prepare_data`` for one record -> ``[2250, 12]`` float32.

    Same function, same constants (``ANCHOR_PREPROC``), so features here match
    the cache the journal heads were fitted on. Raises on unreadable records;
    the caller decides what to do with that.

    A record outside the training formats (non-finite samples; channels other
    than the 12 standard leads in the standard order) first goes through
    :func:`src.container.input_guard.guard` and then the same preprocessing on
    the array (``src.data.signals.preprocess_array_crops``, bit-identical to the
    anchor's path on clean records, ``tests/test_tta.py``). Every other record
    takes the anchor's path unchanged.
    """
    from team.utils_challenge import load_signals
    from src.container.input_guard import guard

    header = load_header(record)
    sr = load_sampling_rate(header)
    p = ANCHOR_PREPROC
    x, fields = load_signals(record)
    fixed = guard(x, fields.get("sig_name"))
    if fixed is None:
        signal = load_and_process_signal(
            "", record, sr, p["target_sr"], p["preprocessing"], p["intensity_normalization"],
            p["standard_len"], p["pad_mode"], None, p["heartbeat_mode"],
        )
    else:
        from src.data.signals import preprocess_array_crops

        y, notes = fixed
        log.warning("record %s: input guard: %s", record, "; ".join(notes))
        signal = torch.from_numpy(preprocess_array_crops(y, sr, ("head",))["head"])
    if tuple(signal.shape) != (p["standard_len"], N_LEADS):
        raise ValueError(f"preprocessed shape {tuple(signal.shape)} != ({p['standard_len']}, {N_LEADS})")
    return signal.to(torch.float32)


def preprocess_record_crops(record: str, crops: Sequence[str]) -> torch.Tensor:
    """``[len(crops), 2250, 12]`` float32: the anchor preprocessing, one window per crop (S5.5).

    Loads with the loader ``load_and_process_signal`` uses and runs
    ``src.data.signals.preprocess_array_crops`` -- the same resampling,
    cleaning and standardisation, once, then each crop's window. The ``head``
    entry equals :func:`preprocess_record` (``tests/test_tta.py``), including
    the input guard.
    """
    from team.utils_challenge import load_signals
    from src.container.input_guard import guard
    from src.data.signals import preprocess_array_crops

    header = load_header(record)
    sr = load_sampling_rate(header)
    x, fields = load_signals(record)
    fixed = guard(x, fields.get("sig_name"))
    if fixed is not None:
        x, notes = fixed
        log.warning("record %s: input guard: %s", record, "; ".join(notes))
    out = preprocess_array_crops(x, sr, tuple(crops))
    return torch.stack([torch.from_numpy(out[c]) for c in crops]).to(torch.float32)


def read_demographics(record: str) -> tuple[Any, Any]:
    """``(age, sex)`` from the record's header via the organisers' ``helper_code``.

    Raw values (``get_age`` returns a float when the field parses, the raw
    string otherwise; ``get_sex`` the raw string or ``None``); the head's
    :func:`src.models.heads.demographic_matrix` turns anything unparseable into
    "missing" with its flag set, so a header without these fields is scored,
    never failed.
    """
    import helper_code  # the organisers' file at the repo root; imported only for demographics slots

    header = load_header(record)
    return helper_code.get_age(header), helper_code.get_sex(header)


def build_slot_head(slot: "Slot", blob: Any, emb_dim: int) -> nn.Module:
    """The head module for ``slot`` with the weights in ``blob`` loaded strictly.

    Anchor kinds without a ``spec`` go through ``EnsembleHead`` (with the
    anchor-layout key remap); anything else is rebuilt from the ``spec`` that
    ``src/run.py::save_head`` writes beside the weights.
    """
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    spec = blob.get("spec") if isinstance(blob, dict) else None
    width = emb_dim * FEATURE_WIDTH_FACTOR[slot.feature]
    if spec is None or (spec.get("kind") in ("non_linear", "linear") and not spec.get("demographics")
                        and spec.get("input_norm", "none") == "none" and not spec.get("rhythm")
                        and spec.get("erase", "none") == "none"):
        _require(slot.head_kind in ("non_linear", "linear") and not slot.demographics and not slot.rhythm
                 and slot.erase == "none",
                 f"slot {slot.name!r}: head_kind {slot.head_kind!r} / demographics / erase need a head file "
                 "with a 'spec'")
        head = EnsembleHead(n_leads=N_LEADS, emb_dim=width, kind=slot.head_kind)
        try:
            head.load_state_dict(_remap_head_keys(state, head.state_dict().keys()), strict=True)
        except RuntimeError as e:  # shape mismatch: torch reports it as RuntimeError
            raise ManifestError(f"slot {slot.name!r}: head {slot.head!r} does not fit an EnsembleHead "
                                f"of width {width}: {e}") from e
        return head
    from src.models.heads import build_head

    spec = dict(spec)
    _require(spec.get("kind") == slot.head_kind,
             f"slot {slot.name!r}: head file kind {spec.get('kind')!r} != manifest head_kind {slot.head_kind!r}")
    _require(bool(spec.get("demographics", False)) == slot.demographics,
             f"slot {slot.name!r}: head file demographics={bool(spec.get('demographics', False))} "
             f"!= manifest demographics={slot.demographics}")
    _require(bool(spec.get("rhythm", False)) == slot.rhythm,
             f"slot {slot.name!r}: head file rhythm={bool(spec.get('rhythm', False))} != manifest rhythm={slot.rhythm}")
    _require(str(spec.get("erase", "none")) == slot.erase,
             f"slot {slot.name!r}: head file erase={spec.get('erase', 'none')!r} != manifest erase={slot.erase!r}")
    _require(int(spec.get("emb_dim", -1)) == width,
             f"slot {slot.name!r}: head emb_dim {spec.get('emb_dim')} != encoder width {emb_dim} x "
             f"{FEATURE_WIDTH_FACTOR[slot.feature]} (feature {slot.feature!r})")
    _require(int(spec.get("n_leads", N_LEADS)) == N_LEADS, f"slot {slot.name!r}: head n_leads must be {N_LEADS}")
    head = build_head(**spec)
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ManifestError(f"slot {slot.name!r}: head {slot.head!r} does not fit its spec {spec}: {e}") from e
    return head


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class JournalModel:
    """What ``team_code.load_model`` returns for a journal model folder.

    Holds one encoder per distinct encoder spec (shared across slots that name
    the same one), one head per slot, and the binarization rule. All modules
    are in eval mode on ``device``.
    """

    #: Marker used by team_code to dispatch without importing this module eagerly.
    is_journal_model = True
    breaker_first_n = BREAKER_FIRST_N
    breaker_min_seen = BREAKER_MIN_SEEN
    breaker_max_fraction = BREAKER_MAX_FRACTION

    def __init__(self, manifest: Manifest, encoders: dict[str, ST_MEM], heads: dict[str, nn.Module],
                 device: torch.device, model_folder: Path):
        self.manifest = manifest
        self.encoders = encoders
        self.heads = heads
        self.device = device
        self.model_folder = model_folder
        #: record path -> {slot: logit} for every record seen, all slots.
        self.extra_scores: dict[str, dict[str, float]] = {}
        self.n_failed = 0
        self.n_seen = 0
        for m in list(encoders.values()) + list(heads.values()):
            m.eval().to(device)
        self._extra_dir = os.environ.get(EXTRA_OUTPUT_ENV) or None
        if self._extra_dir:
            os.makedirs(self._extra_dir, exist_ok=True)
            log.info("extra slot scores will be appended to %s/%s", self._extra_dir, EXTRA_OUTPUT_FILE)

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, model_folder: str | Path, device: str | torch.device | None = None) -> "JournalModel":
        model_folder = Path(model_folder)
        manifest = load_manifest(model_folder)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        encoders: dict[str, ST_MEM] = {}
        heads: dict[str, nn.Module] = {}
        for slot in manifest.slots:
            key = slot.encoder
            if key not in encoders:
                if key == "good_disco":
                    encoders[key] = load_good_disco()
                else:
                    encoders[key] = load_encoder_file(model_folder / key, expected_arch=slot.encoder_arch)
                log.info("slot %s: encoder %s loaded", slot.name, key)
            # The head width comes from the encoder actually loaded, not from the
            # manifest, so a head/encoder mismatch is a strict load error here.
            emb_dim = int(encoders[key].encoder.width)
            _require(emb_dim == int(slot.encoder_arch["embed_dim"]),
                     f"slot {slot.name!r}: loaded encoder width {emb_dim} != manifest embed_dim "
                     f"{slot.encoder_arch['embed_dim']}")
            blob = torch.load(model_folder / slot.head, map_location="cpu", weights_only=True)
            heads[slot.name] = build_slot_head(slot, blob, emb_dim)
        model = cls(manifest, encoders, heads, device, model_folder)
        model.self_test()
        log.info("JournalModel loaded from %s on %s: slots=%s output_slot=%s output=%s binarization=%s%s",
                 model_folder, device, manifest.slot_names, manifest.output_slot, manifest.output, manifest.binarization,
                 " [SMOKE MANIFEST -- plumbing test only]" if manifest.smoke else "")
        if manifest.smoke:
            print("WARNING: journal_manifest.json is flagged 'smoke' -- random heads, plumbing test only. "
                  "Never submit this folder.", file=sys.stderr)
        return model

    @torch.no_grad()
    def self_test(self) -> dict[str, list[float]]:
        """Score two synthetic signals of the contract shape through every slot.

        Raises :class:`SystematicFailure` on any exception or non-finite logit.
        Runs at load so a model that cannot score anything is caught before the
        harness starts consuming records; returns the logits for logging.
        """
        t, l = ANCHOR_PREPROC["standard_len"], N_LEADS
        gen = torch.Generator().manual_seed(0)
        batch = torch.stack([torch.zeros(t, l), torch.randn(t, l, generator=gen)]).to(self.device)
        demo = [(None, None), (55.0, "Female")]  # one record without header demographics, one with
        try:
            scores = self._score_batch(batch, demo) if self.needs_demographics else self._score_batch(batch)
        except Exception as e:  # noqa: BLE001 - re-raised as a loud, typed failure
            raise SystematicFailure(f"self-test forward failed: {type(e).__name__}: {e}") from e
        out = {name: [float(s[name]) for s in scores] for name in self.manifest.slot_names}
        if self.manifest.output is not None:  # S4C B4: the combination the harness will see
            out["__output__"] = [self.manifest.output.combine(sc) for sc in scores]
        bad = {k: v for k, v in out.items() if any(not math.isfinite(x) for x in v)}
        if bad:
            raise SystematicFailure(f"self-test produced non-finite logits: {bad}")
        log.info("self-test passed: %s", out)
        return out

    # -- inference --------------------------------------------------------

    @torch.no_grad()
    def predict(self, records: Sequence[str]) -> list[dict[str, float]]:
        """Per record, ``{slot: logit}`` for every slot. NaN logits mark a failed record.

        Preprocessing failures are per record; encoder/head failures on a chunk
        are retried record by record so one bad signal never takes its
        neighbours down with it.
        """
        records = list(records)
        out: list[dict[str, float]] = [dict() for _ in records]
        signals: list[torch.Tensor | None] = []
        demos: list[tuple[Any, Any]] = []
        need_demo = self.needs_demographics
        tta = self.manifest.tta
        for i, r in enumerate(records):
            try:
                sig = preprocess_record_crops(r, tta) if tta else preprocess_record(r)
                if not torch.isfinite(sig).all():
                    raise ValueError("non-finite values after preprocessing")
                demos.append(read_demographics(r) if need_demo else (None, None))
                signals.append(sig)
            except Exception as e:  # noqa: BLE001 - see "Failure policy" in the module docstring
                self._fail(r, f"preprocessing failed: {type(e).__name__}: {e}")
                signals.append(None)
                demos.append((None, None))
                out[i] = self._nan_scores()

        ok = [i for i, s in enumerate(signals) if s is not None]
        for start in range(0, len(ok), ENCODE_CHUNK):
            idx = ok[start:start + ENCODE_CHUNK]
            batch = torch.stack([signals[i] for i in idx]).to(self.device)
            try:
                scores: list[dict[str, float] | None] = list(self._score_batch(batch, [demos[i] for i in idx])
                                                             if need_demo else self._score_batch(batch))
            except Exception as e:  # noqa: BLE001
                if len(idx) == 1:
                    self._fail(records[idx[0]], f"forward failed: {type(e).__name__}: {e}")
                    out[idx[0]] = self._nan_scores()
                    continue
                scores = []
                for i in idx:  # retry one at a time so a single bad record is isolated
                    try:
                        one = signals[i].unsqueeze(0).to(self.device)
                        scores.append((self._score_batch(one, [demos[i]]) if need_demo
                                       else self._score_batch(one))[0])
                    except Exception as e2:  # noqa: BLE001
                        self._fail(records[i], f"forward failed: {type(e2).__name__}: {e2}")
                        scores.append(None)  # already counted; skip the finite check below
            for i, s in zip(idx, scores):
                if s is None:
                    out[i] = self._nan_scores()
                    continue
                if any(not math.isfinite(v) for v in s.values()):
                    self._fail(records[i], "non-finite logit")
                    s = self._nan_scores()
                out[i] = s

        for r, s in zip(records, out):
            self.extra_scores[r] = s
        self.n_seen += len(records)
        self._dump_extra(records, out)
        self._check_breaker()
        return out

    def predict_challenge(self, records: Sequence[str]) -> tuple[list[int], list[float]]:
        """The harness contract: ``(labels, probabilities)`` for the output slot.

        ``probability = sigmoid(logit)``; label from the manifest's rule. A
        failed record is ``(0, 0.0)``.
        """
        scores = self.predict(records)
        labels: list[int] = []
        probs: list[float] = []
        for s in scores:
            logit = self.output_logit(s)
            if not math.isfinite(logit):
                labels.append(0)
                probs.append(0.0)
                continue
            labels.append(int(self.manifest.binarization.label(logit)))
            probs.append(float(_sigmoid(logit)))
        return labels, probs

    def output_logit(self, scores: dict[str, float]) -> float:
        """The logit the harness output of one record is made of: the manifest's ``output`` combination
        (k slots, S4C_IMPROVE.md B4) when it declares one, else the output slot's logit; NaN for a failed
        record. ``scores`` is one element of :meth:`predict`'s result."""
        combo = self.manifest.output
        return combo.combine(scores) if combo is not None else scores.get(self.manifest.output_slot, float("nan"))

    def predict_record(self, record: str) -> tuple[int, float]:
        labels, probs = self.predict_challenge([record])
        return labels[0], probs[0]

    # -- internals --------------------------------------------------------

    @property
    def needs_demographics(self) -> bool:
        return any(s.demographics for s in self.manifest.slots)

    def _score_batch(self, batch: torch.Tensor, demo: Sequence[tuple[Any, Any]] | None = None) -> list[dict[str, float]]:
        if batch.dim() == 4:  # TTA: [B, n_crops, T, L] -> mean logit over crops per record
            b, k = batch.shape[:2]
            flat = self._score_batch(batch.reshape(b * k, *batch.shape[2:]),
                                     None if demo is None else [d for d in demo for _ in range(k)])
            return [{name: float(np.mean(np.array([flat[i * k + j][name] for j in range(k)], dtype=np.float64)))
                     for name in self.manifest.slot_names} for i in range(b)]
        feats_cache: dict[tuple[str, str], torch.Tensor] = {}
        per_slot: dict[str, torch.Tensor] = {}
        rhythm_raw = None  # descriptors of this batch's preprocessed signals, computed once for all rhythm slots
        for slot in self.manifest.slots:
            key = (slot.encoder, slot.feature)
            if key not in feats_cache:
                if slot.feature == "both" and (slot.encoder, "cls") in feats_cache and (slot.encoder, "mean") in feats_cache:
                    feats_cache[key] = torch.cat([feats_cache[(slot.encoder, "cls")],
                                                  feats_cache[(slot.encoder, "mean")]], dim=-1)
                else:
                    feats_cache[key] = encode(self.encoders[slot.encoder], batch, slot.feature)
            head = self.heads[slot.name]
            if slot.rhythm:
                if rhythm_raw is None:
                    from src.data.rhythm_view import from_signals
                    rhythm_raw = from_signals(batch)
                d = None
                if slot.demographics:
                    if demo is None or len(demo) != batch.shape[0]:
                        raise ValueError(f"slot {slot.name!r} needs one (age, sex) per record")
                    d = head.demo_matrix([a for a, _ in demo], [s for _, s in demo]).to(self.device)
                logits = head(feats_cache[key], d, head.rhythm_matrix(rhythm_raw).to(self.device))
            elif slot.demographics:
                if demo is None or len(demo) != batch.shape[0]:
                    raise ValueError(f"slot {slot.name!r} needs one (age, sex) per record")
                d = head.demo_matrix([a for a, _ in demo], [s for _, s in demo]).to(self.device)
                logits = head(feats_cache[key], d)
            else:
                logits = head(feats_cache[key])
            per_slot[slot.name] = logits.detach().float().cpu()
        n = batch.shape[0]
        return [{name: float(v[i]) for name, v in per_slot.items()} for i in range(n)]

    def _nan_scores(self) -> dict[str, float]:
        return {name: float("nan") for name in self.manifest.slot_names}

    def _fail(self, record: str, why: str) -> None:
        self.n_failed += 1
        log.warning("record %s -> (0, 0.0): %s", record, why)
        print(f"WARNING: record {record} -> (0, 0.0): {why}", file=sys.stderr)

    def _check_breaker(self) -> None:
        """Abort on a failure pattern that no per-record policy should absorb."""
        n, f = self.n_seen, self.n_failed
        if n >= self.breaker_first_n and f >= n and f > 0:
            why = f"all of the first {n} records failed"
        elif n >= self.breaker_min_seen and f > self.breaker_max_fraction * n:
            why = f"{f}/{n} records failed (> {self.breaker_max_fraction:.0%})"
        else:
            return
        msg = f"circuit breaker: {why}; aborting the run so the failure is visible (see stderr warnings)"
        print(f"ERROR: {msg}", file=sys.stderr)
        raise SystematicFailure(msg)

    def _dump_extra(self, records: Sequence[str], scores: Sequence[dict[str, float]]) -> None:
        if not self._extra_dir:
            return
        path = os.path.join(self._extra_dir, EXTRA_OUTPUT_FILE)
        with open(path, "a") as f:
            for r, s in zip(records, scores):
                row = {"record": r, "logits": {k: (None if not math.isfinite(v) else v) for k, v in s.items()},
                       "probabilities": {k: (None if not math.isfinite(v) else _sigmoid(v)) for k, v in s.items()}}
                if self.manifest.output is not None:
                    c = self.manifest.output.combine(s)
                    row["output_logit"] = None if not math.isfinite(c) else c
                f.write(json.dumps(row) + "\n")


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


__all__ = [
    "ANCHOR_ENCODER_ARCH", "ANCHOR_PREPROC", "MANIFEST_FORMAT", "MANIFEST_NAME", "EXTRA_OUTPUT_ENV",
    "EXTRA_OUTPUT_FILE", "OUTPUT_KINDS", "Binarization", "EnsembleHead", "JournalModel", "Manifest", "ManifestError",
    "OutputCombination", "OutputTerm", "Slot", "parse_output",
    "SystematicFailure", "TTA_CROPS", "build_slot_head", "build_st_mem", "encode", "preprocess_record_crops",
    "read_demographics", "load_encoder_file", "load_good_disco", "load_manifest", "read_good_disco_checkpoint",
    "normalize_arch", "preprocess_record", "strip_anchor_prefix", "validate_manifest",
]
