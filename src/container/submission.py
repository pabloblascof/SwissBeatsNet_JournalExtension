"""D-053: make the organisers' ``train_model.py`` / ``run_model.py`` reach the registered model.

Opt-in. With the flag unset nothing here runs and ``team_code`` behaves exactly as
the CinC anchor did. Choosing a mode is the authors' decision (DECISIONS.md
D-053); this module only makes either choice executable and testable.

The problem
-----------
The organisers' 2025 procedure clones the team repository, builds the image,
mounts a host folder over ``/challenge/model`` and runs ``train_model.py`` and
then ``run_model.py`` (python-example-2025 README). Our registered folder
(``model/``, REGISTRATION.md section 2) is git-ignored, and
``team_code.train_model`` runs the CinC recipe, so that procedure scores a
retrained 2813 recipe -- or, if only ``run_model.py`` runs against an empty
mount, crashes on the missing ``model.pth`` (audit 2026-10-06, container
findings 1 and 2).

The flag
--------
``SBN_SUBMISSION_MODE`` (environment variable), read by ``team_code`` at call time:

``unset`` / ``""`` / ``"anchor"``
    The CinC anchor behaviour, unchanged: ``train_model`` runs
    ``main_mm_mae_stripped.main`` (2813's recipe); ``load_model`` serves a folder
    with ``journal_manifest.json`` through ``JournalModel`` and any other folder
    through the anchor path. ``team_code`` does not even import this module.

``"train_registered_recipe"`` -- option (b)
    ``train_model`` fits the registered recipe (``configs/anchor_rebaseline.yaml``:
    frozen ``good_disco``, CLS per lead, 12 non-linear heads, one epoch, seed 0)
    on every labelled record of the data folder it is given, and writes the
    journal folder through the same code as the registered build:
    ``src.run.save_head`` / ``report.write_predictions`` (as ``fit_final``) and
    ``scripts.container.export_model`` (top-5% threshold from the in-sample
    scores). This reproduces the registered *recipe*, not its bytes. Known
    differences from the registered fit:

    * **Labels.** The registered fit used the ``any_positive`` policy (D-025:
      every exam of a CODE-15% patient with any positive exam is positive). That
      needs ``patient_id`` from ``exams.csv``, which the WFDB headers do not carry,
      so here the label is the header's ``# Chagas label`` -- the exam-level
      label (``label_exam_level`` in our metadata). On our copy of the training
      data that is 8,190 positives instead of 11,854: 3,664 of 366,185 records
      (CODE-15% exams of ever-positive patients) flip from positive to negative.
    * **Records.** Whatever the organisers' data folder holds, in
      ``find_records`` order, not our 366,185 metadata rows in metadata order; the
      balanced sampler's seed-0 draws therefore differ even on identical data.
    * **Numerics.** Features come from the container's own encoder pass, not the
      cached GPU features (measured parity: max abs 1e-5 on features).

    The head and threshold are new; ``run_model.py`` then loads them as any
    journal folder. Needs the training data, the train step and roughly
    ``n_records x 36.9 kB`` of temporary disk for features (13.5 GB for 366k
    records; ``SBN_FEATURE_TMPDIR``, default the system temp dir).

``"registered_folder"`` -- option (c)
    The image ships the registered folder at ``/challenge/registered_model``
    (``SBN_REGISTERED_MODEL_DIR`` overrides, for tests and local simulation), a
    path the ``/challenge/model`` mount cannot shadow. ``train_model`` verifies it
    against ``configs/registered_model_pins.json`` (the REGISTRATION.md sha256s
    plus the good_disco chunk sha256), copies the pinned files into the model
    folder and re-verifies the copies -- no training. ``load_model``, given a
    folder with neither ``journal_manifest.json`` nor ``model.pth`` (a fresh
    mount, run-only), verifies and loads the registered folder in place. Either
    flow scores the registered bytes. A folder that holds ``model.pth`` but no
    manifest (say a mount reused from the 2025 run) still goes to the anchor
    path, with a loud warning -- the specified fallback condition; whether
    that case should also be redirected is for the authors. Acceptable only if
    the organisers accept pre-trained heads in the image.

Any other value raises :class:`SubmissionModeError` at the first call.

What the image would need (not done here: Dockerfile and ``model/`` are frozen)
-------------------------------------------------------------------------------
Option (c), on the branch whose URL goes into the submission form:

1. ``git add -f model/journal_manifest.json model/head_final.pt model/export.json``
   (``.gitignore`` excludes ``model/*``; the organisers build from the clone) and
   check the three sha256s in a fresh clone of that commit.
2. Dockerfile, after the three "DO NOT EDIT" lines::

       COPY ./model /challenge/registered_model
       ENV SBN_SUBMISSION_MODE=registered_folder
       RUN SBN_REGISTERED_MODEL_DIR=/challenge/registered_model \\
           python -m src.container.submission verify

   The ``RUN`` fails the build, not the shot, if a byte differs from the pins
   (including the good_disco chunks).

Option (b): only ``ENV SBN_SUBMISSION_MODE=train_registered_recipe``; nothing
model-specific is shipped. Both: ``configs/`` must stay in the build context
(it is; ``.dockerignore`` excludes ``data``, ``docs`` and ``results/runs``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.container import journal_model as JM  # noqa: E402

MODE_ENV = "SBN_SUBMISSION_MODE"
MODE_ANCHOR = "anchor"
MODE_TRAIN = "train_registered_recipe"
MODE_FOLDER = "registered_folder"
MODES = (MODE_ANCHOR, MODE_TRAIN, MODE_FOLDER)

REGISTERED_DIR_ENV = "SBN_REGISTERED_MODEL_DIR"
DEFAULT_REGISTERED_DIR = Path("/challenge/registered_model")
PINS_PATH = REPO_ROOT / "configs" / "registered_model_pins.json"
#: Another pins file (a later registration, e.g. registration-v2's own-encoder
#: folder, D-067); relative paths resolve against the repository root.
PINS_ENV = "SBN_REGISTERED_PINS"

RECIPE_CONFIG = REPO_ROOT / "configs" / "anchor_rebaseline.yaml"
RECIPE_SEED = 0
RECIPE_FRACTION = 0.05
RECIPE_SLOT = "final"
FIT_DIR_NAME = "registered_recipe_fit"
FEATURE_TMP_ENV = "SBN_FEATURE_TMPDIR"
WORKERS_ENV = "SBN_TRAIN_WORKERS"
#: More than this fraction of unreadable training records aborts the fit.
MAX_FAIL_FRACTION = 0.05

LABEL_NOTE = (
    "Labels are the WFDB header's '# Chagas label' (exam level). The registered fit used label_policy "
    "any_positive (D-025), which needs patient_id from exams.csv and cannot be rebuilt from headers; on our copy "
    "of the training data the two differ on 3,664 of 366,185 records (8,190 vs 11,854 positives)."
)


class SubmissionModeError(ValueError):
    """``SBN_SUBMISSION_MODE`` names no mode, or the recipe cannot run in this mode."""


class RegisteredModelError(RuntimeError):
    """The registered folder is missing or its bytes differ from the pins. Never caught here."""


# --------------------------------------------------------------------------
# The flag
# --------------------------------------------------------------------------


def active_mode(environ: Optional[dict] = None) -> Optional[str]:
    """``None`` for the anchor behaviour (unset, empty or ``"anchor"``), else the mode name."""
    raw = (os.environ if environ is None else environ).get(MODE_ENV, "")
    value = str(raw).strip()
    if value in ("", MODE_ANCHOR):
        return None
    if value not in MODES:
        raise SubmissionModeError(f"{MODE_ENV}={raw!r} is not one of {MODES} (or unset)")
    return value


def registered_dir() -> Path:
    return Path(os.environ.get(REGISTERED_DIR_ENV) or DEFAULT_REGISTERED_DIR)


# --------------------------------------------------------------------------
# Hashes
# --------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_chunks(directory: str | Path) -> str:
    """sha256 of the byte-concatenated chunks, in the order ``JM.read_good_disco_checkpoint`` reads them."""
    directory = Path(directory)
    if not directory.is_dir():
        raise RegisteredModelError(f"encoder chunk directory not found: {directory}")
    files = sorted(f for f in os.listdir(directory) if (directory / f).is_file())
    if not files:
        raise RegisteredModelError(f"no encoder chunks in {directory}")
    h = hashlib.sha256()
    for f in files:
        with open(directory / f, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
    return h.hexdigest()


def _is_sha256(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def pins_path() -> Path:
    raw = os.environ.get(PINS_ENV)
    if not raw:
        return PINS_PATH
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def load_pins(path: Optional[str | Path] = None) -> dict[str, Any]:
    """The pins: ``files`` (relative path -> sha256, the manifest included) and either
    ``good_disco_sha256`` (registration-v1: the slot uses the shipped chunks) or
    ``good_disco_sha256: null`` with every encoder byte among ``files`` (registration-v2)."""
    path = Path(path) if path is not None else pins_path()
    try:
        pins = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RegisteredModelError(f"cannot read the registered-model pins {path}: {e}") from e
    files = pins.get("files")
    if not isinstance(files, dict) or not files or not all(_is_sha256(v) for v in files.values()):
        raise RegisteredModelError(f"{path}: 'files' must map file names to sha256 hex digests")
    if JM.MANIFEST_NAME not in files:
        raise RegisteredModelError(f"{path}: the pins must cover {JM.MANIFEST_NAME}")
    if "good_disco_sha256" not in pins or not (pins["good_disco_sha256"] is None
                                               or _is_sha256(pins["good_disco_sha256"])):
        raise RegisteredModelError(f"{path}: 'good_disco_sha256' must be a sha256 hex digest, or null when the "
                                   "encoder is a pinned file of the folder")
    return pins


def _manifest_files(folder: Path) -> tuple[set[str], bool]:
    """Files the folder's manifest makes ``JournalModel`` read, and whether a slot uses good_disco."""
    manifest = json.loads((folder / JM.MANIFEST_NAME).read_text(encoding="utf-8"))
    names: set[str] = set()
    uses_good_disco = False
    for slot in manifest.get("slots", []):
        names.add(str(slot.get("head")))
        enc = str(slot.get("encoder"))
        if enc == "good_disco":
            uses_good_disco = True
            continue
        p = folder / enc
        if p.is_dir():
            names |= {f.relative_to(folder).as_posix() for f in JM.encoder_chunk_files(p)}
        else:
            names.add(enc)
    return names, uses_good_disco


def verify_registered_folder(folder: str | Path, pins: Optional[dict[str, Any]] = None,
                             check_encoder: bool = True, encoder_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Check every pinned file (and the good_disco chunks) byte for byte; raise on any difference.

    Files in ``folder`` that the pins do not name are reported, not copied and
    not loaded (``JournalModel`` reads only what the manifest names, and the
    manifest itself is pinned).
    """
    folder = Path(folder)
    pins = pins if pins is not None else load_pins()
    if not folder.is_dir():
        raise RegisteredModelError(f"registered model folder not found: {folder}")
    got: dict[str, str] = {}
    bad: dict[str, str] = {}
    for name, want in pins["files"].items():
        p = folder / name
        if not p.is_file():
            bad[name] = "missing"
            continue
        got[name] = sha256_file(p)
        if got[name] != want:
            bad[name] = f"sha256 {got[name]} != pinned {want}"
    if not bad:
        # every byte the manifest makes the model read must be pinned (the manifest itself is)
        read, uses_good_disco = _manifest_files(folder)
        unpinned = sorted(read - set(pins["files"]))
        if unpinned:
            bad["unpinned"] = f"the manifest reads files the pins do not cover: {unpinned}"
        if uses_good_disco and pins["good_disco_sha256"] is None:
            bad["good_disco"] = "a slot uses good_disco but the pins carry no good_disco_sha256"
    enc_sha = None
    if check_encoder and pins["good_disco_sha256"] is not None:
        enc_dir = Path(encoder_dir) if encoder_dir is not None else Path(JM.GOOD_DISCO_DIR)
        try:
            enc_sha = sha256_chunks(enc_dir)
        except RegisteredModelError as e:
            bad["good_disco"] = str(e)
        else:
            if enc_sha != pins["good_disco_sha256"]:
                bad["good_disco"] = f"sha256 {enc_sha} != pinned {pins['good_disco_sha256']} ({enc_dir})"
    if bad:
        raise RegisteredModelError(f"registered model folder {folder} does not match the pins: {bad}")
    ignored = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                     if p.is_file() and p.relative_to(folder).as_posix() not in pins["files"])
    return {"folder": str(folder), "files_sha256": got, "good_disco_sha256": enc_sha, "ignored": ignored}


# --------------------------------------------------------------------------
# Option (c): the registered folder
# --------------------------------------------------------------------------


def install_registered_folder(model_folder: str | Path, source: Optional[str | Path] = None,
                              verbose: bool = False) -> dict[str, Any]:
    """Verify ``source`` (default :func:`registered_dir`), copy the pinned files into
    ``model_folder``, verify the copies. Other files in ``model_folder`` are left alone;
    the journal manifest takes precedence over them in ``team_code.load_model``."""
    pins = load_pins()
    src = Path(source) if source is not None else registered_dir()
    info = verify_registered_folder(src, pins)
    dst = Path(model_folder)
    dst.mkdir(parents=True, exist_ok=True)
    pre_existing = sorted(p.name for p in dst.iterdir())
    for name in pins["files"]:
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.partial")
        shutil.copyfile(src / name, tmp)
        os.replace(tmp, target)
    verify_registered_folder(dst, pins, check_encoder=False)
    info.update(installed_to=str(dst), pre_existing=pre_existing)
    msg = (f"[{MODE_FOLDER}] installed the registered model {src} -> {dst} "
           f"({len(pins['files'])} pinned files{' and good_disco' if pins['good_disco_sha256'] else ''} "
           f"verified; no training)")
    print(msg, file=sys.stderr)
    if verbose:
        print(json.dumps(info, indent=2))
    return info


def needs_fallback(model_folder: str | Path) -> bool:
    """A folder ``team_code.load_model`` could not serve: neither a journal manifest nor ``model.pth``."""
    model_folder = Path(model_folder)
    return not (model_folder / JM.MANIFEST_NAME).exists() and not (model_folder / "model.pth").exists()


def load_for_registered_mode(model_folder: str | Path, verbose: bool = False) -> Optional["JM.JournalModel"]:
    """``team_code.load_model`` in option (c), for a folder without a journal manifest.

    Neither manifest nor ``model.pth``: the verified registered model. ``model.pth``
    present (e.g. a mount reused from the 2025 Challenge run): ``None``, so the
    anchor path serves those files as before -- with a loud warning, because
    that is not the registered model.
    """
    if needs_fallback(model_folder):
        return load_registered_model(verbose)
    print(f"WARNING: [{MODE_FOLDER}] {model_folder} holds model.pth but no {JM.MANIFEST_NAME}: serving the CinC "
          f"anchor files found there, NOT the registered model", file=sys.stderr)
    return None


def load_registered_model(verbose: bool = False, device: Optional[str] = None) -> "JM.JournalModel":
    """Verify the registered folder and load it in place (run-only flow, empty mount)."""
    src = registered_dir()
    info = verify_registered_folder(src)
    print(f"[{MODE_FOLDER}] model folder has neither {JM.MANIFEST_NAME} nor model.pth; "
          f"loading the verified registered model from {src}", file=sys.stderr)
    if verbose:
        print(json.dumps(info, indent=2))
    return JM.JournalModel.load(src, device=device)


# --------------------------------------------------------------------------
# Option (b): fit the registered recipe on the given data folder
# --------------------------------------------------------------------------


def _header_fields(record_path: str) -> tuple[Optional[int], str]:
    """``(label or None, source)`` from the header, via the organisers' ``helper_code``."""
    import helper_code

    header = helper_code.load_header(record_path)
    raw, has = helper_code.get_variable(header, helper_code.label_string)
    label = helper_code.sanitize_boolean_value(raw) if has else None
    if label not in (0, 1):
        label = None
    source = helper_code.get_source(header)
    return label, (str(source) if source else "unknown")


class _RecordSignals(torch.utils.data.Dataset):
    """``(index, [2250, 12] signal or None, error)`` -- the container's own preprocessing."""

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            sig = JM.preprocess_record(self.paths[i])
            if not torch.isfinite(sig).all():
                raise ValueError("non-finite values after preprocessing")
            return i, sig, None
        except Exception as e:  # noqa: BLE001 - counted, and the fit aborts above MAX_FAIL_FRACTION
            return i, None, f"{type(e).__name__}: {e}"


def _as_list(batch):
    return batch


def extract_features(paths: Sequence[str], encoder, device: torch.device, out: np.ndarray,
                     workers: int = 0, verbose: bool = False) -> tuple[np.ndarray, dict[str, str]]:
    """Per-lead CLS features of every record into ``out [N, 12, D]``; returns ``(ok mask, failures)``.

    Batches of ``JM.ENCODE_CHUNK`` records through ``JM.encode`` -- the exact
    call ``JournalModel`` scores with.
    """
    loader = torch.utils.data.DataLoader(_RecordSignals(paths), batch_size=JM.ENCODE_CHUNK, shuffle=False,
                                         num_workers=workers, collate_fn=_as_list)
    ok = np.zeros(len(paths), dtype=bool)
    failures: dict[str, str] = {}
    t0 = time.time()
    for b, batch in enumerate(loader):
        good = [(i, s) for i, s, _ in batch if s is not None]
        for i, _, err in batch:
            if err is not None:
                failures[paths[i]] = err
        if good:
            idx = [i for i, _ in good]
            x = torch.stack([s for _, s in good]).to(device)
            out[idx] = JM.encode(encoder, x, "cls").float().cpu().numpy()
            ok[idx] = True
        if verbose and (b % 200 == 0 or b == len(loader) - 1):
            print(f"[{MODE_TRAIN}] features {min((b + 1) * JM.ENCODE_CHUNK, len(paths))}/{len(paths)} "
                  f"({time.time() - t0:.0f}s, {len(failures)} failed)")
    return ok, failures


def _check_container_recipe(cfg: dict[str, Any], head_cfg) -> None:
    if str(cfg.get("model")) != "anchor_good_disco" or str(cfg.get("feature_kind", "cls")) != "cls":
        raise SubmissionModeError(f"this mode encodes with good_disco CLS only; config has model={cfg.get('model')!r} "
                                  f"feature_kind={cfg.get('feature_kind')!r}")
    extra = {k: v for k, v in (("demographics", head_cfg.demographics), ("rhythm", getattr(head_cfg, "rhythm", False)),
                               ("aux_lambda", getattr(head_cfg, "aux_lambda", 0)), ("sampler", head_cfg.sampler),
                               ("loss", head_cfg.loss if head_cfg.loss in ("group_dro", "nnpu") else None)) if v}
    if extra:
        raise SubmissionModeError(f"head config needs inputs this mode does not build: {extra}")


def train_registered_recipe(data_folder: str | Path, model_folder: str | Path, verbose: bool = False,
                            device: Optional[str] = None, config: str | Path = RECIPE_CONFIG,
                            seed: int = RECIPE_SEED, workers: Optional[int] = None) -> dict[str, Any]:
    """Option (b). Returns the fit's ``run.json`` payload; the journal folder lands in ``model_folder``."""
    import pandas as pd

    from scripts.container import export_model as EM
    from scripts.container import fit_final as FF
    from src import run as runmod
    from src.eval import registry as R
    from src.eval import report
    from src.train.head import HeadConfig, fit_head, predict_head
    import helper_code

    t0 = time.time()
    config = Path(config)
    cfg, text, config_sha = runmod.load_config(config)
    head_cfg = HeadConfig.from_dict(cfg.get("head"))
    FF._check_recipe(cfg, head_cfg, config)  # unselected recipe, no train_folds: as the registered fit
    _check_container_recipe(cfg, head_cfg)

    data_folder, model_folder = Path(data_folder), Path(model_folder)
    records = helper_code.find_records(str(data_folder))
    if not records:
        raise SubmissionModeError(f"no records under {data_folder}")
    paths = [str(data_folder / r) for r in records]
    fields = [_header_fields(p) for p in paths]
    labelled = np.array([lab is not None for lab, _ in fields])
    n_unlabelled = int((~labelled).sum())
    keep = np.flatnonzero(labelled)
    paths = [paths[i] for i in keep]
    records = [records[i] for i in keep]
    labels = np.array([fields[i][0] for i in keep], dtype=np.int64)
    sources = np.array([fields[i][1] for i in keep], dtype=object)
    if len(paths) == 0:
        raise SubmissionModeError(f"none of the records under {data_folder} has a '# Chagas label' header line")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if workers is None:
        workers = int(os.environ.get(WORKERS_ENV) or min(8, os.cpu_count() or 1))
    if verbose:
        print(f"[{MODE_TRAIN}] {len(paths)} labelled records ({int(labels.sum())} positive, {n_unlabelled} "
              f"unlabelled skipped) from {data_folder}; device {dev}; {workers} preprocessing workers")
    encoder = JM.load_good_disco().eval().to(dev)
    enc_sha = sha256_chunks(JM.GOOD_DISCO_DIR)
    emb_dim = int(encoder.encoder.width)

    tmp = Path(tempfile.mkdtemp(prefix="sbn_feats_", dir=os.environ.get(FEATURE_TMP_ENV) or None))
    try:
        feats = np.lib.format.open_memmap(tmp / "cls.npy", mode="w+", dtype=np.float32,
                                          shape=(len(paths), JM.N_LEADS, emb_dim))
        t_feat = time.time()
        ok, failures = extract_features(paths, encoder, dev, feats, workers=workers, verbose=verbose)
        t_feat = time.time() - t_feat
        del encoder
        n_fail = int((~ok).sum())
        if n_fail > MAX_FAIL_FRACTION * len(paths):
            raise SubmissionModeError(f"{n_fail}/{len(paths)} training records could not be preprocessed "
                                      f"(> {MAX_FAIL_FRACTION:.0%}); first: {list(failures.items())[:3]}")
        rows = np.flatnonzero(ok)
        y = labels[rows]
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            raise SubmissionModeError(f"training rows hold {n_pos} positives of {len(y)}; a head cannot be fitted")
        view = runmod._RowView(feats, rows)
        t_fit = time.time()
        head, log = fit_head(view, y, head_cfg, seed, dev, verbose=verbose)
        t_fit = time.time() - t_fit
        scores = predict_head(head, view, dev)

        fit_dir = model_folder / FIT_DIR_NAME
        if fit_dir.exists():
            shutil.rmtree(fit_dir)
        fit_dir.mkdir(parents=True)
        (fit_dir / "config.yaml").write_text(text, encoding="utf-8")
        runmod.save_head(fit_dir, head, head_cfg.kind, str(cfg["feature_kind"]))
        frame = pd.DataFrame({"record": [records[i] for i in rows], "patient_key": [records[i] for i in rows],
                              "source": sources[rows].astype(str), "label": y.astype(float),
                              "score": np.asarray(scores, dtype=float)})
        report.write_predictions(fit_dir, FF.PREDS_NAME, frame)
        train_src = pd.Series(sources[rows].astype(str))
        summary: dict[str, Any] = {
            "kind": "container_registered_recipe_fit",
            "registered": False,
            "mode": MODE_TRAIN,
            "decision": "D-053 option (b): the registered recipe refitted on the organisers' data folder",
            "registry_note": FF.REGISTRY_NOTE,
            "fit_id": f"{cfg['run_name']}_container_s{seed}",
            "source_config": str(config.relative_to(REPO_ROOT)) if config.is_relative_to(REPO_ROOT) else str(config),
            "config_sha": config_sha,
            "run_name": cfg["run_name"], "task": cfg.get("task"), "model": str(cfg["model"]),
            "objective": str(cfg["objective"]), "beta": cfg["beta"], "seed": int(seed), "device": str(dev),
            "label_policy": "exam_level",
            "label_policy_registered": cfg["label_policy"],
            "label_source": "wfdb_header",
            "label_note": LABEL_NOTE,
            "feature_kind": str(cfg["feature_kind"]),
            "data_folder": str(data_folder),
            "train_rows": "every record of the data folder with a '# Chagas label' header line that preprocessed",
            "n_records_found": int(len(fields)),
            "n_unlabelled_skipped": n_unlabelled,
            "n_preprocessing_failed": n_fail,
            "failed_records": dict(list(failures.items())[:20]),
            "n_train": int(len(rows)),
            "n_train_pos": n_pos,
            "n_train_by_source": {str(k): int(v) for k, v in train_src.value_counts().sort_index().items()},
            "n_train_pos_by_source": {str(k): int(v) for k, v in
                                      pd.Series(y, index=train_src).groupby(level=0).sum().sort_index().items()},
            "encoder": {"id": str(cfg["model"]), "feature_kind": "cls", "path": None, "sha256": enc_sha},
            "head_file": "head.pt",
            "train_preds": f"preds_{FF.PREDS_NAME}.parquet",
            "config_file": "config.yaml",
            "head_log": log,
            "git_sha": R.git_sha(),
            "timestamp": R.utc_now(),
            "seconds_features": t_feat,
            "seconds_fit": t_fit,
        }
        summary["seconds_total"] = time.time() - t0
        report.write_run_json(fit_dir, summary)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc = EM.main(["--out", str(model_folder), "--slot", f"{RECIPE_SLOT}={fit_dir}", "--output-slot", RECIPE_SLOT,
                  "--binarization", f"top_fraction:{RECIPE_FRACTION}", "--force"])
    if rc != 0:
        raise SubmissionModeError(f"export_model returned {rc}")
    print(f"[{MODE_TRAIN}] fitted the registered recipe on {summary['n_train']} records "
          f"({n_pos} positive; header labels, not any_positive) -> {model_folder}", file=sys.stderr)
    return summary


# --------------------------------------------------------------------------
# team_code entry points
# --------------------------------------------------------------------------


def train_model(mode: str, data_folder: str, model_folder: str, verbose: bool) -> dict[str, Any]:
    if mode == MODE_TRAIN:
        return train_registered_recipe(data_folder, model_folder, verbose=verbose)
    if mode == MODE_FOLDER:
        return install_registered_folder(model_folder, verbose=verbose)
    raise SubmissionModeError(f"no training behaviour for mode {mode!r}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a registered model folder against the pins (build-time check).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("folder", nargs="?", default=None, help=f"default: ${REGISTERED_DIR_ENV} or {DEFAULT_REGISTERED_DIR}")
    v.add_argument("--no-encoder", action="store_true", help="skip the good_disco chunk hash")
    a = ap.parse_args(argv)
    folder = Path(a.folder) if a.folder else registered_dir()
    try:
        info = verify_registered_folder(folder, check_encoder=not a.no_encoder)
    except RegisteredModelError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(dict(info, ok=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
