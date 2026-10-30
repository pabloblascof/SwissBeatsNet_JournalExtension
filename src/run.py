"""One run = one config x one seed x one reporting fold, end to end.

::

    python -m src.run --config configs/anchor_rebaseline.yaml --seed 0 --outer-fold 2
    python -m src.run --config configs/<name>.yaml --seed 0 --dry-run
    python -m src.run --config configs/<name>.yaml --seed 0 --force   # re-run an existing run_id

A run_id that already has a ``run.json`` or registry rows is refused unless
``--force``: the registry is append-only, so a silent re-run would leave two
full sets of rows under one run_id (``paper_ready`` dedupes seeds, a mean
over rows would not). With ``--force`` every new row's ``notes`` names the
superseded timestamp so the digest can tell them apart.

The row-order guard. The feature cache's ``meta.json`` carries
``signal_records_sha256``, the hash of the metadata ``record`` order it was
encoded against. ``run()`` recomputes that hash from the metadata it loaded
and refuses to train on a mismatch: same N with a different order would pair
every feature row with the wrong record silently. Both hashes go into
``run.json``.

Flow: load metadata and the frozen splits -> apply the label policy -> open
the cached features for the config's encoder -> fit the head on every outer
fold but ``k`` (holding out one inner fold for selection if the config asks
for it) -> score fold ``k`` -> run the compass battery -> write
``results/runs/<run_id>/{preds_*.parquet, head.pt, config.yaml, run.json}`` -> append
registry rows. ``run_id = f"{run_name}_f{k}_s{seed}"``; ``config_sha`` is the
sha256 of the YAML text as read, so a re-run of the same file gets the same
sha and an edited file does not.

The leakage guard. Nothing that trains or selects may read a record whose
``outer_fold == k`` (project rule 4, ARCHITECTURE.md). The guard is a
function of the actual training row set, not of the config's intent, and it
raises :class:`LeakageError` -- it is not a warning, because a run that
leaked is a run whose every registry row is wrong.

Imports of ``src.data.metadata`` and ``src.data.labels`` are attempted first;
those modules are being written against ARCHITECTURE.md in parallel. If they
are absent this file falls back to reading ``metadata.parquet`` and
``frozen_splits.json`` directly and to the ``label_<policy>`` columns the
metadata contract already carries. The fallback is intentionally minimal: it
is not a second implementation of either module.

Config fields (YAML)::

    run_name: anchor_rebaseline
    model: anchor_good_disco          # encoder_id -> data/cache/features/<encoder_id>
    objective: mse_align              # registry label only
    beta: 1.0                         # nullable
    label_policy: any_positive
    outer_fold: 0                     # default when --outer-fold is not given
    train_folds: null                 # optional explicit list; must not contain k
    head: {kind: non_linear, epochs: 1, ...}   # src.train.head.HeadConfig fields
    eval: {battery: true, p3_replicates: 5, p3_prevalence: 0.02, n_bootstrap: 1000,
           ci_permutations: 16, point_permutations: 10000,
           interior_point_permutations: 1000, n_jobs: 1,
           metrics: [tpr_at_5, auroc, auprc]}
        # point_permutations: official count for outer_fold_all / cohort_*;
        # interior_point_permutations: P2-P5 frames and scalars; n_jobs:
        # bootstrap workers (-1 = all cores). Every tpr_at_5 row's notes
        # record the counts it was computed with.
    splits: configs/splits/frozen_splits.json
    metadata: data/cache/metadata.parquet
    features_root: data/cache/features
    feature_kind: cls                 # cls | mean | both (CLS and mean concatenated per lead -> 1536)
    results_root: results/runs        # tests point this at tmp_path
    registry_path: null               # tests point this at tmp_path
    provisional: false
    inner_eval_fold: null             # optional j: evaluate on inner fold j of fold k's pool (below)

Inner-fold evaluation (``inner_eval_fold: j``; S4 own-encoder pilot, D-048).
Within-pool comparisons that must not read the reporting fold: the head is
fitted on the pool's other inner folds and scored on inner fold ``j`` of the
pool, with the whole battery. Implemented by relabelling, in memory only:
the reporting fold's rows are removed (``outer_fold = -9``) and the pool's
inner-fold-``j`` rows take its place (``outer_fold = k``), so every guard and
proxy runs unchanged on the pseudo-fold and nothing reads fold ``k``. Rows
go to split ``outer_fold_<k>_inner<j>`` (never pooled with reporting-fold
rows). Not combinable with ``train_folds``. ``select_metric`` is allowed when
its selection fold ``seed % n_inner`` is not ``j`` (S4C wave C, W1, D-089):
the epoch is then selected on another inner fold of the pool, held out of the
training rows, and the scored fold ``j`` is never a selection fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compass.battery import run_battery  # noqa: E402
from src.eval import registry as R  # noqa: E402
from src.eval import report  # noqa: E402
from src.train.head import HeadConfig, fit_head, predict_head  # noqa: E402

LABEL_POLICIES = ("any_positive", "majority_tie_positive", "majority_tie_negative",
                  "drop_conflicting", "exam_level")

#: Metadata column that holds each policy's labels (ARCHITECTURE.md).
POLICY_COLUMN = {
    "any_positive": "label_any_positive",
    "majority_tie_positive": "label_majority_pos",
    "majority_tie_negative": "label_majority_neg",
    "drop_conflicting": "label_drop_conflicting",
    "exam_level": "label_exam_level",
}


class LeakageError(AssertionError):
    """A training or selection row belongs to the reporting fold."""


class ExistingRunError(RuntimeError):
    """``run_id`` already has artefacts or registry rows; pass ``--force`` to supersede."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> tuple[dict[str, Any], str, str]:
    """``(config, text, sha256(text))``; relative paths resolve against the repo root."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    for key in ("run_name", "model", "objective", "label_policy"):
        if key not in cfg:
            raise ValueError(f"config {path} lacks required field {key!r}")
    if cfg["label_policy"] not in LABEL_POLICIES:
        raise ValueError(f"unknown label_policy {cfg['label_policy']!r}; one of {LABEL_POLICIES}")
    cfg.setdefault("splits", "configs/splits/frozen_splits.json")
    cfg.setdefault("metadata", "data/cache/metadata.parquet")
    cfg.setdefault("features_root", "data/cache/features")
    cfg.setdefault("feature_kind", "cls")
    cfg.setdefault("results_root", "results/runs")
    cfg.setdefault("registry_path", None)
    cfg.setdefault("provisional", False)
    cfg.setdefault("beta", None)
    cfg.setdefault("head", {})
    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("battery", True)
    sub = cfg.get("train_subsample")
    if sub is not None and (not isinstance(sub, dict) or set(sub) != {"fraction"}):
        raise ValueError(f"train_subsample must be {{fraction: <(0, 1]>}}, got {sub!r}")
    if cfg.get("label_permutation") is not None and cfg["label_permutation"] not in PERMUTATION_KINDS:
        raise ValueError(f"label_permutation must be one of {PERMUTATION_KINDS}, got {cfg['label_permutation']!r}")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return cfg, text, sha


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# World: metadata, splits, labels, features
# ---------------------------------------------------------------------------


def _fallback_load_metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _fallback_attach_splits(df: pd.DataFrame, splits_path: Path) -> pd.DataFrame:
    """Map ``patient_key`` through the frozen JSON; unassigned patients get -1."""
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    df = df.copy()
    df["outer_fold"] = df["patient_key"].map(splits["outer"]).fillna(-1).astype(int)
    df["inner_fold"] = df["patient_key"].map(splits["inner"]).fillna(-1).astype(int)
    df.attrs["n_outer"] = int(splits.get("n_outer", df.loc[df.outer_fold >= 0, "outer_fold"].nunique()))
    df.attrs["n_inner"] = int(splits.get("n_inner", df.loc[df.inner_fold >= 0, "inner_fold"].nunique()))
    return df


def load_world(cfg: dict[str, Any]) -> pd.DataFrame:
    """Metadata with ``outer_fold``/``inner_fold`` attached from the frozen splits."""
    meta_path, splits_path = _resolve(cfg["metadata"]), _resolve(cfg["splits"])
    try:
        from src.data.metadata import attach_splits, load_metadata  # type: ignore
        df = attach_splits(load_metadata(meta_path), splits_path)
    except ImportError:
        df = _fallback_attach_splits(_fallback_load_metadata(meta_path), splits_path)
    for c in ("record", "patient_key", "source", "outer_fold", "inner_fold"):
        if c not in df.columns:
            raise ValueError(f"metadata lacks column {c!r}")
    if "n_inner" not in df.attrs:
        df.attrs["n_inner"] = int(df.loc[df.inner_fold >= 0, "inner_fold"].nunique() or 4)
    return df.reset_index(drop=True)


def _fallback_select(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    col = POLICY_COLUMN[policy]
    if col not in df.columns:
        raise ValueError(f"metadata lacks {col!r} for label_policy={policy!r}")
    out = df[df[col].notna()].copy()
    out["label"] = out[col].astype(int)
    return out


def select_labels(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Rows retained under ``policy`` with an int ``label`` column; index preserved."""
    try:
        from src.data.labels import select  # type: ignore
        return select(df, policy)
    except ImportError:
        return _fallback_select(df, policy)


FEATURE_KINDS = ("cls", "mean", "both", "tokens")


class ConcatFeatures:
    """``cls`` and ``mean`` caches concatenated per lead: ``[N, 12, 2 * D]``, lazily.

    ``feature_kind: both`` (D-026 A). Indexing returns ``np.concatenate([cls[idx],
    mean[idx]], axis=-1)``; nothing is materialised beyond the rows asked for.
    The order (CLS first) is the container's too (``journal_model.encode``).
    """

    def __init__(self, cls_feats, mean_feats) -> None:
        if cls_feats.shape != mean_feats.shape:
            raise ValueError(f"cls {cls_feats.shape} and mean {mean_feats.shape} caches disagree")
        self._a, self._b = cls_feats, mean_feats
        self.shape = (int(cls_feats.shape[0]), int(cls_feats.shape[1]), 2 * int(cls_feats.shape[2]))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return np.concatenate([np.asarray(self._a[idx], dtype=np.float32),
                               np.asarray(self._b[idx], dtype=np.float32)], axis=-1)


def open_features(features_root: Path | str, encoder_id: str, kind: str = "cls"):
    """``data/cache/features/<encoder_id>/{cls,mean}.f32`` as a read-only memmap.

    Uses ``src.data.features.open_features`` when that module is present and
    falls back to opening the memmap by the ARCHITECTURE.md layout otherwise.
    ``kind="both"`` returns a :class:`ConcatFeatures` over the two memmaps.
    """
    if kind not in FEATURE_KINDS:
        raise ValueError(f"feature_kind must be one of {FEATURE_KINDS}, got {kind!r}")
    if kind == "tokens":  # S4C H6: per-lead tokens + validity channel, from the sibling features_tokens root
        from src.data import features_tokens as FTK

        cache = FTK.open_tokens(encoder_id, root=_resolve(features_root).parent / "features_tokens")
        clean = feature_meta(features_root, encoder_id)
        if clean and cache["meta"].get("encoder_sha256") != clean.get("sha256"):
            raise ValueError(f"{encoder_id}: the token cache was made by encoder {cache['meta'].get('encoder_sha256')}, "
                             f"the CLS cache by {clean.get('sha256')}")
        return FTK.TokenView(cache, int(cache["meta"]["n_cache"]))
    if kind == "both":
        return ConcatFeatures(open_features(features_root, encoder_id, "cls"),
                              open_features(features_root, encoder_id, "mean"))
    try:
        from src.data.features import open_features as _shared  # type: ignore
        return _shared(encoder_id, _resolve(features_root))[kind]
    except ImportError:
        pass
    d = _resolve(features_root) / encoder_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    n = int(meta["n"])
    path = d / f"{kind}.f32"
    expected = n * 12 * 768 * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {expected} for n={n}")
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, 12, 768))


def feature_meta(features_root: Path | str, encoder_id: str) -> dict[str, Any]:
    """The cache's ``meta.json`` (empty dict when absent)."""
    mp = _resolve(features_root) / encoder_id / "meta.json"
    if not mp.is_file():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_records_hash(meta_df: pd.DataFrame, fmeta: dict[str, Any]) -> dict[str, Any]:
    """Compare the metadata row order with the hash the feature cache was built on.

    Returns ``{"metadata_records_sha256", "features_records_sha256", "checked"}``;
    raises ``ValueError`` on a mismatch. ``checked`` is ``False`` when the
    cache predates the hash (nothing to compare against), which is printed
    rather than hidden.
    """
    from src.data.signals import records_sha256

    have = records_sha256([str(r) for r in meta_df["record"].to_numpy()])
    want = fmeta.get("signal_records_sha256")
    out = {"metadata_records_sha256": have, "features_records_sha256": want, "checked": want is not None}
    if want is None:
        print("[run] feature cache meta.json carries no signal_records_sha256; row-order check skipped")
        return out
    if want != have:
        raise ValueError(
            f"feature cache was encoded against record order {want[:12]}..., metadata is "
            f"{have[:12]}...: row-order contract broken; rebuild the cache or the metadata")
    return out


def existing_run(run_dir: Path, run_id: str, registry_path: Optional[Path]) -> Optional[str]:
    """Earliest timestamp of a prior run under ``run_id`` (or ``"run.json"``), else ``None``."""
    stamps = [r["timestamp"] for r in R.read(registry_path) if r.get("run_id") == run_id]
    if stamps:
        return min(stamps)
    if (run_dir / "run.json").is_file():
        try:
            return str(json.loads((run_dir / "run.json").read_text()).get("timestamp", "run.json"))
        except json.JSONDecodeError:
            return "run.json"
    return None


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def assert_no_leakage(train_outer: np.ndarray, k: int, what: str = "training") -> None:
    """Raise :class:`LeakageError` if any ``train_outer`` entry equals ``k``."""
    train_outer = np.asarray(train_outer)
    n_bad = int((train_outer == k).sum())
    if n_bad:
        raise LeakageError(
            f"{n_bad} {what} record(s) belong to reporting fold {k}. Nothing that trains "
            "or selects may read the reporting fold (project rule 4)."
        )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


#: ``outer_fold`` given to the reporting fold's rows under ``inner_eval_fold``:
#: neither the pool (``>= 0``) nor the test fold (``== k``).
INNER_EVAL_REMOVED = -9

#: S9.overfit (D-097): opt-in manipulations of the training rows and labels. Both config keys are absent from every
#: earlier config, so no existing ``config_sha`` changes and no existing run behaves differently.
SUBSAMPLE_SALT = 9097
PERMUTATION_SALT = 9098
PERMUTATION_KINDS = ("within_source", "all")


def patient_table(df: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """One row per patient among ``rows``: ``patient_key``, ``source`` and the patient's label (the maximum over its
    records, which under ``any_positive`` is every record's label). Sorted by ``patient_key``."""
    sub = df.iloc[np.asarray(rows, dtype=np.int64)]
    t = pd.DataFrame({"patient_key": sub["patient_key"].astype(str).to_numpy(), "source": sub["source"].to_numpy(),
                      "label": sub["label"].to_numpy(dtype=float)})
    return t.groupby("patient_key", sort=True).agg(source=("source", "first"), label=("label", "max")).reset_index()


def subsample_train(df: pd.DataFrame, train_rows: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """The records of a random ``fraction`` of the training patients, drawn within every source x label stratum
    (at least one patient per stratum), seeded by ``(seed, fraction)``. ``fraction == 1`` returns ``train_rows``."""
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_subsample.fraction must lie in (0, 1], got {fraction}")
    if fraction == 1.0:
        return train_rows
    pts = patient_table(df, train_rows)
    rng = np.random.default_rng([int(seed), int(round(fraction * 10**6)), SUBSAMPLE_SALT])
    keep = []
    for _, grp in pts.groupby(["source", "label"], sort=True):
        n = max(1, int(round(fraction * len(grp))))
        keep.append(rng.choice(grp["patient_key"].to_numpy(), size=n, replace=False))
    chosen = np.isin(df["patient_key"].astype(str).to_numpy()[train_rows], np.concatenate(keep))
    return train_rows[chosen]


def permute_labels(df: pd.DataFrame, rows: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Training labels shuffled among patients (``kind="all"``) or among the patients of each source
    (``"within_source"``), seeded by ``seed``; every record takes its patient's new label."""
    if kind not in PERMUTATION_KINDS:
        raise ValueError(f"label_permutation must be one of {PERMUTATION_KINDS}, got {kind!r}")
    pts = patient_table(df, rows)
    rng = np.random.default_rng([int(seed), PERMUTATION_SALT])
    new = pts["label"].to_numpy().copy()
    groups = pts["source"].astype(str).to_numpy() if kind == "within_source" else np.zeros(len(pts), dtype=int)
    for g in sorted(set(groups.tolist()), key=str):
        idx = np.flatnonzero(groups == g)
        new[idx] = new[idx][rng.permutation(len(idx))]
    lookup = dict(zip(pts["patient_key"].to_numpy(), new))
    keys = df["patient_key"].astype(str).to_numpy()[np.asarray(rows, dtype=np.int64)]
    return np.array([lookup[p] for p in keys], dtype=int)


def inner_eval_world(df: pd.DataFrame, k: int, j: int) -> pd.DataFrame:
    """The metadata with inner fold ``j`` of fold ``k``'s pool as the scored
    fold and the reporting fold ``k`` removed (module docstring)."""
    outer = df["outer_fold"].to_numpy().copy()
    inner = df["inner_fold"].to_numpy()
    pool = (outer != k) & (outer >= 0)
    if not (pool & (inner == j)).any():
        raise ValueError(f"inner fold {j} has no rows in the pool of outer fold {k}")
    new = outer.copy()
    new[outer == k] = INNER_EVAL_REMOVED
    new[pool & (inner == j)] = k
    out = df.copy()
    out["outer_fold"] = new
    out.attrs.update(df.attrs)
    return out


def plan_rows(df: pd.DataFrame, cfg: dict[str, Any], k: int, seed: int) -> dict[str, np.ndarray]:
    """Positional row sets: ``train``, ``val`` (may be empty), ``test``, ``pool``.

    ``df`` is the full metadata frame (positional index = cache row) with a
    ``label`` column that is NaN where the policy dropped the record.
    """
    outer = df["outer_fold"].to_numpy()
    inner = df["inner_fold"].to_numpy()
    labelled = df["label"].notna().to_numpy()
    n_inner = int(df.attrs.get("n_inner", 4))

    train_folds = cfg.get("train_folds")
    if train_folds is None:
        pool = labelled & (outer != k) & (outer >= 0)
    else:
        pool = labelled & np.isin(outer, list(train_folds))
    test = labelled & (outer == k)

    head_cfg = HeadConfig.from_dict(cfg.get("head"))
    val = np.zeros(len(df), dtype=bool)
    if head_cfg.select_metric is not None:
        j = seed % n_inner
        inner_eval = cfg.get("inner_eval_fold")
        if inner_eval is not None and j == int(inner_eval):  # the scored inner fold is never a selection fold
            raise ValueError(f"selection fold {j} (seed % {n_inner}) is the inner eval fold {int(inner_eval)}")
        val = pool & (inner == j)
        train = pool & (inner != j)
    else:
        train = pool
    if head_cfg.train_sources is not None:  # S4C H1/H2: head-training records from these sources only
        train = train & df["source"].isin(list(head_cfg.train_sources)).to_numpy()
    sub = cfg.get("train_subsample")
    if sub is not None:  # S9.overfit O-2 (D-097): a patient-level, source x label stratified subsample
        keep = subsample_train(df, np.flatnonzero(train), float(sub["fraction"]), seed)
        train = np.zeros(len(df), dtype=bool)
        train[keep] = True

    # The guard, on the actual rows.
    assert_no_leakage(outer[train], k, "training")
    assert_no_leakage(outer[val], k, "selection")
    return {"train": np.flatnonzero(train), "val": np.flatnonzero(val),
            "test": np.flatnonzero(test), "pool": np.flatnonzero(pool)}


def run(
    config_path: Path | str,
    seed: int,
    outer_fold: Optional[int] = None,
    device: str = "cpu",
    dry_run: bool = False,
    verbose: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    cfg, text, config_sha = load_config(config_path)
    k = int(cfg.get("outer_fold", 0) if outer_fold is None else outer_fold)
    run_id = f"{cfg['run_name']}_f{k}_s{seed}"
    run_dir = _resolve(cfg["results_root"]) / run_id
    registry_path = _resolve(cfg["registry_path"]) if cfg.get("registry_path") else None

    meta = load_world(cfg)
    inner_eval = cfg.get("inner_eval_fold")
    if inner_eval is not None:
        if cfg.get("train_folds") is not None:
            raise ValueError("inner_eval_fold is not combinable with train_folds")
        n_inner = int(meta.attrs.get("n_inner", 4))
        if HeadConfig.from_dict(cfg.get("head")).select_metric is not None and seed % n_inner == int(inner_eval):
            raise ValueError(f"inner_eval_fold {int(inner_eval)} would also be head.select_metric's selection fold "
                             f"(seed % {n_inner} = {seed % n_inner}); use a seed whose selection fold differs")
        meta = inner_eval_world(meta, k, int(inner_eval))
    labelled = select_labels(meta, cfg["label_policy"])
    meta["label"] = np.nan
    meta.loc[labelled.index, "label"] = labelled["label"].astype(float)

    rows = plan_rows(meta, cfg, k, seed)
    feats = open_features(cfg["features_root"], cfg["model"], cfg["feature_kind"])
    if feats.shape[0] != len(meta):
        raise ValueError(f"features have {feats.shape[0]} rows, metadata {len(meta)}: row order contract broken")
    records_check = check_records_hash(meta, feature_meta(cfg["features_root"], cfg["model"]))

    prior = existing_run(run_dir, run_id, registry_path)
    supersede_note = None
    if prior is not None and not dry_run:
        if not force:
            raise ExistingRunError(
                f"{run_id} already exists (first registered {prior}); the registry is append-only. "
                "Re-run with --force to supersede it, or choose another run_name/seed/fold.")
        supersede_note = f"supersedes {run_id} registered {prior}"

    summary = {
        "run_id": run_id, "config_sha": config_sha, "seed": seed, "outer_fold": k,
        "n_train": int(len(rows["train"])), "n_val": int(len(rows["val"])),
        "n_test": int(len(rows["test"])),
        "n_train_pos": int(np.nansum(meta["label"].to_numpy()[rows["train"]])),
        "n_train_by_source": {str(k): int(v) for k, v in
                              meta.iloc[rows["train"]]["source"].value_counts().sort_index().items()},
        "n_test_pos": int(np.nansum(meta["label"].to_numpy()[rows["test"]])),
        "run_dir": str(run_dir), "dry_run": dry_run, "records_check": records_check,
        "supersedes": prior if force else None,
        "inner_eval_fold": None if inner_eval is None else int(inner_eval),
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    if dry_run:
        return summary
    if summary["n_train_pos"] == 0 or summary["n_test_pos"] == 0:
        raise ValueError("no positives in train or test rows; check label policy and splits")

    head_cfg = HeadConfig.from_dict(cfg.get("head"))
    labels = meta["label"].to_numpy()
    y_train = labels[rows["train"]].astype(int)
    if cfg.get("label_permutation") is not None:  # S9.overfit O-4 (D-097): the head sees shuffled labels
        y_train = permute_labels(meta, rows["train"], str(cfg["label_permutation"]), seed)
        summary["label_permutation"] = {"kind": str(cfg["label_permutation"]), "n_train_pos_used": int(y_train.sum()),
                                        "n_changed": int((y_train != labels[rows["train"]].astype(int)).sum())}
        if y_train.min() == y_train.max():
            raise ValueError("the permuted training labels hold one class only")
    if cfg.get("train_subsample") is not None:
        summary["train_subsample"] = dict(cfg["train_subsample"])
    train_view = _RowView(feats, rows["train"])
    val = None
    val_groups = None
    val_near_normal = None
    if len(rows["val"]):
        from src.compass.p5_worst_group import groups as p5_groups
        val = (_RowView(feats, rows["val"]), labels[rows["val"]].astype(int))
        val_groups = p5_groups(meta.iloc[rows["val"]])
        if "near_normal_six" in meta.columns:
            val_near_normal = meta["near_normal_six"].to_numpy(dtype=float)[rows["val"]]

    def demo(r):  # raw (age, sex) for rows r; only read when the head uses demographics
        if not head_cfg.demographics:
            return None
        return (meta["age"].to_numpy()[r], meta["sex"].to_numpy()[r])

    rcache = None
    if head_cfg.rhythm:  # hybrid rhythm view: S3.3 descriptors, row order checked against the metadata
        from src.data import rhythm_view
        from src.data.signals import records_sha256
        rcache = rhythm_view.open_cache(_resolve(cfg.get("rhythm_cache", "data/cache/features/classical")),
                                        records_sha256([str(r) for r in meta["record"].to_numpy()]))

    def rhythm(r):  # raw descriptors for rows r; only read when the head uses the rhythm view
        return None if rcache is None else rhythm_view.from_cache(rcache, r)

    t_fit = time.time()
    head, log = fit_head(train_view, y_train, head_cfg, seed, device, val=val,
                         val_groups=val_groups, verbose=verbose, val_near_normal=val_near_normal,
                         demo=demo(rows["train"]), val_demo=demo(rows["val"]) if len(rows["val"]) else None,
                         aux=aux_targets(meta, rows["train"]) if head_cfg.aux_lambda > 0 else None,
                         rhythm=rhythm(rows["train"]),
                         val_rhythm=rhythm(rows["val"]) if len(rows["val"]) else None,
                         **loss_inputs(meta, rows["train"], head_cfg.loss),
                         **sampler_inputs(meta, rows["train"], head_cfg.sampler),
                         **concept_inputs(meta, rows["train"], head_cfg.erase),
                         **tailneg_inputs(meta, rows["train"], head_cfg.neg_weighting, k),
                         **aug_inputs(cfg, meta, rows["train"], head_cfg.aug_views, k, summary))
    t_fit = time.time() - t_fit

    attn_diag = None
    if head_cfg.kind == "attn_pool":  # S4C H6: score and read the attention diagnostics in one pass
        from src.compass import attnpool_diag as AD

        scores, attn_diag = AD.predict_with_attention(head, _RowView(feats, rows["test"]), device)
    else:
        scores = predict_head(head, _RowView(feats, rows["test"]), device, demo=demo(rows["test"]),
                              rhythm=rhythm(rows["test"]))
    scores_by_record = pd.Series(scores.astype(float), index=meta["record"].to_numpy()[rows["test"]])

    ev = cfg["eval"]
    n_bootstrap = int(ev.get("n_bootstrap", 1000))
    ci_perm = int(ev.get("ci_permutations", 16))
    point_perm = int(ev.get("point_permutations", 10**4))
    interior_perm = int(ev.get("interior_point_permutations", 10**3))
    n_jobs = int(ev.get("n_jobs", 1))
    metrics = tuple(ev.get("metrics", report.DEFAULT_METRICS))
    inner_mask = np.zeros(len(meta), dtype=bool)
    inner_mask[rows["pool"]] = True
    # Training-pool scores: the container's top-fraction threshold (written
    # below) and the S1.8 DA_* duration offset, which is estimated on them and
    # never on the reporting fold.
    train_scores = predict_head(head, train_view, device, demo=demo(rows["train"]), rhythm=rhythm(rows["train"]))
    if ev.get("battery", True):
        tokens = cfg["feature_kind"] == "tokens"  # P4 probes the encoder's CLS features: the base run has it
        results = run_battery(run_dir, meta, scores_by_record, None if tokens else feats,
                              None if tokens else inner_mask, seed=seed,
                              n_bootstrap=n_bootstrap, ci_permutations=ci_perm,
                              point_permutations=interior_perm, n_jobs=n_jobs,
                              p3_replicates=int(ev.get("p3_replicates", 5)),
                              p3_prevalence=float(ev.get("p3_prevalence", 0.02)),
                              p4_max_train=int(ev.get("p4_max_train", 20000)),
                              train_scores_by_record=pd.Series(
                                  train_scores.astype(float), index=meta["record"].to_numpy()[rows["train"]]))
    else:
        from src.compass.battery import base_frames, eval_frame
        results = base_frames(eval_frame(meta, scores_by_record))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(text, encoding="utf-8")
    # Container export (scripts/container/export_model.py) needs the fitted
    # head, the training-pool score distribution for the top-fraction
    # threshold, and where the features' encoder lives.
    save_head(run_dir, head, head_cfg.kind, str(cfg["feature_kind"]))
    train_frame = _pred_frame(meta, rows["train"], train_scores)
    if cfg.get("label_permutation") is not None:  # the labels the head was fitted to, beside the true ones
        train_frame["label_used"] = y_train
    report.write_predictions(run_dir, f"train_pool_f{k}", train_frame)
    encoder_info = encoder_provenance(cfg)
    common = dict(model=str(cfg["model"]), objective=str(cfg["objective"]), beta=cfg["beta"],
                  # A sweep that keeps the anchor's (model, objective, beta) must not
                  # pool with it in paper_ready (D-020): its config sets
                  # split_suffix and its rows live on a non-reporting split name.
                  split=f"outer_fold_{k}" + (f"_inner{int(inner_eval)}" if inner_eval is not None else "")
                  + (f"_{cfg['split_suffix']}" if cfg.get("split_suffix") else ""),
                  seed=seed, config_sha=config_sha,
                  provisional=bool(cfg["provisional"]))
    git = R.git_sha()
    # Every row carries the task tag and the label policy in ``notes`` so the
    # digest (scripts/paper/digest.py) can file it, and so the S6.2 sensitivity
    # runs are distinguishable from the primary policy (D-012).
    note_parts = [supersede_note,
                  f"task={cfg['task']}" if cfg.get("task") else None,
                  f"policy={cfg['label_policy']}"]
    run_notes = " ".join(x for x in note_parts if x)
    written = register_results(run_id, run_dir, results, common, metrics=metrics,
                               n_bootstrap=n_bootstrap, ci_permutations=ci_perm,
                               point_permutations=point_perm, interior_point_permutations=interior_perm,
                               n_jobs=n_jobs, registry_path=registry_path, git_sha=git,
                               notes=run_notes)

    if attn_diag is not None:  # B3's reported diagnostics, task S4C.diag
        from src.compass import attnpool_diag as AD

        for sc in AD.summarise(attn_diag, seed=seed, n_bootstrap=n_bootstrap):
            row = report.register_scalar(run_id, eval_set=AD.EVAL_SET, result=sc,
                                         notes=f"task=S4C.diag policy={cfg['label_policy']}",
                                         registry_path=registry_path, git_sha=git, **common)
            if row is not None:
                written.append(row)
    summary.update(git_sha=git, timestamp=R.utc_now(), seconds_total=time.time() - t0, seconds_fit=t_fit,
                   head_log=log, n_registry_rows=len(written),
                   eval_sets=sorted({r["eval_set"] for r in written}),
                   encoder=encoder_info, head_file="head.pt", train_preds=f"preds_train_pool_f{k}.parquet")
    report.write_run_json(run_dir, summary)
    if verbose:
        print(f"[run] {run_id}: {len(written)} registry rows, {summary['seconds_total']:.1f}s")
    return summary


def concept_inputs(meta: pd.DataFrame, rows: np.ndarray, erase: str) -> dict[str, Any]:
    """``fit_head(concept=...)`` for ``head.erase`` (S4C H3): the recording source of each training row."""
    if erase == "none":
        return {}
    if erase == "leace_source":
        return {"concept": meta["source"].to_numpy()[rows].astype(str)}
    raise ValueError(f"no concept defined for erase={erase!r}")


def tailneg_inputs(meta: pd.DataFrame, rows: np.ndarray, neg_weighting: Optional[str], k: int) -> dict[str, Any]:
    """``fit_head(crossfit_groups=..., row_sources=...)`` for ``head.neg_weighting`` (S4C H4 ``tailneg``).

    The stage-1 cross-fit and the tail weights are computed on ``rows`` -- the training rows -- and nothing
    else; the guard re-checks that none of them is in the reporting fold ``k``. Patient keys group the
    cross-fit parts, so a training record is scored by a head that never saw its patient."""
    if neg_weighting is None:
        return {}
    assert_no_leakage(meta["outer_fold"].to_numpy()[rows], k, "tail-weighting cross-fit")
    return {"crossfit_groups": meta["patient_key"].to_numpy()[rows].astype(str),
            "row_sources": meta["source"].to_numpy()[rows].astype(str)}


def aug_inputs(cfg: dict[str, Any], meta: pd.DataFrame, rows: np.ndarray, aug_views: int, k: int,
               summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """``fit_head(aug_feats=...)`` for ``head.aug_views`` (S4C H5 ``aug``): views aligned with the training rows.

    The views come from ``src.data.features_aug`` for the config's encoder (``cfg["model"]``), under
    ``cfg["features_aug_root"]`` (default ``data/cache/features_aug``). Every training row must be covered. The
    cache must have been made by the encoder that made the clean features (same ``sha256``). Only ``rows`` --
    the training rows, re-checked against the reporting fold ``k`` -- are ever mapped, so no other row's view
    can be read. Provenance goes into ``summary["aug"]``."""
    if not aug_views:
        return {}
    from src.data import features_aug as FA

    assert_no_leakage(meta["outer_fold"].to_numpy()[rows], k, "augmented views")
    aug = FA.open_aug(str(cfg["model"]), root=_resolve(cfg.get("features_aug_root", "data/cache/features_aug")))
    if len(aug["views"]) < aug_views:
        raise ValueError(f"{cfg['model']}: {len(aug['views'])} augmented view(s) cached, {aug_views} needed")
    fmeta = feature_meta(cfg["features_root"], cfg["model"])
    if aug["meta"].get("encoder_sha256") != fmeta.get("sha256"):
        raise ValueError(f"{cfg['model']}: augmented views were made by encoder {aug['meta'].get('encoder_sha256')}, "
                         f"the clean features by {fmeta.get('sha256')}")
    rows = np.asarray(rows, dtype=np.int64)
    pos = np.searchsorted(aug["rows"], rows)
    covered = (pos < len(aug["rows"])) & (aug["rows"][np.minimum(pos, len(aug["rows"]) - 1)] == rows)
    if not covered.all():
        raise ValueError(f"{int((~covered).sum())} training rows have no augmented view in {cfg['model']}'s cache")
    if summary is not None:
        summary["aug"] = {"views": int(aug_views), "n_failed_train_rows": int(aug["failed"][pos].sum()),
                          **{key: aug["meta"].get(key) for key in ("version", "block", "train_salt", "aug_prob",
                                                                   "encoder_sha256", "signal_records_sha256")}}
    return {"aug_feats": [_RowView(v, pos) for v in aug["views"][:aug_views]]}


def dro_groups(meta: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """``'<source>|<near_normal|abnormal|unknown>'`` per row: the S5.2 GroupDRO groups."""
    from src.compass.p5_worst_group import abnormality_token

    sub = meta.iloc[rows]
    nn_col = sub["near_normal_six"].to_numpy(dtype=float) if "near_normal_six" in sub.columns \
        else np.full(len(sub), np.nan)
    return np.array([f"{s}|{abnormality_token(f)}" for s, f in zip(sub["source"].to_numpy(), nn_col)])


def loss_inputs(meta: pd.DataFrame, rows: np.ndarray, loss: str) -> dict[str, Any]:
    """Extra per-row inputs a loss needs: GroupDRO groups, or the nnPU unlabelled mask.

    nnPU treats CODE-15% label-0 records as *unlabelled* (CODE-15%'s labels
    are weak, not serology-confirmed like SaMi-Trop's positives); PTB-XL
    label-0 records stay known negatives.
    """
    if loss == "group_dro":
        return {"train_groups": dro_groups(meta, rows)}
    if loss == "nnpu":
        sub = meta.iloc[rows]
        return {"unlabelled": ((sub["source"] == "CODE-15%") & (sub["label"] == 0)).to_numpy()}
    return {}


def sampler_inputs(meta: pd.DataFrame, rows: np.ndarray, sampler: Optional[str]) -> dict[str, Any]:
    """Extra per-row inputs a head sampler needs (``HeadConfig.sampler``); ``{}`` for the anchor's.

    ``age_matched`` (S4C wave C, W4): the sex x age-band cell of every training row
    (:func:`src.compass.matched_frames.age_cells`).

    ``duration_stratified`` (S4X wave 3, W3-3): the record-duration stratum of
    every training row, ``duration_s >= 8`` (1) or ``< 8`` (0, the zero-padded
    7.3 s format) -- the S1.8 split (``src.compass.duration.LONG_SECONDS``).
    Training rows only; a missing duration is an error, not a stratum.
    """
    if sampler is None:
        return {}
    if sampler == "age_matched":  # S4C wave C, W4: the training rows' sex x age-band cells
        from src.compass.matched_frames import MALE, age_cells

        sub = meta.iloc[rows]
        age = sub["age"].to_numpy(dtype=np.float64)
        ok = np.isfinite(age) & sub["sex"].isin(["Male", "Female"]).to_numpy()
        if not ok.all():
            raise ValueError(f"{int((~ok).sum())} training rows lack age or sex; cannot match")
        return {"strata": age_cells(age, (sub["sex"].to_numpy() == MALE).astype(np.int64))}
    if sampler != "duration_stratified":
        raise ValueError(f"no per-row inputs defined for sampler {sampler!r}")
    from src.compass.duration import LONG_SECONDS

    d = meta["duration_s"].to_numpy(dtype=np.float64)[rows]
    if not np.isfinite(d).all():
        raise ValueError(f"{int((~np.isfinite(d)).sum())} training rows lack duration_s; cannot stratify")
    return {"strata": (d >= LONG_SECONDS).astype(np.int64)}


def aux_targets(meta: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """S4.3 auxiliary targets ``[len(rows), 8]`` for the training rows: the six
    conduction labels (NaN for SaMi-Trop, masked), clipped age and sex
    (:func:`src.models.heads.aux_target_matrix`). Training rows only -- the
    reporting fold's labels are never read here.
    """
    from src.models.heads import CONDUCTION_LABELS, aux_target_matrix

    sub = meta.iloc[rows]
    six = np.column_stack([sub[c].to_numpy(dtype=float) if c in sub.columns else np.full(len(sub), np.nan)
                           for c in CONDUCTION_LABELS])
    return aux_target_matrix(six, sub["age"].to_numpy(), sub["sex"].to_numpy())


def save_head(run_dir: Path, head, kind: str, feature: str) -> Path:
    """``head.pt`` = ``{"state_dict", "kind", "feature", "n_leads", "emb_dim", "spec"}`` on CPU.

    The container loads it strictly -- into ``EnsembleHead`` for the anchor's
    kinds, into ``build_head(**spec)`` otherwise; ``kind``, ``feature`` and the
    constructor ``spec`` travel with the weights so the manifest cannot drift
    from the fit (ARCHITECTURE.md, Container).
    """
    from src.models.heads import head_spec

    if hasattr(head, "chagas_head"):  # S4.3: only the Chagas path is ever exported
        head = head.chagas_head()
    path = Path(run_dir) / "head.pt"
    state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    torch.save({"state_dict": state, "kind": str(kind), "feature": str(feature),
                "n_leads": int(getattr(head, "n_leads", 12)), "emb_dim": int(getattr(head, "emb_dim", 0)),
                "spec": head_spec(head)}, path)
    return path


def encoder_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Which encoder produced the cached features this head was fitted on.

    A head run never holds the encoder itself; ``export_model`` resolves it
    from here. ``path`` is ``results/runs/<encoder_id>/encoder.pt`` when the
    encoder came from a pretraining run of this branch (features.py keys the
    cache by that run_id), ``None`` for the shipped anchor (``good_disco``).
    """
    encoder_id = str(cfg["model"])
    info: dict[str, Any] = {"id": encoder_id, "feature_kind": str(cfg["feature_kind"]), "sha256": None, "path": None}
    meta_path = _resolve(cfg["features_root"]) / encoder_id / "meta.json"
    if meta_path.is_file():
        try:
            info["sha256"] = json.loads(meta_path.read_text()).get("sha256")
        except json.JSONDecodeError:
            pass
    candidate = _resolve(cfg["results_root"]) / encoder_id / "encoder.pt"
    if candidate.is_file():
        info["path"] = str(candidate)
    return info


def _pred_frame(meta: pd.DataFrame, rows: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """A ``report.PRED_COLUMNS`` frame for ``rows`` of ``meta`` with ``scores``."""
    sub = meta.iloc[rows]
    return pd.DataFrame({"record": sub["record"].to_numpy(), "patient_key": sub["patient_key"].to_numpy(),
                         "source": sub["source"].to_numpy(), "label": sub["label"].to_numpy(),
                         "score": np.asarray(scores, dtype=float)})


#: Eval sets whose point estimate keeps the official permutation count.
OFFICIAL_POINT_PREFIXES = ("outer_fold_all", "cohort_")

#: Scalar eval sets filed under their own task whichever run produced them.
#: The S1.8 duration-aware sets (src/compass/duration.py) are computed inside
#: every run's battery but form one family across runs, so their rows' notes
#: lead with ``task=S1.8`` and keep the run's own tag as ``source_task=``.
TASK_OF_PREFIX = {"L2_": "S1.8", "DA_": "S1.8"}


def retag_notes(notes: Optional[str], task: str) -> str:
    """Put ``task=<task>`` first in ``notes``; a different ``task=`` token already there becomes ``source_task=``."""
    notes = notes or ""
    if re.search(rf"(?<!\w)task={re.escape(task)}(?![\w.])", notes):
        return notes
    body = re.sub(r"(?<!\w)task=", "source_task=", notes)
    return f"task={task} {body}".strip()


def notes_for(eval_set: str, notes: Optional[str]) -> Optional[str]:
    """``notes`` as filed for ``eval_set`` (see :data:`TASK_OF_PREFIX`)."""
    for prefix, task in TASK_OF_PREFIX.items():
        if eval_set.startswith(prefix):
            return retag_notes(notes, task)
    return notes


def register_results(run_id, run_dir, results: dict[str, Any], common: dict[str, Any], *,
                     metrics, n_bootstrap, ci_permutations, registry_path, git_sha,
                     point_permutations: int = 10**4, interior_point_permutations: int = 10**3,
                     n_jobs: int = 1, notes: Optional[str] = None) -> list[dict]:
    """Frames -> ``report.register``; scalars -> ``report.register_scalar``.

    Replicate frames (``*_r<i>``) are written to disk but not registered: the
    registry carries their mean (``P3_*``), and a per-replicate row would be
    counted by ``paper_ready`` as if it were a seed. ``outer_fold_all`` and
    the ``cohort_*`` sets are scored at ``point_permutations``; the interior
    proxy frames at ``interior_point_permutations`` (the counts are in every
    row's ``notes``). ``notes`` is prepended to every row (the supersede note);
    scalar sets named in :data:`TASK_OF_PREFIX` get it retagged
    (:func:`notes_for`).
    """
    frames = {k: v for k, v in results.items() if isinstance(v, pd.DataFrame)}
    replicate = {k: v for k, v in frames.items() if re.search(r"_r\d+$", k)}
    for k, v in replicate.items():
        report.write_predictions(run_dir, k, v)
        frames.pop(k)
    official = {k: v for k, v in frames.items() if k.startswith(OFFICIAL_POINT_PREFIXES)}
    interior = {k: v for k, v in frames.items() if k not in official}
    written = []
    for group, perm in ((official, point_permutations), (interior, interior_point_permutations)):
        if group:
            written += report.register(run_id, run_dir, eval_frames=group, metrics=metrics,
                                       n_bootstrap=n_bootstrap, ci_permutations=ci_permutations,
                                       point_permutations=perm, n_jobs=n_jobs, notes=notes,
                                       registry_path=registry_path, git_sha=git_sha, **common)
    for key, v in results.items():
        if isinstance(v, pd.DataFrame):
            continue
        eval_set = key.split("__")[0]
        for sc in (v if isinstance(v, (list, tuple)) else [v]):
            row = report.register_scalar(run_id, eval_set=eval_set, result=sc, notes=notes_for(eval_set, notes),
                                         registry_path=registry_path, git_sha=git_sha, **common)
            if row is not None:
                written.append(row)
    return written


class _RowView:
    """A row subset of the feature memmap that indexes lazily.

    ``fit_head``/``predict_head`` only ever ask for ``shape`` and integer or
    slice indexing, so this is all they need to train on an arbitrary row set
    without materialising it.
    """

    def __init__(self, feats, rows: np.ndarray) -> None:
        self._feats = feats
        self._rows = np.asarray(rows, dtype=np.int64)
        self.shape = (len(self._rows), *feats.shape[1:])

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx):
        return np.asarray(self._feats[self._rows[idx]], dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outer-fold", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="supersede an existing run_id (its old registry rows stay; new rows are noted)")
    a = ap.parse_args(argv)
    run(a.config, a.seed, a.outer_fold, a.device, a.dry_run, verbose=not a.quiet, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
