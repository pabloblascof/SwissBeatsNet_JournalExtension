"""Raw rhythm/interval descriptors for the hybrid rhythm view (exploratory head option).

The head option :class:`src.models.heads.WithRhythm` concatenates the ten
:data:`src.models.heads.RHYTHM_VIEW_NAMES` descriptors of a record into the
final layer. They are columns of the S3.3 classical feature cache
(``src/baselines/features_classical.py``), computed on the anchor's
preprocessed 250 Hz signal: lead II heart rate and RR variability, and the dwt
width proxies (not clinical durations; see that module's docstring).

Two producers, one extractor:

* training / evaluation read the cached columns (:func:`from_cache`), after
  checking that the cache was built on the metadata's record order;
* the container recomputes them from the preprocessed record
  (:func:`from_signals`) with :func:`src.baselines.features_classical.extract`
  -- the function that built the cache -- so a record the container
  preprocesses identically gets identical descriptors.

Missing values are NaN (failed rows, fewer than three beats, a failed
delineation); the head imputes them with the training-pool median and a
missingness flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.models.heads import RHYTHM_VIEW_NAMES

FS = 250


def _indices(names: list[str]) -> list[int]:
    missing = [n for n in RHYTHM_VIEW_NAMES if n not in names]
    if missing:
        raise ValueError(f"classical feature layout lacks {missing}")
    return [names.index(n) for n in RHYTHM_VIEW_NAMES]


def open_cache(cache_dir: Optional[Path | str] = None, records_sha256: Optional[str] = None) -> dict[str, Any]:
    """The classical cache (``open_classical_features``) with the row-order check applied.

    ``records_sha256``: the metadata's ``records_sha256`` (``src.data.signals``);
    a mismatch with the cache's ``signal_records_sha256`` raises, because the
    rows would then belong to other records.
    """
    from src.baselines.features_classical import FEATURES_DIR, open_classical_features

    cache = open_classical_features(cache_dir or FEATURES_DIR)
    want = cache["meta"].get("signal_records_sha256")
    if records_sha256 is not None and want is not None and want != records_sha256:
        raise ValueError(f"classical cache was built on record order {want[:12]}..., metadata is "
                         f"{records_sha256[:12]}...: rebuild the cache or the metadata")
    cache["rhythm_idx"] = _indices(list(cache["names"]))
    return cache


def from_cache(cache: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """Raw descriptors ``[len(rows), 10]`` float64 for metadata rows ``rows``; failed rows are NaN."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), len(RHYTHM_VIEW_NAMES)), dtype=np.float64)
    block = np.asarray(cache["feats"][rows[order]][:, cache["rhythm_idx"]], dtype=np.float64)
    out[order] = block
    out[~cache["valid"][rows]] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def from_signals(signals) -> np.ndarray:
    """Raw descriptors ``[B, 10]`` from preprocessed signals ``[B, 2250, 12]`` (the anchor's 250 Hz view).

    Runs :func:`src.baselines.features_classical.extract` per record, as the
    cache builder did; a record the extractor rejects (non-finite, no
    support) gets an all-NaN row, which the head imputes.
    """
    from src.baselines.features_classical import extract, feature_names

    idx = _indices(feature_names(delineate=True))
    x = signals.detach().cpu().numpy() if hasattr(signals, "detach") else np.asarray(signals)
    out = np.full((len(x), len(RHYTHM_VIEW_NAMES)), np.nan, dtype=np.float64)
    for i, rec in enumerate(x):
        try:
            out[i] = np.asarray(extract(np.asarray(rec, dtype=np.float32), fs=FS, delineate=True),
                                dtype=np.float64)[idx]
        except Exception:  # noqa: BLE001 - an unusable signal is a missing descriptor row
            pass
    out[~np.isfinite(out)] = np.nan
    return out


__all__ = ["from_cache", "from_signals", "open_cache"]
