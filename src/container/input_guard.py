"""Input guard: records the training corpora never showed, made scoreable (S4C_IMPROVE.md A2 item 6).

Why. The hidden cohorts were never released. ``scripts/container/probe_robustness.py``
showed three ways a record outside the training formats is lost or misread by
the anchor's loader, which takes the file's channels in file order:

* any non-finite sample (a NaN lead, or a 100 ms NaN burst in one lead) makes
  preprocessing fail or return NaN, and the record is scored ``(0, 0.0)``;
* more than 12 channels fails the shape check, again ``(0, 0.0)``;
* the 12 standard leads in another order are read as if they were in the
  standard order: the logit moved by 1–2 on the probed records.

What it does. :func:`guard` returns ``None`` for a record whose channels are
the 12 standard leads in the standard order (names compared after
:func:`normalise_name`; a record whose names are not recognised but which has
exactly 12 channels also counts, as the anchor assumes) and whose samples are
all finite. That is every training record, and those records keep the
anchor's code path untouched. Otherwise it returns ``[T, 12]`` in the standard
order and notes saying what changed:

1. channels: the 12 standard leads selected and ordered by name; or, when
   only the eight independent leads (I, II, V1–V6) are named, III, aVR, aVL
   and aVF derived from I and II (Einthoven and Goldberger); or, with names
   not recognised, the first 12 channels in file order; fewer than 12
   unrecognised channels are padded with flat leads (a flat lead scores
   finitely, a failed record does not score at all);
2. samples: non-finite samples are linearly interpolated within their lead
   from its finite neighbours (held flat beyond the first and last finite
   sample); a lead without any finite sample becomes flat.

Nothing here learns or depends on a label, and nothing reads a cohort
identity: it is input handling in the sense of amendment A2's scope.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np

STANDARD_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
#: European notation and other spellings of the limb leads.
ALIASES = {"DI": "I", "DII": "II", "DIII": "III", "D1": "I", "D2": "II", "D3": "III"}


def normalise_name(name: object) -> str:
    """Upper case, without a ``lead`` prefix, spaces, ``-``, ``_`` or ``.``; aliases mapped (``aVR`` -> ``AVR``)."""
    s = re.sub(r"[\s_\-.]", "", str(name).upper())
    s = re.sub(r"^LEAD", "", s)
    return ALIASES.get(s, s)


def _interpolate(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.array(x, dtype=np.float64, copy=True)
    notes = []
    for j in range(x.shape[1]):
        col = x[:, j]
        bad = ~np.isfinite(col)
        if not bad.any():
            continue
        good = np.flatnonzero(~bad)
        if good.size == 0:
            x[:, j] = 0.0
            notes.append(f"lead {j}: no finite sample, set flat")
            continue
        x[bad, j] = np.interp(np.flatnonzero(bad), good, col[good])
        notes.append(f"lead {j}: {int(bad.sum())} non-finite sample(s) interpolated")
    return x, notes


def channels(x: np.ndarray, names: Sequence[object]) -> tuple[np.ndarray, list[str]]:
    """``[T, 12]`` in the standard order and notes (empty when ``x`` already is that, by name or by count)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    norm = [normalise_name(n) for n in names] if names is not None and len(names) == x.shape[1] else []
    index = {n: i for i, n in enumerate(norm)} if len(set(norm)) == len(norm) else {}
    if tuple(norm) == STANDARD_LEADS:
        return x, []
    if all(lead in index for lead in STANDARD_LEADS):
        return x[:, [index[lead] for lead in STANDARD_LEADS]], [f"channels {list(names)} selected/reordered by name"]
    independent = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    if all(lead in index for lead in independent):
        i, ii = x[:, index["I"]], x[:, index["II"]]
        derived = {"I": i, "II": ii, "III": ii - i, "AVR": -(i + ii) / 2.0, "AVL": i - ii / 2.0, "AVF": ii - i / 2.0}
        out = np.stack([derived[lead] if lead in derived else x[:, index[lead]] for lead in STANDARD_LEADS], axis=1)
        return out, [f"channels {list(names)}: III, aVR, aVL, aVF derived from I and II"]
    if x.shape[1] == len(STANDARD_LEADS):
        return x, []  # unrecognised names, twelve channels: the anchor's assumption, unchanged
    if x.shape[1] > len(STANDARD_LEADS):
        return x[:, : len(STANDARD_LEADS)], [f"{x.shape[1]} channels {list(names)}: first 12 taken in file order"]
    pad = np.zeros((x.shape[0], len(STANDARD_LEADS) - x.shape[1]))
    return np.concatenate([x, pad], axis=1), [f"{x.shape[1]} channel(s) {list(names)}: padded with flat leads"]


def guard(x: np.ndarray, names: Sequence[object]) -> Optional[tuple[np.ndarray, list[str]]]:
    """``None`` when the record needs nothing (module docstring), else ``([T, 12] float64, notes)``."""
    out, notes = channels(x, names)
    if not np.isfinite(out).all():
        out, more = _interpolate(out)
        notes += more
    return (out, notes) if notes else None


__all__ = ["ALIASES", "STANDARD_LEADS", "channels", "guard", "normalise_name"]
