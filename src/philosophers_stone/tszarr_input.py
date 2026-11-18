"""BDSP primary TS-Zarr stores as Philosopher's Stone input.

BDSP recordings are published as split TS-Zarr stores. A primary
(``<subject>_<session>_task-psg_eeg.zarr``) holds calibrated signals at their
native rates. Philosopher's Stone reads one EEG channel from it, in
microvolts, and prepares it exactly as :func:`infer_brain_health` prepares an
array: resampled to ``cfg.resample_hz`` and truncated to ``cfg.hours_pad``
hours. Prepared H5 and EDF inputs keep working unchanged.

The channel is the configured one (``c4-m1`` by default), or else the first
of the BDSP brain-health priority list. The EEG must be stored in ``uV``: a
store whose EEG unit is undeclared or not a voltage is refused rather than
rescaled by guesswork.

Reading TS-Zarr needs the shared BDSP reader, an optional dependency:
``pip install 'philosophers-stone[zarr]'`` (Python >= 3.11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# The BDSP brain-health input priority (bdsp-sleep-features EEG_CANDIDATES).
EEG_CANDIDATES = ("c4-m1", "c3-m2", "f3-m2", "f4-m1", "o1-m2", "o2-m1", "eeg")


def is_tszarr_store(path: str | Path) -> bool:
    """Whether ``path`` is a local Zarr v3 group tagged TS-Zarr 1.0."""

    root = Path(path)
    if not root.is_dir():
        return False
    try:
        metadata = json.loads((root / "zarr.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    attributes = metadata.get("attributes") or {}
    return (
        metadata.get("zarr_format") == 3
        and metadata.get("node_type") == "group"
        and (attributes.get("tsz") or {}).get("version") == "1.0"
    )


def load_primary_eeg(
    path: str | Path, channel: str = "c4-m1"
) -> tuple[np.ndarray, float, str]:
    """Return ``(eeg_uv, fs_hz, channel)`` from a primary store at its native rate."""

    try:
        from bdsp.bids_zarr import open_primary
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TS-Zarr input needs the BDSP reader: "
            "pip install 'philosophers-stone[zarr]' (Python >= 3.11)"
        ) from exc
    root = Path(path)
    if not is_tszarr_store(root):
        raise ValueError(f"Not a primary TS-Zarr store: {path}")
    with open_primary(root) as record:
        names = {name.lower(): name for name in record.signals}
        wanted = [channel.lower()] + [
            name for name in EEG_CANDIDATES if name != channel.lower()
        ]
        chosen = next((names[name] for name in wanted if name in names), None)
        if chosen is None:
            raise ValueError(
                f"No EEG channel among {', '.join(wanted)} in {root.name}"
            )
        signal = record.signals[chosen]
        unit = str(signal.metadata.get("unit") or "")
        if unit != "uV":
            raise ValueError(
                f"EEG {chosen} is stored in {unit or 'an undeclared unit'}, not uV; "
                "its scale is not guessed"
            )
        if signal.offset_s != 0:
            raise ValueError(f"EEG {chosen} does not start at recording zero")
        samples = np.asarray(signal.samples[:], dtype=np.float64).reshape(-1)
        return samples, float(signal.rate_hz), chosen


def load_tszarr_signals(path: str | Path, cfg: Any) -> tuple[Any, float]:
    """The ``(signals, fs)`` pair ``_load_eeg`` returns, from a primary store."""

    import pandas as pd

    from .philosopher_utils import _resample_1d

    eeg_uv, fs_hz, _ = load_primary_eeg(path, cfg.channel)
    if eeg_uv.size == 0:
        raise ValueError("Input EEG is empty.")
    if not np.isfinite(eeg_uv).all():
        raise ValueError("Input EEG contains NaN or infinite values.")
    eeg = _resample_1d(eeg_uv, fs_hz, cfg.resample_hz)
    max_len = int(cfg.hours_pad * 3600 * cfg.resample_hz)
    if len(eeg) > max_len:
        eeg = eeg[:max_len]
    return pd.DataFrame({cfg.channel: eeg.astype(float)}), float(cfg.resample_hz)


__all__ = [
    "EEG_CANDIDATES",
    "is_tszarr_store",
    "load_primary_eeg",
    "load_tszarr_signals",
]
