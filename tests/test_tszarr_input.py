"""BDSP primary TS-Zarr input (optional; needs the bdsp reader, Python >= 3.11)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sub-I0002123_ses-1_task-psg_eeg.zarr"
MODULE = Path(__file__).parents[1] / "src" / "philosophers_stone" / "tszarr_input.py"


def _module():
    # Load the loader alone, so this test does not need the model stack.
    spec = importlib.util.spec_from_file_location("tszarr_input_alone", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_primary_store_is_recognized_and_h5_is_not(tmp_path: Path) -> None:
    module = _module()
    assert module.is_tszarr_store(FIXTURE)
    (tmp_path / "recording.h5").write_bytes(b"")
    assert not module.is_tszarr_store(tmp_path / "recording.h5")


def test_primary_eeg_is_read_in_microvolts_at_its_native_rate() -> None:
    pytest.importorskip("bdsp.bids_zarr")
    eeg, fs, channel = _module().load_primary_eeg(FIXTURE)
    assert channel == "c4-m1" and fs == 100.0
    assert eeg.shape == (60000,) and np.isfinite(eeg).all()
    # The synthetic fixture stores a 20 uV, 10 Hz sine.
    assert 19.0 < np.abs(eeg).max() <= 20.0


def test_primary_signals_match_the_array_api_preparation() -> None:
    pytest.importorskip("bdsp.bids_zarr")
    pytest.importorskip("torch")
    from philosophers_stone.philosopher_utils import Config, _resample_1d
    from philosophers_stone.tszarr_input import load_primary_eeg, load_tszarr_signals

    cfg = Config()
    signals, fs = load_tszarr_signals(FIXTURE, cfg)
    eeg, native_fs, _ = load_primary_eeg(FIXTURE, cfg.channel)
    expected = _resample_1d(eeg, native_fs, cfg.resample_hz)
    assert fs == cfg.resample_hz
    assert np.array_equal(signals[cfg.channel].to_numpy(), expected.astype(float))
