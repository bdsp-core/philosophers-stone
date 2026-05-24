from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from philosophers_stone import Config, checkpoint_available, infer_brain_health
from philosophers_stone.cli import _read_manifest, parse_args
from philosophers_stone.philosopher_utils import _result_row_from_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_csv_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,age,sex\n/tmp/example.h5,65,1\n", encoding="utf-8")

    df = _read_manifest(manifest)

    assert list(df.columns) == ["filepath", "age", "sex"]
    assert df.loc[0, "filepath"] == "/tmp/example.h5"
    assert df.loc[0, "age"] == 65.0
    assert df.loc[0, "sex"] == 1


def test_manifest_requires_public_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,age\n/tmp/example.h5,65\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        _read_manifest(manifest)


def test_checkpoint_resolution_uses_env_without_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_checkpoint = tmp_path / "missing.ckpt"
    monkeypatch.setenv("PHILOSOPHER_MODEL_FILE", str(missing_checkpoint))

    cfg = Config()
    available, message = checkpoint_available(cfg)

    assert cfg.model_file == str(missing_checkpoint)
    assert available is False
    assert str(missing_checkpoint) in message


def test_result_row_schema_rounding_and_latent_order() -> None:
    pred_df = pd.DataFrame(
        {"y_pred": [0.1234567, 0.9876543, -0.333335, 1.234565, 0.222226]},
        index=["brain_health_score", "cog_total", "cog_fluid", "cog_crystallized", "dx-test"],
    )
    latent = np.array([[0.1, -0.2, 0.333333333]], dtype=np.float32)

    row = _result_row_from_predictions(
        file_id="sample",
        age=70,
        sex=1,
        filepath="/tmp/sample.h5",
        pred_df=pred_df,
        bhs=0.1234567,
        latent=latent,
        collect_head_outputs=True,
    )

    assert list(row) == [
        "file_id",
        "filepath",
        "age",
        "sex",
        "brain_health_score",
        "total_cognition_score",
        "fluid_cognition_score",
        "crystallized_cognition_score",
        "head_dx-test",
        "lhl_1",
        "lhl_2",
        "lhl_3",
    ]
    assert row["brain_health_score"] == 0.12346
    assert row["total_cognition_score"] == 0.98765
    assert row["fluid_cognition_score"] == -0.33334
    assert row["crystallized_cognition_score"] == 1.23456
    assert row["head_dx-test"] == 0.22223
    assert row["lhl_1"] == pytest.approx(0.1)
    assert row["lhl_2"] == pytest.approx(-0.2)
    assert row["lhl_3"] == pytest.approx(0.333333333)


def test_cli_defaults() -> None:
    args = parse_args([])

    assert args.manifest_csv == "phi_manifest.csv"
    assert args.outdir == "phi_out"
    assert args.save_plots is False
    assert args.save_json is False
    assert args.save_summary is True
    assert args.device_id is None


def test_new_and_legacy_api_imports_match() -> None:
    from phi_utils.philosopher_utils import Config as LegacyConfig
    from phi_utils.philosopher_utils import infer_brain_health as legacy_infer_brain_health

    assert LegacyConfig is Config
    assert legacy_infer_brain_health is infer_brain_health


def test_legacy_cli_help_from_source_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, "philosopher.py", "--help"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "usage: philosopher.py" in completed.stdout
    assert "--manifest_csv" in completed.stdout


def test_vendored_timm_model_imports() -> None:
    from philosophers_stone.model_config import SleepPhilosopherSpectral
    from philosophers_stone._vendor.timm.models.maxxvit import MaxxVitStage

    assert SleepPhilosopherSpectral.__name__ == "SleepPhilosopherSpectral"
    assert MaxxVitStage.__name__ == "MaxxVitStage"
