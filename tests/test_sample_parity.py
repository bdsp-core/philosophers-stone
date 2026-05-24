from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from philosophers_stone import Config


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "fixtures" / "phi_results_baseline.csv"


@pytest.mark.skipif(os.getenv("PHILOSOPHER_RUN_SLOW") != "1", reason="slow sample parity test is opt-in")
def test_sample_csv_numeric_parity(tmp_path: Path) -> None:
    checkpoint = Path(Config().model_file)
    if not checkpoint.exists():
        pytest.skip(f"local checkpoint is not available: {checkpoint}")

    outdir = tmp_path / "new-output"
    command = [shutil.which("philosophers-stone") or sys.executable]
    if command[0] == sys.executable:
        command.extend(["-m", "philosophers_stone.cli"])

    subprocess.run(
        command
        + [
            "--manifest_csv",
            "phi_manifest.csv",
            "--outdir",
            str(outdir),
            "--no-save-plots",
            "--no-save-json",
        ],
        cwd=ROOT,
        check=True,
    )

    baseline = pd.read_csv(BASELINE)
    new = pd.read_csv(outdir / "phi_results.csv")

    assert list(new.columns) == list(baseline.columns)
    assert new["file_id"].tolist() == baseline["file_id"].tolist()

    exact_cols = [
        "file_id",
        "age",
        "sex",
        "brain_health_score",
        "total_cognition_score",
        "fluid_cognition_score",
        "crystallized_cognition_score",
    ]
    pd.testing.assert_frame_equal(new[exact_cols], baseline[exact_cols], check_dtype=False)

    latent_cols = [col for col in baseline.columns if col.startswith("lhl_")]
    assert np.allclose(
        new[latent_cols].to_numpy(),
        baseline[latent_cols].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
