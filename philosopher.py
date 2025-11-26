#!/usr/bin/env python
"""
Batch inference for Sleep‑Philosophers‑Stone
------------------------------------------

* Reads a manifest (CSV or DataFrame) with columns: filepath, age, sex
* Keeps the model in memory
* Streams files via a torch Dataset/DataLoader (batch_size = 1)
* Saves one JSON per EEG file plus a summary CSV (optional)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import gc
import psutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")     # headless; avoids GUI backends keeping refs
plt.ioff()

from phi_utils.philosopher_utils import (
    Config as PhilosopherConfig,
    _apply_head_weights,
    _compute_wavelet_specs,
    _load_eeg,
    load_model,
    plot_spectrogram,
    plot_spectrogram_with_stages,
    preprocess_filter,
)

ManifestLike = Union[str, Path, pd.DataFrame]
REQUIRED_COLUMNS = ["filepath", "age", "sex"]


# ---------- Dataset ---------- #
class SleepEEGDataset(Dataset):
    """Loads *one* file per sample and returns (specs, age_z, sex, file_id, age, filepath)."""

    def __init__(self, manifest_df: pd.DataFrame, cfg: PhilosopherConfig):
        self.cfg = cfg
        self.df = _validate_manifest_df(manifest_df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path_file = str(row["filepath"])
        age = float(row["age"])
        sex = int(row["sex"])
        file_id = os.path.basename(path_file)

        # 1. Load + preprocess EEG
        signals, fs_eeg = _load_eeg(path_file, self.cfg)
        signals = preprocess_filter(signals, bandpass_high=self.cfg.f_high)

        # 2. Spectrogram
        specs = _compute_wavelet_specs(signals, fs_eeg, self.cfg)

        # 3. z-scaled age
        age_z = (age - self.cfg.age_mean_tr_data) / self.cfg.age_std_tr_data

        return specs, age_z, sex, file_id, age, path_file


# ---------- Inference helper ---------- #
@torch.no_grad()
def infer_one(model, specs, age_z, sex, cfg: PhilosopherConfig):
    device = torch.device(cfg.device)
    x = torch.tensor(specs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cov = torch.tensor([[age_z, sex]], dtype=torch.float32).to(device)

    yp_reg, yp_clf, yp_stage, latent = model(x, cov, return_features_lhl=True)
    # yp_reg: outputs for regression tasks, e.g. cognitive scores
    # yp_clf: outputs for classification tasks, e.g. disease probabilities
    # yp_stage: outputs for stage classification (30s epochs)
    # latent: brain health latent representation (LHL, 1024-dim) for the EEG signal
    # Note: Currently, the model is NOT EXPECTED TO BE IN EVAL MODE. 
    # Aggregated stats over batchnorm layers need to be updated per sample! NO EVAL MODE!

    del x, cov
    
    return (
        latent.cpu().numpy(),
        yp_reg.cpu().numpy(),
        yp_clf.cpu().numpy(),
        yp_stage.cpu().numpy(),
    )


# ---------- Manifest helpers ---------- #
def _read_manifest(manifest: ManifestLike) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        df = manifest.copy()
        return _validate_manifest_df(df)

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if path.suffix.lower() != ".csv" and not path.exists():
            candidate = Path(f"{path}.csv")
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"CSV manifest input file not found: {path}")
        df = pd.read_csv(path)
        return _validate_manifest_df(df)

    raise TypeError("Manifest must be a path to CSV or a pandas DataFrame.")


def _validate_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest must contain the columns: {REQUIRED_COLUMNS} (missing {missing}).")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Manifest must not contain empty values in required columns.")

    df = df.copy()
    df["filepath"] = df["filepath"].map(str)
    df["age"] = df["age"].astype(float)
    df["sex"] = df["sex"].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df


def _ensure_dir(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# ---------- Public API ---------- #
def run_philosopher(
    manifest: ManifestLike,
    *,
    outdir: Optional[Union[str, Path]] = "phi_out",
    num_workers: int = 0,
    device_id: Optional[int] = None,
    cfg: Optional[PhilosopherConfig] = None,
    model: Optional[torch.nn.Module] = None,
    save_summary: bool = True,
    save_json: bool = False,
    save_plots: bool = True,
    collect_head_outputs: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    progress_desc: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Run Philosopher Stone inference for the files listed in *manifest*.

    Parameters
    ----------
    manifest:
        Path to a CSV file or a ready DataFrame with columns filepath, age, sex.
    outdir:
        Base output directory. When any of ``save_summary``, ``save_json`` or
        ``save_plots`` is ``True`` this directory must be provided. The summary CSV
        is written here, JSON files under ``json/`` and figures under ``figures/``.
    num_workers:
        DataLoader workers (0 = main process).
    device_id:
        Optional CUDA device index to use when instantiating the configuration.
        Ignored if ``cfg`` is provided.
    cfg:
        Optional PhilosopherConfig override.
    model:
        Optional pre-loaded PyTorch model (allows amortising load cost across calls).
    save_summary:
        Persist the summary CSV if True.
    save_json:
        Write per-file JSON outputs if True (stored beneath ``<outdir>/json``).
    save_plots:
        Write spectrogram figures if True (stored beneath ``<outdir>/figures``).
    collect_head_outputs:
        Include flattened head predictions in the returned DataFrame when True.
    show_progress:
        Render tqdm progress bar when True.
    verbose:
        Print status messages.
    progress_desc:
        Custom progress bar description.

    Returns
    -------
    summary_df:
        DataFrame containing brain-health scores, latent embeddings, and
        optionally head predictions.
    summary_path:
        Path to the written CSV or ``None`` if no file was written.
    """

    if cfg is None:
        gpu_id = device_id if device_id is not None else 0
        cfg = PhilosopherConfig(gpu_id=gpu_id)
    elif device_id is not None:
        cfg.gpu_id = device_id
        cfg.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {cfg.device}")

    manifest_df = _read_manifest(manifest)
    dataset = SleepEEGDataset(manifest_df, cfg)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    model = model or load_model(cfg)

    needs_output_dir = any([save_summary, save_json, save_plots])
    summary_dir: Optional[Path] = None
    if needs_output_dir:
        if outdir is None:
            raise ValueError("outdir must be provided when saving summaries, plots, or JSON outputs.")
        summary_dir = _ensure_dir(outdir)
    elif outdir is not None:
        summary_dir = Path(outdir).expanduser().resolve()

    figure_dir: Optional[Path] = None
    if save_plots and summary_dir is not None:
        figure_dir = _ensure_dir(summary_dir / "figures")

    json_dir: Optional[Path] = None
    if save_json and summary_dir is not None:
        json_dir = _ensure_dir(summary_dir / "json")

    t0 = time.time()
    tqdm_desc = progress_desc
    if show_progress and tqdm_desc is None:
        import random
        _tqdm = [
            'QnJld2luZyBub2N0dXJuYWwgbmV1cm8tZWxpeGlycw==',
        'VGVtcGVyaW5nIGRlbHRhIHdhdmVzIGF0IGZvcmdlIHRlbXBlcmF0dXJl',
        'U21lbHRpbmcgRUVHIGludG8gY29nbml0aXZlIGdvbGQ=',
        'RGlzdGlsbGluZyBtaW5kcyB0byBlbWJlZGRpbmdz',
        'RXh0cmFjdGluZyBuZXVyYWwgZXNzZW5jZQ==',
        'Qm9pbGluZyBkcmVhbXMgaW50byB2ZWN0b3Jz',
        'Q3JhZnRpbmcgUGhpbG9zb3BoZXIncyBTdG9uZXM=',
        'Q29va2luZyBicmFpbiBoZWFsdGggc291cA==',
        ]
        tqdm_desc = base64.b64decode(random.choice(_tqdm)).decode("utf-8")

    iterator: Iterable = tqdm(dataloader, desc=f"\033[38;5;220m{tqdm_desc}") if show_progress else dataloader
    results = []

    for batch in iterator:
        specs, age_z, sex, file_id, age_raw, filepath = batch
        # batch_size == 1 ⇒ unwrap
        specs = specs[0].numpy() if hasattr(specs, "numpy") else specs[0]
        age_z_val = float(age_z[0]) if hasattr(age_z, "__len__") else float(age_z)
        sex_val = int(sex[0]) if hasattr(sex, "__len__") else int(sex)
        filename = file_id[0]
        file_id_val = os.path.splitext(filename)[0]
        filepath_val = filepath[0]
        age_val = float(age_raw[0]) if hasattr(age_raw, "__len__") else float(age_raw)

        latent, y_reg, y_clf, stage = infer_one(model, specs, age_z_val, sex_val, cfg)
        pred_df, bhs = _apply_head_weights(latent, cfg, age_z_val, sex_val)

        if save_json and json_dir is not None:
            out_json = {
                "file_id": file_id_val,
                "filename": filename,
                "filepath": filepath_val,
                "brain_health_score": round(float(bhs), 5),
                "pred_df": pred_df.to_dict(),
                "latent": latent.tolist(),
                "stage": stage.tolist(),
            }
            json_path = json_dir / f"{file_id_val}_result.json"
            with open(json_path, "w") as f:
                json.dump(out_json, f, indent=2)

        f_precision = 5
        row = {
            "file_id": file_id_val,
            "filepath": filepath_val,
            "age": age_val,
            "sex": sex_val,
            "brain_health_score": float(round(bhs, f_precision)),
            "total_cognition_score": float(round(pred_df.loc["cog_total", "y_pred"], f_precision)),
            "fluid_cognition_score": float(round(pred_df.loc["cog_fluid", "y_pred"], f_precision)),
            "crystallized_cognition_score": float(round(pred_df.loc["cog_crystallized", "y_pred"], f_precision)),
        }

        if collect_head_outputs:
            # include head predictions except for the summary cognition heads already captured above
            for head_name, value in pred_df["y_pred"].items():
                if head_name in {"cog_total", "cog_fluid", "cog_crystallized"}:
                    continue
                if head_name not in row:
                    row[f"head_{head_name}"] = float(round(value, f_precision))

        latent_flat = latent.flatten()
        for idx, value in enumerate(latent_flat, start=1):
            row[f"lhl_{idx}"] = float(value)

        results.append(row)

        if save_plots and figure_dir is not None:
            hypnogram = stage.squeeze(0)
            hypnogram = np.argmax(hypnogram, axis=0) + 1
            hypnogram[hypnogram == 6] = 5

            sum_power = specs.sum(axis=1)
            last_power = np.where(sum_power)[0][-1]
            specs_trimmed = specs[: last_power + 1, :]
            hypnogram_trimmed = np.repeat(hypnogram, 30)[: last_power + 1]

            fig = plot_spectrogram(
                specs_trimmed,
                dt=1 / cfg.fs_time,
                vmin=0.1,
                vmax=0.97,
                max_freq_to_plot=20,
                title=f"{file_id_val} | Brain health score: {bhs:.2f}",
                colorbar_unit='Amplitude\n(Wavelet)'
            )
            fig.savefig(figure_dir / f"{file_id_val}_spec.png", dpi=300)
            fig.clf()

            fig_with_stages = plot_spectrogram_with_stages(
                specs_trimmed,
                hypnogram_trimmed,
                dt=1 / cfg.fs_time,
                vmin=0.1,
                vmax=0.97,
                max_freq_to_plot=20,
                title=f"{file_id_val} | Brain health score: {bhs:.2f}",
                colorbar_unit='Amplitude\n(Wavelet)'
            )
            fig_with_stages.savefig(figure_dir / f"{file_id_val}_spec_with_stages.png", dpi=300)
            fig_with_stages.clf()
            
        plt.close('all')
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    show_free_memory = False
    if show_free_memory:
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"[mem] RSS={rss:.2f} GB after {len(results)} files")

    if show_progress and isinstance(iterator, tqdm):  # type: ignore[arg-type]
        iterator.close()

    summary_df = pd.DataFrame(results)
    summary_path: Optional[Path] = None
    if save_summary and summary_dir is not None and not summary_df.empty:
        summary_path = summary_dir / "phi_results.csv"
        if summary_path.exists():
            try:
                existing = pd.read_csv(summary_path)
            except Exception:
                existing = pd.DataFrame()
            combined = pd.concat([existing, summary_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["filepath"], keep="last")
            summary_df = combined
        summary_df.to_csv(summary_path, index=False)

    if verbose:
        elapsed = time.time() - t0
        n_files = len(dataset)
        avg = elapsed / n_files if n_files else 0.0
        msg = (
            f"Philosopher's Stone finished. Processed {n_files} file(s) in {elapsed/60:.1f} minutes. "
            f"({avg:.2f} s/file)."
        )
        if summary_path is not None:
            msg += f" Results in: {summary_path}"
        print(msg)

    return summary_df, summary_path


# ---------- CLI ---------- #
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", default='phi_manifest.csv', help="CSV with columns filepath, age, sex")
    parser.add_argument("--outdir", default="phi_out", help="output folder")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = main process)")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true",
                        help="Generate spectrogram figures (stored in <outdir>/figures)")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false",
                        help="Disable spectrogram figure generation")
    parser.add_argument("--save-json", dest="save_json", action="store_true",
                        help="Write per-file JSON outputs (<outdir>/json)")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false",
                        help="Disable JSON output (default)")
    parser.add_argument("--collect-heads", action="store_true", help="Include head predictions in summary output")
    parser.add_argument("--no-summary", action="store_false", dest="save_summary", help="Skip writing summary CSV")
    parser.add_argument("--device-id", type=int, default=None,
                        help="CUDA device index to use (defaults to auto-detected cuda:0 / cpu)")
    parser.set_defaults(save_plots=False, save_json=False)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    manifest_path = args.manifest_csv
    if not os.path.exists(manifest_path) and not os.path.exists(f"{manifest_path}.csv"):
        raise FileNotFoundError(f"The specified manifest CSV file does not exist: {manifest_path}")
    else:
        print(f"Using manifest CSV file: {manifest_path}")

    summary_df, summary_path = run_philosopher(
        manifest_path,
        outdir=args.outdir,
        num_workers=args.num_workers,
        save_summary=args.save_summary,
        save_json=args.save_json,
        save_plots=args.save_plots,
        collect_head_outputs=args.collect_heads,
        device_id=args.device_id,
    )

if __name__ == "__main__":
    main()
