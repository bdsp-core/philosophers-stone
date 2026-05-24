# Sleep‑Philosopher‑Stone – Brain‑Health Inference from Sleep EEG
# Turn a single‑channel overnight EEG (C4‑M1) into a **Brain‑Health Score**, disease
# probabilities, cognitive‑score estimates, and a 1 × 1024 latent embedding – all
# without retraining.

from __future__ import annotations
import os, json, shutil
from dataclasses import dataclass, asdict, field
from typing import Sequence, Optional
from pathlib import Path
from urllib.request import Request, urlopen
from fractions import Fraction
from importlib import resources

import numpy as np
import pandas as pd
import torch
from scipy import signal as sp_signal

torch.manual_seed(9)

# --- helper modules ---
from philosophers_stone.load_data import load_prepared_data
from philosophers_stone.preprocessing_and_spectrograms import (
    plot_spectrogram,
    plot_spectrogram_with_stages,
    preprocess_filter,
    Wavelet,
    compute_wavelet_transform,
    interpolate_wx_2d,
    pad_spectrogram,
)

from philosophers_stone.model_config import SleepPhilosopherSpectral


# ---------------- Config ---------------- #
DEFAULT_CHECKPOINT_FILENAME = "SleepPhilosophersStone.ckpt"
DEFAULT_HUGGINGFACE_REPO_ID = "wolfgang-ganglberger/philosophers-stone"


def _package_resource_path(filename: str) -> str:
    return str(resources.files("philosophers_stone").joinpath(filename))


def _source_checkout_root() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "philosophers_stone").exists():
            return parent
    return None


def _cache_model_file() -> Path:
    cache_dir = Path(
        os.getenv(
            "PHILOSOPHER_CACHE_DIR",
            str(Path.home() / ".cache" / "philosophers-stone"),
        )
    )
    return cache_dir / "model_files" / DEFAULT_CHECKPOINT_FILENAME


def _default_model_file() -> str:
    env_override = os.getenv("PHILOSOPHER_MODEL_FILE")
    if env_override:
        return env_override

    checkout_root = _source_checkout_root()
    if checkout_root is not None:
        repo_candidate = checkout_root / "model_files" / DEFAULT_CHECKPOINT_FILENAME
        if repo_candidate.exists():
            return str(repo_candidate)

    return str(_cache_model_file())


@dataclass
class Config:
    channel: str = "c4-m1"
    resample_hz: int = 200
    fs_time: int = 1
    n_freqs: int = 100
    f_high: int = 50
    hours_pad: int = 11
    wavelet_name: str = "gmw"
    wavelet_gamma: int = 60
    wavelet_beta: int = 30
    nv: int = 32
    model_file: str = field(default_factory=_default_model_file)
    head_weights_csv: str = field(default_factory=lambda: _package_resource_path("head_weights.csv"))
    plot: bool = False
    gpu_id: int = 0
    device: str = field(init=False)
    age_mean_tr_data: float = 59.60 # mean age of training data
    age_std_tr_data: float = 15.18 # std age of training data

    def __post_init__(self):
        self.device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        # Allow environment override for the model file location.
        env_override = os.getenv("PHILOSOPHER_MODEL_FILE")
        if env_override:
            self.model_file = env_override


# a global default for casual users
DefaultConfig = Config


def _build_huggingface_download_url(repo_id: str, filename: str, revision: str = "main") -> str:
    repo_path = "/".join(part.strip("/") for part in repo_id.split("/", 1))
    file_path = "/".join(part.strip("/") for part in filename.split("/"))
    return f"https://huggingface.co/{repo_path}/resolve/{revision}/{file_path}?download=1"


def _default_checkpoint_download_url() -> Optional[str]:
    env_url = os.getenv("PHILOSOPHER_MODEL_URL")
    if env_url:
        return env_url

    repo_id = os.getenv("PHILOSOPHER_MODEL_REPO_ID", DEFAULT_HUGGINGFACE_REPO_ID)
    filename = os.getenv("PHILOSOPHER_MODEL_FILENAME", DEFAULT_CHECKPOINT_FILENAME)
    revision = os.getenv("PHILOSOPHER_MODEL_REVISION", "main")
    return _build_huggingface_download_url(repo_id, filename, revision)


CHECKPOINT_DOWNLOAD_URL = _default_checkpoint_download_url()


def _download_checkpoint(url: str, destination: Path) -> Path:
    """Download checkpoint to destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    token = os.getenv("PHILOSOPHER_MODEL_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(url, headers=headers)
    with urlopen(request) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.replace(destination)
    return destination


def _get_checkpoint_path(path_str: str, *, download_if_missing: bool = True) -> Path:
    """
    Return a local checkpoint path, downloading from the configured URL if missing.
    If path_str is a URL, download to the canonical model_files location.
    """
    download_url = CHECKPOINT_DOWNLOAD_URL

    if path_str.startswith(("http://", "https://")):
        download_url = path_str
        checkpoint_path = _cache_model_file()
    else:
        checkpoint_path = Path(path_str).expanduser().resolve()

    if checkpoint_path.exists():
        return checkpoint_path

    if not download_if_missing:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    if not download_url:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Configure either PHILOSOPHER_MODEL_FILE with a local path or HTTPS URL, "
            "or set PHILOSOPHER_MODEL_REPO_ID (plus optional PHILOSOPHER_MODEL_FILENAME "
            "and PHILOSOPHER_MODEL_REVISION) to auto-download from Hugging Face."
        )

    print(f"[Model] Checkpoint not found at {checkpoint_path} (expected at first run after installation). Downloading from {download_url} ...")
    return _download_checkpoint(download_url, checkpoint_path)


def checkpoint_available(cfg: Optional[Config] = None) -> tuple[bool, str]:
    """Return whether the configured checkpoint exists locally without downloading it."""

    cfg = cfg or Config()
    try:
        return True, str(_get_checkpoint_path(cfg.model_file, download_if_missing=False))
    except FileNotFoundError as exc:
        return False, str(exc)



def load_model(cfg: Config = DefaultConfig()) -> "torch.nn.Module":
    """
    Load the SleepPhilosopherStone model file **once** and return a PyTorch
    module on the correct device in evaluation mode.
    """

    # ensure the model is present locally (download if missing)
    model_path = _get_checkpoint_path(cfg.model_file, download_if_missing=True)

    # Let the checkpoint restore its own hyperparameters. Passing the current
    # source defaults here can override the checkpoint metadata with placeholder
    # list lengths from `default_model_init_vars()`.
    model = SleepPhilosopherSpectral.load_from_checkpoint(
        str(model_path),
        strict=False,
    )
    model.to(cfg.device)
    # Note: We use the model in .train() mode here but deactivate dropout and gradient computation.
    # This is because batchnorm layers need to update their running stats per sample.
    # In future, we might consider using InstanceNorm or GroupNorm instead.
    # Then we could use eval() mode safely.
    model.train()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "drop_prob"):            # timm DropPath
            module.drop_prob = 0.0
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False      # no state updates, use batch stats

    n_heads = len(model.heads)
    n_target_names = len(model.target_names)
    n_target_output_dims = len(model.target_output_dims)
    n_task_types = len(model.task_types)
    if not (n_heads == n_target_names == n_target_output_dims == n_task_types):
        raise RuntimeError(
            "Loaded Philosopher checkpoint/config mismatch: "
            f"heads={n_heads}, target_names={n_target_names}, "
            f"target_output_dims={n_target_output_dims}, task_types={n_task_types}"
        )
                
    return model


def _normalise_channel_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("‑", "-")
        .replace("/", "-")
    )


def _guess_channel(ch_names: Sequence[str], canonical: str) -> Optional[str]:
    """
    Try common aliases for C4‑M1 (case‑ and whitespace‑insensitive).
    Returns the first match or None.
    """
    aliases = {
        "c4-m1", "c4 m1", "c4‑m1", "c4m1",
        "c4-m",  "c4‑m",  "c4mast", "c4a1", "c4-a1",  # A1/M1 often used interchangeably
        "c4-m2", "c4-a2", "c4",
        }
    aliases = {_normalise_channel_name(a) for a in aliases}
    for ch in ch_names:
        norm = _normalise_channel_name(ch)
        if norm in aliases:
            return ch
    return None


def _resample_raw(raw, target_hz: float):
    """Resample an MNE Raw to target_hz with a defensive low-pass to avoid aliasing."""
    current = float(raw.info.get("sfreq", 0.0))
    if current <= 0:
        raise ValueError("EDF sampling rate could not be determined.")
    if abs(current - target_hz) < 1e-6:
        return raw

    # Choose a conservative low-pass: keep any existing low-pass but never exceed Nyquist of target.
    existing_lowpass = raw.info.get("lowpass") or current / 2.0
    lowpass = min(existing_lowpass, 0.99 * target_hz / 2.0)

    # MNE's resample API differs across versions. Apply the anti-alias low-pass
    # explicitly, then call the portable subset of resample arguments.
    if current > target_hz and lowpass > 0:
        raw.filter(l_freq=None, h_freq=lowpass, verbose=False)
    raw.resample(target_hz, npad="auto", window="boxcar", verbose=False)
    return raw

# ---------------- Return type ------------- #
@dataclass
class Result:
    file_id: str
    pred_df: pd.DataFrame  # y_pred for each head
    latent: np.ndarray     # brain health latent space (1, 1024)
    bhs: float             # brain‑health score

    def to_json(self) -> str:
        out = asdict(self)
        out["pred_df"] = self.pred_df.to_dict()
        out["latent"] = self.latent.tolist()
        return json.dumps(out, indent=2)

    def print_json(self):
        print(self.to_json())
        print()


# ============ helper functions ============ #
def _load_eeg(path_file: str, cfg: Config):
    ext = os.path.splitext(path_file)[1].lower()

    if ext == ".h5":
        signals, _, params = load_prepared_data(
            path_file, signals_to_load=[cfg.channel]
        )
        fs_eeg = params["fs"]
        assert fs_eeg == cfg.resample_hz, \
            f"H5 sampling rate {fs_eeg} Hz ≠ expected {cfg.resample_hz} Hz"

    elif ext == ".edf":
        import mne
        raw = mne.io.read_raw_edf(path_file, preload=True, verbose=False)
        ch = _guess_channel(raw.ch_names, cfg.channel)
        assert ch is not None, f"Could not find channel like '{cfg.channel}' in {raw.ch_names}"
        raw.pick_channels([ch])
        raw = _resample_raw(raw, cfg.resample_hz)
        eeg_np = raw.get_data()[0]          # ndarray (volts)
        # Convert to microvolts to match H5 inputs and training scale
        eeg_uv = eeg_np * 1e6
        # Standardise to DataFrame to match the H5 loader
        signals = pd.DataFrame({cfg.channel: eeg_uv.astype(float)})
        fs_eeg = cfg.resample_hz
    else:
        raise NotImplementedError(f"Unsupported extension {ext}")

    # truncate to cfg.hours_pad hours
    max_len = cfg.hours_pad * 3600 * fs_eeg
    if len(signals) > max_len:
        signals = signals[:max_len]

    assert not pd.isna(signals).any().any(), "Input signal contains NaNs"
    return signals, fs_eeg


def _compute_wavelet_specs(signals, fs_eeg, cfg: Config):
    # downsample to 100 Hz to match training
    assert fs_eeg == 200
    signals_ds = signals[::2]
    fs_eeg //= 2

    N_wavelet = int(4 * fs_eeg)
    wavelet = Wavelet(
        (cfg.wavelet_name, {"gamma": cfg.wavelet_gamma, "beta": cfg.wavelet_beta}),
        N=N_wavelet,
    )
    _, specs_raw, ssq_freqs = compute_wavelet_transform(
        signals_ds[cfg.channel].values,
        wavelet=wavelet,
        nv=cfg.nv,
        fs=fs_eeg,
    )

    # kill NaNs
    nan_frac = np.isnan(specs_raw).mean()
    if nan_frac > 0.1:
        print('Warning: NaN fraction in spectrogram is > 10%. Unsual, check EEG and spectrogram.')
    specs_raw[np.isnan(specs_raw)] = 0

    # interpolate + pad
    freq_bins = _make_frequency_grid(cfg.n_freqs)
    specs_interp = interpolate_wx_2d(
        specs_raw, ssq_freqs, freq_bins, fs_eeg, cfg.fs_time
    )
    specs = pad_spectrogram(specs_interp, cfg.fs_time, hours_pad=cfg.hours_pad)

    if cfg.plot:
        plot_spectrogram(
            specs_interp[:, freq_bins <= 20],
            freq_bins[freq_bins <= 20],
            dt=1 / cfg.fs_time,
            vmin=0.1,
            vmax=0.97,
            title=f"{os.path.basename(path_file)} {cfg.channel}",
        )

    # final shape checks (like the notebook)
    exp_len = cfg.hours_pad * 3600 * cfg.fs_time
    assert specs.shape == (exp_len, cfg.n_freqs), \
        f"Spectrogram shape {specs.shape} ≠ {(exp_len, cfg.n_freqs)}"

    return specs  # (T, F)


@torch.no_grad()
def infer_one(model, specs, age_z: float, sex: int, cfg: Config):
    """Run the neural network for one already-computed spectrogram."""

    device = torch.device(cfg.device)
    x = torch.tensor(specs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cov = torch.tensor([[age_z, sex]], dtype=torch.float32).to(device)

    yp_reg, yp_clf, yp_stage, latent = model(x, cov, return_features_lhl=True)

    del x, cov
    return (
        latent.cpu().numpy(),
        yp_reg.cpu().numpy(),
        yp_clf.cpu().numpy(),
        yp_stage.cpu().numpy(),
    )


def _checkpoint_metadata(cfg: Config) -> dict[str, object]:
    try:
        path = _get_checkpoint_path(cfg.model_file, download_if_missing=False)
    except Exception:
        return {
            "model_file": str(cfg.model_file),
            "model_file_available": False,
        }
    try:
        stat = path.stat()
        return {
            "model_file": str(path),
            "model_file_name": path.name,
            "model_file_size": int(stat.st_size),
            "model_file_mtime": float(stat.st_mtime),
            "model_file_available": True,
        }
    except Exception:
        return {
            "model_file": str(path),
            "model_file_available": True,
        }


def _result_row_from_predictions(
    *,
    file_id: str,
    age: float,
    sex: int,
    filepath: str,
    pred_df: pd.DataFrame,
    bhs: float,
    latent: np.ndarray,
    collect_head_outputs: bool = False,
    precision: int = 5,
) -> dict[str, object]:
    row: dict[str, object] = {
        "file_id": file_id,
        "filepath": filepath,
        "age": float(age),
        "sex": int(sex),
        "brain_health_score": float(round(float(bhs), precision)),
        "total_cognition_score": float(round(pred_df.loc["cog_total", "y_pred"], precision)),
        "fluid_cognition_score": float(round(pred_df.loc["cog_fluid", "y_pred"], precision)),
        "crystallized_cognition_score": float(round(pred_df.loc["cog_crystallized", "y_pred"], precision)),
    }

    if collect_head_outputs:
        for head_name, value in pred_df["y_pred"].items():
            if head_name in {"cog_total", "cog_fluid", "cog_crystallized"}:
                continue
            if head_name not in row:
                row[f"head_{head_name}"] = float(round(float(value), precision))

    for idx, value in enumerate(np.asarray(latent).flatten(), start=1):
        row[f"lhl_{idx}"] = float(value)
    return row


def infer_brain_health_from_specs(
    specs: np.ndarray,
    *,
    age: float,
    sex: int,
    file_id: str,
    filepath: str = "",
    cfg: Optional[Config] = None,
    model: Optional[torch.nn.Module] = None,
    collect_head_outputs: bool = False,
) -> dict[str, object]:
    """Run Philosopher's Stone from a precomputed spectrogram.

    This is the shared core used by the CLI and by tests. It performs no disk
    writes and returns plain Python/numpy values.
    """

    cfg = cfg or Config()
    model = model or load_model(cfg)
    age_z = (float(age) - cfg.age_mean_tr_data) / cfg.age_std_tr_data
    sex_int = int(sex)

    latent, y_reg, y_clf, stage = infer_one(model, specs, age_z, sex_int, cfg)
    pred_df, bhs = _apply_head_weights(latent, cfg, age_z, sex_int)
    row = _result_row_from_predictions(
        file_id=file_id,
        age=float(age),
        sex=sex_int,
        filepath=filepath,
        pred_df=pred_df,
        bhs=bhs,
        latent=latent,
        collect_head_outputs=collect_head_outputs,
    )

    return {
        "schema_version": "philosophers_stone_brain_health_v1",
        "status": "ok",
        **row,
        "predictions": {str(k): float(v) for k, v in pred_df["y_pred"].items()},
        "latent": np.asarray(latent).reshape(-1).astype(np.float32),
        "stage_probabilities": np.asarray(stage).squeeze().astype(np.float32),
        "regression_outputs": np.asarray(y_reg).astype(np.float32),
        "classification_outputs": np.asarray(y_clf).astype(np.float32),
        **_checkpoint_metadata(cfg),
    }


def _resample_1d(signal: np.ndarray, fs_hz: float, target_hz: float) -> np.ndarray:
    fs = float(fs_hz)
    target = float(target_hz)
    if fs <= 0:
        raise ValueError("Input sampling rate must be positive.")
    if abs(fs - target) < 1e-6:
        return np.asarray(signal, dtype=float)

    ratio = Fraction(target / fs).limit_denominator(1000)
    return sp_signal.resample_poly(np.asarray(signal, dtype=float), ratio.numerator, ratio.denominator)


def _error_result(file_id: str, age: object, sex: object, exc: Exception) -> dict[str, object]:
    return {
        "schema_version": "philosophers_stone_brain_health_v1",
        "status": "error",
        "file_id": str(file_id),
        "age": age,
        "sex": sex,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def infer_brain_health(
    eeg_uv: np.ndarray,
    *,
    fs_hz: float,
    age: float,
    sex: int,
    file_id: str,
    filepath: str = "",
    cfg: Optional[Config] = None,
    model: Optional[torch.nn.Module] = None,
    collect_head_outputs: bool = False,
) -> dict[str, object]:
    """Run brain-health inference from a single EEG channel in microvolts.

    Parameters are intentionally array-based so applications can use their own
    H5 readers and unit conversion without creating temporary Philosopher-specific
    files.
    """

    cfg = cfg or Config()
    try:
        eeg = np.asarray(eeg_uv, dtype=float).reshape(-1)
        if eeg.size == 0:
            raise ValueError("Input EEG is empty.")
        if not np.isfinite(eeg).all():
            raise ValueError("Input EEG contains NaN or infinite values.")

        eeg_200 = _resample_1d(eeg, fs_hz, cfg.resample_hz)
        max_len = int(cfg.hours_pad * 3600 * cfg.resample_hz)
        if len(eeg_200) > max_len:
            eeg_200 = eeg_200[:max_len]

        signals = pd.DataFrame({cfg.channel: eeg_200.astype(float)})
        signals = preprocess_filter(signals, Fs=cfg.resample_hz, bandpass_high=cfg.f_high)
        specs = _compute_wavelet_specs(signals, cfg.resample_hz, cfg)
        result = infer_brain_health_from_specs(
            specs,
            age=age,
            sex=sex,
            file_id=file_id,
            filepath=filepath,
            cfg=cfg,
            model=model,
            collect_head_outputs=collect_head_outputs,
        )
        result["input_fs_hz"] = float(fs_hz)
        result["model_fs_hz"] = float(cfg.resample_hz)
        result["input_n_samples"] = int(len(eeg))
        result["model_n_samples"] = int(len(eeg_200))
        return result
    except Exception as exc:
        return _error_result(file_id, age, sex, exc)


def _make_frequency_grid(n_freqs: int) -> np.ndarray:
    wavelet_min_freq = 4.7683e-05
    if n_freqs == 100:
        return np.array(
            [wavelet_min_freq, 0.10]
            + list(np.arange(0.25, 21, 0.25))
            + list(np.arange(20, 50, 2))
        )
    raise ValueError("Unsupported n_freqs")


def _apply_head_weights(latent: np.ndarray, cfg: Config, age_z: float, sex: float):
    df_w = pd.read_csv(cfg.head_weights_csv, index_col=0)
    feats = np.concatenate([[age_z, sex], latent.flatten()]) 
    preds = []
    for head, row in df_w.iterrows():
        b = row["bias"]
        w = row.values[1:]
        y = feats @ w + b
        if head.startswith("dx"):  # logistic
            y = 1 / (1 + np.exp(-y))
        preds.append(y)
    df_pred = pd.DataFrame({"y_pred": preds}, index=df_w.index)

    cog_total_cols = ['cog_total_mesa', 'cog_total_mgh-cog', 'cog_total_fhs', 'cog_total_sof', 'cog_total_mros', 'cog_total_koges']
    cog_fluid_cols = ['cog_fluid_mgh-cog', 'cog_fluid_fhs', 'cog_fluid_sof', 'cog_fluid_mros', 'cog_fluid_koges']
    cog_crystallized_cols = ['cog_crystallized_mgh-cog', 'cog_crystallized_fhs', 'cog_crystallized_sof']
    # create new columns for averages, 'head_cog_total', 'head_cog_fluid', 'head_cog_crystallized'
    df_pred.loc['cog_total', 'y_pred'] = df_pred.loc[cog_total_cols, 'y_pred'].mean()
    df_pred.loc['cog_fluid', 'y_pred'] = df_pred.loc[cog_fluid_cols, 'y_pred'].mean()
    df_pred.loc['cog_crystallized', 'y_pred'] = df_pred.loc[cog_crystallized_cols, 'y_pred'].mean()

    bhs = df_pred.loc["brain_health_score", "y_pred"]
    
    df_pred = df_pred.reindex(['brain_health_score', 'cog_total', 'cog_fluid', 'cog_crystallized'] + [i for i in df_pred.index if i not in ['brain_health_score', 'cog_total', 'cog_fluid', 'cog_crystallized']])
    
    return df_pred, float(bhs)
