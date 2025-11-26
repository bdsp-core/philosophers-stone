# Sleep‑Philosopher‑Stone – Brain‑Health Inference from Sleep EEG
# Turn a single‑channel overnight EEG (C4‑M1) into a **Brain‑Health Score**, disease
# probabilities, cognitive‑score estimates, and a 1 × 1024 latent embedding – all
# without retraining.

from __future__ import annotations
import os, json, shutil
from dataclasses import dataclass, asdict, field
from typing import Sequence, Optional
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

torch.manual_seed(9)

# --- helper modules ---
from phi_utils.load_data import load_prepared_data
from phi_utils.preprocessing_and_spectrograms import (
    plot_spectrogram,
    plot_spectrogram_with_stages,
    preprocess_filter,
    Wavelet,
    compute_wavelet_transform,
    interpolate_wx_2d,
    pad_spectrogram,
)

from phi_utils.philosopher_init import (
    default_model_init_vars
)

from phi_utils.model_config import SleepPhilosopherSpectral


# ---------------- Config ---------------- #
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
    model_file: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent / "model_files" / "SleepPhilosophersStone.ckpt"
    ))
    head_weights_csv: str = "./phi_utils/head_weights.csv"
    plot: bool = False
    gpu_id: int = 0
    device: str = field(init=False)
    age_mean_tr_data: float = 59.60 # mean age of training data
    age_std_tr_data: float = 15.18 # std age of training data

    def __post_init__(self):
        self.device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        # Allow environment override for the model file location
        env_override = os.getenv("PHILOSOPHER_MODEL_FILE")
        if env_override:
            self.model_file = env_override


# a global default for casual users
DefaultConfig = Config


CHECKPOINT_DOWNLOAD_URL = (
    "https://www.dropbox.com/scl/fi/xijxzkplnyo1ai3qztscb/"
    "SleepPhilosophersStone.ckpt?rlkey=al1l5ch171fy6n8jwvuhzubvv&st=sbfv32la&dl=1"
)


def _download_checkpoint(url: str, destination: Path) -> Path:
    """Download checkpoint to destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    with urlopen(url) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.replace(destination)
    return destination


def _get_checkpoint_path(path_str: str) -> Path:
    """
    Return a local checkpoint path, downloading from the configured URL if missing.
    If path_str is a URL, download to the canonical model_files location.
    """
    download_url = CHECKPOINT_DOWNLOAD_URL

    if path_str.startswith(("http://", "https://")):
        download_url = path_str
        checkpoint_path = (
            Path(__file__).resolve().parent.parent
            / "model_files"
            / "SleepPhilosophersStone.ckpt"
        )
    else:
        checkpoint_path = Path(path_str).expanduser().resolve()

    if checkpoint_path.exists():
        return checkpoint_path

    print(f"[Model] Checkpoint not found at {checkpoint_path} (expected at first run after installation). Downloading from {download_url} ...")
    return _download_checkpoint(download_url, checkpoint_path)



def load_model(cfg: Config = DefaultConfig()) -> "torch.nn.Module":
    """
    Load the SleepPhilosopherStone model file **once** and return a PyTorch
    module on the correct device in evaluation mode.
    """

    # ensure the model is present locally (download if missing)
    model_path = _get_checkpoint_path(cfg.model_file)

    model_args = default_model_init_vars()
    dim_final_latent_space = model_args.pop("dim_final_latent_space")
    fs_time = model_args.pop("fs_time")

    model = (
        SleepPhilosopherSpectral(
            **model_args,
            dim_final_latent_space=dim_final_latent_space,
            fs_time=fs_time,
        ).load_from_checkpoint(str(model_path))
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
                
    return model


def _guess_channel(ch_names: Sequence[str], canonical: str) -> Optional[str]:
    """
    Try common aliases for C4‑M1 (case‑ and whitespace‑insensitive).
    Returns the first match or None.
    """
    aliases = {
        "c4-m1", "c4 m1", "c4‑m1", "c4m1",
        "c4-m",  "c4‑m",  "c4mast", "c4a1",  # sometimes linked to mastoid
        "c4-m2", "c4-a2", "c4",
        }
    aliases = {a.lower().replace(" ", "").replace("‑", "-") for a in aliases}
    for ch in ch_names:
        norm = ch.lower().replace(" ", "").replace("‑", "-")
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

    raw.resample(target_hz, npad="auto", window="boxcar", npad_max="auto", low_pass=lowpass, verbose=False)
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
