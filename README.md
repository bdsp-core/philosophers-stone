<p align="center">
  <img src="docs/images/logo.png" alt="Philosopher's Stone logo" width="350">
</p>

# Brain-Health Inference from Sleep EEG

**Philosopher's Stone** is an inference tool that converts a single-channel
overnight sleep EEG into a quantitative index of brain health.

It applies a peer-reviewed multi-cohort deep-learning model trained on 36,000
sleep recordings to estimate a Brain Health Score, cognition-related outputs,
disease-related outputs, mortality-related physiological patterns, and a
1024-dimensional latent embedding for research workflows.

<p align="center">
  <img src="docs/images/overview.jpg" alt="Overview" width="500">
</p>

## Scientific Study

If you use or reference this tool, please cite:

    Ganglberger, W., Sun, H., Turley, N., ... & Westover, M. B. (2026).
    Brain Health from Sleep EEG: A Multicohort, Deep Learning Biomarker for
    Cognition, Disease, and Mortality.
    NEJM AI, 3(3), AIoa2500487.

Article DOI: https://doi.org/10.1056/AIoa2500487

Repository copies:

- [Paper PDF](Ganglberger_2026_NEJMAI_Brain_Health_from_Sleep_EEG.pdf)
- [Supplementary Appendix](Ganglberger_2026_NEJMAI_Appendix.pdf)

## Intended Use

This repository is intended for research use by sleep scientists, neurology and
aging researchers, psychiatry researchers, clinical-trial teams, and data
scientists working with physiological signals.

This software is not a medical device and is not intended for diagnosis,
treatment decisions, or clinical use without independent validation and
appropriate regulatory review.

## Installation

Python 3.10+ is required. Conda is recommended because the stack includes
PyTorch, MNE, scipy, and EEG signal-processing dependencies.

Run the bundled sample end to end:

```bash
./run_sample.sh
```

For application development:

```bash
pip install -e .
```

The package installs a CLI:

```bash
philosophers-stone --manifest_csv phi_manifest.csv --outdir phi_out
```

The legacy source-checkout command remains available:

```bash
python philosopher.py --manifest_csv phi_manifest.csv --outdir phi_out
```

Docker is also supported. See [docker/README.md](docker/README.md).

## Model Checkpoint

The model checkpoint is about 2.3 GB and is intentionally not tracked in Git.

Checkpoint lookup order:

1. `PHILOSOPHER_MODEL_FILE`, if set.
2. `./model_files/SleepPhilosophersStone.ckpt` in a source checkout.
3. `~/.cache/philosophers-stone/model_files/SleepPhilosophersStone.ckpt`.

If the checkpoint is missing, inference can auto-download it from:

```text
https://huggingface.co/wolfgang-ganglberger/philosophers-stone
```

## Inputs

The CLI takes a manifest CSV with these columns:

- `filepath`: path to an EEG file.
- `age`: age in years.
- `sex`: `0` for female, `1` for male.

Supported EEG formats:

| Format | Requirements |
| --- | --- |
| HDF5 (`.h5`) | Dataset `signals/c4-m1`, 1-D float array, full night; attributes `sampling_rate=200` and `unit_voltage="uV"`. |
| EDF (`.edf`) | Must contain a C4-M1 channel or accepted label variant; any sampling rate is resampled to 200 Hz with anti-aliasing. |

Sample full-night EEG files are included under `sample-data/`.

## Python API

Preferred import path:

```python
from philosophers_stone import Config, infer_brain_health

result = infer_brain_health(
    eeg_uv,
    fs_hz=200,
    age=65,
    sex=1,
    file_id="study-001",
    cfg=Config(),
)
```

The legacy import path remains available for compatibility:

```python
from phi_utils.philosopher_utils import Config, infer_brain_health
```

`eeg_uv` must be one overnight EEG channel in microvolts. The array-based API
does not write files; callers decide where to persist scores, latent features,
and optional stage probabilities.

## Outputs

Default summary CSV:

```text
phi_out/phi_results.csv
```

Columns include:

```text
file_id, filepath, age, sex, brain_health_score,
total_cognition_score, fluid_cognition_score,
crystallized_cognition_score, lhl_1 ... lhl_1024
```

Optional outputs:

- JSON summaries under `<outdir>/json/`.
- Spectrogram figures under `<outdir>/figures/`.

## Repository Layout

- `src/philosophers_stone/`: runtime package.
- `src/philosophers_stone/_vendor/timm/`: minimal vendored `timm` runtime subset.
- `phi_utils/`: source-checkout compatibility shims for historic imports.
- `sample-data/`: bundled sample EEG files.
- `paper/`: paper and reproducibility assets, not required at runtime.
- `docker/`: Docker build and usage files.

## License

CC BY-NC 4.0 - Attribution-NonCommercial 4.0 International. See
[LICENSE](LICENSE).
