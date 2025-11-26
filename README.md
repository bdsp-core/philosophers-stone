# Philosophers-Stone — Brain-Health Inference from Single-Channel Sleep EEG

**Philosophers-Stone** is a lightweight inference tool that converts a single-channel overnight sleep EEG into a quantitative index of brain health.  
It applies a validated multi-cohort deep-learning model trained on **36,000 sleep recordings** to estimate cognitive performance, disease likelihoods, and mortality-related physiological patterns.  
The tool runs in seconds and outputs both a **single Brain Health Score** and a **1024-dimensional latent embedding** suitable for research and biomarker discovery.

---

## Who is this for?

- Sleep scientists  
- Neurologists and dementia researchers  
- Aging and cognitive-decline investigators  
- Psychiatry researchers  
- Data scientists working with physiological signals  
- Clinical-trial teams exploring EEG-based biomarkers  

---

## What you get

- **Brain Health Score** (single interpretable metric)
- **1×1024 latent brain-health embedding**  (AI-derived sleep features)
- **Predictions** for cognition, disease risk, and mortality-related physiology  
- **Optional outputs**: spectrograms and per-recording JSON summaries  

---

## Model provenance

This tool implements the multi-task deep-learning framework described in:

Ganglberger W. et al., *Brain health from sleep EEG: A multi-cohort, deep learning biomarker for cognition, disease and mortality*, 2025.

---

## Requirements

- Python ≥ 3.10  
- PyTorch 2.x (CUDA recommended)  
- pandas, numpy, mne (for EDF), h5py, matplotlib, tqdm, psutil  

Install dependencies:

    pip install torch pandas numpy mne h5py matplotlib tqdm psutil

### Model file

Auto-download when first running the code.

---

## Inputs

### Manifest CSV

A CSV with columns:

- `filepath`
- `age` (years)
- `sex` (0=female, 1=male)

### EEG File Requirements

Philosophers-Stone accepts **single-channel overnight EEG** in **HDF5 (.h5)** or **EDF (.edf)** format.  
Preferred channel: **C4-M1**.

| Format        | Requirements |
|---------------|-------------|
| **HDF5 (.h5)** | - Dataset: `signals/c4-m1` (1-D float array, full night) <br> - Attributes: `sampling_rate=200`, `unit_voltage="uV"` <br> - Extra channels/annotations ignored <br> - Manifest uses absolute paths |
| **EDF (.edf)** | - Must contain a C4-M1 channel (label variants allowed) <br> - Any sampling rate accepted; auto-resampled to 200 Hz with anti-aliasing |

Sample full-night EEG data is included under `./sample-data/`.

---

## Quick start (CLI)

    python philosopher.py \
      --manifest_csv phi_manifest.csv

---

## Outputs

- **Summary CSV** (`phi_out/phi_results.csv`)  
  Columns include:  
  `file_id, filepath, age, sex, brain_health_score, total_cognition_score, fluid_cognition_score, crystallized_cognition_score, lhl_1…lhl_1024`

- **Latent embedding (`lhl_1…lhl_1024`)**  
  A 1024-dimensional vector summarizing brain-health-relevant EEG patterns.

- **Optional JSON files** under `phi_out/json/`  
- **Optional spectrograms** under `phi_out/figures/`

---

## Performance tips

- Use a **GPU** if available  
- Keep `batch_size=1`


---

## Citation

If you use this tool in academic work, please cite:

Ganglberger W. et al. (2025). *Brain health from sleep EEG: A multi-cohort, deep learning biomarker for cognition, disease and mortality.*

---

## License

**CC BY-NC 4.0** — Attribution-NonCommercial 4.0 International.  
See the `LICENSE` file for details.
