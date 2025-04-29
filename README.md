# Audio Feature Extractor ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

A Python tool for extracting acoustic features (MFCCs, Chroma, Spectral, and Temporal features) from audio files (`.wav`/`.mp3`) and exporting them to CSV for machine learning applications.

---

## Features

- **Comprehensive Feature Extraction**: MFCCs, Chroma, Spectral Centroid/Rolloff, Zero Crossing Rate, Tempo, Harmonic/Percussive separation.
- **Parallel Processing**: Utilizes `joblib` for faster extraction on large datasets.
- **Robust Output**: CSV format compatible with Pandas, Scikit-learn, and TensorFlow.
- **Error Resilient**: Skips corrupt files and logs errors.

## Installation and Run

```bash
pip install -r requirements.txt
```

```bash
python extract_features.py /path/to/audio_dir
```

## Advanced Options

| Argument   | Description                         | Default        |
| ---------- | ----------------------------------- | -------------- |
| `--sr`     | Sample rate (Hz)                    | `22050`        |
| `--n_mfcc` | Number of MFCC coefficients         | `13`           |
| `--n_jobs` | CPU cores to use (`-1` = all cores) | `-1`           |
| `--output` | Output CSV filename                 | `features.csv` |

## Example

```bash
python extract_features.py ./audio_samples --sr 44100 --n_mfcc 20 --output high_res_features.csv
```

## Output CSV Columns

| Feature Type | Columns                       | Description                             |
| ------------ | ----------------------------- | --------------------------------------- |
| **Metadata** | `file_name`, `duration`       | Filename and audio length (seconds)     |
| **MFCCs**    | `mfcc_{1-13}_mean/std`        | Mel-frequency cepstral coefficients     |
| **Chroma**   | `chroma_{1-12}_mean/std`      | Pitch class energies (12 semitones)     |
| **Spectral** | `spectral_centroid_mean/std`  | Spectral centroid frequency (Hz)        |
| **Temporal** | `tempo`, `zero_crossing_rate` | Beats-per-minute and Zero Crossing Rate |
