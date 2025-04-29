import os
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")  # Suppress Librosa warnings


def extract_features(file_path, sr=22050, n_mfcc=13, hop_length=512):
    """Extract audio features with error handling and extended stats."""
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # Compute features
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length)
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

        # Spectral Bandwidth and Contrast (additional features)
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length)
        spec_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=hop_length)

        # Harmonic/Percussive Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Avoid division by zero
        harmonic_ratio = np.mean(y_harmonic / (y + 1e-6))

        # Aggregate statistics (mean, std, skew)
        features = {
            'file_name': os.path.basename(file_path),
            'duration': librosa.get_duration(y=y, sr=sr),
            'tempo': tempo,
            'harmonic_ratio': harmonic_ratio,
            'spectral_centroid_mean': np.mean(spec_centroid),
            'spectral_centroid_std': np.std(spec_centroid),
            'spectral_rolloff_mean': np.mean(spec_rolloff),
            'spectral_rolloff_std': np.std(spec_rolloff),
            'zero_crossing_rate_mean': np.mean(zcr),
            'zero_crossing_rate_std': np.std(zcr),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_contrast_mean': np.mean(spec_contrast),
        }

        # Dynamic MFCC/Chroma stats
        for i in range(n_mfcc):
            features.update({
                f'mfcc_{i+1}_mean': np.mean(mfccs[i]),
                f'mfcc_{i+1}_std': np.std(mfccs[i]),
            })

        for i in range(chroma.shape[0]):
            features.update({
                f'chroma_{i+1}_mean': np.mean(chroma[i]),
                f'chroma_{i+1}_std': np.std(chroma[i]),
            })

        return features

    except Exception as e:
        print(f"\nError processing {file_path}: {str(e)}")
        return None


def main(audio_dir, sr=22050, n_mfcc=13, n_jobs=-1, output_csv='audio_features.csv'):
    """Process all audio files in parallel and save to CSV."""
    files = glob.glob(os.path.join(audio_dir, '*.wav')) + \
        glob.glob(os.path.join(audio_dir, '*.mp3'))
    print(f"Found {len(files)} audio files in {audio_dir}")

    # Parallel feature extraction
    all_features = Parallel(n_jobs=n_jobs)(
        delayed(extract_features)(f, sr=sr, n_mfcc=n_mfcc)
        for f in tqdm(files, desc="Extracting features")
    )

    # Remove failed extractions
    all_features = [f for f in all_features if f is not None]
    df = pd.DataFrame(all_features)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} records to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Advanced Audio Feature Extractor')
    parser.add_argument('audio_dir', type=str,
                        help='Directory containing audio files')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate (Hz)')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='CPU cores to use (-1 for all)')
    parser.add_argument(
        '--output', type=str, default='audio_features.csv', help='Output CSV filename')

    args = parser.parse_args()
    main(
        audio_dir=args.audio_dir,
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_jobs=args.n_jobs,
        output_csv=args.output
    )
