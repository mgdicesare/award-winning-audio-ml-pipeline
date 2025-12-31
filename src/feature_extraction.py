"""
Feature extraction script for cepstral analysis of speech signals.

This script extracts Gammatone Frequency Cepstral Coefficients (GTCC)
and their temporal derivatives (ΔGTCC) from audio files, following
the methodology described in the associated research study.

The output is a CSV file containing aggregated cepstral features
(mean over time) and a class label.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from gammatone.gtgram import gtgram


def extract_gtcc(
    signal: np.ndarray,
    sr: int,
    n_filters: int = 64,
    n_coeffs: int = 13,
    window_time: float = 0.025,
    hop_time: float = 0.010,
) -> np.ndarray:
    """
    Extract GTCC features from an audio signal.

    Parameters
    ----------
    signal : np.ndarray
        Audio time series.
    sr : int
        Sampling rate.
    n_filters : int
        Number of gammatone filters.
    n_coeffs : int
        Number of cepstral coefficients to retain.
    window_time : float
        Analysis window length in seconds.
    hop_time : float
        Hop size in seconds.

    Returns
    -------
    gtcc : np.ndarray
        GTCC matrix (n_coeffs x frames).
    """
    gtg = gtgram(
        signal,
        sr,
        window_time,
        hop_time,
        n_filters,
        fmin=50,
    )

    log_gtg = np.log(gtg + np.finfo(float).eps)
    gtcc = librosa.feature.mfcc(S=log_gtg, n_mfcc=n_coeffs)

    return gtcc


def aggregate_features(gtcc: np.ndarray) -> np.ndarray:
    """
    Aggregate cepstral features by computing the mean over time.

    Parameters
    ----------
    gtcc : np.ndarray
        Cepstral feature matrix.

    Returns
    -------
    features : np.ndarray
        Aggregated feature vector.
    """
    delta = librosa.feature.delta(gtcc)

    gtcc_mean = np.mean(gtcc, axis=1)
    delta_mean = np.mean(delta, axis=1)

    features = np.concatenate([gtcc_mean, delta_mean])
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract GTCC and ΔGTCC features from a directory of WAV files."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--sr", type=int, default=16000)

    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.input_dir, "*.wav"))
    if len(wav_files) == 0:
        raise RuntimeError("No WAV files found in the specified directory.")

    data = []

    for wav_path in tqdm(wav_files, desc="Extracting features"):
        try:
            signal, sr = librosa.load(wav_path, sr=args.sr)
            gtcc = extract_gtcc(signal, sr)
            features = aggregate_features(gtcc)
            row = np.concatenate([features, [args.label]])
            data.append(row)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    n_features = data[0].shape[0] - 1
    columns = (
        [f"GTCC_{i+1}" for i in range(n_features // 2)]
        + [f"DeltaGTCC_{i+1}" for i in range(n_features // 2)]
        + ["Label"]
    )

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved features to {args.output_csv}")


if __name__ == "__main__":
    main()
