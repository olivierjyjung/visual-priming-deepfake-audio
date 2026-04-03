#!/usr/bin/env python3
"""
EEG Preprocessing Pipeline
===========================

Processes XDF format EEG data with standard preprocessing steps.

Usage:
    python eeg_analysis.py --data-dir /path/to/eeg --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# EEG processing libraries
try:
    import pyxdf
    import mne
    from scipy import signal, stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyxdf mne numpy pandas scipy matplotlib seaborn")
    HAS_DEPENDENCIES = False


# =============================================================================
# Configuration
# =============================================================================

# Default EEG parameters (can be overridden)
DEFAULT_SFREQ = 300  # Sampling frequency (Hz)
LOWCUT = 0.1         # High-pass filter cutoff (Hz)
HIGHCUT = 40         # Low-pass filter cutoff (Hz)
NOTCH_FREQ = 60      # Notch filter frequency (Hz) - use 50 for Korea/Europe

# Epoch parameters
EPOCH_TMIN = -0.2    # Pre-stimulus baseline (seconds)
EPOCH_TMAX = 0.8     # Post-stimulus window (seconds)
BASELINE = (-0.2, 0) # Baseline correction window


# =============================================================================
# Data Loading Functions
# =============================================================================

def find_xdf_files(data_dir: Path) -> List[Path]:
    """
    Find all XDF files in the data directory.
    Supports BIDS-style directory structure.
    """
    xdf_files = []

    # Check for BIDS structure
    for subject_dir in sorted(data_dir.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            eeg_dir = subject_dir / 'ses-S001' / 'eeg'
            if eeg_dir.exists():
                xdf_files.extend(eeg_dir.glob('*.xdf'))

    # Also check for flat structure
    if not xdf_files:
        xdf_files = list(data_dir.glob('**/*.xdf'))

    return sorted(xdf_files)


def load_xdf_data(xdf_path: Path) -> Tuple[Optional[np.ndarray], float, List, Optional[np.ndarray]]:
    """
    Load EEG data from XDF file.

    Returns:
        eeg_data: EEG data array (channels x samples)
        sfreq: Sampling frequency (Hz)
        markers: Event markers as list of (timestamp, marker_string) tuples
        eeg_times: EEG timestamps array
    """
    try:
        streams, header = pyxdf.load_xdf(str(xdf_path))

        eeg_data = None
        sfreq = DEFAULT_SFREQ
        markers = []
        eeg_times = None

        for stream in streams:
            stream_type = stream['info']['type'][0].lower()

            if 'eeg' in stream_type or stream['info']['type'][0] == 'EEG':
                eeg_data = np.array(stream['time_series']).T
                eeg_times = np.array(stream['time_stamps'])

                try:
                    sfreq = float(stream['info']['nominal_srate'][0])
                    if sfreq <= 0:
                        sfreq = DEFAULT_SFREQ
                except (KeyError, IndexError, ValueError, TypeError):
                    pass

            elif 'marker' in stream_type or 'event' in stream_type:
                markers = list(zip(
                    stream['time_stamps'],
                    [str(m[0]) if isinstance(m, list) else str(m) for m in stream['time_series']]
                ))

        return eeg_data, sfreq, markers, eeg_times

    except Exception as e:
        print(f"Error loading {xdf_path}: {e}")
        return None, DEFAULT_SFREQ, [], None


def extract_subject_id(xdf_path: Path) -> str:
    """Extract subject ID from file path."""
    for part in xdf_path.parts:
        if part.startswith('sub-'):
            return part
    return xdf_path.stem


# =============================================================================
# Preprocessing Functions
# =============================================================================

def create_mne_raw(data: np.ndarray, sfreq: float, ch_names: Optional[List[str]] = None) -> mne.io.RawArray:
    """
    Create MNE Raw object from numpy array.

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency
        ch_names: Channel names (optional)

    Returns:
        MNE Raw object
    """
    n_channels = data.shape[0]

    if ch_names is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    # Scale data to Volts (MNE expects Volts)
    data_volts = data * 1e-6

    raw = mne.io.RawArray(data_volts, info)
    return raw


def preprocess_eeg(raw: mne.io.RawArray,
                   lowcut: float = LOWCUT,
                   highcut: float = HIGHCUT,
                   notch_freq: float = NOTCH_FREQ) -> mne.io.RawArray:
    """
    Apply standard EEG preprocessing pipeline.

    Steps:
    1. Bandpass filter
    2. Notch filter (power line noise)
    3. Re-reference to average
    """
    raw = raw.copy()

    # Bandpass filter
    raw.filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin', verbose=False)

    # Notch filter for power line noise
    raw.notch_filter(freqs=notch_freq, verbose=False)

    # Re-reference to average
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()

    return raw


def detect_bad_channels(raw: mne.io.RawArray, threshold: float = 3.0) -> List[str]:
    """
    Detect bad channels based on variance.

    Args:
        raw: MNE Raw object
        threshold: Z-score threshold for marking bad channels

    Returns:
        List of bad channel names
    """
    data = raw.get_data()
    variances = np.var(data, axis=1)
    z_scores = stats.zscore(variances)
    bad_idx = np.where(np.abs(z_scores) > threshold)[0]
    bad_channels = [raw.ch_names[i] for i in bad_idx]
    return bad_channels


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_psd(raw: mne.io.RawArray, fmin: float = 0.5, fmax: float = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density.

    Returns:
        psds: Power spectral density array
        freqs: Frequency array
    """
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds = spectrum.get_data()
    freqs = spectrum.freqs
    return psds, freqs


def extract_band_power(psds: np.ndarray, freqs: np.ndarray,
                       band: Tuple[float, float]) -> float:
    """
    Extract power in a frequency band.

    Args:
        psds: Power spectral density array
        freqs: Frequency array
        band: (low, high) frequency bounds

    Returns:
        Mean power in the band
    """
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return np.mean(psds[:, idx])


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_subject(xdf_path: Path, output_dir: Path) -> Optional[Dict]:
    """
    Process a single subject's EEG data.

    Args:
        xdf_path: Path to XDF file
        output_dir: Directory for saving results

    Returns:
        Dictionary with subject results
    """
    subject_id = extract_subject_id(xdf_path)
    print(f"\nProcessing: {subject_id}")

    # Load data
    data, sfreq, markers, eeg_times = load_xdf_data(xdf_path)

    if data is None or eeg_times is None:
        print(f"  Skipping {subject_id}: Could not load data")
        return None

    print(f"  Data shape: {data.shape}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Markers found: {len(markers)}")

    if data.shape[1] == 0:
        print(f"  Skipping {subject_id}: Empty data file")
        return None

    # Create MNE Raw object
    raw = create_mne_raw(data, sfreq)

    # Detect bad channels
    bad_channels = detect_bad_channels(raw)
    if bad_channels:
        print(f"  Bad channels detected: {bad_channels}")
        raw.info['bads'] = bad_channels

    # Preprocess
    raw_preprocessed = preprocess_eeg(raw)

    # Compute PSD
    psds, freqs = compute_psd(raw_preprocessed)

    # Extract band powers
    bands = {
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    band_powers = {}
    for band_name, band_range in bands.items():
        band_powers[band_name] = extract_band_power(psds, freqs, band_range)

    # Store results
    results = {
        'subject_id': subject_id,
        'n_channels': data.shape[0],
        'n_samples': data.shape[1],
        'sfreq': sfreq,
        'duration_s': data.shape[1] / sfreq,
        'n_markers': len(markers),
        'n_bad_channels': len(bad_channels),
        **{f'{k}_power': v for k, v in band_powers.items()}
    }

    return results


def run_batch_processing(data_dir: Path, output_dir: Path):
    """
    Run batch processing on all subjects.
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    # Find all XDF files
    xdf_files = find_xdf_files(data_dir)
    print(f"Found {len(xdf_files)} XDF files")

    if not xdf_files:
        print(f"No XDF files found in {data_dir}")
        return None

    # Process each subject
    all_results = []

    for xdf_path in xdf_files:
        result = process_subject(xdf_path, output_dir)
        if result:
            all_results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    # Save summary
    summary_path = output_dir / 'eeg_processing_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total subjects processed: {len(df)}")
    print(f"Average duration: {df['duration_s'].mean():.1f} seconds")
    print(f"Subjects with bad channels: {(df['n_bad_channels'] > 0).sum()}")

    return df


# =============================================================================
# Entry Point
# =============================================================================

def main(data_dir: Path, output_dir: Path):
    """Main entry point."""
    print("=" * 60)
    print("EEG Preprocessing Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    if not HAS_DEPENDENCIES:
        print("\nPlease install required dependencies and try again.")
        return

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        return

    results_df = run_batch_processing(data_dir, output_dir)

    print("\nAnalysis complete!")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG preprocessing pipeline for Visual Priming & Deepfake Audio study"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="Directory containing EEG data files (XDF format)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Directory for output files (default: results/)"
    )

    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
