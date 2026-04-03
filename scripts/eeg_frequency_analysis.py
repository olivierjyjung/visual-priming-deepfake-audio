#!/usr/bin/env python3
"""
EEG Frequency Band Analysis
============================

Analyzes power spectral density in different frequency bands.

Frequency bands:
- Delta (1-4 Hz)
- Theta (4-8 Hz)
- Alpha (8-13 Hz)
- SMR (12-15 Hz)
- Beta (13-30 Hz)

Usage:
    python eeg_frequency_analysis.py --data-dir /path/to/eeg --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import pyxdf
    import mne
    from scipy import stats
    from scipy.signal import welch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    HAS_DEPENDENCIES = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SFREQ = 300
LOWPASS = 40
HIGHPASS = 0.1

FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'smr': (12, 15),
    'beta': (13, 30),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
}

# Channel groups
LEFT_FRONTAL = ['EEG001', 'EEG002', 'EEG003']
RIGHT_FRONTAL = ['EEG004', 'EEG005', 'EEG006']
CENTRAL_CHANNELS = ['EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012']
PARIETAL_CHANNELS = ['EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018']


# =============================================================================
# Data Loading
# =============================================================================

def find_xdf_files(data_dir: Path) -> List[Path]:
    """Find all XDF files in the data directory."""
    xdf_files = []
    for subject_dir in sorted(data_dir.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            eeg_dir = subject_dir / 'ses-S001' / 'eeg'
            if eeg_dir.exists():
                xdf_files.extend(eeg_dir.glob('*.xdf'))
    if not xdf_files:
        xdf_files = list(data_dir.glob('**/*.xdf'))
    return sorted(xdf_files)


def load_xdf_data(filepath: Path):
    """Load XDF file and extract EEG data."""
    try:
        streams, header = pyxdf.load_xdf(str(filepath))

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
                except:
                    pass

            elif 'marker' in stream_type or 'event' in stream_type:
                markers = [{'time': t, 'value': str(m[0]) if isinstance(m, list) else str(m)}
                           for t, m in zip(stream['time_stamps'], stream['time_series'])]

        return eeg_data, sfreq, markers, eeg_times

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, DEFAULT_SFREQ, [], None


def extract_subject_id(xdf_path: Path) -> str:
    """Extract subject ID from file path."""
    for part in xdf_path.parts:
        if part.startswith('sub-'):
            return part
    return xdf_path.stem


def get_condition(subject_id: str) -> str:
    """Determine condition based on subject ID pattern."""
    match = re.search(r'sub-\d+_\d+_\d+_(\d+)_\d+', subject_id)
    if match:
        subj_num = int(match.group(1))
        return 'Treatment' if subj_num <= 8 else 'Neutral'
    return 'Unknown'


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Apply basic preprocessing to EEG data.

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency

    Returns:
        Filtered data
    """
    n_channels = data.shape[0]
    ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data * 1e-6, info)

    raw.filter(l_freq=HIGHPASS, h_freq=LOWPASS, fir_design='firwin', verbose=False)

    return raw.get_data()


# =============================================================================
# Frequency Analysis
# =============================================================================

def compute_band_power(data: np.ndarray, sfreq: float,
                       band: Tuple[float, float]) -> float:
    """
    Compute power in a frequency band using Welch's method.

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency
        band: (low, high) frequency bounds

    Returns:
        Mean band power across channels
    """
    powers = []

    for ch_data in data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))

        # Find frequency indices
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

        if len(idx) > 0:
            band_power = np.mean(psd[idx])
            powers.append(band_power)

    return np.mean(powers) if powers else np.nan


def compute_relative_power(data: np.ndarray, sfreq: float,
                           band: Tuple[float, float],
                           total_band: Tuple[float, float] = (1, 40)) -> float:
    """
    Compute relative band power (band power / total power).

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency
        band: Target frequency band
        total_band: Total frequency range for normalization

    Returns:
        Relative band power
    """
    band_power = compute_band_power(data, sfreq, band)
    total_power = compute_band_power(data, sfreq, total_band)

    if total_power > 0:
        return band_power / total_power
    return np.nan


def compute_frontal_alpha_asymmetry(data: np.ndarray, sfreq: float,
                                     left_idx: List[int],
                                     right_idx: List[int]) -> float:
    """
    Compute frontal alpha asymmetry (FAA).

    FAA = ln(right alpha) - ln(left alpha)
    Positive values indicate greater left frontal activity (approach motivation)

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency
        left_idx: Left frontal channel indices
        right_idx: Right frontal channel indices

    Returns:
        Frontal alpha asymmetry score
    """
    alpha_band = FREQ_BANDS['alpha']

    left_power = compute_band_power(data[left_idx, :], sfreq, alpha_band)
    right_power = compute_band_power(data[right_idx, :], sfreq, alpha_band)

    if left_power > 0 and right_power > 0:
        return np.log(right_power) - np.log(left_power)
    return np.nan


def analyze_subject(data: np.ndarray, sfreq: float) -> Dict:
    """
    Perform full frequency analysis for a subject.

    Returns:
        Dictionary with all frequency metrics
    """
    results = {}

    # Absolute band powers
    for band_name, band_range in FREQ_BANDS.items():
        results[f'{band_name}_power'] = compute_band_power(data, sfreq, band_range)

    # Relative band powers
    for band_name, band_range in FREQ_BANDS.items():
        results[f'{band_name}_relative'] = compute_relative_power(data, sfreq, band_range)

    # Frontal alpha asymmetry
    n_channels = data.shape[0]
    if n_channels >= 6:
        left_idx = [0, 1, 2]
        right_idx = [3, 4, 5]
        results['frontal_alpha_asymmetry'] = compute_frontal_alpha_asymmetry(
            data, sfreq, left_idx, right_idx
        )

    # Theta/Beta ratio (attention index)
    if results.get('theta_power') and results.get('beta_power'):
        results['theta_beta_ratio'] = results['theta_power'] / results['beta_power']

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================

def compare_conditions(treatment_data: List[Dict], neutral_data: List[Dict]) -> Dict:
    """Compare frequency metrics between conditions."""
    results = {}

    # Get all metric keys
    all_keys = set()
    for d in treatment_data + neutral_data:
        all_keys.update(d.keys())

    numeric_keys = [k for k in all_keys if k not in ['subject_id', 'condition']]

    for key in numeric_keys:
        treat_vals = [d[key] for d in treatment_data if key in d and not np.isnan(d[key])]
        neut_vals = [d[key] for d in neutral_data if key in d and not np.isnan(d[key])]

        if len(treat_vals) < 3 or len(neut_vals) < 3:
            continue

        t_stat, p_val = stats.ttest_ind(treat_vals, neut_vals)

        pooled_std = np.sqrt(
            ((len(treat_vals) - 1) * np.var(treat_vals) +
             (len(neut_vals) - 1) * np.var(neut_vals)) /
            (len(treat_vals) + len(neut_vals) - 2)
        )
        cohens_d = (np.mean(treat_vals) - np.mean(neut_vals)) / pooled_std if pooled_std > 0 else 0

        results[key] = {
            'treatment_mean': np.mean(treat_vals),
            'treatment_sd': np.std(treat_vals),
            'neutral_mean': np.mean(neut_vals),
            'neutral_sd': np.std(neut_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d
        }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_band_power_comparison(treatment_data: List[Dict], neutral_data: List[Dict],
                                output_dir: Path):
    """Plot frequency band power comparison."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    bands = list(FREQ_BANDS.keys())

    for idx, band in enumerate(bands):
        if idx >= len(axes):
            break

        key = f'{band}_relative'

        treat_vals = [d[key] for d in treatment_data if key in d and not np.isnan(d[key])]
        neut_vals = [d[key] for d in neutral_data if key in d and not np.isnan(d[key])]

        ax = axes[idx]

        data = [treat_vals, neut_vals]
        bp = ax.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')

        ax.set_xticklabels(['Treatment', 'Neutral'])
        ax.set_ylabel('Relative Power')
        ax.set_title(f'{band.capitalize()} Band ({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz)')

    # Hide extra subplots
    for idx in range(len(bands), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Frequency Band Power: Treatment vs Neutral', fontsize=14)
    plt.tight_layout()

    save_path = output_dir / 'figures' / 'frequency_band_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def process_all_subjects(data_dir: Path, output_dir: Path):
    """Process all subjects and compare conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    xdf_files = find_xdf_files(data_dir)
    print(f"Found {len(xdf_files)} XDF files")

    if not xdf_files:
        return

    treatment_results = []
    neutral_results = []
    all_results = []

    for xdf_path in xdf_files:
        subject_id = extract_subject_id(xdf_path)
        print(f"Processing: {subject_id}")

        data, sfreq, markers, eeg_times = load_xdf_data(xdf_path)

        if data is None:
            continue

        # Preprocess
        data = preprocess_eeg(data, sfreq)

        # Analyze
        freq_results = analyze_subject(data, sfreq)
        freq_results['subject_id'] = subject_id
        freq_results['condition'] = get_condition(subject_id)

        all_results.append(freq_results)

        if freq_results['condition'] == 'Treatment':
            treatment_results.append(freq_results)
        elif freq_results['condition'] == 'Neutral':
            neutral_results.append(freq_results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'eeg_frequency_results.csv', index=False)
    print(f"\nResults saved: {output_dir / 'eeg_frequency_results.csv'}")

    # Statistical comparison
    if treatment_results and neutral_results:
        stats_results = compare_conditions(treatment_results, neutral_results)

        print("\n" + "=" * 60)
        print("FREQUENCY ANALYSIS: Treatment vs Neutral")
        print("=" * 60)

        for metric, stats in stats_results.items():
            if stats['p_value'] < 0.1:  # Show trending or significant results
                print(f"\n{metric}:")
                print(f"  Treatment: M = {stats['treatment_mean']:.4f}")
                print(f"  Neutral:   M = {stats['neutral_mean']:.4f}")
                print(f"  t = {stats['t_statistic']:.2f}, p = {stats['p_value']:.4f}, d = {stats['cohens_d']:.2f}")

        # Plot
        plot_band_power_comparison(treatment_results, neutral_results, output_dir)

    return df


def main(data_dir: Path, output_dir: Path):
    """Main entry point."""
    print("=" * 60)
    print("EEG Frequency Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    if not HAS_DEPENDENCIES:
        print("\nPlease install required dependencies.")
        return

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        return

    results = process_all_subjects(data_dir, output_dir)
    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG frequency analysis for Visual Priming study"
    )
    parser.add_argument("--data-dir", "-d", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results"))

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
