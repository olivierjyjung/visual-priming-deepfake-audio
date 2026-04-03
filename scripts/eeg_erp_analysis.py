#!/usr/bin/env python3
"""
EEG ERP Analysis Script
========================

Performs event-related potential (ERP) analysis comparing Treatment vs Neutral conditions.

Components analyzed:
- MMN (Mismatch Negativity): 100-250ms
- N250: 200-300ms (Voice familiarity processing)
- P300: 250-500ms
- LPP (Late Positive Potential): 400-700ms

Usage:
    python eeg_erp_analysis.py --data-dir /path/to/eeg --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import pyxdf
    import mne
    from scipy import stats
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

# EEG parameters
DEFAULT_SFREQ = 300
LOWCUT = 0.1
HIGHCUT = 40
NOTCH_FREQ = 60

# Epoch parameters
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
BASELINE = (-0.2, 0)

# ERP windows (in seconds)
ERP_WINDOWS = {
    'MMN': (0.100, 0.250),
    'N250': (0.200, 0.300),
    'P300': (0.250, 0.500),
    'N400': (0.300, 0.500),
    'LPP': (0.400, 0.700),
}

# Channel regions (generic 24-channel layout)
FRONTAL_CHANNELS = ['EEG001', 'EEG002', 'EEG003', 'EEG004', 'EEG005', 'EEG006']
CENTRAL_CHANNELS = ['EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012']
PARIETAL_CHANNELS = ['EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018']

COMPONENT_CHANNELS = {
    'MMN': FRONTAL_CHANNELS,
    'N250': FRONTAL_CHANNELS + CENTRAL_CHANNELS,
    'P300': PARIETAL_CHANNELS,
    'N400': CENTRAL_CHANNELS + PARIETAL_CHANNELS,
    'LPP': CENTRAL_CHANNELS + PARIETAL_CHANNELS,
}


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


def load_xdf_data(xdf_path: Path):
    """Load EEG data from XDF file."""
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
                except:
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
# Preprocessing
# =============================================================================

def create_mne_raw(data: np.ndarray, sfreq: float) -> mne.io.RawArray:
    """Create MNE Raw object from numpy array."""
    n_channels = data.shape[0]
    ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    data_volts = data * 1e-6

    return mne.io.RawArray(data_volts, info)


def preprocess_eeg(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Apply standard EEG preprocessing."""
    raw = raw.copy()
    raw.filter(l_freq=LOWCUT, h_freq=HIGHCUT, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()
    return raw


# =============================================================================
# ERP Analysis
# =============================================================================

def extract_erp_amplitude(data: np.ndarray, times: np.ndarray,
                          time_window: Tuple[float, float],
                          channels: List[int] = None) -> Dict[str, float]:
    """
    Extract ERP amplitude and latency from a time window.

    Args:
        data: EEG data (channels x samples)
        times: Time array
        time_window: (start, end) in seconds
        channels: Channel indices to use

    Returns:
        Dictionary with mean amplitude, peak amplitude, and peak latency
    """
    tmin, tmax = time_window
    idx_start = np.argmin(np.abs(times - tmin))
    idx_end = np.argmin(np.abs(times - tmax))

    if channels is not None:
        data = data[channels, :]

    window_data = data[:, idx_start:idx_end]
    window_times = times[idx_start:idx_end]

    # Average across channels
    avg_data = np.mean(window_data, axis=0)

    # Mean amplitude
    mean_amp = np.mean(avg_data) * 1e6  # Convert to µV

    # Peak amplitude and latency
    peak_idx = np.argmax(np.abs(avg_data))
    peak_amp = avg_data[peak_idx] * 1e6
    peak_latency = window_times[peak_idx] * 1000  # Convert to ms

    return {
        'mean_amplitude': mean_amp,
        'peak_amplitude': peak_amp,
        'peak_latency': peak_latency
    }


def analyze_erp_components(raw: mne.io.RawArray, sfreq: float) -> Dict:
    """
    Analyze all ERP components for a subject.

    Returns:
        Dictionary with component metrics
    """
    data = raw.get_data()
    times = np.arange(data.shape[1]) / sfreq

    results = {}

    for component, window in ERP_WINDOWS.items():
        # Get channel indices for this component
        component_ch = COMPONENT_CHANNELS.get(component, None)
        if component_ch:
            ch_indices = [i for i, ch in enumerate(raw.ch_names) if ch in component_ch]
        else:
            ch_indices = None

        erp_metrics = extract_erp_amplitude(data, times, window, ch_indices)

        results[f'{component}_mean_amp'] = erp_metrics['mean_amplitude']
        results[f'{component}_peak_amp'] = erp_metrics['peak_amplitude']
        results[f'{component}_peak_lat'] = erp_metrics['peak_latency']

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================

def compare_conditions(treatment_data: List[Dict], neutral_data: List[Dict]) -> Dict:
    """
    Compare ERP components between conditions using t-tests.

    Returns:
        Dictionary with statistical results for each component
    """
    results = {}

    for component in ERP_WINDOWS.keys():
        key = f'{component}_mean_amp'

        treat_vals = [d[key] for d in treatment_data if key in d]
        neut_vals = [d[key] for d in neutral_data if key in d]

        if len(treat_vals) < 3 or len(neut_vals) < 3:
            continue

        # t-test
        t_stat, p_val = stats.ttest_ind(treat_vals, neut_vals)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(treat_vals) - 1) * np.var(treat_vals) +
             (len(neut_vals) - 1) * np.var(neut_vals)) /
            (len(treat_vals) + len(neut_vals) - 2)
        )
        cohens_d = (np.mean(treat_vals) - np.mean(neut_vals)) / pooled_std if pooled_std > 0 else 0

        results[component] = {
            'treatment_mean': np.mean(treat_vals),
            'treatment_sd': np.std(treat_vals),
            'neutral_mean': np.mean(neut_vals),
            'neutral_sd': np.std(neut_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_treatment': len(treat_vals),
            'n_neutral': len(neut_vals)
        }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_erp_comparison(treatment_data: List[Dict], neutral_data: List[Dict],
                        output_dir: Path):
    """Plot ERP component comparison between conditions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, component in enumerate(ERP_WINDOWS.keys()):
        if idx >= len(axes):
            break

        key = f'{component}_mean_amp'

        treat_vals = [d[key] for d in treatment_data if key in d]
        neut_vals = [d[key] for d in neutral_data if key in d]

        ax = axes[idx]

        positions = [1, 2]
        data = [treat_vals, neut_vals]

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')

        ax.set_xticklabels(['Treatment', 'Neutral'])
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'{component} Component')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Hide empty subplots
    for idx in range(len(ERP_WINDOWS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('ERP Component Comparison: Treatment vs Neutral', fontsize=14)
    plt.tight_layout()

    save_path = output_dir / 'figures' / 'erp_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def get_condition(subject_id: str) -> str:
    """Determine condition based on subject ID pattern."""
    import re
    match = re.search(r'sub-\d+_\d+_\d+_(\d+)_\d+', subject_id)
    if match:
        subj_num = int(match.group(1))
        return 'Treatment' if subj_num <= 8 else 'Neutral'
    return 'Unknown'


def process_all_subjects(data_dir: Path, output_dir: Path):
    """Process all subjects and compare conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    xdf_files = find_xdf_files(data_dir)
    print(f"Found {len(xdf_files)} XDF files")

    if not xdf_files:
        print(f"No XDF files found in {data_dir}")
        return

    treatment_results = []
    neutral_results = []
    all_results = []

    for xdf_path in xdf_files:
        subject_id = extract_subject_id(xdf_path)
        print(f"\nProcessing: {subject_id}")

        data, sfreq, markers, eeg_times = load_xdf_data(xdf_path)

        if data is None:
            continue

        raw = create_mne_raw(data, sfreq)
        raw = preprocess_eeg(raw)

        erp_results = analyze_erp_components(raw, sfreq)
        erp_results['subject_id'] = subject_id
        erp_results['condition'] = get_condition(subject_id)

        all_results.append(erp_results)

        if erp_results['condition'] == 'Treatment':
            treatment_results.append(erp_results)
        elif erp_results['condition'] == 'Neutral':
            neutral_results.append(erp_results)

    # Save individual results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'eeg_erp_results.csv', index=False)
    print(f"\nResults saved: {output_dir / 'eeg_erp_results.csv'}")

    # Statistical comparison
    if treatment_results and neutral_results:
        stats_results = compare_conditions(treatment_results, neutral_results)

        print("\n" + "=" * 60)
        print("ERP COMPONENT COMPARISON: Treatment vs Neutral")
        print("=" * 60)

        for component, stats in stats_results.items():
            print(f"\n{component}:")
            print(f"  Treatment: M = {stats['treatment_mean']:.2f}, SD = {stats['treatment_sd']:.2f}")
            print(f"  Neutral:   M = {stats['neutral_mean']:.2f}, SD = {stats['neutral_sd']:.2f}")
            print(f"  t({stats['n_treatment'] + stats['n_neutral'] - 2}) = {stats['t_statistic']:.2f}, p = {stats['p_value']:.4f}")
            print(f"  Cohen's d = {stats['cohens_d']:.2f}")

        # Plot comparison
        plot_erp_comparison(treatment_results, neutral_results, output_dir)

    return df


def main(data_dir: Path, output_dir: Path):
    """Main entry point."""
    print("=" * 60)
    print("EEG ERP Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    if not HAS_DEPENDENCIES:
        print("\nPlease install required dependencies and try again.")
        return

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        return

    results = process_all_subjects(data_dir, output_dir)
    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG ERP analysis for Visual Priming & Deepfake Audio study"
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
