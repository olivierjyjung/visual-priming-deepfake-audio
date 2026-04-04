#!/usr/bin/env python3
"""
EEG ERP Analysis Script
========================

Performs event-related potential (ERP) analysis comparing Treatment vs Neutral conditions.

Components analyzed:
- MMN (Mismatch Negativity): 100-250ms
- N250: 200-300ms (Voice familiarity processing)
- Frontal N2: 200-350ms (Conflict monitoring)
- P300: 250-500ms
- LPP (Late Positive Potential): 400-700ms

Preprocessing includes:
- Bandpass filtering (0.1-40 Hz)
- Notch filtering (60 Hz)
- Epoching around stimulus onset
- Artifact rejection (±100μV threshold)
- ICA-based eye artifact removal
- Bad epoch rejection

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
    from mne.preprocessing import ICA, create_eog_epochs
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

# Artifact rejection thresholds
# Note: Using 300μV for dry electrode system (DSI-24), consistent with validated threshold
# from prior DSI-24 studies (see ICA 2026 submission using same equipment)
REJECT_CRITERIA = {
    'eeg': 300e-6  # ±300μV threshold (validated for DSI-24 dry electrodes)
}
FLAT_CRITERIA = {
    'eeg': 1e-6   # Flat signal threshold
}

# ICA parameters
ICA_N_COMPONENTS = 15
ICA_RANDOM_STATE = 42

# ERP windows (in seconds)
ERP_WINDOWS = {
    'MMN': (0.100, 0.250),
    'N250': (0.200, 0.300),
    'Frontal_N2': (0.200, 0.350),
    'P300': (0.250, 0.500),
    'LPP': (0.400, 0.700),
}

# Channel regions (generic 24-channel layout)
FRONTAL_CHANNELS = ['EEG001', 'EEG002', 'EEG003', 'EEG004', 'EEG005', 'EEG006']
CENTRAL_CHANNELS = ['EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012']
TEMPORAL_CHANNELS = ['EEG007', 'EEG008', 'EEG011', 'EEG012']  # Approximate temporal
PARIETAL_CHANNELS = ['EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018']

COMPONENT_CHANNELS = {
    'MMN': FRONTAL_CHANNELS,
    'N250': list(set(TEMPORAL_CHANNELS + CENTRAL_CHANNELS)),  # Temporal for voice processing (deduplicated)
    'Frontal_N2': FRONTAL_CHANNELS,  # Frontal for conflict monitoring
    'P300': PARIETAL_CHANNELS,
    'LPP': list(set(CENTRAL_CHANNELS + PARIETAL_CHANNELS)),  # Deduplicated
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
# Preprocessing with Artifact Rejection
# =============================================================================

def create_mne_raw(data: np.ndarray, sfreq: float) -> mne.io.RawArray:
    """Create MNE Raw object from numpy array."""
    n_channels = data.shape[0]
    ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    data_volts = data * 1e-6

    return mne.io.RawArray(data_volts, info, verbose=False)


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


def preprocess_continuous(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    Apply preprocessing to continuous data before epoching.

    Steps:
    1. Bandpass filter (0.1-40 Hz)
    2. Notch filter (60 Hz)
    3. Detect and interpolate bad channels
    """
    raw = raw.copy()

    # Bandpass filter
    raw.filter(l_freq=LOWCUT, h_freq=HIGHCUT, fir_design='firwin', verbose=False)

    # Notch filter for power line noise
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    # Detect bad channels
    bad_channels = detect_bad_channels(raw)
    if bad_channels:
        print(f"    Bad channels detected: {bad_channels}")
        raw.info['bads'] = bad_channels
        # Interpolate bad channels if montage is available
        try:
            raw.interpolate_bads(verbose=False)
        except:
            pass  # Skip interpolation if no montage

    return raw


def apply_ica_artifact_removal(raw: mne.io.RawArray, n_components: int = ICA_N_COMPONENTS) -> mne.io.RawArray:
    """
    Apply ICA to remove eye artifacts.

    Args:
        raw: Preprocessed MNE Raw object
        n_components: Number of ICA components

    Returns:
        Raw object with artifacts removed
    """
    raw = raw.copy()

    # Fit ICA
    ica = ICA(n_components=n_components, random_state=ICA_RANDOM_STATE, verbose=False)

    try:
        ica.fit(raw, verbose=False)

        # Find EOG-related components using variance-based detection
        # (since we don't have dedicated EOG channels)
        # Components with high frontal activity are likely eye artifacts

        # Get ICA sources
        sources = ica.get_sources(raw).get_data()

        # Find components with high correlation to frontal channels
        frontal_idx = [i for i, ch in enumerate(raw.ch_names) if ch in FRONTAL_CHANNELS]
        if frontal_idx:
            frontal_data = raw.get_data()[frontal_idx, :]

            exclude_components = []
            for comp_idx in range(sources.shape[0]):
                # Check correlation with frontal activity
                for f_idx in range(frontal_data.shape[0]):
                    corr = np.abs(np.corrcoef(sources[comp_idx, :], frontal_data[f_idx, :])[0, 1])
                    if corr > 0.7:  # High correlation threshold
                        if comp_idx not in exclude_components:
                            exclude_components.append(comp_idx)
                        break

            # Limit to max 2 components to avoid over-removal
            ica.exclude = exclude_components[:2]

            if ica.exclude:
                print(f"    ICA: Excluding components {ica.exclude}")
                raw = ica.apply(raw, verbose=False)

    except Exception as e:
        print(f"    ICA failed: {e}")

    return raw


def create_epochs(raw: mne.io.RawArray, markers: List, eeg_times: np.ndarray,
                  tmin: float = EPOCH_TMIN, tmax: float = EPOCH_TMAX) -> Optional[mne.Epochs]:
    """
    Create epochs from continuous data, time-locked to stimulus onset.

    Args:
        raw: Preprocessed MNE Raw object
        markers: List of (timestamp, marker_string) tuples
        eeg_times: EEG timestamp array
        tmin: Epoch start time relative to event (seconds)
        tmax: Epoch end time relative to event (seconds)

    Returns:
        MNE Epochs object or None if no valid epochs
    """
    if not markers or eeg_times is None:
        print("    No markers found for epoching")
        return None

    # Filter for audio stimulus onset markers
    # Adjust marker names based on your experiment's marker scheme
    audio_markers = ['audio_start', 'stimulus_onset', 'audio', 'stim', 'test_audio']

    events_list = []
    for timestamp, marker in markers:
        # Check if this is an audio onset marker
        marker_lower = marker.lower()
        is_audio_marker = any(m in marker_lower for m in audio_markers)

        # Also accept numeric markers or any marker if no specific audio markers found
        if is_audio_marker or marker.isdigit():
            # Find closest sample index
            sample_idx = np.argmin(np.abs(eeg_times - timestamp))
            events_list.append([sample_idx, 0, 1])  # [sample, 0, event_id]

    if not events_list:
        # If no specific markers found, use all markers
        print("    No audio markers found, using all markers")
        for timestamp, marker in markers:
            sample_idx = np.argmin(np.abs(eeg_times - timestamp))
            events_list.append([sample_idx, 0, 1])

    if not events_list:
        print("    No events found for epoching")
        return None

    events = np.array(events_list)
    print(f"    Found {len(events)} events for epoching")

    # Create epochs with artifact rejection
    try:
        epochs = mne.Epochs(
            raw, events,
            event_id={'stimulus': 1},
            tmin=tmin, tmax=tmax,
            baseline=BASELINE,
            reject=REJECT_CRITERIA,
            flat=FLAT_CRITERIA,
            preload=True,
            verbose=False
        )

        # Report rejection statistics
        n_original = len(events)
        n_kept = len(epochs)
        n_rejected = n_original - n_kept
        rejection_rate = (n_rejected / n_original) * 100 if n_original > 0 else 0

        print(f"    Epochs: {n_kept}/{n_original} kept ({rejection_rate:.1f}% rejected)")

        if n_kept < 3:
            print("    Warning: Too few epochs remaining after rejection")
            return None

        return epochs

    except Exception as e:
        print(f"    Epoching failed: {e}")
        return None


def reject_bad_epochs_by_variance(epochs: mne.Epochs, threshold: float = 3.0) -> mne.Epochs:
    """
    Additional epoch rejection based on variance across epochs.

    Args:
        epochs: MNE Epochs object
        threshold: Z-score threshold for rejection

    Returns:
        Epochs with bad epochs dropped
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Compute variance for each epoch
    epoch_vars = np.var(data, axis=(1, 2))  # Variance across channels and time
    z_scores = stats.zscore(epoch_vars)

    # Find bad epochs
    bad_epochs = np.where(np.abs(z_scores) > threshold)[0]

    if len(bad_epochs) > 0:
        print(f"    Variance-based rejection: {len(bad_epochs)} additional epochs")
        epochs = epochs.drop(bad_epochs, verbose=False)

    return epochs


# =============================================================================
# ERP Analysis
# =============================================================================

def extract_erp_amplitude(epochs: mne.Epochs,
                          time_window: Tuple[float, float],
                          channel_names: List[str] = None) -> Dict[str, float]:
    """
    Extract ERP amplitude and latency from epochs.

    Args:
        epochs: MNE Epochs object
        time_window: (start, end) in seconds
        channel_names: Channel names to use (None = all)

    Returns:
        Dictionary with mean amplitude, peak amplitude, and peak latency
    """
    tmin, tmax = time_window

    # Pick channels
    if channel_names:
        available_channels = [ch for ch in channel_names if ch in epochs.ch_names]
        if available_channels:
            epochs_subset = epochs.copy().pick_channels(available_channels, verbose=False)
        else:
            epochs_subset = epochs
    else:
        epochs_subset = epochs

    # Get evoked (averaged) response
    evoked = epochs_subset.average()

    # Crop to time window
    evoked_window = evoked.copy().crop(tmin=tmin, tmax=tmax)

    # Get data
    data = evoked_window.get_data()  # (n_channels, n_times)
    times = evoked_window.times

    # Average across channels
    avg_data = np.mean(data, axis=0)

    # Mean amplitude in window (in μV)
    mean_amp = np.mean(avg_data) * 1e6

    # Peak amplitude and latency
    # For negative components (MMN, N250, N2), find negative peak
    # For positive components (P300, LPP), find positive peak
    neg_peak_idx = np.argmin(avg_data)
    pos_peak_idx = np.argmax(avg_data)

    neg_peak_amp = avg_data[neg_peak_idx] * 1e6
    pos_peak_amp = avg_data[pos_peak_idx] * 1e6

    neg_peak_latency = times[neg_peak_idx] * 1000  # ms
    pos_peak_latency = times[pos_peak_idx] * 1000  # ms

    return {
        'mean_amplitude': mean_amp,
        'neg_peak_amplitude': neg_peak_amp,
        'pos_peak_amplitude': pos_peak_amp,
        'neg_peak_latency': neg_peak_latency,
        'pos_peak_latency': pos_peak_latency,
        'n_epochs': len(epochs_subset)
    }


def analyze_erp_components(epochs: mne.Epochs) -> Dict:
    """
    Analyze all ERP components from epochs.

    Returns:
        Dictionary with component metrics
    """
    results = {}

    for component, window in ERP_WINDOWS.items():
        # Get channel names for this component
        component_ch = COMPONENT_CHANNELS.get(component, None)

        erp_metrics = extract_erp_amplitude(epochs, window, component_ch)

        # Store appropriate peak based on component polarity
        if component in ['MMN', 'N250', 'Frontal_N2']:
            # Negative components
            results[f'{component}_mean_amp'] = erp_metrics['mean_amplitude']
            results[f'{component}_peak_amp'] = erp_metrics['neg_peak_amplitude']
            results[f'{component}_peak_lat'] = erp_metrics['neg_peak_latency']
        else:
            # Positive components (P300, LPP)
            results[f'{component}_mean_amp'] = erp_metrics['mean_amplitude']
            results[f'{component}_peak_amp'] = erp_metrics['pos_peak_amplitude']
            results[f'{component}_peak_lat'] = erp_metrics['pos_peak_latency']

        results[f'{component}_n_epochs'] = erp_metrics['n_epochs']

    return results


def compute_signal_variance(epochs: mne.Epochs) -> float:
    """
    Compute within-trial signal variance as a measure of neural dynamics.

    Args:
        epochs: MNE Epochs object

    Returns:
        Mean signal variance across epochs
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Compute variance within each epoch (across time), then average
    epoch_variances = np.var(data, axis=2)  # (n_epochs, n_channels)
    mean_variance = np.mean(epoch_variances)

    return mean_variance


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

        treat_vals = [d[key] for d in treatment_data if key in d and not np.isnan(d[key])]
        neut_vals = [d[key] for d in neutral_data if key in d and not np.isnan(d[key])]

        if len(treat_vals) < 3 or len(neut_vals) < 3:
            continue

        # t-test
        t_stat, p_val = stats.ttest_ind(treat_vals, neut_vals)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(treat_vals) - 1) * np.var(treat_vals, ddof=1) +
             (len(neut_vals) - 1) * np.var(neut_vals, ddof=1)) /
            (len(treat_vals) + len(neut_vals) - 2)
        )
        cohens_d = (np.mean(treat_vals) - np.mean(neut_vals)) / pooled_std if pooled_std > 0 else 0

        results[component] = {
            'treatment_mean': np.mean(treat_vals),
            'treatment_sd': np.std(treat_vals, ddof=1),
            'neutral_mean': np.mean(neut_vals),
            'neutral_sd': np.std(neut_vals, ddof=1),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_treatment': len(treat_vals),
            'n_neutral': len(neut_vals)
        }

    # Also compare peak latencies for key components
    for component in ['N250', 'Frontal_N2']:
        key = f'{component}_peak_lat'

        treat_vals = [d[key] for d in treatment_data if key in d and not np.isnan(d[key])]
        neut_vals = [d[key] for d in neutral_data if key in d and not np.isnan(d[key])]

        if len(treat_vals) >= 3 and len(neut_vals) >= 3:
            t_stat, p_val = stats.ttest_ind(treat_vals, neut_vals)

            pooled_std = np.sqrt(
                ((len(treat_vals) - 1) * np.var(treat_vals, ddof=1) +
                 (len(neut_vals) - 1) * np.var(neut_vals, ddof=1)) /
                (len(treat_vals) + len(neut_vals) - 2)
            )
            cohens_d = (np.mean(treat_vals) - np.mean(neut_vals)) / pooled_std if pooled_std > 0 else 0

            results[f'{component}_latency'] = {
                'treatment_mean': np.mean(treat_vals),
                'treatment_sd': np.std(treat_vals, ddof=1),
                'neutral_mean': np.mean(neut_vals),
                'neutral_sd': np.std(neut_vals, ddof=1),
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

        treat_vals = [d[key] for d in treatment_data if key in d and not np.isnan(d[key])]
        neut_vals = [d[key] for d in neutral_data if key in d and not np.isnan(d[key])]

        if not treat_vals or not neut_vals:
            continue

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

    plt.suptitle('ERP Component Comparison: Treatment vs Neutral\n(After Artifact Rejection)', fontsize=14)
    plt.tight_layout()

    save_path = output_dir / 'figures' / 'erp_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def load_condition_mapping(csv_path: Path = None) -> Dict[str, str]:
    """
    Load condition mapping from CSV file.

    Args:
        csv_path: Path to participant_group CSV file

    Returns:
        Dictionary mapping participant ID to condition ('Treatment' or 'Neutral')
    """
    # Default path - adjust as needed
    if csv_path is None:
        # Try common locations
        possible_paths = [
            Path('/Users/user/Desktop/audio_ai/stuff/participant_group for Audio Task.csv'),
            Path('participant_group for Audio Task.csv'),
            Path('../data/participant_group.csv'),
        ]
        for p in possible_paths:
            if p.exists():
                csv_path = p
                break

    condition_map = {}

    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                participant_id = str(row['Participant ID']).strip()
                condition_num = row['Condition']
                if pd.notna(condition_num):
                    condition_num = int(condition_num)
                    # Groups 1-8 = Treatment, Groups 9-16 = Neutral
                    condition = 'Treatment' if condition_num <= 8 else 'Neutral'
                    condition_map[participant_id] = condition
        except Exception as e:
            print(f"Warning: Could not load condition mapping: {e}")

    return condition_map


# Global condition mapping (loaded once)
CONDITION_MAP = None


def get_condition(subject_id: str) -> str:
    """Determine condition based on participant ID using CSV mapping."""
    global CONDITION_MAP

    if CONDITION_MAP is None:
        CONDITION_MAP = load_condition_mapping()

    # Extract participant ID from subject_id (remove 'sub-' prefix)
    participant_id = subject_id.replace('sub-', '')

    if participant_id in CONDITION_MAP:
        return CONDITION_MAP[participant_id]

    # Fallback: try without any prefix variations
    for key in CONDITION_MAP:
        if key in participant_id or participant_id in key:
            return CONDITION_MAP[key]

    return 'Unknown'


def process_subject(xdf_path: Path) -> Optional[Dict]:
    """
    Process a single subject with full artifact rejection pipeline.

    Args:
        xdf_path: Path to XDF file

    Returns:
        Dictionary with ERP results or None if processing failed
    """
    subject_id = extract_subject_id(xdf_path)
    print(f"\nProcessing: {subject_id}")

    # Load data
    data, sfreq, markers, eeg_times = load_xdf_data(xdf_path)

    if data is None:
        print(f"  Skipping: Could not load data")
        return None

    # Check for empty data
    if data.shape[1] == 0:
        print(f"  Skipping: Empty data file")
        return None

    print(f"  Data shape: {data.shape}, Sfreq: {sfreq} Hz")
    print(f"  Markers: {len(markers)}")

    # Create MNE Raw object
    raw = create_mne_raw(data, sfreq)

    # Step 1: Preprocess continuous data
    print("  Preprocessing continuous data...")
    raw = preprocess_continuous(raw)

    # Step 2: Apply ICA for artifact removal
    print("  Applying ICA...")
    raw = apply_ica_artifact_removal(raw)

    # Step 3: Re-reference to average
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()

    # Step 4: Create epochs with artifact rejection
    print("  Creating epochs...")
    epochs = create_epochs(raw, markers, eeg_times)

    if epochs is None or len(epochs) < 3:
        print(f"  Skipping: Insufficient epochs")
        return None

    # Step 5: Additional variance-based epoch rejection
    epochs = reject_bad_epochs_by_variance(epochs)

    if len(epochs) < 3:
        print(f"  Skipping: Insufficient epochs after variance rejection")
        return None

    # Step 6: Analyze ERP components
    print("  Analyzing ERP components...")
    erp_results = analyze_erp_components(epochs)

    # Step 7: Compute signal variance
    signal_var = compute_signal_variance(epochs)
    erp_results['signal_variance'] = signal_var

    # Add metadata
    erp_results['subject_id'] = subject_id
    erp_results['condition'] = get_condition(subject_id)
    erp_results['n_epochs_final'] = len(epochs)
    erp_results['n_channels'] = len(epochs.ch_names)

    print(f"  Complete: {len(epochs)} epochs, condition = {erp_results['condition']}")

    return erp_results


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
        result = process_subject(xdf_path)

        if result is None:
            continue

        all_results.append(result)

        if result['condition'] == 'Treatment':
            treatment_results.append(result)
        elif result['condition'] == 'Neutral':
            neutral_results.append(result)

    # Save individual results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'eeg_erp_results.csv', index=False)
    print(f"\nResults saved: {output_dir / 'eeg_erp_results.csv'}")

    # Statistical comparison
    if treatment_results and neutral_results:
        stats_results = compare_conditions(treatment_results, neutral_results)

        print("\n" + "=" * 70)
        print("ERP COMPONENT COMPARISON: Treatment vs Neutral")
        print("(With proper epoching and artifact rejection)")
        print("=" * 70)

        for component, stat in stats_results.items():
            sig_marker = "*" if stat['p_value'] < 0.05 else ""
            print(f"\n{component}:")
            print(f"  Treatment: M = {stat['treatment_mean']:.2f}, SD = {stat['treatment_sd']:.2f} (n={stat['n_treatment']})")
            print(f"  Neutral:   M = {stat['neutral_mean']:.2f}, SD = {stat['neutral_sd']:.2f} (n={stat['n_neutral']})")
            print(f"  t = {stat['t_statistic']:.2f}, p = {stat['p_value']:.4f}{sig_marker}")
            print(f"  Cohen's d = {stat['cohens_d']:.2f}")

        # Save statistics
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(output_dir / 'eeg_erp_statistics.csv')
        print(f"\nStatistics saved: {output_dir / 'eeg_erp_statistics.csv'}")

        # Plot comparison
        plot_erp_comparison(treatment_results, neutral_results, output_dir)

    return df


def main(data_dir: Path, output_dir: Path):
    """Main entry point."""
    print("=" * 70)
    print("EEG ERP Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 70)
    print("\nPreprocessing steps:")
    print("  1. Bandpass filter (0.1-40 Hz)")
    print("  2. Notch filter (60 Hz)")
    print("  3. Bad channel detection")
    print("  4. ICA-based artifact removal")
    print("  5. Epoching around stimulus onset")
    print(f"  6. Artifact rejection (±{REJECT_CRITERIA['eeg']*1e6:.0f}μV threshold)")
    print("  7. Variance-based epoch rejection")
    print("=" * 70)

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
