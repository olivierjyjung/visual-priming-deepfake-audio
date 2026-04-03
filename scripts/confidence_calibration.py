#!/usr/bin/env python3
"""
Confidence Calibration Analysis
================================

Analyzes the relationship between confidence and accuracy in audio
authenticity judgments.

Outputs:
1. Confidence calibration curves
2. Overconfidence quantification (Brier score, ECE)
3. Condition-specific calibration analysis

Usage:
    python confidence_calibration.py --data-dir /path/to/data --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    HAS_DEPENDENCIES = False


# =============================================================================
# Data Loading
# =============================================================================

def load_behavioral_data(data_path: Path) -> pd.DataFrame:
    """Load the processed behavioral data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations")
    return df


def prepare_calibration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for calibration analysis.

    Creates:
    - condition: Treatment vs Neutral
    - predicted_ai: Whether participant judged audio as AI
    - actual_ai: Ground truth (based on clip number)
    - correct: Whether judgment was correct
    - confidence_normalized: Confidence on 0-1 scale
    """
    df = df.copy()

    # Add condition based on group_range or group_number
    if 'group_range' in df.columns:
        df['condition'] = df['group_range'].apply(
            lambda x: 'Treatment' if x == '1-8' else 'Neutral'
        )
    elif 'group_number' in df.columns:
        df['condition'] = df['group_number'].apply(
            lambda x: 'Treatment' if x <= 8 else 'Neutral'
        )

    # Create binary judgment variable
    if 'Q3_judgment_category' in df.columns:
        df['predicted_ai'] = df['Q3_judgment_category'].apply(
            lambda x: 1 if x == 'AI' else 0 if x == 'Real' else np.nan
        )

    # Determine actual audio type from clip_number
    # Based on experimental design: clips 2, 3 are typically AI-generated
    if 'clip_number' in df.columns:
        df['actual_ai'] = df['clip_number'].apply(
            lambda x: 1 if x in [2, 3] else 0
        )

    # Create accuracy variable
    df['correct'] = (df['predicted_ai'] == df['actual_ai']).astype(float)
    df.loc[df['predicted_ai'].isna(), 'correct'] = np.nan

    # Normalize confidence to 0-1 scale (from 1-5 scale)
    if 'Q4_confidence_num' in df.columns:
        df['confidence_normalized'] = (df['Q4_confidence_num'] - 1) / 4

    return df


# =============================================================================
# Calibration Analysis
# =============================================================================

def compute_calibration_metrics(df: pd.DataFrame) -> dict:
    """
    Compute calibration metrics.

    Returns:
        Dictionary with Brier score, ECE, and calibration data
    """
    valid_df = df.dropna(subset=['correct', 'confidence_normalized'])

    if len(valid_df) < 10:
        return None

    y_true = valid_df['correct'].values
    y_prob = valid_df['confidence_normalized'].values

    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=5, strategy='uniform'
        )
    except:
        fraction_of_positives = None
        mean_predicted_value = None

    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_prob, n_bins=5)

    # Overconfidence: mean confidence - mean accuracy
    overconfidence = np.mean(y_prob) - np.mean(y_true)

    return {
        'brier_score': brier,
        'ece': ece,
        'overconfidence': overconfidence,
        'mean_confidence': np.mean(y_prob),
        'mean_accuracy': np.mean(y_true),
        'n_observations': len(valid_df),
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum(|accuracy_i - confidence_i| * n_i / N)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_prob[in_bin])
            ece += np.abs(bin_accuracy - bin_confidence) * (n_in_bin / total_samples)

    return ece


def analyze_by_confidence_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze accuracy at each confidence level.

    Returns:
        DataFrame with accuracy by confidence level
    """
    results = []

    if 'Q4_confidence_num' not in df.columns:
        return pd.DataFrame()

    for conf_level in range(1, 6):
        subset = df[df['Q4_confidence_num'] == conf_level]
        valid_subset = subset.dropna(subset=['correct'])

        if len(valid_subset) > 0:
            results.append({
                'confidence_level': conf_level,
                'n_observations': len(valid_subset),
                'accuracy': valid_subset['correct'].mean(),
                'expected_accuracy': (conf_level - 1) / 4  # Normalized confidence
            })

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================

def plot_calibration_curve(df: pd.DataFrame, output_dir: Path):
    """Plot calibration curves by condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall calibration curve
    ax = axes[0]

    for condition in ['Treatment', 'Neutral']:
        cond_df = df[df['condition'] == condition].dropna(subset=['correct', 'confidence_normalized'])

        if len(cond_df) < 10:
            continue

        try:
            fraction_pos, mean_pred = calibration_curve(
                cond_df['correct'].values,
                cond_df['confidence_normalized'].values,
                n_bins=5
            )

            color = '#3498db' if condition == 'Treatment' else '#e74c3c'
            ax.plot(mean_pred, fraction_pos, marker='o', linewidth=2,
                    label=condition, color=color)

        except Exception as e:
            print(f"Could not compute calibration for {condition}: {e}")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction Correct')
    ax.set_title('Calibration Curve by Condition')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Accuracy by confidence level
    ax = axes[1]

    confidence_analysis = analyze_by_confidence_level(df)

    if not confidence_analysis.empty:
        ax.bar(confidence_analysis['confidence_level'] - 0.15,
               confidence_analysis['accuracy'],
               width=0.3, label='Actual Accuracy', color='#3498db')

        ax.bar(confidence_analysis['confidence_level'] + 0.15,
               confidence_analysis['expected_accuracy'],
               width=0.3, label='Expected (if calibrated)', color='#95a5a6', alpha=0.7)

        ax.set_xlabel('Confidence Level (1-5)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Confidence Level')
        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()

    save_path = output_dir / 'figures' / 'calibration_analysis.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


def plot_confidence_accuracy_scatter(df: pd.DataFrame, output_dir: Path):
    """Plot confidence vs accuracy scatter by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate by participant
    if 'participant_idx' in df.columns:
        participant_data = df.groupby(['participant_idx', 'condition']).agg({
            'correct': 'mean',
            'confidence_normalized': 'mean'
        }).reset_index()
    else:
        participant_data = df.groupby('condition').agg({
            'correct': 'mean',
            'confidence_normalized': 'mean'
        }).reset_index()

    for condition in ['Treatment', 'Neutral']:
        cond_data = participant_data[participant_data['condition'] == condition]
        color = '#3498db' if condition == 'Treatment' else '#e74c3c'
        ax.scatter(cond_data['confidence_normalized'], cond_data['correct'],
                   alpha=0.6, label=condition, color=color, s=100)

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Confidence-Accuracy Relationship by Participant')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_path = output_dir / 'figures' / 'confidence_accuracy_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate calibration analysis report."""
    report = []
    report.append("=" * 70)
    report.append("CONFIDENCE CALIBRATION ANALYSIS REPORT")
    report.append("Visual Priming & Deepfake Audio Perception Study")
    report.append("=" * 70)

    # Overall metrics
    report.append("\n1. OVERALL CALIBRATION METRICS")
    report.append("-" * 50)

    overall_metrics = compute_calibration_metrics(df)
    if overall_metrics:
        report.append(f"Brier Score: {overall_metrics['brier_score']:.4f}")
        report.append(f"Expected Calibration Error (ECE): {overall_metrics['ece']:.4f}")
        report.append(f"Overconfidence: {overall_metrics['overconfidence']:.4f}")
        report.append(f"Mean Confidence: {overall_metrics['mean_confidence']:.4f}")
        report.append(f"Mean Accuracy: {overall_metrics['mean_accuracy']:.4f}")

    # By condition
    report.append("\n2. CALIBRATION BY CONDITION")
    report.append("-" * 50)

    for condition in ['Treatment', 'Neutral']:
        cond_df = df[df['condition'] == condition]
        metrics = compute_calibration_metrics(cond_df)

        if metrics:
            report.append(f"\n{condition}:")
            report.append(f"  Brier Score: {metrics['brier_score']:.4f}")
            report.append(f"  ECE: {metrics['ece']:.4f}")
            report.append(f"  Overconfidence: {metrics['overconfidence']:.4f}")
            report.append(f"  N: {metrics['n_observations']}")

    # Accuracy by confidence level
    report.append("\n3. ACCURACY BY CONFIDENCE LEVEL")
    report.append("-" * 50)

    conf_analysis = analyze_by_confidence_level(df)
    if not conf_analysis.empty:
        for _, row in conf_analysis.iterrows():
            report.append(
                f"Level {int(row['confidence_level'])}: "
                f"Accuracy = {row['accuracy']:.3f}, "
                f"N = {int(row['n_observations'])}"
            )

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)

    report_path = output_dir / 'reports' / 'confidence_calibration_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"Report saved: {report_path}")
    print(report_text)


# =============================================================================
# Main Pipeline
# =============================================================================

def main(data_dir: Path, output_dir: Path):
    """Main entry point."""
    print("=" * 60)
    print("Confidence Calibration Analysis")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    if not HAS_DEPENDENCIES:
        print("\nPlease install required dependencies.")
        return

    # Find processed data file
    data_path = data_dir / 'processed_data.csv'
    if not data_path.exists():
        data_path = data_dir / 'processed_real_data.csv'
    if not data_path.exists():
        print(f"Error: Could not find processed data in {data_dir}")
        return

    # Load and prepare data
    df = load_behavioral_data(data_path)
    df = prepare_calibration_data(df)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    # Run analyses
    plot_calibration_curve(df, output_dir)
    plot_confidence_accuracy_scatter(df, output_dir)
    generate_report(df, output_dir)

    # Save analysis data
    conf_analysis = analyze_by_confidence_level(df)
    conf_analysis.to_csv(output_dir / 'calibration_by_confidence.csv', index=False)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confidence calibration analysis"
    )
    parser.add_argument("--data-dir", "-d", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results"))

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
