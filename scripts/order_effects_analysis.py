#!/usr/bin/env python3
"""
Order Effects Analysis
=======================

Analyzes potential order effects and performs robustness checks
for the main findings.

Tests:
1. Clip presentation order effects (1-4)
2. First idol presentation order (Hanni vs Jennie)
3. Audio type order (Real vs AI first)
4. Cochran-Mantel-Haenszel test for condition effect controlling for order

Usage:
    python order_effects_analysis.py --data-dir /path/to/data --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# Statistical Functions
# =============================================================================

def cochran_mantel_haenszel(df: pd.DataFrame, exposure: str,
                            outcome: str, strata: str) -> tuple:
    """
    Compute Cochran-Mantel-Haenszel test statistic.

    Tests for association between exposure and outcome, controlling for strata.

    Args:
        df: DataFrame with data
        exposure: Column name for exposure variable (e.g., 'condition')
        outcome: Column name for binary outcome (e.g., 'Q1_false_memory_binary')
        strata: Column name for stratification variable (e.g., 'clip_number')

    Returns:
        (CMH statistic, p-value)
    """
    strata_vals = df[strata].unique()

    numerator = 0
    variance = 0

    for s in strata_vals:
        df_s = df[df[strata] == s]

        # Create 2x2 table
        a = len(df_s[(df_s[exposure] == 'Treatment') & (df_s[outcome] == 1)])
        b = len(df_s[(df_s[exposure] == 'Treatment') & (df_s[outcome] == 0)])
        c = len(df_s[(df_s[exposure] == 'Neutral') & (df_s[outcome] == 1)])
        d = len(df_s[(df_s[exposure] == 'Neutral') & (df_s[outcome] == 0)])

        n = a + b + c + d
        if n == 0:
            continue

        n1 = a + b  # Treatment total
        n0 = c + d  # Neutral total
        m1 = a + c  # Outcome = 1 total
        m0 = b + d  # Outcome = 0 total

        E_a = (n1 * m1) / n
        Var_a = (n1 * n0 * m1 * m0) / (n**2 * (n - 1)) if n > 1 else 0

        numerator += (a - E_a)
        variance += Var_a

    if variance > 0:
        cmh_stat = (abs(numerator) - 0.5)**2 / variance  # with continuity correction
        p_value = 1 - stats.chi2.cdf(cmh_stat, 1)
        return cmh_stat, p_value

    return None, None


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_clip_order(df: pd.DataFrame) -> dict:
    """Analyze false memory rate by clip order."""
    results = {}

    order_fm = df.groupby('clip_number').agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean']
    })
    order_fm.columns = ['false_memory_count', 'total', 'rate']
    results['by_clip'] = order_fm.to_dict()

    # Chi-square test
    contingency = pd.crosstab(df['clip_number'], df['Q1_false_memory_binary'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    results['chi2'] = chi2
    results['p_value'] = p
    results['dof'] = dof

    return results


def analyze_condition_by_order(df: pd.DataFrame) -> dict:
    """Analyze condition effect controlling for clip order."""
    cond_order = df.groupby(['condition', 'clip_number']).agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean']
    })
    cond_order.columns = ['fm_count', 'total', 'rate']

    cmh_stat, cmh_p = cochran_mantel_haenszel(
        df, 'condition', 'Q1_false_memory_binary', 'clip_number'
    )

    return {
        'by_condition_order': cond_order.to_dict(),
        'cmh_statistic': cmh_stat,
        'cmh_p_value': cmh_p
    }


def analyze_first_idol_effect(df: pd.DataFrame) -> dict:
    """Analyze first idol presentation order effect."""
    # Based on experimental design
    # Groups 1,2,5,6,9,10,13,14 = Hanni first
    # Groups 3,4,7,8,11,12,15,16 = Jennie first
    df = df.copy()
    df['first_idol'] = df['group_number'].apply(
        lambda x: 'Hanni' if x in [1, 2, 5, 6, 9, 10, 13, 14] else 'Jennie'
    )

    idol_fm = df.groupby('first_idol').agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean']
    })
    idol_fm.columns = ['fm_count', 'total', 'rate']

    contingency = pd.crosstab(df['first_idol'], df['Q1_false_memory_binary'])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    return {
        'by_first_idol': idol_fm.to_dict(),
        'chi2': chi2,
        'p_value': p
    }


def analyze_audio_order_effect(df: pd.DataFrame) -> dict:
    """Analyze audio type order effect (Real first vs AI first)."""
    df = df.copy()
    # Based on experimental design:
    # Odd groups = Real first, Even groups = AI first
    df['audio_order'] = df['group_number'].apply(
        lambda x: 'Real_first' if x % 2 == 1 else 'AI_first'
    )

    audio_fm = df.groupby('audio_order').agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean']
    })
    audio_fm.columns = ['fm_count', 'total', 'rate']

    contingency = pd.crosstab(df['audio_order'], df['Q1_false_memory_binary'])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    return {
        'by_audio_order': audio_fm.to_dict(),
        'chi2': chi2,
        'p_value': p
    }


def analyze_group_variance(df: pd.DataFrame) -> dict:
    """Analyze variance in false memory rates across groups."""
    group_fm = df.groupby(['group_number', 'condition']).agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean']
    })
    group_fm.columns = ['fm_count', 'total', 'rate']
    group_fm = group_fm.reset_index()

    treatment_rates = group_fm[group_fm['condition'] == 'Treatment']['rate'].values
    neutral_rates = group_fm[group_fm['condition'] == 'Neutral']['rate'].values

    # Levene's test for equality of variance
    if len(treatment_rates) > 1 and len(neutral_rates) > 1:
        levene_stat, levene_p = stats.levene(treatment_rates, neutral_rates)
    else:
        levene_stat, levene_p = None, None

    return {
        'treatment_sd': np.std(treatment_rates) if len(treatment_rates) > 0 else None,
        'treatment_range': (treatment_rates.min(), treatment_rates.max()) if len(treatment_rates) > 0 else None,
        'neutral_sd': np.std(neutral_rates) if len(neutral_rates) > 0 else None,
        'neutral_range': (neutral_rates.min(), neutral_rates.max()) if len(neutral_rates) > 0 else None,
        'levene_stat': levene_stat,
        'levene_p': levene_p
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate order effects analysis report."""
    report = []
    report.append("=" * 70)
    report.append("ORDER EFFECTS ANALYSIS REPORT")
    report.append("Visual Priming & Deepfake Audio Perception Study")
    report.append("=" * 70)

    # 1. Clip order effect
    report.append("\n1. FALSE MEMORY RATE BY CLIP ORDER")
    report.append("-" * 50)

    clip_results = analyze_clip_order(df)
    for clip in sorted(df['clip_number'].unique()):
        rate = clip_results['by_clip']['rate'].get(clip, 0)
        total = clip_results['by_clip']['total'].get(clip, 0)
        report.append(f"  Clip {clip}: {rate*100:.1f}% (n={total})")

    report.append(f"\nChi-square test: χ²({clip_results['dof']}) = {clip_results['chi2']:.3f}, p = {clip_results['p_value']:.4f}")

    # 2. Condition effect controlling for order
    report.append("\n2. CONDITION EFFECT CONTROLLING FOR CLIP ORDER (CMH Test)")
    report.append("-" * 50)

    cond_order_results = analyze_condition_by_order(df)
    if cond_order_results['cmh_statistic']:
        report.append(f"CMH statistic: χ²(1) = {cond_order_results['cmh_statistic']:.3f}")
        report.append(f"p-value: {cond_order_results['cmh_p_value']:.4f}")

        if cond_order_results['cmh_p_value'] < 0.05:
            report.append("→ Main effect REMAINS SIGNIFICANT after controlling for clip order")
        else:
            report.append("→ Main effect becomes non-significant after controlling for clip order")

    # 3. First idol effect
    report.append("\n3. FIRST IDOL PRESENTATION ORDER EFFECT")
    report.append("-" * 50)

    idol_results = analyze_first_idol_effect(df)
    for idol in ['Hanni', 'Jennie']:
        rate = idol_results['by_first_idol']['rate'].get(idol, 0)
        total = idol_results['by_first_idol']['total'].get(idol, 0)
        report.append(f"  {idol} first: {rate*100:.1f}% (n={total})")

    report.append(f"\nChi-square test: χ²(1) = {idol_results['chi2']:.3f}, p = {idol_results['p_value']:.4f}")

    # 4. Audio order effect
    report.append("\n4. AUDIO TYPE ORDER EFFECT")
    report.append("-" * 50)

    audio_results = analyze_audio_order_effect(df)
    for order in ['Real_first', 'AI_first']:
        rate = audio_results['by_audio_order']['rate'].get(order, 0)
        total = audio_results['by_audio_order']['total'].get(order, 0)
        report.append(f"  {order}: {rate*100:.1f}% (n={total})")

    report.append(f"\nChi-square test: χ²(1) = {audio_results['chi2']:.3f}, p = {audio_results['p_value']:.4f}")

    # 5. Group variance
    report.append("\n5. GROUP-LEVEL VARIANCE IN FALSE MEMORY RATES")
    report.append("-" * 50)

    var_results = analyze_group_variance(df)
    if var_results['treatment_sd'] is not None:
        report.append(f"  Treatment groups: SD = {var_results['treatment_sd']:.3f}")
    if var_results['neutral_sd'] is not None:
        report.append(f"  Neutral groups: SD = {var_results['neutral_sd']:.3f}")
    if var_results['levene_stat'] is not None:
        report.append(f"\nLevene's test: F = {var_results['levene_stat']:.3f}, p = {var_results['levene_p']:.4f}")

    # 6. Summary
    report.append("\n" + "=" * 70)
    report.append("SUMMARY: ROBUSTNESS OF MAIN FINDING")
    report.append("=" * 70)

    orig_treat = df[df['condition'] == 'Treatment']['Q1_false_memory_binary']
    orig_neut = df[df['condition'] == 'Neutral']['Q1_false_memory_binary']

    report.append(f"\nOriginal finding (without order control):")
    report.append(f"  Treatment: {orig_treat.mean()*100:.1f}% ({orig_treat.sum()}/{len(orig_treat)})")
    report.append(f"  Neutral: {orig_neut.mean()*100:.1f}% ({orig_neut.sum()}/{len(orig_neut)})")

    contingency = pd.crosstab(df['condition'], df['Q1_false_memory_binary'])
    chi2, p, _, _ = stats.chi2_contingency(contingency, correction=True)
    report.append(f"  χ²(1) = {chi2:.3f}, p = {p:.4f}")

    if cond_order_results['cmh_p_value']:
        report.append(f"\nAfter controlling for clip order (CMH test):")
        report.append(f"  χ²(1) = {cond_order_results['cmh_statistic']:.3f}, p = {cond_order_results['cmh_p_value']:.4f}")

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)

    # Save report
    report_path = output_dir / 'reports' / 'order_effects_report.txt'
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
    print("Order Effects Analysis")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Find processed data file
    data_path = data_dir / 'processed_data.csv'
    if not data_path.exists():
        data_path = data_dir / 'processed_real_data.csv'
    if not data_path.exists():
        print(f"Error: Could not find processed data in {data_dir}")
        return

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations")

    # Add condition variable if not present
    if 'condition' not in df.columns:
        if 'group_number' in df.columns:
            df['condition'] = df['group_number'].apply(
                lambda x: 'Treatment' if x <= 8 else 'Neutral'
            )
        elif 'group_range' in df.columns:
            df['condition'] = df['group_range'].apply(
                lambda x: 'Treatment' if x == '1-8' else 'Neutral'
            )

    # Filter valid data
    df_valid = df[df['Q1_false_memory_binary'].notna()].copy()
    print(f"Valid observations: {len(df_valid)}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    # Generate report
    generate_report(df_valid, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Order effects analysis for Visual Priming study"
    )
    parser.add_argument("--data-dir", "-d", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results"))

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
