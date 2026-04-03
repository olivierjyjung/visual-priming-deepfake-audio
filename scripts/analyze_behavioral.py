#!/usr/bin/env python3
"""
Behavioral Data Analysis Script
================================

Analyzes survey responses for false memory rates, audio source judgments,
and perceptual factors.

Usage:
    python analyze_behavioral.py --data-dir /path/to/data --output-dir results/

Author: Visual Priming & Deepfake Audio Perception Study
"""

import argparse
import warnings
from pathlib import Path
import re

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

# Non-interactive matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================

# Group conditions mapping
# Groups 1-8: Treatment (Visual Priming with idol-related content)
# Groups 9-16: Neutral (Control with unrelated content)
TREATMENT_GROUPS = [1, 2, 3, 4, 5, 6, 7, 8]
NEUTRAL_GROUPS = [9, 10, 11, 12, 13, 14, 15, 16]


# =============================================================================
# Data Loading
# =============================================================================

def load_all_data(data_dir: Path):
    """
    Load and combine all response data.

    Expected files:
        - Group #1-8 (Responses).xlsx
        - Group #9-16 (Responses).xlsx
        - participant_group for Audio Task.csv (optional)
        - Errors from Audio task.xlsx (optional)
    """
    data_dir = Path(data_dir)

    # Load response files
    df1_path = data_dir / "Group #1-8 (Responses).xlsx"
    df2_path = data_dir / "Group #9-16 (Responses).xlsx"

    if not df1_path.exists() or not df2_path.exists():
        raise FileNotFoundError(
            f"Required data files not found in {data_dir}\n"
            "Expected: 'Group #1-8 (Responses).xlsx' and 'Group #9-16 (Responses).xlsx'"
        )

    df1 = pd.read_excel(df1_path)
    df2 = pd.read_excel(df2_path)

    print(f"Group 1-8 responses: {len(df1)}")
    print(f"Group 9-16 responses: {len(df2)}")

    # Optional: participant assignments
    participants = None
    participants_path = data_dir / "participant_group for Audio Task.csv"
    if participants_path.exists():
        participants = pd.read_csv(participants_path)
        print(f"Participant assignments: {len(participants)}")

    # Optional: error records
    errors = None
    errors_path = data_dir / "Errors from Audio task.xlsx"
    if errors_path.exists():
        errors = pd.read_excel(errors_path)
        print(f"Error records: {len(errors)}")

    return df1, df2, participants, errors


def reshape_to_long_format(df, group_range="1-8"):
    """
    Reshape wide format data to long format.
    Each participant has 4 audio clip evaluations.
    """
    records = []

    for idx, row in df.iterrows():
        group_num = row['Group number']
        timestamp = row['Timestamp']

        # 4 audio clips per participant
        for clip_num in range(1, 5):
            suffix = "" if clip_num == 1 else f" {clip_num}"

            record = {
                'participant_idx': idx,
                'group_number': group_num,
                'timestamp': timestamp,
                'clip_number': clip_num,
                'group_range': group_range,
            }

            # Q1: False memory
            q1_col = f"1.해당 가수가 이 노래를 부른 영상을 본 적이 있습니까?{suffix}"
            if q1_col in df.columns:
                record['Q1_false_memory'] = row[q1_col]

            # Q2: Confidence in Q1
            q2_col = f"2. 위 질문(1번 문항)에 대한 답변에 얼마나 확신하십니까?{suffix}"
            if q2_col in df.columns:
                record['Q2_confidence'] = row[q2_col]

            # Q3: Audio source judgment
            q3_col = f"3. 방금 들은 노래는 어떤 방식으로 만들어졌다고 생각하십니까?(복수 선택 불가){suffix}"
            if q3_col in df.columns:
                record['Q3_audio_judgment'] = row[q3_col]

            # Q4: Confidence in Q3
            q4_col = f"4. 3번 문항에 대한 답변에 얼마나 확신하십니까?{suffix}"
            if q4_col in df.columns:
                record['Q4_confidence'] = row[q4_col]

            # Q5: Judgment factors
            factors = ['발음', '억양', '감정 표현', '음질', '박자감(리듬감)', '숨소리나 호흡의 자연스러운 정도']
            for factor in factors:
                col = f"5. 다음 항목들이 본인의 판단(4번 문항의 응답)에 어느 정도 영향을 미쳤는지 평가해주세요. [{factor}]{suffix}"
                if col in df.columns:
                    record[f'Q5_{factor}'] = row[col]

            # Q6: Likeability
            q6_col = f"6. 방금 들은 노래에 대한 전반적인 호감도를 선택해 주십시오.{suffix}"
            if q6_col in df.columns:
                record['Q6_likeability'] = row[q6_col]

            records.append(record)

        # Q7-Q9 are per-singer (2 singers), add to first 2 clips
        for singer_num in range(1, 3):
            suffix = "" if singer_num == 1 else " 2"

            q7_col = f"7.  평소 해당 가수의 목소리 또는 노래를 얼마나 자주 접합니까?{suffix}"
            if q7_col in df.columns:
                for rec in records[-4:]:
                    if rec['clip_number'] in [singer_num, singer_num + 2]:
                        rec['Q7_listening_freq'] = row[q7_col]

            q8_col = f"8. 본인이 해당 가수의 목소리(또는 노래 스타일)을 얼마나 잘 알고 있다고 생각합니까?{suffix}"
            if q8_col in df.columns:
                for rec in records[-4:]:
                    if rec['clip_number'] in [singer_num, singer_num + 2]:
                        rec['Q8_voice_familiarity'] = row[q8_col]

    return pd.DataFrame(records)


def parse_responses(df):
    """Parse and clean response values."""
    df = df.copy()

    def extract_number(x):
        """Extract number from string like '(4) 꽤 확신한다'."""
        if pd.isna(x):
            return np.nan
        match = re.search(r'\((\d+)\)', str(x))
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)', str(x))
        if match:
            return int(match.group(1))
        return np.nan

    # Q1: False memory (예/아니오 -> 1/0)
    if 'Q1_false_memory' in df.columns:
        df['Q1_false_memory_binary'] = df['Q1_false_memory'].apply(
            lambda x: 1 if '예' in str(x) or '있' in str(x) else 0 if pd.notna(x) else np.nan
        )

    # Q2: Confidence in false memory
    if 'Q2_confidence' in df.columns:
        df['Q2_confidence_num'] = df['Q2_confidence'].apply(extract_number)

    # Q3: Audio judgment - categorize as AI vs Real
    if 'Q3_audio_judgment' in df.columns:
        def categorize_judgment(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x)
            if x_str.startswith('(1)') or x_str.startswith('(2)'):
                return 'AI'
            elif x_str.startswith('(3)') or x_str.startswith('(4)'):
                return 'Other'
            elif x_str.startswith('(5)') or x_str.startswith('(6)'):
                return 'Real'
            x_lower = x_str.lower()
            if 'ai' in x_lower or '딥러닝' in x_lower or '합성' in x_lower:
                return 'AI'
            elif '실제' in x_lower:
                return 'Real'
            else:
                return 'Other'

        df['Q3_judgment_category'] = df['Q3_audio_judgment'].apply(categorize_judgment)
        df['Q3_judgment_num'] = df['Q3_audio_judgment'].apply(extract_number)

    # Q4: Confidence in audio judgment
    if 'Q4_confidence' in df.columns:
        df['Q4_confidence_num'] = df['Q4_confidence'].apply(extract_number)

    # Q6: Likeability
    if 'Q6_likeability' in df.columns:
        df['Q6_likeability_num'] = df['Q6_likeability'].apply(extract_number)

    # Q7, Q8: Familiarity
    if 'Q7_listening_freq' in df.columns:
        df['Q7_listening_freq_num'] = df['Q7_listening_freq'].apply(extract_number)

    if 'Q8_voice_familiarity' in df.columns:
        df['Q8_voice_familiarity_num'] = df['Q8_voice_familiarity'].apply(extract_number)

    # Q5 factors
    for col in df.columns:
        if col.startswith('Q5_') and '_num' not in col:
            df[col + '_num'] = df[col].apply(extract_number)

    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_by_condition(df):
    """Analyze Treatment vs Neutral condition differences."""
    print("\n=== CONDITION-LEVEL ANALYSIS (Treatment vs Neutral) ===")

    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    df_valid = df[df['Q1_false_memory_binary'].notna()].copy()

    print(f"\nValid observations: {len(df_valid)}")
    print(f"  Treatment: {len(df_valid[df_valid['condition'] == 'Treatment'])}")
    print(f"  Neutral: {len(df_valid[df_valid['condition'] == 'Neutral'])}")

    # False Memory by condition
    print("\n--- False Memory by Condition ---")
    fm_by_cond = df_valid.groupby('condition')['Q1_false_memory_binary'].agg(['sum', 'count', 'mean'])
    fm_by_cond['rate_pct'] = fm_by_cond['mean'] * 100
    print(fm_by_cond)

    # Chi-square test
    treatment = df_valid[df_valid['condition'] == 'Treatment']['Q1_false_memory_binary']
    neutral = df_valid[df_valid['condition'] == 'Neutral']['Q1_false_memory_binary']

    contingency = np.array([
        [treatment.sum(), len(treatment) - treatment.sum()],
        [neutral.sum(), len(neutral) - neutral.sum()]
    ])

    chi2_yates, p_yates, dof, _ = stats.chi2_contingency(contingency, correction=True)
    cramers_v = np.sqrt(chi2_yates / len(df_valid))

    print(f"\nChi-square (Yates): χ²(1) = {chi2_yates:.2f}, p = {p_yates:.4f}")
    print(f"Cramér's V = {cramers_v:.3f}")

    # Audio judgment by condition
    if 'Q3_judgment_category' in df_valid.columns:
        print("\n--- Audio Judgment by Condition ---")
        judgment_pct = pd.crosstab(
            df_valid['condition'],
            df_valid['Q3_judgment_category'],
            normalize='index'
        ) * 100
        print(judgment_pct.round(1))

    return df_valid


def analyze_judgment_factors(df):
    """Analyze which factors influenced judgments."""
    print("\n=== Judgment Factor Analysis ===")

    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]

    if not factor_cols:
        print("No Q5 factor columns found")
        return None

    factor_means = df[factor_cols].mean().sort_values(ascending=False)

    print("\nMean Factor Importance:")
    for col, val in factor_means.items():
        factor_name = col.replace('Q5_', '').replace('_num', '')
        print(f"  {factor_name}: {val:.2f}")

    return factor_means


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_condition_comparison(df, output_dir):
    """Plot false memory comparison between conditions."""
    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # False memory rates
    fm_rates = df.groupby('condition')['Q1_false_memory_binary'].mean() * 100
    colors = ['#3498db', '#e74c3c']

    bars = axes[0].bar(fm_rates.index, fm_rates.values, color=colors, edgecolor='black')
    axes[0].set_ylabel('False Memory Rate (%)')
    axes[0].set_title('False Memory Rate by Condition')
    axes[0].set_ylim(0, 30)

    for bar, val in zip(bars, fm_rates.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Audio judgment distribution
    if 'Q3_judgment_category' in df.columns:
        judgment_counts = pd.crosstab(df['condition'], df['Q3_judgment_category'])
        judgment_counts.plot(kind='bar', ax=axes[1], color=['#E24A33', '#988ED5', '#348ABD'])
        axes[1].set_ylabel('Count')
        axes[1].set_title('Audio Source Judgment by Condition')
        axes[1].legend(title='Judgment')
        axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()

    save_path = output_dir / 'figures' / 'condition_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


def plot_factor_importance(df, output_dir):
    """Plot judgment factor importance."""
    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]

    if not factor_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    factor_means = df[factor_cols].mean().sort_values(ascending=True)
    factor_names = [col.replace('Q5_', '').replace('_num', '') for col in factor_means.index]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(factor_means)))

    bars = ax.barh(factor_names, factor_means.values, color=colors, edgecolor='black')
    ax.set_xlabel('Mean Importance Rating')
    ax.set_title('Factors Influencing Audio Authenticity Judgment')

    for bar, val in zip(bars, factor_means.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center')

    plt.tight_layout()

    save_path = output_dir / 'figures' / 'factor_importance.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(df, output_path):
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("BEHAVIORAL DATA ANALYSIS REPORT")
    report.append("Visual Priming & Deepfake Audio Perception Study")
    report.append("=" * 80)
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Sample characteristics
    report.append("\n" + "-" * 80)
    report.append("1. SAMPLE CHARACTERISTICS")
    report.append("-" * 80)

    n_participants = df['participant_idx'].nunique()
    n_observations = len(df)
    n_groups = df['group_number'].nunique()

    report.append(f"\nTotal participants: {n_participants}")
    report.append(f"Total observations (clips evaluated): {n_observations}")
    report.append(f"Number of experimental groups: {n_groups}")

    # False Memory Analysis
    report.append("\n" + "-" * 80)
    report.append("2. FALSE MEMORY ANALYSIS")
    report.append("-" * 80)

    if 'Q1_false_memory_binary' in df.columns:
        overall_fm_rate = df['Q1_false_memory_binary'].mean() * 100
        report.append(f"\nOverall false memory rate: {overall_fm_rate:.1f}%")

        df_temp = df.copy()
        df_temp['condition'] = df_temp['group_number'].apply(
            lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
        )

        for cond in ['Treatment', 'Neutral']:
            cond_data = df_temp[df_temp['condition'] == cond]
            rate = cond_data['Q1_false_memory_binary'].mean() * 100
            n = len(cond_data)
            report.append(f"  {cond}: {rate:.1f}% (n={n})")

    # Audio Judgment Analysis
    report.append("\n" + "-" * 80)
    report.append("3. AUDIO SOURCE JUDGMENT ANALYSIS")
    report.append("-" * 80)

    if 'Q3_judgment_category' in df.columns:
        report.append("\nBy category (AI vs Real):")
        cat_counts = df['Q3_judgment_category'].value_counts()
        for cat, count in cat_counts.items():
            pct = count / len(df) * 100
            report.append(f"  {cat}: {count} ({pct:.1f}%)")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved: {output_path}")
    return report_text


# =============================================================================
# Main Execution
# =============================================================================

def main(data_dir: Path, output_dir: Path):
    """Main analysis pipeline."""
    print("=" * 60)
    print("Behavioral Data Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n>>> Loading data...")
    df1, df2, participants, errors = load_all_data(data_dir)

    # Reshape to long format
    print("\n>>> Reshaping data to long format...")
    df1_long = reshape_to_long_format(df1, "1-8")
    df2_long = reshape_to_long_format(df2, "9-16")

    # Combine
    df_combined = pd.concat([df1_long, df2_long], ignore_index=True)
    print(f"Combined dataset: {len(df_combined)} observations")

    # Parse responses
    print("\n>>> Parsing responses...")
    df_combined = parse_responses(df_combined)

    # Save processed data
    processed_path = output_dir / 'processed_data.csv'
    df_combined.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"Processed data saved: {processed_path}")

    # Run analyses
    print("\n>>> Running analyses...")
    analyze_by_condition(df_combined)
    analyze_judgment_factors(df_combined)

    # Generate visualizations
    print("\n>>> Generating visualizations...")
    plot_condition_comparison(df_combined, output_dir)
    plot_factor_importance(df_combined, output_dir)

    # Generate report
    print("\n>>> Generating report...")
    generate_report(df_combined, output_dir / 'reports' / 'behavioral_analysis_report.txt')

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return df_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavioral data analysis for Visual Priming & Deepfake Audio study"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Directory for output files (default: results/)"
    )

    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
