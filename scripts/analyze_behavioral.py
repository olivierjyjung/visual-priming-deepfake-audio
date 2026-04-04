#!/usr/bin/env python3
"""
Behavioral Data Analysis Script
================================

Analyzes survey responses for false memory rates, audio source judgments,
and perceptual factors using appropriate statistical methods.

Key methodological improvements:
- Mixed-effects models to account for nested structure (multiple clips per participant)
- Multiple comparison correction (Benjamini-Hochberg FDR)
- Robust text parsing for Korean responses

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

# Mixed-effects models
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    print("Warning: statsmodels not installed. Mixed-effects models unavailable.")
    print("Install with: pip install statsmodels")
    HAS_STATSMODELS = False


# =============================================================================
# Configuration
# =============================================================================

# Group conditions mapping
# Groups 1-8: Treatment (Visual Priming with idol-related content)
# Groups 9-16: Neutral (Control with unrelated content)
TREATMENT_GROUPS = [1, 2, 3, 4, 5, 6, 7, 8]
NEUTRAL_GROUPS = [9, 10, 11, 12, 13, 14, 15, 16]

# Alpha level
ALPHA = 0.05


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
        participants = pd.read_csv(participants_path, encoding='utf-8-sig')
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

    Creates unique participant IDs for mixed-effects modeling.
    """
    records = []

    for idx, row in df.iterrows():
        group_num = row['Group number']
        timestamp = row['Timestamp']

        # Create unique participant ID combining group_range and index
        participant_id = f"{group_range}_{idx}"

        # 4 audio clips per participant
        for clip_num in range(1, 5):
            suffix = "" if clip_num == 1 else f" {clip_num}"

            record = {
                'participant_id': participant_id,  # Unique participant identifier
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


def parse_korean_yes_no(x):
    """
    Robustly parse Korean yes/no responses.

    Handles various formats:
    - '예', '네', 'Yes', 'yes'
    - '아니오', '아니요', 'No', 'no'
    - Responses with parentheses like '(1) 예'
    """
    if pd.isna(x):
        return np.nan

    x_str = str(x).strip().lower()

    # Yes indicators
    yes_patterns = ['예', '네', 'yes', '있다', '있습니다', '봤다', '봤습니다']
    for pattern in yes_patterns:
        if pattern in x_str:
            return 1

    # No indicators
    no_patterns = ['아니', 'no', '없다', '없습니다', '못', '안']
    for pattern in no_patterns:
        if pattern in x_str:
            return 0

    # Check for number in parentheses - (1) often means yes, (2) means no
    match = re.search(r'\((\d+)\)', x_str)
    if match:
        num = int(match.group(1))
        if num == 1:
            return 1
        elif num == 2:
            return 0

    return np.nan


def parse_responses(df):
    """Parse and clean response values with robust Korean text handling."""
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

    # Q1: False memory - using robust Korean parsing
    if 'Q1_false_memory' in df.columns:
        df['Q1_false_memory_binary'] = df['Q1_false_memory'].apply(parse_korean_yes_no)

    # Q2: Confidence in false memory
    if 'Q2_confidence' in df.columns:
        df['Q2_confidence_num'] = df['Q2_confidence'].apply(extract_number)

    # Q3: Audio judgment - categorize as AI vs Real
    if 'Q3_audio_judgment' in df.columns:
        def categorize_judgment(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x)

            # Check for parenthesized numbers first
            match = re.search(r'\((\d+)\)', x_str)
            if match:
                num = int(match.group(1))
                if num in [1, 2]:
                    return 'AI'
                elif num in [3, 4]:
                    return 'Other'
                elif num in [5, 6]:
                    return 'Real'

            # Fallback to keyword matching
            x_lower = x_str.lower()
            if 'ai' in x_lower or '딥러닝' in x_lower or '합성' in x_lower or '인공' in x_lower:
                return 'AI'
            elif '실제' in x_lower or '본인' in x_lower or '아이돌' in x_lower:
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
# Statistical Analysis Functions
# =============================================================================

def analyze_with_mixed_effects(df):
    """
    Analyze Treatment vs Neutral using mixed-effects models.

    This properly accounts for the nested structure where multiple
    observations (clips) come from each participant.

    Model: outcome ~ condition + (1 | participant_id)
    """
    print("\n" + "=" * 70)
    print("MIXED-EFFECTS MODEL ANALYSIS")
    print("(Accounting for nested structure: clips within participants)")
    print("=" * 70)

    if not HAS_STATSMODELS:
        print("\nERROR: statsmodels not available. Cannot run mixed-effects models.")
        print("Falling back to simple analysis (NOT RECOMMENDED for publication).")
        return analyze_by_condition_simple(df)

    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    # Create binary condition variable (Treatment = 1, Neutral = 0)
    df['condition_binary'] = (df['condition'] == 'Treatment').astype(int)

    results = {}
    p_values = []
    test_names = []

    # =========================================================================
    # 1. FALSE MEMORY - Mixed-effects logistic regression
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. FALSE MEMORY ANALYSIS (Mixed-Effects Logistic Regression)")
    print("-" * 70)

    df_fm = df[df['Q1_false_memory_binary'].notna()].copy()

    n_participants = df_fm['participant_id'].nunique()
    n_observations = len(df_fm)

    print(f"\nSample: {n_participants} participants, {n_observations} observations")
    print(f"  (Average {n_observations/n_participants:.1f} clips per participant)")

    # Descriptive statistics
    fm_by_cond = df_fm.groupby('condition').agg({
        'Q1_false_memory_binary': ['sum', 'count', 'mean'],
        'participant_id': 'nunique'
    })
    fm_by_cond.columns = ['n_yes', 'n_obs', 'rate', 'n_participants']
    fm_by_cond['rate_pct'] = fm_by_cond['rate'] * 100

    print("\nDescriptive Statistics:")
    print(fm_by_cond[['n_participants', 'n_obs', 'n_yes', 'rate_pct']].to_string())

    # Mixed-effects logistic regression
    try:
        # Using GEE (Generalized Estimating Equations) as alternative to GLMM
        # GEE is more robust and handles binary outcomes well
        model_fm = smf.gee(
            "Q1_false_memory_binary ~ condition_binary",
            groups="participant_id",
            data=df_fm,
            family=sm.families.Binomial()
        )
        result_fm = model_fm.fit()

        # Extract results
        coef = result_fm.params['condition_binary']
        se = result_fm.bse['condition_binary']
        z_val = result_fm.tvalues['condition_binary']
        p_val = result_fm.pvalues['condition_binary']

        # Odds ratio
        odds_ratio = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)

        print(f"\nMixed-Effects Model Results (GEE):")
        print(f"  Coefficient (Treatment): {coef:.3f} (SE = {se:.3f})")
        print(f"  z = {z_val:.2f}, p = {p_val:.4f}")
        print(f"  Odds Ratio: {odds_ratio:.2f} [95% CI: {ci_low:.2f}, {ci_high:.2f}]")

        if odds_ratio < 1:
            print(f"  → Treatment REDUCES false memory (OR < 1)")
        else:
            print(f"  → Treatment INCREASES false memory (OR > 1)")

        results['false_memory'] = {
            'coefficient': coef,
            'se': se,
            'z': z_val,
            'p': p_val,
            'odds_ratio': odds_ratio,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'treatment_rate': fm_by_cond.loc['Treatment', 'rate_pct'],
            'neutral_rate': fm_by_cond.loc['Neutral', 'rate_pct'],
        }

        p_values.append(p_val)
        test_names.append('False Memory')

    except Exception as e:
        print(f"\nMixed-effects model failed: {e}")
        print("Falling back to chi-square test (not recommended)")

        # Fallback to chi-square
        treatment = df_fm[df_fm['condition'] == 'Treatment']['Q1_false_memory_binary']
        neutral = df_fm[df_fm['condition'] == 'Neutral']['Q1_false_memory_binary']

        contingency = np.array([
            [treatment.sum(), len(treatment) - treatment.sum()],
            [neutral.sum(), len(neutral) - neutral.sum()]
        ])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency, correction=True)

        print(f"\nChi-square (Yates): χ²(1) = {chi2:.2f}, p = {p_val:.4f}")
        print("WARNING: This does not account for nested structure!")

        p_values.append(p_val)
        test_names.append('False Memory (chi-square)')

    # =========================================================================
    # 2. AUDIO JUDGMENT - Mixed-effects ordinal/multinomial
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. AUDIO SOURCE JUDGMENT ANALYSIS")
    print("-" * 70)

    if 'Q3_judgment_num' in df.columns:
        df_aj = df[df['Q3_judgment_num'].notna()].copy()

        # Descriptive
        judgment_by_cond = pd.crosstab(
            df_aj['condition'],
            df_aj['Q3_judgment_category'],
            normalize='index'
        ) * 100

        print("\nAudio Judgment Distribution (%):")
        print(judgment_by_cond.round(1).to_string())

        # Mixed-effects linear model on judgment scale (1-6)
        try:
            model_aj = smf.mixedlm(
                "Q3_judgment_num ~ condition_binary",
                data=df_aj,
                groups="participant_id"
            )
            result_aj = model_aj.fit(reml=True)

            coef = result_aj.fe_params['condition_binary']
            se = result_aj.bse_fe['condition_binary']
            p_val = result_aj.pvalues['condition_binary']

            print(f"\nMixed-Effects Linear Model:")
            print(f"  Coefficient (Treatment): {coef:.3f} (SE = {se:.3f})")
            print(f"  p = {p_val:.4f}")

            if coef > 0:
                print(f"  → Treatment shifts judgment toward 'Real' (higher scores)")
            else:
                print(f"  → Treatment shifts judgment toward 'AI' (lower scores)")

            results['audio_judgment'] = {
                'coefficient': coef,
                'se': se,
                'p': p_val,
            }

            p_values.append(p_val)
            test_names.append('Audio Judgment')

        except Exception as e:
            print(f"\nMixed-effects model failed: {e}")

    # =========================================================================
    # 3. CONFIDENCE RATINGS - Mixed-effects linear
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. CONFIDENCE ANALYSIS")
    print("-" * 70)

    for conf_var, conf_name in [('Q2_confidence_num', 'False Memory Confidence'),
                                 ('Q4_confidence_num', 'Audio Judgment Confidence')]:
        if conf_var in df.columns:
            df_conf = df[df[conf_var].notna()].copy()

            # Descriptive
            conf_by_cond = df_conf.groupby('condition')[conf_var].agg(['mean', 'std', 'count'])
            print(f"\n{conf_name}:")
            print(conf_by_cond.round(2).to_string())

            try:
                model_conf = smf.mixedlm(
                    f"{conf_var} ~ condition_binary",
                    data=df_conf,
                    groups="participant_id"
                )
                result_conf = model_conf.fit(reml=True)

                coef = result_conf.fe_params['condition_binary']
                p_val = result_conf.pvalues['condition_binary']

                print(f"  Mixed-effects: β = {coef:.3f}, p = {p_val:.4f}")

                p_values.append(p_val)
                test_names.append(conf_name)

            except Exception as e:
                print(f"  Model failed: {e}")

    # =========================================================================
    # 4. MULTIPLE COMPARISON CORRECTION
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg FDR)")
    print("-" * 70)

    if len(p_values) > 1:
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method='fdr_bh')

        print(f"\nNumber of tests: {len(p_values)}")
        print(f"Alpha level: {ALPHA}")
        print("\nResults after FDR correction:")
        print("-" * 50)

        for name, p_orig, p_corr, sig in zip(test_names, p_values, p_corrected, rejected):
            sig_marker = "***" if sig else ""
            print(f"  {name}:")
            print(f"    Original p = {p_orig:.4f}")
            print(f"    Corrected p = {p_corr:.4f} {sig_marker}")

        results['multiple_comparison'] = {
            'test_names': test_names,
            'p_original': p_values,
            'p_corrected': list(p_corrected),
            'significant': list(rejected),
            'method': 'Benjamini-Hochberg FDR'
        }

    return df, results


def analyze_by_condition_simple(df):
    """
    Simple analysis without mixed-effects (fallback).

    WARNING: This treats observations as independent, which is incorrect
    when participants provide multiple responses.
    """
    print("\n" + "=" * 70)
    print("SIMPLE ANALYSIS (NOT RECOMMENDED - ignores nested structure)")
    print("=" * 70)
    print("\nWARNING: This analysis treats all observations as independent.")
    print("With 4 clips per participant, this inflates Type I error rate.")
    print("Use mixed-effects models for publication-quality analysis.\n")

    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    df_valid = df[df['Q1_false_memory_binary'].notna()].copy()

    print(f"Total observations: {len(df_valid)}")
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
    print("\n⚠️  WARNING: p-value is likely UNDERESTIMATED due to ignored clustering!")

    return df_valid, {}


def analyze_judgment_factors(df):
    """Analyze which factors influenced judgments."""
    print("\n" + "=" * 70)
    print("JUDGMENT FACTOR ANALYSIS")
    print("=" * 70)

    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]

    if not factor_cols:
        print("No Q5 factor columns found")
        return None

    factor_means = df[factor_cols].mean().sort_values(ascending=False)

    print("\nMean Factor Importance (scale 1-5):")
    for col, val in factor_means.items():
        factor_name = col.replace('Q5_', '').replace('_num', '')
        print(f"  {factor_name}: {val:.2f}")

    return factor_means


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_condition_comparison(df, output_dir, results=None):
    """Plot false memory comparison between conditions with proper error bars."""
    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate participant-level means for proper error bars
    participant_means = df.groupby(['participant_id', 'condition'])['Q1_false_memory_binary'].mean().reset_index()

    # False memory rates with proper CI
    colors = ['#3498db', '#e74c3c']

    cond_stats = participant_means.groupby('condition')['Q1_false_memory_binary'].agg(['mean', 'std', 'count'])
    cond_stats['se'] = cond_stats['std'] / np.sqrt(cond_stats['count'])
    cond_stats['ci95'] = 1.96 * cond_stats['se']

    x_pos = [0, 1]
    means = [cond_stats.loc['Treatment', 'mean'] * 100, cond_stats.loc['Neutral', 'mean'] * 100]
    errors = [cond_stats.loc['Treatment', 'ci95'] * 100, cond_stats.loc['Neutral', 'ci95'] * 100]

    bars = axes[0].bar(x_pos, means, color=colors, edgecolor='black', yerr=errors, capsize=5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(['Treatment', 'Neutral'])
    axes[0].set_ylabel('False Memory Rate (%)')
    axes[0].set_title('False Memory Rate by Condition\n(Error bars: 95% CI)')
    axes[0].set_ylim(0, max(means) + max(errors) + 10)

    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add significance annotation if results available
    if results and 'false_memory' in results:
        p_val = results['false_memory'].get('p', 1.0)
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'

        max_y = max(means) + max(errors) + 5
        axes[0].plot([0, 0, 1, 1], [max_y, max_y + 2, max_y + 2, max_y], 'k-', linewidth=1)
        axes[0].text(0.5, max_y + 3, sig_text, ha='center', va='bottom', fontsize=12)

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

def generate_report(df, results, output_path):
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("BEHAVIORAL DATA ANALYSIS REPORT")
    report.append("Visual Priming & Deepfake Audio Perception Study")
    report.append("=" * 80)
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append("\n" + "-" * 80)
    report.append("METHODOLOGICAL NOTES")
    report.append("-" * 80)
    report.append("\nThis analysis uses mixed-effects models (GEE/GLMM) to properly")
    report.append("account for the nested data structure where multiple observations")
    report.append("(4 audio clips) come from each participant.")
    report.append("\nMultiple comparison correction: Benjamini-Hochberg FDR")

    # Sample characteristics
    report.append("\n" + "-" * 80)
    report.append("1. SAMPLE CHARACTERISTICS")
    report.append("-" * 80)

    n_participants = df['participant_id'].nunique()
    n_observations = len(df)
    n_groups = df['group_number'].nunique()

    report.append(f"\nTotal participants: {n_participants}")
    report.append(f"Total observations (clips evaluated): {n_observations}")
    report.append(f"Observations per participant: {n_observations / n_participants:.1f}")
    report.append(f"Number of experimental groups: {n_groups}")

    # False Memory Analysis
    if 'false_memory' in results:
        report.append("\n" + "-" * 80)
        report.append("2. FALSE MEMORY ANALYSIS (PRIMARY OUTCOME)")
        report.append("-" * 80)

        fm = results['false_memory']
        report.append(f"\nTreatment rate: {fm['treatment_rate']:.1f}%")
        report.append(f"Neutral rate: {fm['neutral_rate']:.1f}%")
        report.append(f"\nMixed-effects logistic regression (GEE):")
        report.append(f"  Odds Ratio: {fm['odds_ratio']:.2f} [95% CI: {fm['ci_low']:.2f}, {fm['ci_high']:.2f}]")
        report.append(f"  z = {fm['z']:.2f}, p = {fm['p']:.4f}")

    # Multiple comparison results
    if 'multiple_comparison' in results:
        report.append("\n" + "-" * 80)
        report.append("3. MULTIPLE COMPARISON CORRECTION")
        report.append("-" * 80)

        mc = results['multiple_comparison']
        report.append(f"\nMethod: {mc['method']}")
        report.append("\nCorrected p-values:")
        for name, p_orig, p_corr, sig in zip(mc['test_names'], mc['p_original'],
                                               mc['p_corrected'], mc['significant']):
            sig_marker = "*" if sig else ""
            report.append(f"  {name}: p = {p_corr:.4f} {sig_marker}")

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
    print("=" * 70)
    print("Behavioral Data Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 70)
    print("\nMethodological improvements in this version:")
    print("  ✓ Mixed-effects models for nested data structure")
    print("  ✓ Multiple comparison correction (FDR)")
    print("  ✓ Robust Korean text parsing")
    print("=" * 70)

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
    print(f"Unique participants: {df_combined['participant_id'].nunique()}")

    # Parse responses
    print("\n>>> Parsing responses...")
    df_combined = parse_responses(df_combined)

    # Save processed data
    processed_path = output_dir / 'processed_data.csv'
    df_combined.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"Processed data saved: {processed_path}")

    # Run analyses with mixed-effects models
    print("\n>>> Running mixed-effects analysis...")
    df_analyzed, results = analyze_with_mixed_effects(df_combined)

    # Factor analysis
    analyze_judgment_factors(df_combined)

    # Generate visualizations
    print("\n>>> Generating visualizations...")
    plot_condition_comparison(df_combined, output_dir, results)
    plot_factor_importance(df_combined, output_dir)

    # Generate report
    print("\n>>> Generating report...")
    generate_report(df_combined, results, output_dir / 'reports' / 'behavioral_analysis_report.txt')

    # Save statistical results
    if results:
        results_df = pd.DataFrame([{
            'analysis': 'false_memory',
            **results.get('false_memory', {})
        }])
        results_df.to_csv(output_dir / 'statistical_results.csv', index=False)
        print(f"Statistical results saved: {output_dir / 'statistical_results.csv'}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return df_combined, results


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
