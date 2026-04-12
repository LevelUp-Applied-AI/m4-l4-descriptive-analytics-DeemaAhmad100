"""Tier 3 — Statistical Simulation and Power Analysis
Hashemite Technical University - Student Performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.stats.power as smp
from statsmodels.stats.power import TTestIndPower


def bootstrap_ci(df, column='gpa', group_col='has_internship', n_boot=10000, alpha=0.05):
    """Bootstrap 95% Confidence Interval for mean difference"""
    print("\n🔬 Tier 3 - Part 1: Bootstrap Confidence Interval")
    
    yes = df[df[group_col] == 'Yes'][column].dropna().values
    no = df[df[group_col] == 'No'][column].dropna().values
    
    boot_diffs = []
    for _ in range(n_boot):
        sample_yes = np.random.choice(yes, size=len(yes), replace=True)
        sample_no = np.random.choice(no, size=len(no), replace=True)
        boot_diffs.append(np.mean(sample_yes) - np.mean(sample_no))
    
    boot_diffs = np.array(boot_diffs)
    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    mean_diff = np.mean(boot_diffs)
    
    print(f"Mean Difference (Yes - No): {mean_diff:.4f}")
    print(f"95% Bootstrap CI: [{lower:.4f}, {upper:.4f}]")
    
    # Compare with parametric t-test
    t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False)
    print(f"Parametric t-test p-value: {p_val:.5f}")
    
    return {"mean_diff": mean_diff, "ci_lower": lower, "ci_upper": upper}


def power_analysis(df, effect_size=None):
    """Power Analysis - What sample size do we need?"""
    print("\n🔬 Tier 3 - Part 2: Power Analysis")
    
    yes = df[df['has_internship'] == 'Yes']['gpa'].dropna()
    no = df[df['has_internship'] == 'No']['gpa'].dropna()
    
    if effect_size is None:
        # Calculate Cohen's d as effect size
        mean_diff = yes.mean() - no.mean()
        pooled_std = np.sqrt((yes.var() * (len(yes)-1) + no.var() * (len(no)-1)) / (len(yes) + len(no) - 2))
        effect_size = mean_diff / pooled_std if pooled_std != 0 else 0.5
    
    print(f"Observed Effect Size (Cohen's d): {effect_size:.3f}")
    
    # Power analysis
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        power=0.80,
        alpha=0.05,
        ratio=1,          # equal group sizes
        alternative='two-sided'
    )
    
    print(f"Required sample size per group for 80% power: {int(np.ceil(sample_size))}")
    print(f"Total required sample: {int(np.ceil(sample_size * 2))}")
    
    return sample_size


def simulation_false_positive(n_sim=5000, alpha=0.05):
    """Simulation: Check if false positive rate matches alpha"""
    print("\n🔬 Tier 3 - Part 3: False Positive Rate Simulation")
    
    false_positives = 0
    for _ in range(n_sim):
        # Generate two groups with NO real difference (same distribution)
        group1 = np.random.normal(loc=3.0, scale=0.6, size=200)
        group2 = np.random.normal(loc=3.0, scale=0.6, size=200)
        
        _, p_value = stats.ttest_ind(group1, group2)
        if p_value < alpha:
            false_positives += 1
    
    observed_rate = false_positives / n_sim
    print(f"Simulated False Positive Rate: {observed_rate:.4f} (expected ≈ {alpha})")
    print(f"Number of false positives: {false_positives} out of {n_sim} simulations")
    
    return observed_rate


def main():
    print("🚀 Starting Tier 3 — Advanced Statistical Analysis\n")
    
    # Load data
    df = pd.read_csv("data/student_performance.csv")
    
    # Run all Tier 3 components
    bootstrap_ci(df)
    power_analysis(df)
    simulation_false_positive(n_sim=5000)
    
    print("\n" + "="*80)
    print("🎉 TIER 3 COMPLETED SUCCESSFULLY!")
    print("All three parts (Bootstrap, Power Analysis, Simulation) are done.")
    print("="*80)


if __name__ == "__main__":
    main()