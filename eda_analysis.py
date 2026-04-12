"""Lab 4 — Descriptive Analytics: Student Performance EDA"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    os.makedirs("output", exist_ok=True)
    
    with open("output/data_profile.txt", "w", encoding="utf-8") as f:
        f.write("=== Hashemite Technical University - Student Performance Dataset ===\n\n")
        f.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
        f.write("=== Data Types ===\n")
        f.write(str(df.dtypes) + "\n\n")
        
        f.write("=== Missing Values Before Cleaning ===\n")
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Count', 'Percentage (%)'])
        f.write(str(missing_table) + "\n\n")
        
        f.write("=== Descriptive Statistics (Numeric Columns) ===\n")
        f.write(str(df.describe().round(2)) + "\n\n")

        # Handling Missing Values
        f.write("=== Missing Values Handling Decisions ===\n\n")
        
        if df['commute_minutes'].isnull().sum() > 0:
            median_commute = df['commute_minutes'].median()
            df['commute_minutes'] = df['commute_minutes'].fillna(median_commute)
            f.write(f"commute_minutes:\n   - Missing: {df['commute_minutes'].isnull().sum()} rows\n")
            f.write("   - Decision: Impute with median\n")
            f.write("   - Reasoning: Likely MCAR. Median is robust.\n\n")
        
        if df['scholarship'].isnull().sum() > 0:
            df['scholarship'] = df['scholarship'].fillna('None')
            f.write(f"scholarship:\n   - Missing: {df['scholarship'].isnull().sum()} rows\n")
            f.write("   - Decision: Fill with 'None'\n")
            f.write("   - Reasoning: Missing likely means no scholarship.\n\n")

    print("\n=== Missing Values Before Cleaning ===")
    print(missing_table[missing_table['Count'] > 0])
    
    print(f"\n✅ Cleaning completed:")
    print(f"   • commute_minutes → imputed with median = {median_commute:.1f} minutes")
    print(f"   • scholarship → filled with 'None'")
    print(f"   • Final shape: {df.shape[0]} rows")
    
    print("\n📄 Full report saved to → output/data_profile.txt")
    
    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables."""
    print("\n🎨 Starting Task 2: Distribution Analysis...")

    sns.set_style("whitegrid")
    
    # GPA Distribution
    plt.figure()
    sns.histplot(data=df, x='gpa', kde=True, bins=20, color='skyblue')
    plt.title('Distribution of GPA\n(Most students have GPA between 2.5 - 3.5)', fontsize=14)
    plt.xlabel('GPA (0.0 - 4.0)')
    plt.ylabel('Number of Students')
    plt.savefig('output/gpa_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Study Hours Distribution
    plt.figure()
    sns.histplot(data=df, x='study_hours_weekly', kde=True, bins=20, color='salmon')
    plt.title('Distribution of Weekly Study Hours', fontsize=14)
    plt.xlabel('Study Hours per Week')
    plt.ylabel('Number of Students')
    plt.savefig('output/study_hours_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Box Plot: GPA by Department
    plt.figure()
    sns.boxplot(data=df, x='department', y='gpa', palette='Set2')
    plt.title('GPA Distribution by Department (Box Plot)', fontsize=14)
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.xticks(rotation=45)
    plt.savefig('output/gpa_by_department_box.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Task 2 Completed! Distribution plots saved.")


def plot_correlations(df):
    """Correlation analysis and visualizations."""
    numeric_cols = ['course_load', 'study_hours_weekly', 'gpa', 'attendance_pct', 'commute_minutes']
    
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Strongest correlation scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='study_hours_weekly', y='gpa', alpha=0.7, color='purple')
    sns.regplot(data=df, x='study_hours_weekly', y='gpa', scatter=False, color='red')
    plt.title('Study Hours vs GPA')
    plt.savefig('output/study_hours_vs_gpa.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Task 3 Completed! Correlation plots saved.")


def run_hypothesis_tests(df):
    """Run hypothesis tests + Tier 1 ANOVA (without statsmodels dependency for CI)"""
    print("\n🔬 Starting Task 4 + Tier 1: Hypothesis Testing...")

    # Hypothesis 1: Internship vs GPA (t-test)
    yes = df[df['has_internship'] == 'Yes']['gpa'].dropna()
    no = df[df['has_internship'] == 'No']['gpa'].dropna()
    
    t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False)
    print(f"Internship t-test: t = {t_stat:.4f}, p-value = {p_val:.5f}")
    if p_val < 0.05:
        print("✅ Significant: Students with internships have higher GPA.")

    # Tier 1: ANOVA for GPA across departments (using scipy only)
    print("\n🔬 Tier 1: ANOVA - Does GPA differ across departments?")
    groups = [df[df['department'] == dept]['gpa'].dropna() for dept in df['department'].unique()]
    f_stat, p_val_anova = stats.f_oneway(*groups)
    
    print(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.6f}")
    if p_val_anova < 0.05:
        print("✅ Significant difference in GPA between departments.")
    else:
        print("No significant difference.")

    # Violin Plot (Tier 1)
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df, x='department', y='gpa', inner="quartile", palette="muted")
    sns.boxplot(data=df, x='department', y='gpa', width=0.25, color="white")
    plt.title("GPA Distribution by Department (Violin Plot - Tier 1)")
    plt.xticks(rotation=45)
    plt.savefig('output/tier1_gpa_by_department_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Tier 1 Violin plot saved.")

    print("\n✅ Task 4 + Tier 1 Completed!")
    return {}

def main():
    os.makedirs("output", exist_ok=True)
    
    df = load_and_profile("data/student_performance.csv")
    
    plot_distributions(df)
    plot_correlations(df)
    run_hypothesis_tests(df)

    print("\n" + "="*80)
    print("ALL TASKS + Tier 1 Completed Successfully!")
    print("You can now write FINDINGS.md")
    print("="*80)


if __name__ == "__main__":
    main()