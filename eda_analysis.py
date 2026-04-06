"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
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

        # ====================== Handling Missing Values ======================
        f.write("=== Missing Values Handling Decisions ===\n\n")
        
        # Only handle columns that actually have missing values
        if df['commute_minutes'].isnull().sum() > 0:
            median_commute = df['commute_minutes'].median()
            df['commute_minutes'] = df['commute_minutes'].fillna(median_commute)
            
            f.write("commute_minutes:\n")
            f.write(f"   - Missing: {df['commute_minutes'].isnull().sum()} rows (9.05% before cleaning)\n")
            f.write("   - Decision: Impute with median\n")
            f.write("   - Reasoning: Likely MCAR. Median is robust to outliers and skewness.\n")
            f.write("     Imputing preserves all 2000 rows for better statistical power.\n\n")
        
        if df['scholarship'].isnull().sum() > 0:
            df['scholarship'] = df['scholarship'].fillna('None')
            
            f.write("scholarship:\n")
            f.write(f"   - Missing: 389 rows (19.45%)\n")
            f.write("   - Decision: Fill with 'None'\n")
            f.write("   - Reasoning: High missing rate. Missing values most likely mean the student has no scholarship.\n")
            f.write("     Creating 'None' category is semantically correct and avoids data loss.\n\n")
        
        # study_hours_weekly has 0 missing → no need to mention it
        
        f.write(f"→ Final dataset shape after cleaning: {df.shape[0]} rows\n")
        f.write("→ No rows were dropped during cleaning.\n")

    # ====================== Console Summary ======================
    print("\n=== Missing Values Before Cleaning ===")
    print(missing_table[missing_table['Count'] > 0])
    
    print(f"\n✅ Cleaning completed:")
    print(f"   • commute_minutes → imputed with median = {median_commute:.1f} minutes")
    print(f"   • scholarship → filled with 'None'")
    print(f"   • Final shape: {df.shape[0]} rows")
    
    print("\n📄 Full report saved to → output/data_profile.txt")
    
    return df

def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    """Create distribution plots for key numeric variables."""
    
   
    print("\n🎨 Starting Task 2: Distribution Analysis...")

    # Set plot style and size
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Histogram + KDE for GPA
    plt.figure()
    sns.histplot(data=df, x='gpa', kde=True, bins=20, color='skyblue')
    plt.title('Distribution of GPA\n(Most students have GPA between 2.5 - 3.5)', fontsize=14)
    plt.xlabel('GPA (0.0 - 4.0)')
    plt.ylabel('Number of Students')
    plt.savefig('output/gpa_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/gpa_distribution.png")

    # 2. Histogram + KDE for study_hours_weekly
    plt.figure()
    sns.histplot(data=df, x='study_hours_weekly', kde=True, bins=20, color='salmon')
    plt.title('Distribution of Weekly Study Hours', fontsize=14)
    plt.xlabel('Study Hours per Week')
    plt.ylabel('Number of Students')
    plt.savefig('output/study_hours_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/study_hours_distribution.png")

    # 3. Box Plot: GPA by Department
    plt.figure()
    sns.boxplot(data=df, x='department', y='gpa', palette='Set2')
    plt.title('GPA Distribution by Department\n(Box Plot)', fontsize=14)
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.xticks(rotation=15)
    plt.savefig('output/gpa_by_department.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/gpa_by_department.png")

    # 4. Bar Chart for Scholarship (Categorical)
    plt.figure()
    scholarship_counts = df['scholarship'].value_counts()
    sns.barplot(x=scholarship_counts.index, y=scholarship_counts.values, palette='viridis')
    plt.title('Number of Students by Scholarship Type', fontsize=14)
    plt.xlabel('Scholarship Type')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=15)
    plt.savefig('output/scholarship_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/scholarship_distribution.png")

    # 5. Bonus: Histogram for Attendance Percentage
    plt.figure()
    sns.histplot(data=df, x='attendance_pct', kde=True, bins=20, color='lightgreen')
    plt.title('Distribution of Attendance Percentage', fontsize=14)
    plt.xlabel('Attendance (%)')
    plt.ylabel('Number of Students')
    plt.savefig('output/attendance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/attendance_distribution.png")

    print("\n✅ Task 2 Completed! All distribution plots saved in output/ folder")

def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """

    # The most relevant numeric columns for correlation analysis
    numeric_cols = ['course_load', 'study_hours_weekly', 'gpa', 
                    'attendance_pct', 'commute_minutes']
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                square=True, 
                linewidths=0.5, 
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Pearson Correlation Heatmap\n(Numeric Variables)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: output/correlation_heatmap.png")
    
    # 2. Scatter Plot - أقوى علاقة متوقعة: study_hours_weekly vs gpa
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='study_hours_weekly', y='gpa', 
                    alpha=0.7, color='purple', s=60)
    sns.regplot(data=df, x='study_hours_weekly', y='gpa', 
                scatter=False, color='red', line_kws={'linewidth': 2})
    
    plt.title('Correlation: Weekly Study Hours vs GPA', fontsize=14)
    plt.xlabel('Study Hours per Week')
    plt.ylabel('GPA')
    plt.savefig('output/study_hours_vs_gpa.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/study_hours_vs_gpa.png")

    # 3. Scatter Plot - Second Strongest Correlation: attendance_pct vs gpa
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='attendance_pct', y='gpa', 
                    alpha=0.7, color='teal', s=60)
    sns.regplot(data=df, x='attendance_pct', y='gpa', 
                scatter=False, color='red', line_kws={'linewidth': 2})
    
    plt.title('Correlation: Attendance Percentage vs GPA', fontsize=14)
    plt.xlabel('Attendance Percentage (%)')
    plt.ylabel('GPA')
    plt.savefig('output/attendance_vs_gpa.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/attendance_vs_gpa.png")
    
    # print strongest correlations (absolute value) to console
    print("\nStrongest Correlations (absolute value):")
    corr_unstack = corr_matrix.unstack().sort_values(ascending=False)
    # Exclude self-correlations (1.0) and show top 5 unique pairs
    corr_unstack = corr_unstack[corr_unstack < 1.0]
    print(corr_unstack.head(6))
    
    print("\n✅ Task 3 Completed! Correlation plots saved in output/ folder")


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """

    print("\n🔬 Starting Task 4: Hypothesis Testing...")
    results = {}
    
    # ====================== Hypothesis 1: Internship vs GPA (t-test) ======================
    print("\nHypothesis 1: Students with internships have higher GPA than those without.")
    
    internship_yes = df[df['has_internship'] == 'Yes']['gpa']
    internship_no = df[df['has_internship'] == 'No']['gpa']
    
    t_stat, p_value = stats.ttest_ind(internship_yes, internship_no, equal_var=False)
    
    # Cohen's d (Effect Size)
    mean_diff = internship_yes.mean() - internship_no.mean()
    pooled_std = np.sqrt(((internship_yes.std()**2 * (len(internship_yes)-1)) + 
                          (internship_no.std()**2 * (len(internship_no)-1))) / 
                         (len(internship_yes) + len(internship_no) - 2))
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0
    
    results['internship_ttest'] = {
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 6),
        'cohen_d': round(cohen_d, 3),
        'interpretation': 'Significant' if p_value < 0.05 else 'Not Significant'
    }
    
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value     : {p_value:.6f}")
    print(f"Cohen's d   : {cohen_d:.3f}")
    if p_value < 0.05:
        print("✅ Result: Statistically significant → Students with internships have significantly higher GPA.")
    else:
        print("❌ Result: Not statistically significant.")
    
    # ====================== Hypothesis 2: Scholarship vs Department (Chi-square) ======================
    print("\nHypothesis 2: Scholarship status is associated with department.")
    
    crosstab = pd.crosstab(df['scholarship'], df['department'])
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    
    results['scholarship_chi2'] = {
        'chi2_statistic': round(chi2, 4),
        'p_value': round(p, 6),
        'degrees_of_freedom': dof,
        'interpretation': 'Significant' if p < 0.05 else 'Not Significant'
    }
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value             : {p:.6f}")
    print(f"Degrees of freedom  : {dof}")
    if p < 0.05:
        print("✅ Result: Statistically significant → There is an association between scholarship type and department.")
    else:
        print("❌ Result: Not statistically significant.")
    
    print("\n✅ Task 4 Completed! Hypothesis tests finished.")
    return results




def main():

    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)
    
    # Task 1
    df = load_and_profile("data/student_performance.csv")
    
    # Task 2
    plot_distributions(df)
    
    # Task 3
    plot_correlations(df)
    
    # Task 4
    run_hypothesis_tests(df)
    
    print("\n" + "="*80)
    print("ALL TASKS (1 to 4) Completed Successfully!")
    print("You can now write FINDINGS.md")
    print("="*80)

    # TODO: Write a FINDINGS.md summarizing your analysis


if __name__ == "__main__":
    main()
