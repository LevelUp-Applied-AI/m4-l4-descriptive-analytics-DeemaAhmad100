"""Tier 2 — Automated EDA Report Generator

Reusable module that performs full Exploratory Data Analysis on any DataFrame.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')   # ← important
import matplotlib.pyplot as plt

def generate_eda_report(df, 
                       output_dir="output/automated_eda",
                       numeric_cols=None,
                       title="Automated EDA Report",
                       style="whitegrid"):
    """
    Generate a complete automated EDA report for any DataFrame.
    """
    import os
    import seaborn as sns
    import numpy as np
    from datetime import datetime

    matplotlib.use('Agg')   # Force non-interactive backend
    sns.set_style(style)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 Starting Automated EDA Report: {title}")
    print("=" * 70)
    
    # 1. Data Profile
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'string', 'category']).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    }
    
    with open(f"{output_dir}/data_profile.txt", "w", encoding="utf-8") as f:
        f.write(f"Automated EDA Report - {title}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Shape: {profile['shape'][0]} rows × {profile['shape'][1]} columns\n\n")
        f.write("Numeric Columns: " + str(profile['numeric_columns']) + "\n\n")
        f.write("Missing Values:\n" + str(pd.Series(profile['missing_values'])) + "\n")
    
    print(f"✅ Data profile saved")

    # 2. Distribution Plots
    if numeric_cols is None:
        numeric_cols = profile['numeric_columns']
    
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.savefig(f"{output_dir}/dist_{col}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Distribution plots saved for {len(numeric_cols)} columns")

    # 3. Correlation Heatmap
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Correlation heatmap saved")

    # 4. Missing Data Visualization
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if not missing_pct.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, hue=missing_pct.index, palette='Reds_r', legend=False)
        plt.title('Missing Values Percentage by Column')
        plt.xlabel('Missing %')
        plt.savefig(f"{output_dir}/missing_data.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Missing data visualization saved")

    # 5. Outlier Summary
    outlier_summary = {}
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            outlier_summary[col] = outliers
    
    with open(f"{output_dir}/outlier_summary.txt", "w", encoding="utf-8") as f:
        f.write("Outlier Summary (IQR Method)\n\n")
        for col, count in outlier_summary.items():
            f.write(f"{col}: {count} outliers\n")
    
    print("✅ Outlier summary saved")
    print(f"🎉 Automated EDA Report completed! Files saved in: {output_dir}/")
    
    return profile


# ====================== Example Usage ======================
if __name__ == "__main__":
    # Test with student performance data
    df = pd.read_csv("data/student_performance.csv")
    generate_eda_report(
        df=df,
        output_dir="output/automated_eda",
        title="Hashemite Technical University - Student Performance",
        style="whitegrid"
    )