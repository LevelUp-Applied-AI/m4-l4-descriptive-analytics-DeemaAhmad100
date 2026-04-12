"""Tests for Tier 2 — Automated EDA Report Generator"""

import pytest
import pandas as pd
import numpy as np
from eda_report import generate_eda_report
from pathlib import Path


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing"""
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 30, 100),
        'gpa': np.random.uniform(2.0, 4.0, 100),
        'study_hours': np.random.uniform(5, 40, 100),
        'department': np.random.choice(['CS', 'Engineering', 'Business', 'Biology'], 100),
        'has_internship': np.random.choice(['Yes', 'No'], 100),
        'missing_col': [np.nan if i % 7 == 0 else i for i in range(100)]
    }
    return pd.DataFrame(data)


def test_generate_eda_report_creates_output(sample_df, tmp_path):
    """Test that generate_eda_report creates the output directory and files"""
    output_dir = tmp_path / "test_eda"
    
    profile = generate_eda_report(
        df=sample_df,
        output_dir=str(output_dir),
        title="Test EDA Report",
        style="whitegrid"
    )
    
    # Check directory exists
    assert output_dir.exists()
    
    # Check important files were created
    assert (output_dir / "data_profile.txt").exists()
    assert (output_dir / "correlation_heatmap.png").exists()
    assert (output_dir / "outlier_summary.txt").exists()
    
    # Check profile returned correctly
    assert isinstance(profile, dict)
    assert "shape" in profile
    assert "numeric_columns" in profile


def test_handles_different_data_types(sample_df):
    """Test that the function handles mixed data types and missing values"""
    profile = generate_eda_report(
        df=sample_df,
        output_dir="output/test_mixed",
        numeric_cols=['age', 'gpa', 'study_hours']
    )
    
    assert len(profile["numeric_columns"]) >= 3
    assert "missing_col" in profile["missing_values"]


def test_outlier_detection(sample_df):
    """Test outlier summary logic"""
    # Add obvious outlier
    sample_df.loc[0, 'gpa'] = 5.0  # unrealistic GPA
    
    profile = generate_eda_report(
        df=sample_df,
        output_dir="output/test_outlier"
    )
    
    # Just check it runs without error
    assert Path("output/test_outlier/outlier_summary.txt").exists()


def test_empty_or_small_dataframe():
    """Test edge case with small/empty DataFrame"""
    small_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    profile = generate_eda_report(small_df, output_dir="output/test_small")
    
    assert profile["shape"] == (3, 2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])