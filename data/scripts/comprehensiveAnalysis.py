#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis
Analyzes the credit risk dataset in all possible ways
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def loadData():
    """Load the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('../raw/creditRiskDataset.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basicInfo(df):
    """Basic dataset information"""
    print("\n" + "="*60)
    print("BASIC DATASET INFORMATION")
    print("="*60)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data Types:\n{df.dtypes}")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())

def missingValuesAnalysis(df):
    """Comprehensive missing values analysis"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    })
    
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing_df['Missing Count'].sum() == 0:
        print("No missing values found in the dataset")
    
    # Check for different types of missing values
    print(f"\nChecking for different missing value representations...")
    missing_representations = ['', ' ', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN']
    
    for col in df.columns:
        for missing_val in missing_representations:
            count = (df[col] == missing_val).sum()
            if count > 0:
                print(f"Column '{col}' has {count} instances of '{missing_val}'")

def dataTypesAnalysis(df):
    """Analyze data types and conversions needed"""
    print("\n" + "="*60)
    print("DATA TYPES ANALYSIS")
    print("="*60)
    
    print("Current data types:")
    print(df.dtypes)
    
    print(f"\nData type distribution:")
    print(df.dtypes.value_counts())
    
    # Check for potential type conversions
    print(f"\nPotential data type issues:")
    
    # Check numeric columns that might be strings
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            print(f"Column '{col}' contains only numeric data but is stored as object")
        except:
            print(f"Column '{col}' contains non-numeric data")
    
    # Check for mixed types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_types = df[col].apply(lambda x: type(x).__name__).unique()
            if len(unique_types) > 1:
                print(f"Column '{col}' has mixed data types: {unique_types}")

def numericColumnsAnalysis(df):
    """Comprehensive analysis of numeric columns"""
    print("\n" + "="*60)
    print("NUMERIC COLUMNS ANALYSIS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {list(numeric_cols)}")
    
    print(f"\nDescriptive Statistics:")
    print(df[numeric_cols].describe())
    
    print(f"\nSkewness Analysis:")
    for col in numeric_cols:
        skewness = stats.skew(df[col])
        print(f"{col}: {skewness:.3f} ({'Right skewed' if skewness > 1 else 'Left skewed' if skewness < -1 else 'Approximately normal'})")
    
    print(f"\nKurtosis Analysis:")
    for col in numeric_cols:
        kurtosis = stats.kurtosis(df[col])
        print(f"{col}: {kurtosis:.3f} ({'Heavy tails' if kurtosis > 3 else 'Light tails' if kurtosis < -1 else 'Normal tails'})")
    
    # Outlier detection using IQR method
    print(f"\nOutlier Analysis (IQR Method):")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(df)) * 100
        
        print(f"{col}: {outlier_count} outliers ({outlier_percent:.2f}%) - Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

def categoricalColumnsAnalysis(df):
    """Comprehensive analysis of categorical columns"""
    print("\n" + "="*60)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(f"Unique values: {df[col].nunique()}")
        print(f"Value counts:")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))
        
        # Check for high cardinality
        if df[col].nunique() > 50:
            print(f"WARNING: High cardinality column ({df[col].nunique()} unique values)")
        
        # Check for imbalanced categories
        if df[col].nunique() <= 10:
            max_count = value_counts.iloc[0]
            min_count = value_counts.iloc[-1]
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 10:
                print(f"WARNING: Highly imbalanced categories (ratio: {imbalance_ratio:.1f}:1)")

def targetVariableAnalysis(df):
    """Detailed analysis of the target variable"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    target_col = 'loan_status'
    
    print(f"Target variable: {target_col}")
    print(f"Value counts:")
    target_counts = df[target_col].value_counts()
    print(target_counts)
    
    print(f"\nPercentage distribution:")
    target_percent = df[target_col].value_counts(normalize=True) * 100
    print(target_percent)
    
    # Check for class imbalance
    imbalance_ratio = target_counts.max() / target_counts.min()
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("WARNING: Significant class imbalance detected")
    else:
        print("Class distribution is relatively balanced")

def correlationAnalysis(df):
    """Correlation analysis between variables"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Numeric correlations
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Find high correlations
    print(f"\nHigh Correlations (|r| > 0.7):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                print(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_value:.3f}")

def featureTargetAnalysis(df):
    """Analysis of features vs target variable"""
    print("\n" + "="*60)
    print("FEATURE vs TARGET ANALYSIS")
    print("="*60)
    
    target_col = 'loan_status'
    
    # Numeric features vs target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    print("Numeric Features vs Target:")
    for col in numeric_cols:
        grouped = df.groupby(target_col)[col].agg(['mean', 'std', 'count'])
        print(f"\n{col}:")
        print(grouped)
    
    # Categorical features vs target
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\nCategorical Features vs Target:")
    for col in categorical_cols:
        print(f"\n{col}:")
        cross_tab = pd.crosstab(df[col], df[target_col], normalize='index')
        print(cross_tab.round(3))

def dataQualityIssues(df):
    """Identify potential data quality issues"""
    print("\n" + "="*60)
    print("DATA QUALITY ISSUES")
    print("="*60)
    
    issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for negative values where they shouldn't be
    if (df['person_age'] < 0).any():
        issues.append("Negative ages found")
    
    if (df['person_income'] < 0).any():
        issues.append("Negative incomes found")
    
    if (df['loan_amnt'] < 0).any():
        issues.append("Negative loan amounts found")
    
    # Check for unrealistic values
    if (df['person_age'] > 100).any():
        issues.append("Ages over 100 found")
    
    if (df['person_age'] < 18).any():
        issues.append("Ages under 18 found")
    
    # Check for missing values in critical columns
    critical_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_status']
    for col in critical_cols:
        if df[col].isnull().any():
            issues.append(f"Missing values in critical column: {col}")
    
    if issues:
        print("Data Quality Issues Found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("No major data quality issues found")

def statisticalTests(df):
    """Perform statistical tests"""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # Normality tests for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'loan_status']
    
    print("Normality Tests (Shapiro-Wilk):")
    for col in numeric_cols[:5]:  # Test first 5 numeric columns
        sample = df[col].sample(min(5000, len(df)))  # Sample for large datasets
        statistic, p_value = stats.shapiro(sample)
        print(f"{col}: p-value = {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Not normal'})")

def generateSummaryReport(df):
    """Generate a summary report"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print(f"Dataset Summary:")
    print(f"- Total records: {len(df):,}")
    print(f"- Total features: {len(df.columns)}")
    print(f"- Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"- Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Duplicate rows: {df.duplicated().sum()}")
    
    # Target distribution
    target_dist = df['loan_status'].value_counts(normalize=True)
    print(f"- Default rate: {target_dist[1]*100:.1f}%")
    print(f"- Non-default rate: {target_dist[0]*100:.1f}%")

def main():
    """Main analysis function"""
    print("COMPREHENSIVE DATA QUALITY ANALYSIS")
    print("Credit Risk Dataset")
    
    # Load data
    df = loadData()
    
    # Run all analyses
    basicInfo(df)
    missingValuesAnalysis(df)
    dataTypesAnalysis(df)
    numericColumnsAnalysis(df)
    categoricalColumnsAnalysis(df)
    targetVariableAnalysis(df)
    correlationAnalysis(df)
    featureTargetAnalysis(df)
    dataQualityIssues(df)
    statisticalTests(df)
    generateSummaryReport(df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
