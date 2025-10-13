#!/usr/bin/env python3
"""
Data Validation Script
Basic validation and statistics for the credit risk dataset
"""

import pandas as pd
import numpy as np

def validateDataset(csvFile):
    """
    Validate and analyze the credit risk dataset
    """
    print(f"Analyzing {csvFile}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csvFile)
        
        print(f"\nDataset Overview:")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nColumn Information:")
        for col in df.columns:
            dtype = df[col].dtype
            nulls = df[col].isnull().sum()
            null_pct = (nulls / len(df)) * 100
            print(f"{col}: {dtype} | Nulls: {nulls} ({null_pct:.1f}%)")
        
        print(f"\nTarget Variable Analysis (loan_status):")
        target_counts = df['loan_status'].value_counts()
        print(f"Non-default (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"Default (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        print(f"\nNumeric Columns Summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe().round(2))
        
        print(f"\nCategorical Columns Summary:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:  # Show values if few unique
                print(f"  Values: {list(df[col].unique())}")
        
        # Check for potential issues
        print(f"\nData Quality Checks:")
        issues = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"Missing values in: {missing_cols}")
        else:
            print("No missing values found")
        
        # Check for negative values where they shouldn't be
        if (df['person_age'] < 0).any():
            issues.append("Negative ages found")
        if (df['person_income'] < 0).any():
            issues.append("Negative incomes found")
        if (df['loan_amnt'] < 0).any():
            issues.append("Negative loan amounts found")
        
        if not issues:
            print("No obvious data quality issues found")
        else:
            for issue in issues:
                print(f"Warning: {issue}")
        
        return df
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

def main():
    csvFile = "creditRiskDataset.csv"
    df = validateDataset(csvFile)
    
    if df is not None:
        print(f"\nDataset validation completed successfully!")
        print(f"Ready for next step: Data folder structure setup")

if __name__ == "__main__":
    main()
