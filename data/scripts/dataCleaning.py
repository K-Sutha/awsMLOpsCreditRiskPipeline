#!/usr/bin/env python3
"""
Production-Level Data Cleaning Script
Addresses all data quality issues with production-ready approaches
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

def setupLogging():
    """Setup production-level logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../processed/cleaned/dataCleaning.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def loadData():
    """Load the raw dataset with production-level validation"""
    logger = logging.getLogger(__name__)
    logger.info("Loading raw dataset...")
    
    try:
        df = pd.read_csv('../raw/creditRiskDataset.csv')
        logger.info(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def removeDuplicates(df):
    """Remove duplicate rows with production-level tracking"""
    logger = logging.getLogger(__name__)
    logger.info("Removing duplicate rows...")
    
    initial_count = len(df)
    df = df.drop_duplicates(keep='first')  # Keep first occurrence
    removed_count = initial_count - len(df)
    
    logger.info(f"Removed {removed_count} duplicate rows")
    logger.info(f"Remaining rows: {len(df)}")
    
    # Track cleaning statistics
    cleaning_stats = {
        'duplicates_removed': removed_count,
        'rows_after_dedup': len(df)
    }
    
    return df, cleaning_stats

def handleOutliers(df):
    """Handle outliers using production-level statistical methods"""
    logger = logging.getLogger(__name__)
    logger.info("Handling outliers using statistical methods...")
    
    outlier_stats = {}
    
    # 1. Age outliers - Cap at realistic maximum
    age_outliers = df[df['person_age'] > 100]
    logger.info(f"Found {len(age_outliers)} records with age > 100")
    df['person_age'] = df['person_age'].clip(upper=100)
    outlier_stats['age_capped'] = len(age_outliers)
    
    # 2. Income outliers - Use robust statistical method
    # Remove negative values first
    negative_income = df[df['person_income'] < 0]
    logger.info(f"Found {len(negative_income)} records with negative income")
    df['person_income'] = df['person_income'].clip(lower=0)
    outlier_stats['negative_income_fixed'] = len(negative_income)
    
    # Use 99th percentile for income capping (production approach)
    income_99th = df['person_income'].quantile(0.99)
    income_outliers = df[df['person_income'] > income_99th]
    logger.info(f"Found {len(income_outliers)} income outliers (>{income_99th:,.0f})")
    df['person_income'] = df['person_income'].clip(upper=income_99th)
    outlier_stats['income_capped'] = len(income_outliers)
    
    # 3. Loan amount outliers
    negative_loans = df[df['loan_amnt'] < 0]
    logger.info(f"Found {len(negative_loans)} records with negative loan amounts")
    df['loan_amnt'] = df['loan_amnt'].clip(lower=0)
    outlier_stats['negative_loans_fixed'] = len(negative_loans)
    
    loan_99th = df['loan_amnt'].quantile(0.99)
    loan_outliers = df[df['loan_amnt'] > loan_99th]
    logger.info(f"Found {len(loan_outliers)} loan amount outliers (>{loan_99th:,.0f})")
    df['loan_amnt'] = df['loan_amnt'].clip(upper=loan_99th)
    outlier_stats['loan_capped'] = len(loan_outliers)
    
    # 4. Employment length outliers
    emp_outliers = df[df['person_emp_length'].astype(str).str.contains('123')]
    logger.info(f"Found {len(emp_outliers)} records with impossible employment length")
    outlier_stats['emp_length_impossible'] = len(emp_outliers)
    
    return df, outlier_stats

def handleMissingValues(df):
    """Handle missing values using production-level imputation strategies"""
    logger = logging.getLogger(__name__)
    logger.info("Handling missing values with advanced imputation...")
    
    missing_stats = {}
    
    # 1. Handle missing interest rates (marked as '?')
    missing_rates = df[df['loan_int_rate'] == '?']
    logger.info(f"Found {len(missing_rates)} records with missing interest rates")
    missing_stats['interest_rates_missing'] = len(missing_rates)
    
    # Convert to numeric, replacing '?' with NaN
    df['loan_int_rate'] = pd.to_numeric(df['loan_int_rate'], errors='coerce')
    
    # Advanced imputation: Use median by loan grade and income level
    df['income_quartile'] = pd.qcut(df['person_income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    median_rates = df.groupby(['loan_grade', 'income_quartile'])['loan_int_rate'].median()
    
    # Fill missing values using stratified median
    for grade in df['loan_grade'].unique():
        for quartile in df['income_quartile'].unique():
            mask = (df['loan_grade'] == grade) & (df['income_quartile'] == quartile) & (df['loan_int_rate'].isna())
            if len(mask) > 0:
                df.loc[mask, 'loan_int_rate'] = median_rates.get((grade, quartile), df['loan_int_rate'].median())
    
    # Fill any remaining missing values with overall median
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df = df.drop('income_quartile', axis=1)  # Clean up temporary column
    
    # 2. Handle missing employment length
    df['person_emp_length'] = pd.to_numeric(df['person_emp_length'], errors='coerce')
    missing_emp = df['person_emp_length'].isna().sum()
    logger.info(f"Found {missing_emp} records with missing employment length")
    missing_stats['employment_missing'] = missing_emp
    
    # Fill with median by age group
    df['age_group'] = pd.cut(df['person_age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '45+'])
    median_emp_by_age = df.groupby('age_group')['person_emp_length'].median()
    
    for age_group in df['age_group'].unique():
        mask = (df['age_group'] == age_group) & (df['person_emp_length'].isna())
        df.loc[mask, 'person_emp_length'] = median_emp_by_age[age_group]
    
    # Fill any remaining missing values with overall median
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df = df.drop('age_group', axis=1)  # Clean up temporary column
    
    # 3. Cap employment length at realistic maximum
    emp_extreme = df[df['person_emp_length'] > 50]
    logger.info(f"Found {len(emp_extreme)} records with employment length > 50 years")
    df['person_emp_length'] = df['person_emp_length'].clip(upper=50)
    missing_stats['employment_capped'] = len(emp_extreme)
    
    return df, missing_stats

def createProductionFeatures(df):
    """Create production-ready features with business logic"""
    logger = logging.getLogger(__name__)
    logger.info("Creating production-ready features...")
    
    feature_stats = {}
    
    # 1. Age-based features
    df['age_group'] = pd.cut(df['person_age'], 
                            bins=[0, 25, 35, 45, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior'])
    feature_stats['age_groups_created'] = df['age_group'].nunique()
    
    # 2. Income-based features
    df['income_quartile'] = pd.qcut(df['person_income'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    df['has_high_income'] = (df['person_income'] > df['person_income'].quantile(0.75)).astype(int)
    feature_stats['income_features_created'] = 2
    
    # 3. Employment stability features
    df['employment_stability'] = df['person_emp_length'] / (df['person_age'] - 18).clip(lower=1)
    df['has_stable_employment'] = (df['employment_stability'] > 0.3).astype(int)
    feature_stats['employment_features_created'] = 2
    
    # 4. Loan risk features
    df['debt_to_income_ratio'] = df['loan_percent_income']
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt'].clip(lower=1)
    df['has_high_debt_ratio'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
    feature_stats['loan_features_created'] = 3
    
    # 5. Risk indicators (production-level business logic)
    df['risk_grade_group'] = df['loan_grade'].map({
        'A': 'LowRisk', 'B': 'LowRisk',
        'C': 'MediumRisk', 'D': 'MediumRisk', 
        'E': 'HighRisk', 'F': 'HighRisk', 'G': 'HighRisk'
    })
    
    # 6. Historical default indicator
    df['has_historical_default'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
    
    logger.info(f"Created {sum(feature_stats.values())} new features")
    return df, feature_stats

def encodeCategoricalVariables(df):
    """Encode categorical variables using production-level techniques"""
    logger = logging.getLogger(__name__)
    logger.info("Encoding categorical variables...")
    
    encoding_stats = {}
    
    # 1. One-hot encode high-cardinality categorical variables
    categorical_cols = ['person_home_ownership', 'loan_intent', 'age_group', 'income_quartile', 'risk_grade_group']
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)  # Drop first to avoid multicollinearity
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
            encoding_stats[f'{col}_encoded'] = len(dummies.columns)
            logger.info(f"One-hot encoded: {col} -> {len(dummies.columns)} features")
    
    # 2. Binary encode binary categorical variables
    binary_cols = ['cb_person_default_on_file']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'N': 0, 'Y': 1})
            encoding_stats[f'{col}_binary_encoded'] = 1
            logger.info(f"Binary encoded: {col}")
    
    # 3. Ordinal encode loan grade (maintain order)
    if 'loan_grade' in df.columns:
        grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df['loan_grade_encoded'] = df['loan_grade'].map(grade_order)
        df = df.drop('loan_grade', axis=1)
        encoding_stats['loan_grade_ordinal_encoded'] = 1
        logger.info("Ordinal encoded: loan_grade")
    
    return df, encoding_stats

def finalValidation(df):
    """Final production-level data validation"""
    logger = logging.getLogger(__name__)
    logger.info("Performing final data validation...")
    
    validation_results = {}
    
    # Basic statistics
    validation_results['final_shape'] = df.shape
    validation_results['missing_values'] = df.isnull().sum().sum()
    validation_results['duplicate_rows'] = df.duplicated().sum()
    
    # Data type validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    validation_results['numeric_columns'] = len(numeric_cols)
    validation_results['categorical_columns'] = len(categorical_cols)
    
    # Check for infinite values
    infinite_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    validation_results['infinite_values'] = infinite_count
    
    # Target variable distribution
    if 'loan_status' in df.columns:
        target_dist = df['loan_status'].value_counts(normalize=True)
        validation_results['default_rate'] = target_dist[1]
        validation_results['non_default_rate'] = target_dist[0]
        validation_results['class_imbalance_ratio'] = target_dist[0] / target_dist[1]
    
    # Memory usage
    validation_results['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    logger.info(f"Final validation results: {validation_results}")
    return validation_results

def saveCleaningReport(all_stats):
    """Save comprehensive cleaning report"""
    logger = logging.getLogger(__name__)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all numpy types in stats
    cleaned_stats = {k: convert_numpy_types(v) for k, v in all_stats.items()}
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'cleaning_summary': cleaned_stats,
        'data_quality_score': calculateDataQualityScore(cleaned_stats)
    }
    
    # Save to JSON
    with open('../processed/cleaned/cleaningReport.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Cleaning report saved to cleaningReport.json")

def calculateDataQualityScore(stats):
    """Calculate overall data quality score"""
    # Simple scoring system
    score = 100
    
    # Deduct points for issues
    if stats.get('duplicates_removed', 0) > 0:
        score -= 5
    if stats.get('missing_values', 0) > 0:
        score -= 10
    if stats.get('infinite_values', 0) > 0:
        score -= 15
    
    return max(score, 0)

def main():
    """Main production-level cleaning function"""
    # Setup logging
    logger = setupLogging()
    
    logger.info("Starting production-level data cleaning process")
    
    try:
        # Load data
        df = loadData()
        
        # Apply cleaning steps with statistics tracking
        all_stats = {}
        
        # Step 1: Remove duplicates
        df, dup_stats = removeDuplicates(df)
        all_stats.update(dup_stats)
        
        # Step 2: Handle outliers
        df, outlier_stats = handleOutliers(df)
        all_stats.update(outlier_stats)
        
        # Step 3: Handle missing values
        df, missing_stats = handleMissingValues(df)
        all_stats.update(missing_stats)
        
        # Step 4: Create production features
        df, feature_stats = createProductionFeatures(df)
        all_stats.update(feature_stats)
        
        # Step 5: Encode categorical variables
        df, encoding_stats = encodeCategoricalVariables(df)
        all_stats.update(encoding_stats)
        
        # Step 6: Final validation
        validation_results = finalValidation(df)
        all_stats.update(validation_results)
        
        # Save cleaned data
        output_path = '../processed/cleaned/cleanedDataset.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")
        
        # Save cleaning report
        saveCleaningReport(all_stats)
        
        logger.info("Data cleaning process completed successfully")
        logger.info(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Data quality score: {calculateDataQualityScore(all_stats)}/100")
        
    except Exception as e:
        logger.error(f"Error in data cleaning process: {e}")
        raise

if __name__ == "__main__":
    main()
