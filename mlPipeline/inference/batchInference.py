#!/usr/bin/env python3
"""
Batch Inference Script for Credit Risk Platform
Production-ready batch ML inference with SageMaker Processing
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
import boto3
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchInference:
    """Production-ready batch inference for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info("Batch Inference initialized")
    
    def load_config(self, config_path):
        """Load inference configuration"""
        default_config = {
            "model_name": "credit-risk-ensemble",
            "model_version": "1.0",
            "s3_bucket": "credit-risk-features",
            "input_path": "/opt/ml/input/data/",
            "output_path": "/opt/ml/output/data/",
            "batch_size": 1000,
            "feature_engineering": {
                "create_age_groups": True,
                "create_income_quartiles": True,
                "create_employment_stability": True,
                "create_debt_ratios": True,
                "create_risk_indicators": True
            },
            "risk_thresholds": {
                "low_risk": 0.3,
                "medium_risk": 0.7,
                "high_risk": 1.0
            },
            "output_formats": ["csv", "json", "parquet"]
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_model(self):
        """Load the trained model and artifacts"""
        logger.info("Loading model and artifacts...")
        
        try:
            bucket = self.config['s3_bucket']
            prefix = f"models/{self.config['model_name']}/{self.config['model_version']}/"
            
            # Download model files from S3
            model_key = f"{prefix}ensemble_model.joblib"
            scaler_key = f"{prefix}scaler.joblib"
            features_key = f"{prefix}feature_names.json"
            
            # Create local paths
            local_model_path = "/tmp/ensemble_model.joblib"
            local_scaler_path = "/tmp/scaler.joblib"
            local_features_path = "/tmp/feature_names.json"
            
            # Download from S3
            self.s3_client.download_file(bucket, model_key, local_model_path)
            self.s3_client.download_file(bucket, scaler_key, local_scaler_path)
            self.s3_client.download_file(bucket, features_key, local_features_path)
            
            # Load model and scaler
            self.model = joblib.load(local_model_path)
            self.scaler = joblib.load(local_scaler_path)
            
            # Load feature names
            with open(local_features_path, 'r') as f:
                self.feature_names = json.load(f)
            
            logger.info(f"Model loaded successfully with {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load batch data for inference"""
        logger.info("Loading batch data...")
        
        try:
            input_path = self.config['input_path']
            
            # Look for input files
            input_files = []
            if os.path.exists(input_path):
                for file in os.listdir(input_path):
                    if file.endswith(('.csv', '.json', '.parquet')):
                        input_files.append(os.path.join(input_path, file))
            
            if not input_files:
                raise FileNotFoundError("No input files found")
            
            # Load the first file (assuming single input file)
            input_file = input_files[0]
            logger.info(f"Loading data from: {input_file}")
            
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file)
            elif input_file.endswith('.parquet'):
                df = pd.read_parquet(input_file)
            
            logger.info(f"Data loaded: {df.shape[0]} records, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input data"""
        logger.info("Validating input data...")
        
        try:
            # Check required columns
            required_columns = [
                'person_age', 'person_income', 'person_emp_length',
                'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                'cb_person_default_on_file', 'cb_person_cred_hist_length',
                'person_home_ownership', 'loan_intent', 'loan_grade'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate numeric columns
            numeric_columns = ['person_age', 'person_income', 'person_emp_length', 
                             'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                             'cb_person_cred_hist_length']
            
            for col in numeric_columns:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            df = df.fillna(df.median())
            
            logger.info("Data validation completed")
            return df
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from input data"""
        logger.info("Engineering features...")
        
        try:
            # Age-based features
            if self.config['feature_engineering']['create_age_groups']:
                df['age_group'] = pd.cut(df['person_age'], 
                                       bins=[0, 25, 35, 45, 100], 
                                       labels=['18-25', '26-35', '36-45', '45+'])
            
            # Income-based features
            if self.config['feature_engineering']['create_income_quartiles']:
                df['income_quartile'] = pd.qcut(df['person_income'], 
                                              q=4, 
                                              labels=['Low', 'Medium', 'High', 'VeryHigh'])
            
            # Employment stability
            if self.config['feature_engineering']['create_employment_stability']:
                df['employment_stability'] = df['person_emp_length'] / (df['person_age'] - 18)
                df['has_stable_employment'] = (df['employment_stability'] > 0.3).astype(int)
            
            # Debt ratios
            if self.config['feature_engineering']['create_debt_ratios']:
                df['debt_to_income_ratio'] = df['loan_percent_income']
                df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
                df['has_high_debt_ratio'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
            
            # Risk indicators
            if self.config['feature_engineering']['create_risk_indicators']:
                df['has_high_income'] = (df['person_income'] > df['person_income'].quantile(0.75)).astype(int)
                df['has_historical_default'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
                
                # Risk grade grouping
                df['risk_grade_group'] = df['loan_grade'].map({
                    'A': 'LowRisk', 'B': 'LowRisk',
                    'C': 'MediumRisk', 'D': 'MediumRisk',
                    'E': 'HighRisk', 'F': 'HighRisk', 'G': 'HighRisk'
                })
            
            # One-hot encode categorical variables
            categorical_columns = [
                'person_home_ownership', 'loan_intent', 'age_group', 
                'income_quartile', 'risk_grade_group'
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    df = pd.get_dummies(df, columns=[col], prefix=col)
            
            # Binary encode loan_grade
            if 'loan_grade' in df.columns:
                df['loan_grade_encoded'] = df['loan_grade'].map({
                    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
                })
                df = df.drop('loan_grade', axis=1)
            
            # Binary encode cb_person_default_on_file
            if 'cb_person_default_on_file' in df.columns:
                df['cb_person_default_on_file'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
            
            logger.info(f"Feature engineering completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model prediction"""
        logger.info("Preparing features for prediction...")
        
        try:
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Handle missing values
            df = df.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            logger.info(f"Features prepared. Shape: {X_scaled.shape}")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise
    
    def predict_batch(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make batch predictions"""
        logger.info(f"Making batch predictions for {X.shape[0]} records...")
        
        try:
            # Make predictions
            predictions_proba = self.model.predict_proba(X)
            predictions_class = self.model.predict(X)
            
            # Calculate risk scores
            risk_scores = predictions_proba[:, 1]  # Probability of default
            
            # Determine risk levels
            risk_levels = np.where(
                risk_scores < self.config['risk_thresholds']['low_risk'], 'LOW',
                np.where(
                    risk_scores < self.config['risk_thresholds']['medium_risk'], 'MEDIUM', 'HIGH'
                )
            )
            
            # Calculate confidence scores
            confidence_scores = np.max(predictions_proba, axis=1)
            
            results = {
                'risk_scores': risk_scores,
                'risk_levels': risk_levels,
                'predictions_class': predictions_class,
                'confidence_scores': confidence_scores,
                'predictions_proba': predictions_proba
            }
            
            logger.info("Batch predictions completed")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def create_output_dataframe(self, original_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create output DataFrame with predictions"""
        logger.info("Creating output DataFrame...")
        
        try:
            # Create output DataFrame
            output_df = original_df.copy()
            
            # Add prediction columns
            output_df['prediction_id'] = [f"pred_{i}_{int(datetime.now().timestamp())}" 
                                        for i in range(len(output_df))]
            output_df['risk_score'] = predictions['risk_scores']
            output_df['risk_level'] = predictions['risk_levels']
            output_df['prediction_class'] = predictions['predictions_class']
            output_df['confidence_score'] = predictions['confidence_scores']
            output_df['model_name'] = self.config['model_name']
            output_df['model_version'] = self.config['model_version']
            output_df['prediction_timestamp'] = datetime.now().isoformat()
            
            # Add probability columns
            output_df['prob_no_default'] = predictions['predictions_proba'][:, 0]
            output_df['prob_default'] = predictions['predictions_proba'][:, 1]
            
            logger.info(f"Output DataFrame created with {len(output_df)} records")
            return output_df
            
        except Exception as e:
            logger.error(f"Output DataFrame creation failed: {e}")
            raise
    
    def save_results(self, output_df: pd.DataFrame):
        """Save prediction results"""
        logger.info("Saving prediction results...")
        
        try:
            output_path = self.config['output_path']
            os.makedirs(output_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save in different formats
            if 'csv' in self.config['output_formats']:
                csv_path = os.path.join(output_path, f"batch_predictions_{timestamp}.csv")
                output_df.to_csv(csv_path, index=False)
                logger.info(f"Results saved to CSV: {csv_path}")
            
            if 'json' in self.config['output_formats']:
                json_path = os.path.join(output_path, f"batch_predictions_{timestamp}.json")
                output_df.to_json(json_path, orient='records', indent=2)
                logger.info(f"Results saved to JSON: {json_path}")
            
            if 'parquet' in self.config['output_formats']:
                parquet_path = os.path.join(output_path, f"batch_predictions_{timestamp}.parquet")
                output_df.to_parquet(parquet_path, index=False)
                logger.info(f"Results saved to Parquet: {parquet_path}")
            
            # Save summary statistics
            summary = {
                'total_records': len(output_df),
                'risk_level_distribution': output_df['risk_level'].value_counts().to_dict(),
                'average_risk_score': float(output_df['risk_score'].mean()),
                'average_confidence': float(output_df['confidence_score'].mean()),
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            summary_path = os.path.join(output_path, f"batch_summary_{timestamp}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved to: {summary_path}")
            
            return {
                'output_path': output_path,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Results saving failed: {e}")
            raise
    
    def upload_to_s3(self, local_paths: Dict[str, str]):
        """Upload results to S3"""
        logger.info("Uploading results to S3...")
        
        try:
            bucket = self.config['s3_bucket']
            prefix = f"batch-inference/{self.config['model_name']}/{datetime.now().strftime('%Y/%m/%d')}/"
            
            uploaded_files = {}
            
            for file_type, local_path in local_paths.items():
                if os.path.exists(local_path):
                    s3_key = f"{prefix}{os.path.basename(local_path)}"
                    
                    self.s3_client.upload_file(local_path, bucket, s3_key)
                    uploaded_files[file_type] = f"s3://{bucket}/{s3_key}"
                    
                    logger.info(f"Uploaded {file_type}: {uploaded_files[file_type]}")
            
            return uploaded_files
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def run_batch_inference(self):
        """Run the complete batch inference pipeline"""
        logger.info("Starting batch inference pipeline...")
        
        try:
            # Step 1: Load model
            self.load_model()
            
            # Step 2: Load data
            df = self.load_data()
            
            # Step 3: Validate data
            df = self.validate_data(df)
            
            # Step 4: Engineer features
            df = self.engineer_features(df)
            
            # Step 5: Prepare features
            X = self.prepare_features(df)
            
            # Step 6: Make predictions
            predictions = self.predict_batch(X)
            
            # Step 7: Create output DataFrame
            output_df = self.create_output_dataframe(df, predictions)
            
            # Step 8: Save results
            saved_paths = self.save_results(output_df)
            
            # Step 9: Upload to S3 (if in SageMaker environment)
            if os.path.exists('/opt/ml/'):
                s3_paths = self.upload_to_s3(saved_paths)
                logger.info(f"S3 upload completed: {s3_paths}")
            
            logger.info("Batch inference completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'total_records': len(output_df),
                'summary': saved_paths['summary'],
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Batch inference for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to input data file')
    parser.add_argument('--output', type=str, help='Path to output directory')
    
    args = parser.parse_args()
    
    # Initialize batch inference
    batch_inference = BatchInference(config_path=args.config)
    
    # Run batch inference
    result = batch_inference.run_batch_inference()
    
    # Log final result
    logger.info(f"Batch inference result: {result['status']}")
    
    if result['status'] == 'SUCCESS':
        logger.info("Batch inference completed successfully!")
        logger.info(f"Total records processed: {result['total_records']}")
        sys.exit(0)
    else:
        logger.error("Batch inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
