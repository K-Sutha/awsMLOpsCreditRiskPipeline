#!/usr/bin/env python3
"""
Real-time Inference Script for Credit Risk Platform
Production-ready real-time ML inference with caching
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeInference:
    """Production-ready real-time inference for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.cache = {}
        self.cache_ttl = {}
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3')
        self.dynamodb_client = boto3.client('dynamodb')
        
        logger.info("Real-time Inference initialized")
    
    def load_config(self, config_path):
        """Load inference configuration"""
        default_config = {
            "model_name": "credit-risk-ensemble",
            "model_version": "1.0",
            "s3_bucket": "credit-risk-features",
            "dynamodb_table": "credit-risk-risk-cache",
            "cache_ttl_hours": 168,  # 7 days
            "max_cache_size": 10000,
            "feature_engineering": {
                "create_age_groups": True,
                "create_income_quartiles": True,
                "create_employment_stability": True,
                "create_debt_ratios": True,
                "create_risk_indicators": True
            },
            "validation_rules": {
                "person_age": {"min": 18, "max": 100},
                "person_income": {"min": 0, "max": 1000000},
                "loan_amnt": {"min": 0, "max": 1000000},
                "loan_int_rate": {"min": 0, "max": 50},
                "person_emp_length": {"min": 0, "max": 50}
            },
            "risk_thresholds": {
                "low_risk": 0.3,
                "medium_risk": 0.7,
                "high_risk": 1.0
            }
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
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        logger.info("Validating input data...")
        
        try:
            validation_errors = []
            
            # Check required fields
            required_fields = [
                'person_age', 'person_income', 'person_emp_length',
                'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                'cb_person_default_on_file', 'cb_person_cred_hist_length',
                'person_home_ownership', 'loan_intent', 'loan_grade'
            ]
            
            for field in required_fields:
                if field not in input_data:
                    validation_errors.append(f"Missing required field: {field}")
            
            # Validate numeric fields
            for field, rules in self.config['validation_rules'].items():
                if field in input_data:
                    value = input_data[field]
                    if not isinstance(value, (int, float)):
                        validation_errors.append(f"Field {field} must be numeric")
                    elif value < rules['min'] or value > rules['max']:
                        validation_errors.append(f"Field {field} out of range [{rules['min']}, {rules['max']}]")
            
            # Validate categorical fields
            categorical_fields = {
                'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
                'loan_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'cb_person_default_on_file': ['Y', 'N']
            }
            
            for field, valid_values in categorical_fields.items():
                if field in input_data:
                    value = input_data[field]
                    if value not in valid_values:
                        validation_errors.append(f"Field {field} has invalid value: {value}")
            
            if validation_errors:
                raise ValueError(f"Validation errors: {validation_errors}")
            
            logger.info("Input validation passed")
            return input_data
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
    
    def engineer_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Engineer features from input data"""
        logger.info("Engineering features...")
        
        try:
            # Create a DataFrame from input data
            df = pd.DataFrame([input_data])
            
            # Age-based features
            if self.config['feature_engineering']['create_age_groups']:
                df['age_group'] = pd.cut(df['person_age'], 
                                       bins=[0, 25, 35, 45, 100], 
                                       labels=['18-25', '26-35', '36-45', '45+'])
            
            # Income-based features
            if self.config['feature_engineering']['create_income_quartiles']:
                # For real-time inference, we'll use fixed quartile thresholds
                # In production, these would be calculated from training data
                income_thresholds = [25000, 50000, 75000, 1000000]
                df['income_quartile'] = pd.cut(df['person_income'], 
                                             bins=[0] + income_thresholds, 
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
                df['has_high_income'] = (df['person_income'] > 75000).astype(int)
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
    
    def generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key for input data"""
        # Create a hash of the input data for caching
        input_str = json.dumps(input_data, sort_keys=True)
        cache_key = hashlib.md5(input_str.encode()).hexdigest()
        return cache_key
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache"""
        try:
            # Check in-memory cache first
            if cache_key in self.cache:
                if cache_key in self.cache_ttl:
                    if datetime.now() < self.cache_ttl[cache_key]:
                        logger.info("Cache hit (in-memory)")
                        return self.cache[cache_key]
                    else:
                        # Remove expired cache entry
                        del self.cache[cache_key]
                        del self.cache_ttl[cache_key]
            
            # Check DynamoDB cache
            response = self.dynamodb_client.get_item(
                TableName=self.config['dynamodb_table'],
                Key={'cacheKey': {'S': cache_key}}
            )
            
            if 'Item' in response:
                item = response['Item']
                ttl = int(item['ttl']['N'])
                
                if time.time() < ttl:
                    prediction = json.loads(item['prediction']['S'])
                    logger.info("Cache hit (DynamoDB)")
                    
                    # Store in in-memory cache
                    self.cache[cache_key] = prediction
                    self.cache_ttl[cache_key] = datetime.now() + timedelta(hours=1)
                    
                    return prediction
                else:
                    # Remove expired item
                    self.dynamodb_client.delete_item(
                        TableName=self.config['dynamodb_table'],
                        Key={'cacheKey': {'S': cache_key}}
                    )
            
            logger.info("Cache miss")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    def save_to_cache(self, cache_key: str, prediction: Dict[str, Any]):
        """Save prediction to cache"""
        try:
            # Save to in-memory cache
            self.cache[cache_key] = prediction
            self.cache_ttl[cache_key] = datetime.now() + timedelta(hours=1)
            
            # Limit cache size
            if len(self.cache) > self.config['max_cache_size']:
                # Remove oldest entries
                oldest_key = min(self.cache_ttl.keys(), key=lambda k: self.cache_ttl[k])
                del self.cache[oldest_key]
                del self.cache_ttl[oldest_key]
            
            # Save to DynamoDB cache
            ttl = int(time.time()) + (self.config['cache_ttl_hours'] * 3600)
            
            self.dynamodb_client.put_item(
                TableName=self.config['dynamodb_table'],
                Item={
                    'cacheKey': {'S': cache_key},
                    'prediction': {'S': json.dumps(prediction)},
                    'ttl': {'N': str(ttl)},
                    'timestamp': {'S': datetime.now().isoformat()}
                }
            )
            
            logger.info("Prediction saved to cache")
            
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data"""
        logger.info("Making prediction...")
        
        try:
            # Generate cache key
            cache_key = self.generate_cache_key(input_data)
            
            # Check cache first
            cached_prediction = self.get_from_cache(cache_key)
            if cached_prediction:
                return cached_prediction
            
            # Validate input
            validated_data = self.validate_input(input_data)
            
            # Engineer features
            df = self.engineer_features(validated_data)
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X)[0]
            prediction_class = self.model.predict(X)[0]
            
            # Calculate risk score
            risk_score = prediction_proba[1]  # Probability of default
            
            # Determine risk level
            if risk_score < self.config['risk_thresholds']['low_risk']:
                risk_level = 'LOW'
            elif risk_score < self.config['risk_thresholds']['medium_risk']:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            # Create prediction result
            prediction = {
                'prediction_id': f"pred_{int(time.time())}_{cache_key[:8]}",
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'prediction_class': int(prediction_class),
                'confidence': float(max(prediction_proba)),
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'feature_importance': self.get_feature_importance(validated_data)
            }
            
            # Save to cache
            self.save_to_cache(cache_key, prediction)
            
            logger.info(f"Prediction completed. Risk level: {risk_level}, Score: {risk_score:.4f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_feature_importance(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Get feature importance for the prediction"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importance_dict = dict(zip(self.feature_names, abs(self.model.coef_[0])))
            else:
                # Default: equal importance
                importance_dict = {feature: 1.0 / len(self.feature_names) for feature in self.feature_names}
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Return top 10 features
            return dict(list(sorted_importance.items())[:10])
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def batch_predict(self, input_data_list: list) -> list:
        """Make batch predictions"""
        logger.info(f"Making batch predictions for {len(input_data_list)} records...")
        
        try:
            predictions = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    prediction = self.predict(input_data)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Prediction failed for record {i}: {e}")
                    predictions.append({
                        'error': str(e),
                        'record_index': i
                    })
            
            logger.info(f"Batch predictions completed. {len(predictions)} results")
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time inference for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to input JSON file')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = RealtimeInference(config_path=args.config)
    
    # Load model
    inference.load_model()
    
    # Test prediction
    if args.input:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    else:
        # Sample input data
        input_data = {
            "person_age": 35,
            "person_income": 60000,
            "person_emp_length": 5,
            "loan_amnt": 15000,
            "loan_int_rate": 12.5,
            "loan_percent_income": 0.25,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 8,
            "person_home_ownership": "RENT",
            "loan_intent": "PERSONAL",
            "loan_grade": "B"
        }
    
    # Make prediction
    result = inference.predict(input_data)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
