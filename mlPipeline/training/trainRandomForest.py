#!/usr/bin/env python3
"""
Random Forest Training Script for Credit Risk Platform
Production-ready ML training with SageMaker
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import boto3
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestTrainer:
    """Production-ready Random Forest trainer for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info("Random Forest Trainer initialized")
    
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "model_name": "credit-risk-random-forest",
            "model_version": "1.0",
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 5,
            "hyperparameters": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['sqrt', 'log2', None],
                "bootstrap": [True, False],
                "class_weight": ['balanced', None]
            },
            "s3_bucket": "credit-risk-features",
            "s3_prefix": "train/",
            "output_path": "/opt/ml/model",
            "metrics_path": "/opt/ml/output/data/metrics.json"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_data(self):
        """Load training data from S3 or local path"""
        logger.info("Loading training data...")
        
        try:
            # Try to load from S3 first (for SageMaker training)
            if os.path.exists('/opt/ml/input/data/training/'):
                data_path = '/opt/ml/input/data/training/'
                logger.info("Loading data from SageMaker input path")
            else:
                # Local development path
                data_path = '../data/processed/cleaned/'
                logger.info("Loading data from local path")
            
            # Load the cleaned dataset
            train_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if not train_files:
                raise FileNotFoundError("No CSV files found in data path")
            
            # Load the first CSV file (should be cleanedDataset.csv)
            data_file = os.path.join(data_path, train_files[0])
            df = pd.read_csv(data_file)
            
            logger.info(f"Data loaded: {df.shape[0]} records, {df.shape[1]} columns")
            
            # Separate features and target
            target_column = 'loan_status'
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Get feature columns (exclude target and metadata columns)
            exclude_columns = [
                'loan_status', 'processing_date', 'job_name', 'quality_score'
            ]
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            X = df[feature_columns]
            y = df[target_column]
            
            # Store feature names for later use
            self.feature_names = feature_columns
            
            logger.info(f"Features: {len(feature_columns)} columns")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, X, y):
        """Preprocess the training data"""
        logger.info("Preprocessing data...")
        
        try:
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert back to DataFrame to maintain column names
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model with hyperparameter tuning"""
        logger.info("Training Random Forest model...")
        
        try:
            # Create base model
            base_model = RandomForestClassifier(
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            # Hyperparameter tuning with GridSearchCV
            logger.info("Starting hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model,
                self.config['hyperparameters'],
                cv=self.config['cv_folds'],
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            self.model = grid_search.best_estimator_
            
            # Store best parameters
            self.training_metrics['best_parameters'] = grid_search.best_params_
            self.training_metrics['best_cv_score'] = grid_search.best_score_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model, X_test, y_test, 
                cv=self.config['cv_folds'], 
                scoring='roc_auc'
            )
            
            # Store metrics
            self.training_metrics.update({
                'auc_score': auc_score,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std(),
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
                'model_type': 'RandomForestClassifier',
                'training_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"AUC Score: {auc_score:.4f}")
            logger.info(f"CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Log feature importance
            feature_importance = sorted(
                zip(self.feature_names, self.model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            logger.info("Top 10 Feature Importances:")
            for feature, importance in feature_importance:
                logger.info(f"  {feature}: {importance:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self):
        """Save the trained model and artifacts"""
        logger.info("Saving model and artifacts...")
        
        try:
            # Create output directory
            os.makedirs(self.config['output_path'], exist_ok=True)
            
            # Save model
            model_path = os.path.join(self.config['output_path'], 'model.joblib')
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.config['output_path'], 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature names
            features_path = os.path.join(self.config['output_path'], 'feature_names.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)
            
            # Save model metadata
            metadata = {
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'model_type': 'RandomForestClassifier',
                'feature_count': len(self.feature_names),
                'training_timestamp': datetime.now().isoformat(),
                'hyperparameters': self.training_metrics['best_parameters'],
                'performance_metrics': {
                    'auc_score': self.training_metrics['auc_score'],
                    'cv_score': self.training_metrics['best_cv_score'],
                    'cv_scores_mean': self.training_metrics['cv_scores_mean'],
                    'cv_scores_std': self.training_metrics['cv_scores_std']
                }
            }
            
            metadata_path = os.path.join(self.config['output_path'], 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save training metrics
            metrics_path = self.config['metrics_path']
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Scaler saved to: {scaler_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'metrics_path': metrics_path
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def upload_to_s3(self, local_paths):
        """Upload model artifacts to S3"""
        logger.info("Uploading model artifacts to S3...")
        
        try:
            bucket = self.config['s3_bucket']
            prefix = f"models/{self.config['model_name']}/{self.config['model_version']}/"
            
            uploaded_files = {}
            
            for artifact_type, local_path in local_paths.items():
                if os.path.exists(local_path):
                    s3_key = f"{prefix}{os.path.basename(local_path)}"
                    
                    self.s3_client.upload_file(local_path, bucket, s3_key)
                    uploaded_files[artifact_type] = f"s3://{bucket}/{s3_key}"
                    
                    logger.info(f"Uploaded {artifact_type}: {uploaded_files[artifact_type]}")
            
            return uploaded_files
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting Random Forest training pipeline...")
        
        try:
            # Step 1: Load data
            X, y = self.load_data()
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Step 3: Train model
            model = self.train_model(X_train, y_train)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Step 5: Save model
            saved_paths = self.save_model()
            
            # Step 6: Upload to S3 (if in SageMaker environment)
            if os.path.exists('/opt/ml/'):
                s3_paths = self.upload_to_s3(saved_paths)
                logger.info(f"S3 upload completed: {s3_paths}")
            
            logger.info("Random Forest training completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'model': model,
                'metrics': metrics,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Random Forest model for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--local', action='store_true', help='Run in local development mode')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RandomForestTrainer(config_path=args.config)
    
    # Run training
    result = trainer.run_training()
    
    # Log final result
    logger.info(f"Training result: {result['status']}")
    
    if result['status'] == 'SUCCESS':
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
