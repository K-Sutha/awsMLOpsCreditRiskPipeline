#!/usr/bin/env python3
"""
Model Ensemble Script for Credit Risk Platform
Combines multiple models for improved performance
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import boto3
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Production-ready model ensemble for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        self.individual_models = {}
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info("Model Ensemble initialized")
    
    def load_config(self, config_path):
        """Load ensemble configuration"""
        default_config = {
            "model_name": "credit-risk-ensemble",
            "model_version": "1.0",
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 5,
            "ensemble_methods": {
                "voting": {
                    "enabled": True,
                    "voting_type": "soft",
                    "weights": [0.3, 0.4, 0.3]  # RF, XGB, NN
                },
                "stacking": {
                    "enabled": True,
                    "final_estimator": "LogisticRegression",
                    "cv_folds": 5
                }
            },
            "base_models": {
                "random_forest": {
                    "enabled": True,
                    "model_path": "models/credit-risk-random-forest/1.0/model.joblib",
                    "scaler_path": "models/credit-risk-random-forest/1.0/scaler.joblib"
                },
                "xgboost": {
                    "enabled": True,
                    "model_path": "models/credit-risk-xgboost/1.0/model.joblib",
                    "scaler_path": "models/credit-risk-xgboost/1.0/scaler.joblib"
                },
                "neural_network": {
                    "enabled": True,
                    "model_path": "models/credit-risk-neural-network/1.0/model.h5",
                    "scaler_path": "models/credit-risk-neural-network/1.0/scaler.joblib"
                }
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
        """Load training data"""
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
    
    def load_base_models(self):
        """Load pre-trained base models"""
        logger.info("Loading base models...")
        
        try:
            bucket = self.config['s3_bucket']
            base_models = {}
            
            for model_name, model_config in self.config['base_models'].items():
                if not model_config['enabled']:
                    continue
                
                logger.info(f"Loading {model_name} model...")
                
                # Download model files from S3
                model_key = model_config['model_path']
                scaler_key = model_config['scaler_path']
                
                # Create local paths
                local_model_path = f"/tmp/{model_name}_model.joblib"
                local_scaler_path = f"/tmp/{model_name}_scaler.joblib"
                
                # Download from S3
                self.s3_client.download_file(bucket, model_key, local_model_path)
                self.s3_client.download_file(bucket, scaler_key, local_scaler_path)
                
                # Load model and scaler
                model = joblib.load(local_model_path)
                scaler = joblib.load(local_scaler_path)
                
                base_models[model_name] = {
                    'model': model,
                    'scaler': scaler
                }
                
                logger.info(f"Loaded {model_name} model successfully")
            
            self.individual_models = base_models
            logger.info(f"Loaded {len(base_models)} base models")
            
            return base_models
            
        except Exception as e:
            logger.error(f"Error loading base models: {e}")
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
    
    def create_voting_ensemble(self, base_models):
        """Create voting ensemble"""
        logger.info("Creating voting ensemble...")
        
        try:
            # Prepare base models for voting
            estimators = []
            for model_name, model_data in base_models.items():
                estimators.append((model_name, model_data['model']))
            
            # Create voting classifier
            voting_type = self.config['ensemble_methods']['voting']['voting_type']
            weights = self.config['ensemble_methods']['voting']['weights']
            
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting_type,
                weights=weights
            )
            
            logger.info(f"Created voting ensemble with {len(estimators)} models")
            logger.info(f"Voting type: {voting_type}")
            logger.info(f"Weights: {weights}")
            
            return voting_ensemble
            
        except Exception as e:
            logger.error(f"Error creating voting ensemble: {e}")
            raise
    
    def create_stacking_ensemble(self, base_models):
        """Create stacking ensemble"""
        logger.info("Creating stacking ensemble...")
        
        try:
            # Prepare base models for stacking
            estimators = []
            for model_name, model_data in base_models.items():
                estimators.append((model_name, model_data['model']))
            
            # Create stacking classifier
            from sklearn.linear_model import LogisticRegression
            
            stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=self.config['random_state']),
                cv=self.config['ensemble_methods']['stacking']['cv_folds']
            )
            
            logger.info(f"Created stacking ensemble with {len(estimators)} models")
            logger.info(f"Final estimator: LogisticRegression")
            logger.info(f"CV folds: {self.config['ensemble_methods']['stacking']['cv_folds']}")
            
            return stacking_ensemble
            
        except Exception as e:
            logger.error(f"Error creating stacking ensemble: {e}")
            raise
    
    def train_ensemble(self, X_train, y_train):
        """Train the ensemble model"""
        logger.info("Training ensemble model...")
        
        try:
            # Load base models
            base_models = self.load_base_models()
            
            # Create ensembles
            ensembles = {}
            
            # Voting ensemble
            if self.config['ensemble_methods']['voting']['enabled']:
                logger.info("Training voting ensemble...")
                voting_ensemble = self.create_voting_ensemble(base_models)
                voting_ensemble.fit(X_train, y_train)
                ensembles['voting'] = voting_ensemble
                
                # Evaluate voting ensemble
                voting_scores = cross_val_score(
                    voting_ensemble, X_train, y_train,
                    cv=self.config['cv_folds'], scoring='roc_auc'
                )
                logger.info(f"Voting ensemble CV scores: {voting_scores.mean():.4f} (+/- {voting_scores.std() * 2:.4f})")
            
            # Stacking ensemble
            if self.config['ensemble_methods']['stacking']['enabled']:
                logger.info("Training stacking ensemble...")
                stacking_ensemble = self.create_stacking_ensemble(base_models)
                stacking_ensemble.fit(X_train, y_train)
                ensembles['stacking'] = stacking_ensemble
                
                # Evaluate stacking ensemble
                stacking_scores = cross_val_score(
                    stacking_ensemble, X_train, y_train,
                    cv=self.config['cv_folds'], scoring='roc_auc'
                )
                logger.info(f"Stacking ensemble CV scores: {stacking_scores.mean():.4f} (+/- {stacking_scores.std() * 2:.4f})")
            
            # Choose best ensemble
            best_ensemble = None
            best_score = 0
            best_name = None
            
            for name, ensemble in ensembles.items():
                scores = cross_val_score(
                    ensemble, X_train, y_train,
                    cv=self.config['cv_folds'], scoring='roc_auc'
                )
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_ensemble = ensemble
                    best_name = name
            
            self.ensemble_model = best_ensemble
            
            logger.info(f"Best ensemble: {best_name} with CV score: {best_score:.4f}")
            
            # Store training metrics
            self.training_metrics.update({
                'ensemble_type': best_name,
                'best_cv_score': best_score,
                'all_ensembles': {
                    name: {
                        'cv_scores_mean': cross_val_score(
                            ensemble, X_train, y_train,
                            cv=self.config['cv_folds'], scoring='roc_auc'
                        ).mean(),
                        'cv_scores_std': cross_val_score(
                            ensemble, X_train, y_train,
                            cv=self.config['cv_folds'], scoring='roc_auc'
                        ).std()
                    } for name, ensemble in ensembles.items()
                },
                'base_models_used': list(base_models.keys()),
                'training_timestamp': datetime.now().isoformat()
            })
            
            return self.ensemble_model
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate the ensemble model"""
        logger.info("Evaluating ensemble model...")
        
        try:
            # Make predictions
            y_pred = self.ensemble_model.predict(X_test)
            y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store metrics
            self.training_metrics.update({
                'auc_score': auc_score,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'model_type': 'Ensemble',
                'test_predictions': y_pred_proba.tolist()
            })
            
            logger.info(f"Ensemble AUC Score: {auc_score:.4f}")
            logger.info(f"Ensemble Test Accuracy: {class_report['accuracy']:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            raise
    
    def save_ensemble(self):
        """Save the ensemble model and artifacts"""
        logger.info("Saving ensemble model and artifacts...")
        
        try:
            # Create output directory
            os.makedirs(self.config['output_path'], exist_ok=True)
            
            # Save ensemble model
            ensemble_path = os.path.join(self.config['output_path'], 'ensemble_model.joblib')
            joblib.dump(self.ensemble_model, ensemble_path)
            
            # Save scaler
            scaler_path = os.path.join(self.config['output_path'], 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature names
            features_path = os.path.join(self.config['output_path'], 'feature_names.json')
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)
            
            # Save ensemble metadata
            metadata = {
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'model_type': 'Ensemble',
                'ensemble_type': self.training_metrics['ensemble_type'],
                'feature_count': len(self.feature_names),
                'training_timestamp': datetime.now().isoformat(),
                'base_models': self.training_metrics['base_models_used'],
                'performance_metrics': {
                    'auc_score': self.training_metrics['auc_score'],
                    'best_cv_score': self.training_metrics['best_cv_score'],
                    'all_ensemble_scores': self.training_metrics['all_ensembles']
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
            
            logger.info(f"Ensemble model saved to: {ensemble_path}")
            logger.info(f"Scaler saved to: {scaler_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            
            return {
                'ensemble_path': ensemble_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'metrics_path': metrics_path
            }
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            raise
    
    def upload_to_s3(self, local_paths):
        """Upload ensemble artifacts to S3"""
        logger.info("Uploading ensemble artifacts to S3...")
        
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
    
    def run_ensemble_training(self):
        """Run the complete ensemble training pipeline"""
        logger.info("Starting ensemble training pipeline...")
        
        try:
            # Step 1: Load data
            X, y = self.load_data()
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Step 3: Train ensemble
            ensemble_model = self.train_ensemble(X_train, y_train)
            
            # Step 4: Evaluate ensemble
            metrics = self.evaluate_ensemble(X_test, y_test)
            
            # Step 5: Save ensemble
            saved_paths = self.save_ensemble()
            
            # Step 6: Upload to S3 (if in SageMaker environment)
            if os.path.exists('/opt/ml/'):
                s3_paths = self.upload_to_s3(saved_paths)
                logger.info(f"S3 upload completed: {s3_paths}")
            
            logger.info("Ensemble training completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'ensemble_model': ensemble_model,
                'metrics': metrics,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train ensemble model for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--local', action='store_true', help='Run in local development mode')
    
    args = parser.parse_args()
    
    # Initialize ensemble
    ensemble = ModelEnsemble(config_path=args.config)
    
    # Run ensemble training
    result = ensemble.run_ensemble_training()
    
    # Log final result
    logger.info(f"Ensemble training result: {result['status']}")
    
    if result['status'] == 'SUCCESS':
        logger.info("Ensemble training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Ensemble training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
