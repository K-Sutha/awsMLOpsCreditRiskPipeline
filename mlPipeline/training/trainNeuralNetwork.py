#!/usr/bin/env python3
"""
Neural Network Training Script for Credit Risk Platform
Production-ready deep learning with TensorFlow/Keras
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import boto3
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNetworkTrainer:
    """Production-ready Neural Network trainer for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}
        self.history = None
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['random_state'])
        tf.random.set_seed(self.config['random_state'])
        
        logger.info("Neural Network Trainer initialized")
    
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "model_name": "credit-risk-neural-network",
            "model_version": "1.0",
            "random_state": 42,
            "test_size": 0.2,
            "validation_size": 0.1,
            "hyperparameters": {
                "hidden_layers": [2, 3, 4],
                "hidden_units": [64, 128, 256],
                "dropout_rate": [0.2, 0.3, 0.5],
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [32, 64, 128],
                "epochs": [50, 100, 200],
                "activation": ['relu', 'tanh', 'elu']
            },
            "early_stopping": {
                "patience": 10,
                "monitor": "val_auc",
                "mode": "max",
                "restore_best_weights": True
            },
            "model_checkpoint": {
                "monitor": "val_auc",
                "mode": "max",
                "save_best_only": True
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
            
            # Further split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['validation_size'],
                random_state=self.config['random_state'],
                stratify=y_train
            )
            
            logger.info(f"Train set: {X_train.shape[0]} samples")
            logger.info(f"Validation set: {X_val.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def create_model(self, input_dim, hyperparams):
        """Create neural network model architecture"""
        logger.info("Creating neural network model...")
        
        try:
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Dense(
                hyperparams['hidden_units'],
                activation=hyperparams['activation'],
                input_shape=(input_dim,),
                name='input_layer'
            ))
            model.add(layers.Dropout(hyperparams['dropout_rate']))
            
            # Hidden layers
            for i in range(hyperparams['hidden_layers'] - 1):
                model.add(layers.Dense(
                    hyperparams['hidden_units'],
                    activation=hyperparams['activation'],
                    name=f'hidden_layer_{i+1}'
                ))
                model.add(layers.Dropout(hyperparams['dropout_rate']))
            
            # Output layer
            model.add(layers.Dense(
                1,
                activation='sigmoid',
                name='output_layer'
            ))
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            logger.info(f"Model created with {model.count_params()} parameters")
            logger.info(f"Architecture: {hyperparams['hidden_layers']} hidden layers, {hyperparams['hidden_units']} units each")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def train_model(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train the neural network model"""
        logger.info("Training neural network model...")
        
        try:
            # Create model
            self.model = self.create_model(X_train.shape[1], hyperparams)
            
            # Define callbacks
            callbacks_list = []
            
            # Early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor=self.config['early_stopping']['monitor'],
                patience=self.config['early_stopping']['patience'],
                mode=self.config['early_stopping']['mode'],
                restore_best_weights=self.config['early_stopping']['restore_best_weights'],
                verbose=1
            )
            callbacks_list.append(early_stopping)
            
            # Model checkpoint
            checkpoint_path = os.path.join(self.config['output_path'], 'best_model.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            model_checkpoint = callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=self.config['model_checkpoint']['monitor'],
                mode=self.config['model_checkpoint']['mode'],
                save_best_only=self.config['model_checkpoint']['save_best_only'],
                verbose=1
            )
            callbacks_list.append(model_checkpoint)
            
            # Reduce learning rate on plateau
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
            callbacks_list.append(reduce_lr)
            
            # Train model
            logger.info(f"Training with {hyperparams['epochs']} epochs, batch size {hyperparams['batch_size']}")
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Store training metrics
            self.training_metrics.update({
                'hyperparameters': hyperparams,
                'training_history': {
                    'loss': self.history.history['loss'],
                    'accuracy': self.history.history['accuracy'],
                    'auc': self.history.history['auc'],
                    'val_loss': self.history.history['val_loss'],
                    'val_accuracy': self.history.history['val_accuracy'],
                    'val_auc': self.history.history['val_auc']
                },
                'best_epoch': len(self.history.history['loss']),
                'best_val_auc': max(self.history.history['val_auc'])
            })
            
            logger.info(f"Training completed. Best validation AUC: {self.training_metrics['best_val_auc']:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        try:
            # Make predictions
            y_pred_proba = self.model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
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
                'model_type': 'NeuralNetwork',
                'training_timestamp': datetime.now().isoformat(),
                'test_predictions': y_pred_proba.tolist()
            })
            
            logger.info(f"AUC Score: {auc_score:.4f}")
            logger.info(f"Test Accuracy: {class_report['accuracy']:.4f}")
            
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
            
            # Save model in TensorFlow format
            model_path = os.path.join(self.config['output_path'], 'model')
            self.model.save(model_path)
            
            # Save model in H5 format for compatibility
            h5_path = os.path.join(self.config['output_path'], 'model.h5')
            self.model.save(h5_path)
            
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
                'model_type': 'NeuralNetwork',
                'feature_count': len(self.feature_names),
                'training_timestamp': datetime.now().isoformat(),
                'hyperparameters': self.training_metrics['hyperparameters'],
                'performance_metrics': {
                    'auc_score': self.training_metrics['auc_score'],
                    'best_val_auc': self.training_metrics['best_val_auc'],
                    'best_epoch': self.training_metrics['best_epoch']
                },
                'model_architecture': {
                    'input_shape': (len(self.feature_names),),
                    'hidden_layers': self.training_metrics['hyperparameters']['hidden_layers'],
                    'hidden_units': self.training_metrics['hyperparameters']['hidden_units'],
                    'total_parameters': self.model.count_params()
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
            
            # Save training history
            history_path = os.path.join(self.config['output_path'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_metrics['training_history'], f, indent=2)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"H5 model saved to: {h5_path}")
            logger.info(f"Scaler saved to: {scaler_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            logger.info(f"Training history saved to: {history_path}")
            
            return {
                'model_path': model_path,
                'h5_path': h5_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path,
                'metrics_path': metrics_path,
                'history_path': history_path
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
        logger.info("Starting Neural Network training pipeline...")
        
        try:
            # Step 1: Load data
            X, y = self.load_data()
            
            # Step 2: Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
            
            # Step 3: Train model (using default hyperparameters for now)
            default_hyperparams = {
                'hidden_layers': 3,
                'hidden_units': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 100,
                'activation': 'relu'
            }
            
            model = self.train_model(X_train, y_train, X_val, y_val, default_hyperparams)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Step 5: Save model
            saved_paths = self.save_model()
            
            # Step 6: Upload to S3 (if in SageMaker environment)
            if os.path.exists('/opt/ml/'):
                s3_paths = self.upload_to_s3(saved_paths)
                logger.info(f"S3 upload completed: {s3_paths}")
            
            logger.info("Neural Network training completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'model': model,
                'metrics': metrics,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            logger.error(f"Neural Network training failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Neural Network model for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--local', action='store_true', help='Run in local development mode')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NeuralNetworkTrainer(config_path=args.config)
    
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
