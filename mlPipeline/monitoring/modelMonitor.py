#!/usr/bin/env python3
"""
Model Monitoring Script for Credit Risk Platform
Production-ready model monitoring with drift detection and performance tracking
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
from typing import Dict, Any, List, Optional
import argparse
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Production-ready model monitoring for credit risk"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.baseline_data = None
        self.baseline_metrics = {}
        self.current_metrics = {}
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3')
        self.cloudwatch_client = boto3.client('cloudwatch')
        self.sns_client = boto3.client('sns')
        
        logger.info("Model Monitor initialized")
    
    def load_config(self, config_path):
        """Load monitoring configuration"""
        default_config = {
            "model_name": "credit-risk-ensemble",
            "model_version": "1.0",
            "s3_bucket": "credit-risk-features",
            "baseline_data_path": "baseline/training_data.csv",
            "monitoring_window_days": 7,
            "drift_threshold": 0.1,
            "performance_threshold": 0.05,
            "alerts": {
                "enabled": True,
                "sns_topic": "credit-risk-model-alerts",
                "email_recipients": ["admin@company.com"]
            },
            "metrics": {
                "accuracy": {"threshold": 0.05, "direction": "decrease"},
                "precision": {"threshold": 0.05, "direction": "decrease"},
                "recall": {"threshold": 0.05, "direction": "decrease"},
                "f1_score": {"threshold": 0.05, "direction": "decrease"},
                "auc_score": {"threshold": 0.05, "direction": "decrease"}
            },
            "drift_detection": {
                "enabled": True,
                "methods": ["ks_test", "psi", "chi_square"],
                "thresholds": {
                    "ks_test": 0.05,
                    "psi": 0.2,
                    "chi_square": 0.05
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_baseline_data(self):
        """Load baseline data for comparison"""
        logger.info("Loading baseline data...")
        
        try:
            bucket = self.config['s3_bucket']
            baseline_key = self.config['baseline_data_path']
            
            # Download baseline data from S3
            local_path = "/tmp/baseline_data.csv"
            self.s3_client.download_file(bucket, baseline_key, local_path)
            
            # Load baseline data
            self.baseline_data = pd.read_csv(local_path)
            
            logger.info(f"Baseline data loaded: {self.baseline_data.shape[0]} records")
            
            return self.baseline_data
            
        except Exception as e:
            logger.error(f"Error loading baseline data: {e}")
            raise
    
    def load_current_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load current data for monitoring period"""
        logger.info(f"Loading current data from {start_date} to {end_date}...")
        
        try:
            bucket = self.config['s3_bucket']
            prefix = "inference/"
            
            # List objects in S3
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            current_data = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Check if object is within date range
                    if start_date <= obj['LastModified'].replace(tzinfo=None) <= end_date:
                        # Download and load data
                        local_path = f"/tmp/current_data_{obj['Key'].split('/')[-1]}"
                        self.s3_client.download_file(bucket, obj['Key'], local_path)
                        
                        if obj['Key'].endswith('.csv'):
                            df = pd.read_csv(local_path)
                        elif obj['Key'].endswith('.json'):
                            df = pd.read_json(local_path)
                        elif obj['Key'].endswith('.parquet'):
                            df = pd.read_parquet(local_path)
                        
                        current_data.append(df)
            
            if current_data:
                current_df = pd.concat(current_data, ignore_index=True)
                logger.info(f"Current data loaded: {current_df.shape[0]} records")
                return current_df
            else:
                logger.warning("No current data found for monitoring period")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading current data: {e}")
            raise
    
    def calculate_baseline_metrics(self):
        """Calculate baseline metrics from training data"""
        logger.info("Calculating baseline metrics...")
        
        try:
            if self.baseline_data is None:
                self.load_baseline_data()
            
            # Calculate baseline statistics for each feature
            baseline_stats = {}
            
            numeric_columns = self.baseline_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                baseline_stats[col] = {
                    'mean': float(self.baseline_data[col].mean()),
                    'std': float(self.baseline_data[col].std()),
                    'min': float(self.baseline_data[col].min()),
                    'max': float(self.baseline_data[col].max()),
                    'median': float(self.baseline_data[col].median()),
                    'q25': float(self.baseline_data[col].quantile(0.25)),
                    'q75': float(self.baseline_data[col].quantile(0.75))
                }
            
            # Calculate baseline distribution for categorical columns
            categorical_columns = self.baseline_data.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                baseline_stats[col] = {
                    'distribution': self.baseline_data[col].value_counts(normalize=True).to_dict()
                }
            
            self.baseline_metrics = baseline_stats
            
            logger.info(f"Baseline metrics calculated for {len(baseline_stats)} features")
            
            return baseline_stats
            
        except Exception as e:
            logger.error(f"Error calculating baseline metrics: {e}")
            raise
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between baseline and current data"""
        logger.info("Detecting data drift...")
        
        try:
            drift_results = {}
            
            if self.baseline_metrics is None:
                self.calculate_baseline_metrics()
            
            # Check numeric features
            numeric_columns = current_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in self.baseline_metrics:
                    drift_results[col] = {}
                    
                    # Kolmogorov-Smirnov test
                    if 'ks_test' in self.config['drift_detection']['methods']:
                        ks_stat, ks_pvalue = stats.ks_2samp(
                            self.baseline_data[col].dropna(),
                            current_data[col].dropna()
                        )
                        drift_results[col]['ks_test'] = {
                            'statistic': float(ks_stat),
                            'p_value': float(ks_pvalue),
                            'drift_detected': ks_pvalue < self.config['drift_detection']['thresholds']['ks_test']
                        }
                    
                    # Population Stability Index (PSI)
                    if 'psi' in self.config['drift_detection']['methods']:
                        psi = self.calculate_psi(
                            self.baseline_data[col].dropna(),
                            current_data[col].dropna()
                        )
                        drift_results[col]['psi'] = {
                            'value': float(psi),
                            'drift_detected': psi > self.config['drift_detection']['thresholds']['psi']
                        }
            
            # Check categorical features
            categorical_columns = current_data.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if col in self.baseline_metrics:
                    drift_results[col] = {}
                    
                    # Chi-square test
                    if 'chi_square' in self.config['drift_detection']['methods']:
                        chi2_stat, chi2_pvalue = self.chi_square_test(
                            self.baseline_data[col].dropna(),
                            current_data[col].dropna()
                        )
                        drift_results[col]['chi_square'] = {
                            'statistic': float(chi2_stat),
                            'p_value': float(chi2_pvalue),
                            'drift_detected': chi2_pvalue < self.config['drift_detection']['thresholds']['chi_square']
                        }
            
            # Overall drift assessment
            total_features = len(drift_results)
            drifted_features = sum(1 for feature_results in drift_results.values() 
                                 if any(method.get('drift_detected', False) 
                                       for method in feature_results.values()))
            
            drift_summary = {
                'total_features': total_features,
                'drifted_features': drifted_features,
                'drift_percentage': (drifted_features / total_features) * 100 if total_features > 0 else 0,
                'overall_drift_detected': drifted_features > (total_features * self.config['drift_threshold'])
            }
            
            logger.info(f"Data drift detection completed. {drifted_features}/{total_features} features drifted")
            
            return {
                'drift_results': drift_results,
                'drift_summary': drift_summary
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            raise
    
    def calculate_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins
            baseline_min, baseline_max = baseline.min(), baseline.max()
            current_min, current_max = current.min(), current.max()
            
            min_val = min(baseline_min, current_min)
            max_val = max(baseline_max, current_max)
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate distributions
            baseline_dist = np.histogram(baseline, bins=bin_edges)[0]
            current_dist = np.histogram(current, bins=bin_edges)[0]
            
            # Normalize distributions
            baseline_dist = baseline_dist / baseline_dist.sum()
            current_dist = current_dist / current_dist.sum()
            
            # Calculate PSI
            psi = 0
            for i in range(len(baseline_dist)):
                if baseline_dist[i] > 0 and current_dist[i] > 0:
                    psi += (current_dist[i] - baseline_dist[i]) * np.log(current_dist[i] / baseline_dist[i])
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def chi_square_test(self, baseline: pd.Series, current: pd.Series) -> tuple:
        """Perform chi-square test for categorical data"""
        try:
            # Get unique values
            all_values = set(baseline.unique()) | set(current.unique())
            
            # Create contingency table
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()
            
            observed = []
            expected = []
            
            for value in all_values:
                baseline_count = baseline_counts.get(value, 0)
                current_count = current_counts.get(value, 0)
                
                observed.append([baseline_count, current_count])
                
                # Expected counts
                total_baseline = baseline_counts.sum()
                total_current = current_counts.sum()
                total = total_baseline + total_current
                
                expected_baseline = (baseline_count + current_count) * total_baseline / total
                expected_current = (baseline_count + current_count) * total_current / total
                
                expected.append([expected_baseline, expected_current])
            
            # Calculate chi-square statistic
            chi2_stat = 0
            for i in range(len(observed)):
                for j in range(len(observed[i])):
                    if expected[i][j] > 0:
                        chi2_stat += (observed[i][j] - expected[i][j]) ** 2 / expected[i][j]
            
            # Degrees of freedom
            df = (len(all_values) - 1) * (2 - 1)
            
            # Calculate p-value
            chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)
            
            return chi2_stat, chi2_pvalue
            
        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return 0.0, 1.0
    
    def calculate_performance_metrics(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate current performance metrics"""
        logger.info("Calculating performance metrics...")
        
        try:
            # Check if we have ground truth labels
            if 'loan_status' not in current_data.columns:
                logger.warning("No ground truth labels found. Skipping performance calculation.")
                return {}
            
            # Check if we have predictions
            if 'prediction_class' not in current_data.columns:
                logger.warning("No predictions found. Skipping performance calculation.")
                return {}
            
            y_true = current_data['loan_status']
            y_pred = current_data['prediction_class']
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Calculate AUC if we have probability scores
            if 'prob_default' in current_data.columns:
                metrics['auc_score'] = roc_auc_score(y_true, current_data['prob_default'])
            
            self.current_metrics = metrics
            
            logger.info(f"Performance metrics calculated: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def check_performance_degradation(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check for performance degradation"""
        logger.info("Checking performance degradation...")
        
        try:
            # Load baseline performance metrics (would be stored from training)
            baseline_metrics = {
                'accuracy': 0.85,  # Example baseline values
                'precision': 0.82,
                'recall': 0.80,
                'f1_score': 0.81,
                'auc_score': 0.83
            }
            
            performance_alerts = {}
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    threshold = self.config['metrics'][metric_name]['threshold']
                    direction = self.config['metrics'][metric_name]['direction']
                    
                    if direction == 'decrease':
                        degradation = baseline_value - current_value
                        alert_triggered = degradation > threshold
                    else:  # increase
                        degradation = current_value - baseline_value
                        alert_triggered = degradation > threshold
                    
                    performance_alerts[metric_name] = {
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'degradation': degradation,
                        'threshold': threshold,
                        'alert_triggered': alert_triggered
                    }
            
            # Overall performance assessment
            total_metrics = len(performance_alerts)
            degraded_metrics = sum(1 for alert in performance_alerts.values() if alert['alert_triggered'])
            
            performance_summary = {
                'total_metrics': total_metrics,
                'degraded_metrics': degraded_metrics,
                'degradation_percentage': (degraded_metrics / total_metrics) * 100 if total_metrics > 0 else 0,
                'overall_degradation_detected': degraded_metrics > (total_metrics * self.config['performance_threshold'])
            }
            
            logger.info(f"Performance degradation check completed. {degraded_metrics}/{total_metrics} metrics degraded")
            
            return {
                'performance_alerts': performance_alerts,
                'performance_summary': performance_summary
            }
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            raise
    
    def send_alerts(self, drift_results: Dict[str, Any], performance_results: Dict[str, Any]):
        """Send alerts for detected issues"""
        logger.info("Sending alerts...")
        
        try:
            if not self.config['alerts']['enabled']:
                logger.info("Alerts disabled. Skipping alert sending.")
                return
            
            alerts = []
            
            # Data drift alerts
            if drift_results['drift_summary']['overall_drift_detected']:
                alerts.append({
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH',
                    'message': f"Data drift detected: {drift_results['drift_summary']['drift_percentage']:.1f}% of features drifted",
                    'details': drift_results['drift_summary']
                })
            
            # Performance degradation alerts
            if performance_results['performance_summary']['overall_degradation_detected']:
                alerts.append({
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'HIGH',
                    'message': f"Performance degradation detected: {performance_results['performance_summary']['degradation_percentage']:.1f}% of metrics degraded",
                    'details': performance_results['performance_summary']
                })
            
            # Send alerts
            if alerts:
                for alert in alerts:
                    self.send_sns_alert(alert)
            
            logger.info(f"Sent {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    def send_sns_alert(self, alert: Dict[str, Any]):
        """Send alert via SNS"""
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'alert': alert
            }
            
            self.sns_client.publish(
                TopicArn=f"arn:aws:sns:us-east-1:123456789012:{self.config['alerts']['sns_topic']}",
                Subject=f"Model Alert: {alert['type']}",
                Message=json.dumps(message, indent=2)
            )
            
            logger.info(f"Alert sent: {alert['type']}")
            
        except Exception as e:
            logger.error(f"Error sending SNS alert: {e}")
    
    def save_monitoring_report(self, drift_results: Dict[str, Any], performance_results: Dict[str, Any]):
        """Save monitoring report"""
        logger.info("Saving monitoring report...")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config['model_name'],
                'model_version': self.config['model_version'],
                'monitoring_window_days': self.config['monitoring_window_days'],
                'drift_results': drift_results,
                'performance_results': performance_results,
                'summary': {
                    'data_drift_detected': drift_results['drift_summary']['overall_drift_detected'],
                    'performance_degradation_detected': performance_results['performance_summary']['overall_degradation_detected'],
                    'overall_status': 'HEALTHY' if not (drift_results['drift_summary']['overall_drift_detected'] or 
                                                      performance_results['performance_summary']['overall_degradation_detected']) else 'ISSUES_DETECTED'
                }
            }
            
            # Save to local file
            report_path = f"/tmp/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Upload to S3
            bucket = self.config['s3_bucket']
            s3_key = f"monitoring/reports/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.upload_file(report_path, bucket, s3_key)
            
            logger.info(f"Monitoring report saved: s3://{bucket}/{s3_key}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error saving monitoring report: {e}")
            raise
    
    def run_monitoring(self):
        """Run the complete monitoring pipeline"""
        logger.info("Starting model monitoring pipeline...")
        
        try:
            # Step 1: Load baseline data and metrics
            self.load_baseline_data()
            self.calculate_baseline_metrics()
            
            # Step 2: Load current data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['monitoring_window_days'])
            
            current_data = self.load_current_data(start_date, end_date)
            
            if current_data.empty:
                logger.warning("No current data found. Skipping monitoring.")
                return {'status': 'SKIPPED', 'reason': 'No current data'}
            
            # Step 3: Detect data drift
            drift_results = self.detect_data_drift(current_data)
            
            # Step 4: Calculate performance metrics
            current_metrics = self.calculate_performance_metrics(current_data)
            performance_results = self.check_performance_degradation(current_metrics)
            
            # Step 5: Send alerts
            self.send_alerts(drift_results, performance_results)
            
            # Step 6: Save monitoring report
            report = self.save_monitoring_report(drift_results, performance_results)
            
            logger.info("Model monitoring completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'drift_results': drift_results,
                'performance_results': performance_results,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Model monitoring failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Model monitoring for credit risk')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--days', type=int, default=7, help='Monitoring window in days')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ModelMonitor(config_path=args.config)
    
    # Run monitoring
    result = monitor.run_monitoring()
    
    # Log final result
    logger.info(f"Monitoring result: {result['status']}")
    
    if result['status'] == 'SUCCESS':
        logger.info("Monitoring completed successfully!")
        sys.exit(0)
    else:
        logger.error("Monitoring failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
