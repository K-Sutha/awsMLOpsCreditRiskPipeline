#!/usr/bin/env python3
"""
AWS Glue Job: Data Quality Check
Production-ready ETL job for monitoring data quality and generating alerts
"""

import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import boto3
from datetime import datetime
import json

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Glue context
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

class DataQualityCheckJob:
    """Production-ready data quality monitoring job"""
    
    def __init__(self, args):
        self.args = args
        self.job_name = args['JOB_NAME']
        self.s3_bucket = args.get('S3_BUCKET', 'credit-risk-processed-data')
        self.quality_threshold = float(args.get('QUALITY_THRESHOLD', '0.95'))
        self.alert_threshold = float(args.get('ALERT_THRESHOLD', '0.90'))
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info(f"Starting Data Quality Check Job: {self.job_name}")
        logger.info(f"S3 bucket: {self.s3_bucket}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Alert threshold: {self.alert_threshold}")
    
    def readDataForQualityCheck(self, data_path):
        """Read data from specified S3 path for quality check"""
        logger.info(f"Reading data from: {data_path}")
        
        try:
            # Read data based on file extension
            if data_path.endswith('.parquet'):
                data = spark.read.parquet(data_path)
            elif data_path.endswith('.csv'):
                data = spark.read.format("csv") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .load(data_path)
            else:
                # Default to parquet
                data = spark.read.parquet(data_path)
            
            logger.info(f"Data loaded: {data.count()} records, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            logger.error(f"Error reading data from {data_path}: {e}")
            raise
    
    def checkCompleteness(self, data):
        """Check data completeness (missing values)"""
        logger.info("Checking data completeness...")
        
        try:
            total_records = data.count()
            completeness_checks = {}
            
            for col_name in data.columns:
                # Count missing values
                missing_count = data.filter(col(col_name).isNull()).count()
                missing_percentage = (missing_count / total_records) * 100
                
                # Count empty strings (for string columns)
                empty_count = 0
                if data.select(col_name).dtypes[0][1] == 'string':
                    empty_count = data.filter(col(col_name) == "").count()
                
                total_missing = missing_count + empty_count
                total_missing_percentage = (total_missing / total_records) * 100
                
                completeness_checks[col_name] = {
                    "missing_values": missing_count,
                    "empty_strings": empty_count,
                    "total_missing": total_missing,
                    "missing_percentage": total_missing_percentage,
                    "completeness_score": 1 - (total_missing_percentage / 100)
                }
            
            # Overall completeness score
            overall_completeness = sum(check["completeness_score"] for check in completeness_checks.values()) / len(completeness_checks)
            
            logger.info(f"Overall completeness score: {overall_completeness:.3f}")
            
            return {
                "overall_completeness": overall_completeness,
                "column_checks": completeness_checks
            }
            
        except Exception as e:
            logger.error(f"Error checking completeness: {e}")
            raise
    
    def checkValidity(self, data):
        """Check data validity (data types, ranges, formats)"""
        logger.info("Checking data validity...")
        
        try:
            total_records = data.count()
            validity_checks = {}
            
            for col_name, col_type in data.dtypes:
                validity_checks[col_name] = {
                    "data_type": col_type,
                    "valid_records": 0,
                    "invalid_records": 0,
                    "validity_score": 1.0
                }
                
                # Check for data type violations
                if col_type == 'int':
                    invalid_count = data.filter(col(col_name).isNull() | col(col_name).isNaN()).count()
                elif col_type == 'double':
                    invalid_count = data.filter(col(col_name).isNull() | col(col_name).isNaN()).count()
                elif col_type == 'string':
                    invalid_count = data.filter(col(col_name).isNull()).count()
                else:
                    invalid_count = 0
                
                validity_checks[col_name]["invalid_records"] = invalid_count
                validity_checks[col_name]["valid_records"] = total_records - invalid_count
                validity_checks[col_name]["validity_score"] = (total_records - invalid_count) / total_records
                
                # Additional range checks for specific columns
                if col_name == "person_age":
                    age_outliers = data.filter((col("person_age") < 18) | (col("person_age") > 100)).count()
                    validity_checks[col_name]["age_outliers"] = age_outliers
                
                elif col_name == "person_income":
                    negative_income = data.filter(col("person_income") < 0).count()
                    validity_checks[col_name]["negative_income"] = negative_income
                
                elif col_name == "loan_amnt":
                    negative_loans = data.filter(col("loan_amnt") < 0).count()
                    validity_checks[col_name]["negative_loans"] = negative_loans
                
                elif col_name == "loan_int_rate":
                    invalid_rates = data.filter((col("loan_int_rate") < 0) | (col("loan_int_rate") > 50)).count()
                    validity_checks[col_name]["invalid_rates"] = invalid_rates
            
            # Overall validity score
            overall_validity = sum(check["validity_score"] for check in validity_checks.values()) / len(validity_checks)
            
            logger.info(f"Overall validity score: {overall_validity:.3f}")
            
            return {
                "overall_validity": overall_validity,
                "column_checks": validity_checks
            }
            
        except Exception as e:
            logger.error(f"Error checking validity: {e}")
            raise
    
    def checkConsistency(self, data):
        """Check data consistency (relationships, business rules)"""
        logger.info("Checking data consistency...")
        
        try:
            total_records = data.count()
            consistency_checks = {}
            
            # 1. Check age vs employment length consistency
            age_emp_inconsistent = data.filter(col("person_emp_length") > (col("person_age") - 18)).count()
            consistency_checks["age_employment_consistency"] = {
                "inconsistent_records": age_emp_inconsistent,
                "percentage": (age_emp_inconsistent / total_records) * 100,
                "consistency_score": 1 - (age_emp_inconsistent / total_records)
            }
            
            # 2. Check loan amount vs income consistency
            high_debt_ratio = data.filter(col("loan_percent_income") > 1.0).count()
            consistency_checks["debt_income_consistency"] = {
                "high_debt_records": high_debt_ratio,
                "percentage": (high_debt_ratio / total_records) * 100,
                "consistency_score": 1 - (high_debt_ratio / total_records)
            }
            
            # 3. Check loan grade vs interest rate consistency
            if "loan_grade" in data.columns and "loan_int_rate" in data.columns:
                # Higher grades should have lower interest rates
                grade_rate_inconsistent = data.filter(
                    (col("loan_grade") == "A") & (col("loan_int_rate") > 15) |
                    (col("loan_grade") == "B") & (col("loan_int_rate") > 20) |
                    (col("loan_grade") == "C") & (col("loan_int_rate") > 25)
                ).count()
                
                consistency_checks["grade_rate_consistency"] = {
                    "inconsistent_records": grade_rate_inconsistent,
                    "percentage": (grade_rate_inconsistent / total_records) * 100,
                    "consistency_score": 1 - (grade_rate_inconsistent / total_records)
                }
            
            # 4. Check for duplicate records
            duplicates = total_records - data.dropDuplicates().count()
            consistency_checks["duplicate_consistency"] = {
                "duplicate_records": duplicates,
                "percentage": (duplicates / total_records) * 100,
                "consistency_score": 1 - (duplicates / total_records)
            }
            
            # Overall consistency score
            overall_consistency = sum(check["consistency_score"] for check in consistency_checks.values()) / len(consistency_checks)
            
            logger.info(f"Overall consistency score: {overall_consistency:.3f}")
            
            return {
                "overall_consistency": overall_consistency,
                "consistency_checks": consistency_checks
            }
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            raise
    
    def checkUniqueness(self, data):
        """Check data uniqueness"""
        logger.info("Checking data uniqueness...")
        
        try:
            total_records = data.count()
            unique_records = data.dropDuplicates().count()
            duplicate_count = total_records - unique_records
            
            uniqueness_score = unique_records / total_records
            
            logger.info(f"Uniqueness score: {uniqueness_score:.3f}")
            logger.info(f"Duplicate records: {duplicate_count}")
            
            return {
                "total_records": total_records,
                "unique_records": unique_records,
                "duplicate_records": duplicate_count,
                "uniqueness_score": uniqueness_score
            }
            
        except Exception as e:
            logger.error(f"Error checking uniqueness: {e}")
            raise
    
    def checkFreshness(self, data):
        """Check data freshness (timestamps, processing dates)"""
        logger.info("Checking data freshness...")
        
        try:
            # Check if processing_date column exists
            if "processing_date" in data.columns:
                # Get latest processing date
                latest_date = data.select(max("processing_date")).collect()[0][0]
                current_date = datetime.now().date()
                
                # Calculate days difference
                days_old = (current_date - latest_date).days
                
                # Freshness score (1.0 for same day, decreasing for older data)
                freshness_score = max(0, 1 - (days_old / 30))  # 30 days threshold
                
                logger.info(f"Latest processing date: {latest_date}")
                logger.info(f"Days old: {days_old}")
                logger.info(f"Freshness score: {freshness_score:.3f}")
                
                return {
                    "latest_processing_date": str(latest_date),
                    "days_old": days_old,
                    "freshness_score": freshness_score
                }
            else:
                logger.warning("No processing_date column found, skipping freshness check")
                return {
                    "freshness_score": 1.0,
                    "note": "No processing_date column found"
                }
                
        except Exception as e:
            logger.error(f"Error checking freshness: {e}")
            raise
    
    def calculateOverallQualityScore(self, completeness, validity, consistency, uniqueness, freshness):
        """Calculate overall data quality score"""
        logger.info("Calculating overall quality score...")
        
        try:
            # Weighted average of all quality dimensions
            weights = {
                "completeness": 0.25,
                "validity": 0.25,
                "consistency": 0.20,
                "uniqueness": 0.15,
                "freshness": 0.15
            }
            
            overall_score = (
                completeness["overall_completeness"] * weights["completeness"] +
                validity["overall_validity"] * weights["validity"] +
                consistency["overall_consistency"] * weights["consistency"] +
                uniqueness["uniqueness_score"] * weights["uniqueness"] +
                freshness["freshness_score"] * weights["freshness"]
            )
            
            logger.info(f"Overall quality score: {overall_score:.3f}")
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error calculating overall quality score: {e}")
            raise
    
    def generateQualityReport(self, data_path, quality_metrics):
        """Generate comprehensive quality report"""
        logger.info("Generating quality report...")
        
        try:
            # Create quality report
            quality_report = {
                "timestamp": datetime.now().isoformat(),
                "job_name": self.job_name,
                "data_path": data_path,
                "quality_metrics": quality_metrics,
                "quality_threshold": self.quality_threshold,
                "alert_threshold": self.alert_threshold,
                "quality_passed": quality_metrics["overall_score"] >= self.quality_threshold,
                "alert_triggered": quality_metrics["overall_score"] < self.alert_threshold
            }
            
            # Add recommendations
            recommendations = []
            
            if quality_metrics["completeness"]["overall_completeness"] < 0.95:
                recommendations.append("Data completeness is low - investigate missing value patterns")
            
            if quality_metrics["validity"]["overall_validity"] < 0.95:
                recommendations.append("Data validity issues detected - check data types and ranges")
            
            if quality_metrics["consistency"]["overall_consistency"] < 0.90:
                recommendations.append("Data consistency issues found - review business rules")
            
            if quality_metrics["uniqueness"]["uniqueness_score"] < 0.99:
                recommendations.append("Duplicate records detected - implement deduplication")
            
            if quality_metrics["freshness"]["freshness_score"] < 0.8:
                recommendations.append("Data is stale - check processing pipeline")
            
            quality_report["recommendations"] = recommendations
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            raise
    
    def saveQualityReport(self, quality_report):
        """Save quality report to S3"""
        logger.info("Saving quality report to S3...")
        
        try:
            # Convert to JSON
            report_json = json.dumps(quality_report, indent=2, default=str)
            
            # Save to S3
            report_key = f"quality/data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=report_key,
                Body=report_json,
                ContentType='application/json'
            )
            
            logger.info(f"Quality report saved to: s3://{self.s3_bucket}/{report_key}")
            
            return f"s3://{self.s3_bucket}/{report_key}"
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise
    
    def run(self, data_path=None):
        """Main execution method"""
        try:
            logger.info("Starting data quality check process...")
            
            # Default data path if not provided
            if not data_path:
                data_path = f"s3://{self.s3_bucket}/cleaned/"
            
            # Step 1: Read data
            data = self.readDataForQualityCheck(data_path)
            
            # Step 2: Check completeness
            completeness = self.checkCompleteness(data)
            
            # Step 3: Check validity
            validity = self.checkValidity(data)
            
            # Step 4: Check consistency
            consistency = self.checkConsistency(data)
            
            # Step 5: Check uniqueness
            uniqueness = self.checkUniqueness(data)
            
            # Step 6: Check freshness
            freshness = self.checkFreshness(data)
            
            # Step 7: Calculate overall score
            overall_score = self.calculateOverallQualityScore(completeness, validity, consistency, uniqueness, freshness)
            
            # Step 8: Generate report
            quality_metrics = {
                "completeness": completeness,
                "validity": validity,
                "consistency": consistency,
                "uniqueness": uniqueness,
                "freshness": freshness,
                "overall_score": overall_score
            }
            
            quality_report = self.generateQualityReport(data_path, quality_metrics)
            
            # Step 9: Save report
            report_path = self.saveQualityReport(quality_report)
            
            logger.info("Data quality check completed successfully!")
            logger.info(f"Overall quality score: {overall_score:.3f}")
            logger.info(f"Quality passed: {quality_report['quality_passed']}")
            logger.info(f"Alert triggered: {quality_report['alert_triggered']}")
            logger.info(f"Report saved to: {report_path}")
            
            return {
                "status": "SUCCESS",
                "quality_score": overall_score,
                "quality_passed": quality_report['quality_passed'],
                "alert_triggered": quality_report['alert_triggered'],
                "report_path": report_path,
                "recommendations": quality_report['recommendations']
            }
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }

def main():
    """Main entry point for Glue job"""
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    # Add additional arguments with defaults
    args['S3_BUCKET'] = args.get('S3_BUCKET', 'credit-risk-processed-data')
    args['QUALITY_THRESHOLD'] = args.get('QUALITY_THRESHOLD', '0.95')
    args['ALERT_THRESHOLD'] = args.get('ALERT_THRESHOLD', '0.90')
    args['DATA_PATH'] = args.get('DATA_PATH', None)
    
    # Initialize and run job
    job = DataQualityCheckJob(args)
    result = job.run(args.get('DATA_PATH'))
    
    # Log final result
    logger.info(f"Job completed with result: {result}")
    
    # Exit with appropriate code
    if result["status"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
