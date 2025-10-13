#!/usr/bin/env python3
"""
AWS Glue Job: Data Ingestion
Production-ready ETL job for cleaning and preparing raw credit risk data
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

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Glue context
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

class DataIngestionJob:
    """Production-ready data ingestion job for credit risk data"""
    
    def __init__(self, args):
        self.args = args
        self.job_name = args['JOB_NAME']
        self.s3_raw_bucket = args.get('S3_RAW_BUCKET', 'credit-risk-raw-data')
        self.s3_processed_bucket = args.get('S3_PROCESSED_BUCKET', 'credit-risk-processed-data')
        self.quality_threshold = float(args.get('QUALITY_THRESHOLD', '0.95'))
        
        # Initialize S3 client for logging
        self.s3_client = boto3.client('s3')
        
        logger.info(f"Starting Data Ingestion Job: {self.job_name}")
        logger.info(f"Raw bucket: {self.s3_raw_bucket}")
        logger.info(f"Processed bucket: {self.s3_processed_bucket}")
    
    def readRawData(self):
        """Read raw CSV data from S3"""
        logger.info("Reading raw data from S3...")
        
        try:
            # Read CSV from S3 with proper schema
            raw_data = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("nullValue", "?") \
                .load(f"s3://{self.s3_raw_bucket}/raw/")
            
            logger.info(f"Raw data loaded: {raw_data.count()} records, {len(raw_data.columns)} columns")
            return raw_data
            
        except Exception as e:
            logger.error(f"Error reading raw data: {e}")
            raise
    
    def cleanData(self, raw_data):
        """Clean and validate raw data"""
        logger.info("Starting data cleaning process...")
        
        try:
            initial_count = raw_data.count()
            logger.info(f"Initial record count: {initial_count}")
            
            # 1. Remove duplicates
            cleaned_data = raw_data.dropDuplicates()
            after_dedup = cleaned_data.count()
            duplicates_removed = initial_count - after_dedup
            logger.info(f"Duplicates removed: {duplicates_removed}")
            
            # 2. Fix age outliers (cap at 100)
            age_outliers = cleaned_data.filter(col("person_age") > 100)
            age_outlier_count = age_outliers.count()
            logger.info(f"Age outliers found: {age_outlier_count}")
            
            cleaned_data = cleaned_data.withColumn("person_age", 
                                                 when(col("person_age") > 100, 100)
                                                 .otherwise(col("person_age")))
            
            # 3. Fix negative income values
            negative_income = cleaned_data.filter(col("person_income") < 0)
            negative_income_count = negative_income.count()
            logger.info(f"Negative income records: {negative_income_count}")
            
            cleaned_data = cleaned_data.withColumn("person_income", 
                                                 when(col("person_income") < 0, 0)
                                                 .otherwise(col("person_income")))
            
            # 4. Fix negative loan amounts
            negative_loans = cleaned_data.filter(col("loan_amnt") < 0)
            negative_loan_count = negative_loans.count()
            logger.info(f"Negative loan records: {negative_loan_count}")
            
            cleaned_data = cleaned_data.withColumn("loan_amnt", 
                                                 when(col("loan_amnt") < 0, 0)
                                                 .otherwise(col("loan_amnt")))
            
            # 5. Cap income at 99th percentile
            income_stats = cleaned_data.select(expr("percentile_approx(person_income, 0.99)").alias("p99")).collect()
            income_99th = income_stats[0]["p99"]
            logger.info(f"Income 99th percentile: {income_99th}")
            
            income_outliers = cleaned_data.filter(col("person_income") > income_99th)
            income_outlier_count = income_outliers.count()
            logger.info(f"Income outliers capped: {income_outlier_count}")
            
            cleaned_data = cleaned_data.withColumn("person_income", 
                                                 when(col("person_income") > income_99th, income_99th)
                                                 .otherwise(col("person_income")))
            
            # 6. Cap loan amount at 99th percentile
            loan_stats = cleaned_data.select(expr("percentile_approx(loan_amnt, 0.99)").alias("p99")).collect()
            loan_99th = loan_stats[0]["p99"]
            logger.info(f"Loan amount 99th percentile: {loan_99th}")
            
            loan_outliers = cleaned_data.filter(col("loan_amnt") > loan_99th)
            loan_outlier_count = loan_outliers.count()
            logger.info(f"Loan amount outliers capped: {loan_outlier_count}")
            
            cleaned_data = cleaned_data.withColumn("loan_amnt", 
                                                 when(col("loan_amnt") > loan_99th, loan_99th)
                                                 .otherwise(col("loan_amnt")))
            
            # 7. Fix employment length outliers
            emp_outliers = cleaned_data.filter(col("person_emp_length") > 50)
            emp_outlier_count = emp_outliers.count()
            logger.info(f"Employment length outliers: {emp_outlier_count}")
            
            cleaned_data = cleaned_data.withColumn("person_emp_length", 
                                                 when(col("person_emp_length") > 50, 50)
                                                 .otherwise(col("person_emp_length")))
            
            # 8. Handle missing interest rates
            missing_rates = cleaned_data.filter(col("loan_int_rate").isNull())
            missing_rate_count = missing_rates.count()
            logger.info(f"Missing interest rates: {missing_rate_count}")
            
            # Fill missing rates with median by loan grade
            median_rates = cleaned_data.groupBy("loan_grade") \
                                     .agg(expr("percentile_approx(loan_int_rate, 0.5)").alias("median_rate")) \
                                     .collect()
            
            median_rate_dict = {row["loan_grade"]: row["median_rate"] for row in median_rates}
            
            for grade, median_rate in median_rate_dict.items():
                cleaned_data = cleaned_data.withColumn("loan_int_rate", 
                                                     when((col("loan_grade") == grade) & col("loan_int_rate").isNull(), 
                                                          median_rate)
                                                     .otherwise(col("loan_int_rate")))
            
            # 9. Handle missing employment length
            missing_emp = cleaned_data.filter(col("person_emp_length").isNull())
            missing_emp_count = missing_emp.count()
            logger.info(f"Missing employment length: {missing_emp_count}")
            
            # Fill missing employment with median by age group
            cleaned_data = cleaned_data.withColumn("age_group", 
                                                 when(col("person_age") <= 25, "18-25")
                                                 .when(col("person_age") <= 35, "26-35")
                                                 .when(col("person_age") <= 45, "36-45")
                                                 .otherwise("45+"))
            
            median_emp_by_age = cleaned_data.groupBy("age_group") \
                                          .agg(expr("percentile_approx(person_emp_length, 0.5)").alias("median_emp")) \
                                          .collect()
            
            median_emp_dict = {row["age_group"]: row["median_emp"] for row in median_emp_by_age}
            
            for age_group, median_emp in median_emp_dict.items():
                cleaned_data = cleaned_data.withColumn("person_emp_length", 
                                                     when((col("age_group") == age_group) & col("person_emp_length").isNull(), 
                                                          median_emp)
                                                     .otherwise(col("person_emp_length")))
            
            # Remove temporary age_group column
            cleaned_data = cleaned_data.drop("age_group")
            
            final_count = cleaned_data.count()
            logger.info(f"Final cleaned record count: {final_count}")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise
    
    def validateDataQuality(self, cleaned_data):
        """Validate data quality and generate quality report"""
        logger.info("Validating data quality...")
        
        try:
            total_records = cleaned_data.count()
            
            # Check for missing values
            missing_values = {}
            for col_name in cleaned_data.columns:
                missing_count = cleaned_data.filter(col(col_name).isNull()).count()
                missing_percentage = (missing_count / total_records) * 100
                missing_values[col_name] = {
                    "count": missing_count,
                    "percentage": missing_percentage
                }
            
            # Check data types
            data_types = dict(cleaned_data.dtypes)
            
            # Check target variable distribution
            target_distribution = cleaned_data.groupBy("loan_status") \
                                            .count() \
                                            .withColumn("percentage", 
                                                      col("count") / total_records * 100) \
                                            .collect()
            
            # Generate quality score
            quality_score = 1.0
            for col_name, stats in missing_values.items():
                if stats["percentage"] > 5:  # More than 5% missing
                    quality_score -= 0.1
            
            # Create quality report
            quality_report = {
                "timestamp": datetime.now().isoformat(),
                "job_name": self.job_name,
                "total_records": total_records,
                "quality_score": quality_score,
                "missing_values": missing_values,
                "data_types": data_types,
                "target_distribution": [{"loan_status": row["loan_status"], 
                                       "count": row["count"], 
                                       "percentage": row["percentage"]} 
                                      for row in target_distribution],
                "quality_threshold": self.quality_threshold,
                "quality_passed": quality_score >= self.quality_threshold
            }
            
            logger.info(f"Data quality score: {quality_score:.3f}")
            logger.info(f"Quality threshold: {self.quality_threshold}")
            logger.info(f"Quality passed: {quality_report['quality_passed']}")
            
            if not quality_report['quality_passed']:
                logger.warning("Data quality below threshold!")
                # In production, you might want to send alerts here
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            raise
    
    def saveCleanedData(self, cleaned_data, quality_report):
        """Save cleaned data to S3"""
        logger.info("Saving cleaned data to S3...")
        
        try:
            # Add metadata columns
            cleaned_data_with_meta = cleaned_data.withColumn("processing_date", current_date()) \
                                                .withColumn("job_name", lit(self.job_name)) \
                                                .withColumn("quality_score", lit(quality_report["quality_score"]))
            
            # Save as Parquet for better performance
            output_path = f"s3://{self.s3_processed_bucket}/cleaned/"
            cleaned_data_with_meta.write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(output_path)
            
            logger.info(f"Cleaned data saved to: {output_path}")
            
            # Save quality report
            quality_report_path = f"s3://{self.s3_processed_bucket}/quality/ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert report to JSON and save
            import json
            report_json = json.dumps(quality_report, indent=2)
            
            # Save to S3
            self.s3_client.put_object(
                Bucket=self.s3_processed_bucket,
                Key=f"quality/ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                Body=report_json,
                ContentType='application/json'
            )
            
            logger.info(f"Quality report saved to: {quality_report_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise
    
    def run(self):
        """Main execution method"""
        try:
            logger.info("Starting data ingestion process...")
            
            # Step 1: Read raw data
            raw_data = self.readRawData()
            
            # Step 2: Clean data
            cleaned_data = self.cleanData(raw_data)
            
            # Step 3: Validate quality
            quality_report = self.validateDataQuality(cleaned_data)
            
            # Step 4: Save cleaned data
            output_path = self.saveCleanedData(cleaned_data, quality_report)
            
            logger.info("Data ingestion completed successfully!")
            logger.info(f"Output path: {output_path}")
            logger.info(f"Quality score: {quality_report['quality_score']:.3f}")
            
            return {
                "status": "SUCCESS",
                "output_path": output_path,
                "quality_score": quality_report["quality_score"],
                "records_processed": quality_report["total_records"]
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }

def main():
    """Main entry point for Glue job"""
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    # Add additional arguments with defaults
    args['S3_RAW_BUCKET'] = args.get('S3_RAW_BUCKET', 'credit-risk-raw-data')
    args['S3_PROCESSED_BUCKET'] = args.get('S3_PROCESSED_BUCKET', 'credit-risk-processed-data')
    args['QUALITY_THRESHOLD'] = args.get('QUALITY_THRESHOLD', '0.95')
    
    # Initialize and run job
    job = DataIngestionJob(args)
    result = job.run()
    
    # Log final result
    logger.info(f"Job completed with result: {result}")
    
    # Exit with appropriate code
    if result["status"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
