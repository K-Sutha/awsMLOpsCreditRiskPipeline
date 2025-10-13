#!/usr/bin/env python3
"""
AWS Glue Job: Feature Engineering
Production-ready ETL job for creating ML-ready features from cleaned data
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
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Glue context
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

class FeatureEngineeringJob:
    """Production-ready feature engineering job for credit risk data"""
    
    def __init__(self, args):
        self.args = args
        self.job_name = args['JOB_NAME']
        self.s3_processed_bucket = args.get('S3_PROCESSED_BUCKET', 'credit-risk-processed-data')
        self.s3_features_bucket = args.get('S3_FEATURES_BUCKET', 'credit-risk-features')
        self.train_test_split = float(args.get('TRAIN_TEST_SPLIT', '0.8'))
        self.random_seed = int(args.get('RANDOM_SEED', '42'))
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info(f"Starting Feature Engineering Job: {self.job_name}")
        logger.info(f"Processed bucket: {self.s3_processed_bucket}")
        logger.info(f"Features bucket: {self.s3_features_bucket}")
        logger.info(f"Train/test split: {self.train_test_split}")
    
    def readCleanedData(self):
        """Read cleaned data from S3"""
        logger.info("Reading cleaned data from S3...")
        
        try:
            # Read cleaned Parquet data
            cleaned_data = spark.read.parquet(f"s3://{self.s3_processed_bucket}/cleaned/")
            
            logger.info(f"Cleaned data loaded: {cleaned_data.count()} records, {len(cleaned_data.columns)} columns")
            
            # Show sample of data
            logger.info("Sample data:")
            cleaned_data.select("person_age", "person_income", "loan_amnt", "loan_status").show(5)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error reading cleaned data: {e}")
            raise
    
    def createDerivedFeatures(self, data):
        """Create derived features from existing columns"""
        logger.info("Creating derived features...")
        
        try:
            # 1. Age-based features
            data = data.withColumn("age_group", 
                                 when(col("person_age") <= 25, "Young")
                                 .when(col("person_age") <= 35, "Adult")
                                 .when(col("person_age") <= 45, "Middle")
                                 .otherwise("Senior"))
            
            # 2. Income-based features
            data = data.withColumn("income_quartile",
                                 when(col("person_income") <= data.select(expr("percentile_approx(person_income, 0.25)")).collect()[0][0], "Low")
                                 .when(col("person_income") <= data.select(expr("percentile_approx(person_income, 0.5)")).collect()[0][0], "Medium")
                                 .when(col("person_income") <= data.select(expr("percentile_approx(person_income, 0.75)")).collect()[0][0], "High")
                                 .otherwise("VeryHigh"))
            
            data = data.withColumn("has_high_income", 
                                 when(col("person_income") > data.select(expr("percentile_approx(person_income, 0.75)")).collect()[0][0], 1)
                                 .otherwise(0))
            
            # 3. Employment stability features
            data = data.withColumn("employment_stability", 
                                 col("person_emp_length") / (col("person_age") - 18).cast("double"))
            
            data = data.withColumn("has_stable_employment", 
                                 when(col("employment_stability") > 0.3, 1)
                                 .otherwise(0))
            
            # 4. Loan risk features
            data = data.withColumn("debt_to_income_ratio", col("loan_percent_income"))
            data = data.withColumn("income_to_loan_ratio", 
                                 col("person_income") / col("loan_amnt"))
            data = data.withColumn("has_high_debt_ratio", 
                                 when(col("debt_to_income_ratio") > 0.4, 1)
                                 .otherwise(0))
            
            # 5. Risk grade grouping
            data = data.withColumn("risk_grade_group", 
                                 when(col("loan_grade").isin(["A", "B"]), "LowRisk")
                                 .when(col("loan_grade").isin(["C", "D"]), "MediumRisk")
                                 .otherwise("HighRisk"))
            
            # 6. Historical default indicator
            data = data.withColumn("has_historical_default", 
                                 when(col("cb_person_default_on_file") == "Y", 1)
                                 .otherwise(0))
            
            # 7. Loan amount features
            data = data.withColumn("loan_amount_category",
                                 when(col("loan_amnt") <= 5000, "Small")
                                 .when(col("loan_amnt") <= 15000, "Medium")
                                 .when(col("loan_amnt") <= 25000, "Large")
                                 .otherwise("VeryLarge"))
            
            # 8. Interest rate features
            data = data.withColumn("interest_rate_category",
                                 when(col("loan_int_rate") <= 10, "Low")
                                 .when(col("loan_int_rate") <= 15, "Medium")
                                 .otherwise("High"))
            
            logger.info("Derived features created successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            raise
    
    def encodeCategoricalFeatures(self, data):
        """Encode categorical features for ML"""
        logger.info("Encoding categorical features...")
        
        try:
            # Define categorical columns
            categorical_columns = [
                "person_home_ownership",
                "loan_intent", 
                "age_group",
                "income_quartile",
                "risk_grade_group",
                "loan_amount_category",
                "interest_rate_category"
            ]
            
            # Create pipeline for encoding
            stages = []
            
            for col_name in categorical_columns:
                # String indexing
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
                stages.append(indexer)
                
                # One-hot encoding
                encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", 
                                      outputCol=f"{col_name}_encoded")
                stages.append(encoder)
            
            # Create and fit pipeline
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(data)
            
            # Transform data
            encoded_data = model.transform(data)
            
            logger.info("Categorical features encoded successfully")
            return encoded_data, model
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            raise
    
    def scaleNumericalFeatures(self, data):
        """Scale numerical features for ML"""
        logger.info("Scaling numerical features...")
        
        try:
            # Define numerical columns to scale
            numerical_columns = [
                "person_age",
                "person_income", 
                "person_emp_length",
                "loan_amnt",
                "loan_int_rate",
                "loan_percent_income",
                "cb_person_cred_hist_length",
                "employment_stability",
                "income_to_loan_ratio"
            ]
            
            # Create vector assembler
            assembler = VectorAssembler(
                inputCols=numerical_columns,
                outputCol="numerical_features"
            )
            
            # Create scaler
            scaler = StandardScaler(
                inputCol="numerical_features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler])
            model = pipeline.fit(data)
            
            # Transform data
            scaled_data = model.transform(data)
            
            logger.info("Numerical features scaled successfully")
            return scaled_data, model
            
        except Exception as e:
            logger.error(f"Error scaling numerical features: {e}")
            raise
    
    def createMLFeatures(self, data):
        """Create final ML-ready feature vector"""
        logger.info("Creating ML-ready feature vector...")
        
        try:
            # Get all encoded categorical columns
            categorical_encoded_cols = [col for col in data.columns if col.endswith("_encoded")]
            
            # Combine scaled numerical features with encoded categorical features
            feature_assembler = VectorAssembler(
                inputCols=["scaled_features"] + categorical_encoded_cols,
                outputCol="features"
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[feature_assembler])
            model = pipeline.fit(data)
            
            # Transform data
            ml_data = model.transform(data)
            
            # Select only necessary columns for ML
            ml_columns = ["features", "loan_status", "processing_date", "job_name"]
            ml_data = ml_data.select(*ml_columns)
            
            logger.info("ML features created successfully")
            logger.info(f"Feature vector dimension: {ml_data.select('features').first()[0].size}")
            
            return ml_data, model
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            raise
    
    def splitTrainTest(self, ml_data):
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets...")
        
        try:
            # Split data
            train_data, test_data = ml_data.randomSplit([self.train_test_split, 1-self.train_test_split], 
                                                      seed=self.random_seed)
            
            train_count = train_data.count()
            test_count = test_data.count()
            
            logger.info(f"Train set: {train_count} records")
            logger.info(f"Test set: {test_count} records")
            
            # Check target distribution in both sets
            train_distribution = train_data.groupBy("loan_status").count().collect()
            test_distribution = test_data.groupBy("loan_status").count().collect()
            
            logger.info("Train set distribution:")
            for row in train_distribution:
                logger.info(f"  Loan status {row['loan_status']}: {row['count']} records")
            
            logger.info("Test set distribution:")
            for row in test_distribution:
                logger.info(f"  Loan status {row['loan_status']}: {row['count']} records")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting train/test data: {e}")
            raise
    
    def saveFeatures(self, train_data, test_data, feature_models):
        """Save processed features to S3"""
        logger.info("Saving features to S3...")
        
        try:
            # Save train data
            train_path = f"s3://{self.s3_features_bucket}/train/"
            train_data.write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(train_path)
            
            logger.info(f"Train data saved to: {train_path}")
            
            # Save test data
            test_path = f"s3://{self.s3_features_bucket}/test/"
            test_data.write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(test_path)
            
            logger.info(f"Test data saved to: {test_path}")
            
            # Save feature metadata
            feature_metadata = {
                "timestamp": datetime.now().isoformat(),
                "job_name": self.job_name,
                "train_records": train_data.count(),
                "test_records": test_data.count(),
                "feature_dimension": train_data.select("features").first()[0].size,
                "train_test_split": self.train_test_split,
                "random_seed": self.random_seed,
                "models_saved": list(feature_models.keys())
            }
            
            # Save metadata to S3
            import json
            metadata_json = json.dumps(feature_metadata, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.s3_features_bucket,
                Key=f"metadata/feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                Body=metadata_json,
                ContentType='application/json'
            )
            
            logger.info("Feature metadata saved successfully")
            
            return {
                "train_path": train_path,
                "test_path": test_path,
                "metadata": feature_metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise
    
    def run(self):
        """Main execution method"""
        try:
            logger.info("Starting feature engineering process...")
            
            # Step 1: Read cleaned data
            cleaned_data = self.readCleanedData()
            
            # Step 2: Create derived features
            derived_data = self.createDerivedFeatures(cleaned_data)
            
            # Step 3: Encode categorical features
            encoded_data, categorical_model = self.encodeCategoricalFeatures(derived_data)
            
            # Step 4: Scale numerical features
            scaled_data, scaling_model = self.scaleNumericalFeatures(encoded_data)
            
            # Step 5: Create ML features
            ml_data, feature_model = self.createMLFeatures(scaled_data)
            
            # Step 6: Split train/test
            train_data, test_data = self.splitTrainTest(ml_data)
            
            # Step 7: Save features
            feature_models = {
                "categorical_encoder": categorical_model,
                "numerical_scaler": scaling_model,
                "feature_assembler": feature_model
            }
            
            result = self.saveFeatures(train_data, test_data, feature_models)
            
            logger.info("Feature engineering completed successfully!")
            logger.info(f"Train records: {result['metadata']['train_records']}")
            logger.info(f"Test records: {result['metadata']['test_records']}")
            logger.info(f"Feature dimension: {result['metadata']['feature_dimension']}")
            
            return {
                "status": "SUCCESS",
                "train_path": result["train_path"],
                "test_path": result["test_path"],
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }

def main():
    """Main entry point for Glue job"""
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    
    # Add additional arguments with defaults
    args['S3_PROCESSED_BUCKET'] = args.get('S3_PROCESSED_BUCKET', 'credit-risk-processed-data')
    args['S3_FEATURES_BUCKET'] = args.get('S3_FEATURES_BUCKET', 'credit-risk-features')
    args['TRAIN_TEST_SPLIT'] = args.get('TRAIN_TEST_SPLIT', '0.8')
    args['RANDOM_SEED'] = args.get('RANDOM_SEED', '42')
    
    # Initialize and run job
    job = FeatureEngineeringJob(args)
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
