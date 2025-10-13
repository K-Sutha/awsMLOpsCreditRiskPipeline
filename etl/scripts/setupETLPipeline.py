#!/usr/bin/env python3
"""
ETL Pipeline Setup Script
Production-ready script for setting up AWS Glue ETL pipeline
"""

import boto3
import json
import logging
import time
from datetime import datetime
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLPipelineSetup:
    """Setup and configure AWS Glue ETL pipeline"""
    
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.glue_client = boto3.client('glue', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.iam_client = boto3.client('iam', region_name=region_name)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region_name)
        self.sns_client = boto3.client('sns', region_name=region_name)
        
        logger.info(f"Initialized ETL Pipeline Setup for region: {region_name}")
    
    def createS3Buckets(self):
        """Create S3 buckets for ETL pipeline"""
        logger.info("Creating S3 buckets for ETL pipeline...")
        
        buckets = [
            'credit-risk-raw-data',
            'credit-risk-processed-data',
            'credit-risk-features',
            'credit-risk-glue-scripts',
            'credit-risk-glue-temp'
        ]
        
        created_buckets = []
        
        for bucket_name in buckets:
            try:
                # Create bucket
                if self.region_name == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region_name}
                    )
                
                # Enable versioning
                self.s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                # Set lifecycle policy
                lifecycle_config = {
                    'Rules': [
                        {
                            'ID': f'{bucket_name}-lifecycle',
                            'Status': 'Enabled',
                            'Transitions': [
                                {
                                    'Days': 30,
                                    'StorageClass': 'STANDARD_IA'
                                },
                                {
                                    'Days': 90,
                                    'StorageClass': 'GLACIER'
                                }
                            ]
                        }
                    ]
                }
                
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=bucket_name,
                    LifecycleConfiguration=lifecycle_config
                )
                
                logger.info(f"Created bucket: {bucket_name}")
                created_buckets.append(bucket_name)
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'BucketAlreadyExists':
                    logger.info(f"Bucket already exists: {bucket_name}")
                    created_buckets.append(bucket_name)
                else:
                    logger.error(f"Error creating bucket {bucket_name}: {e}")
                    raise
        
        return created_buckets
    
    def createIAMRole(self):
        """Create IAM role for AWS Glue"""
        logger.info("Creating IAM role for AWS Glue...")
        
        role_name = 'AWSGlueServiceRole'
        
        try:
            # Check if role exists
            self.iam_client.get_role(RoleName=role_name)
            logger.info(f"IAM role already exists: {role_name}")
            return f"arn:aws:iam::{self.iam_client.get_caller_identity()['Account']}:role/{role_name}"
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                # Create role
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "glue.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                self.iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description='IAM role for AWS Glue service'
                )
                
                # Attach policies
                policies = [
                    'arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole',
                    'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                    'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
                ]
                
                for policy_arn in policies:
                    self.iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                
                logger.info(f"Created IAM role: {role_name}")
                return f"arn:aws:iam::{self.iam_client.get_caller_identity()['Account']}:role/{role_name}"
                
            else:
                logger.error(f"Error creating IAM role: {e}")
                raise
    
    def createGlueDatabases(self):
        """Create Glue databases"""
        logger.info("Creating Glue databases...")
        
        databases = [
            {
                'name': 'credit_risk_raw',
                'description': 'Database for raw credit risk data',
                'location_uri': 's3://credit-risk-raw-data/raw/'
            },
            {
                'name': 'credit_risk_processed',
                'description': 'Database for processed credit risk data',
                'location_uri': 's3://credit-risk-processed-data/cleaned/'
            },
            {
                'name': 'credit_risk_features',
                'description': 'Database for ML features',
                'location_uri': 's3://credit-risk-features/'
            },
            {
                'name': 'credit_risk_quality',
                'description': 'Database for quality reports',
                'location_uri': 's3://credit-risk-processed-data/quality/'
            }
        ]
        
        created_databases = []
        
        for db in databases:
            try:
                self.glue_client.create_database(
                    DatabaseInput={
                        'Name': db['name'],
                        'Description': db['description'],
                        'LocationUri': db['location_uri']
                    }
                )
                logger.info(f"Created database: {db['name']}")
                created_databases.append(db['name'])
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'AlreadyExistsException':
                    logger.info(f"Database already exists: {db['name']}")
                    created_databases.append(db['name'])
                else:
                    logger.error(f"Error creating database {db['name']}: {e}")
                    raise
        
        return created_databases
    
    def createGlueJobs(self, role_arn):
        """Create Glue jobs"""
        logger.info("Creating Glue jobs...")
        
        jobs = [
            {
                'name': 'data-ingestion-job',
                'description': 'Data ingestion and cleaning job',
                'script_location': 's3://credit-risk-glue-scripts/dataIngestion.py',
                'role': role_arn,
                'max_capacity': 2,
                'timeout': 60,
                'arguments': {
                    'S3_RAW_BUCKET': 'credit-risk-raw-data',
                    'S3_PROCESSED_BUCKET': 'credit-risk-processed-data',
                    'QUALITY_THRESHOLD': '0.95'
                }
            },
            {
                'name': 'feature-engineering-job',
                'description': 'Feature engineering job',
                'script_location': 's3://credit-risk-glue-scripts/featureEngineering.py',
                'role': role_arn,
                'max_capacity': 2,
                'timeout': 60,
                'arguments': {
                    'S3_PROCESSED_BUCKET': 'credit-risk-processed-data',
                    'S3_FEATURES_BUCKET': 'credit-risk-features',
                    'TRAIN_TEST_SPLIT': '0.8',
                    'RANDOM_SEED': '42'
                }
            },
            {
                'name': 'data-quality-check-job',
                'description': 'Data quality monitoring job',
                'script_location': 's3://credit-risk-glue-scripts/dataQualityCheck.py',
                'role': role_arn,
                'max_capacity': 2,
                'timeout': 60,
                'arguments': {
                    'S3_BUCKET': 'credit-risk-processed-data',
                    'QUALITY_THRESHOLD': '0.95',
                    'ALERT_THRESHOLD': '0.90'
                }
            }
        ]
        
        created_jobs = []
        
        for job in jobs:
            try:
                self.glue_client.create_job(
                    Name=job['name'],
                    Description=job['description'],
                    Role=job['role'],
                    Command={
                        'Name': 'glueetl',
                        'ScriptLocation': job['script_location']
                    },
                    MaxCapacity=job['max_capacity'],
                    Timeout=job['timeout'],
                    DefaultArguments=job['arguments']
                )
                logger.info(f"Created job: {job['name']}")
                created_jobs.append(job['name'])
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'AlreadyExistsException':
                    logger.info(f"Job already exists: {job['name']}")
                    created_jobs.append(job['name'])
                else:
                    logger.error(f"Error creating job {job['name']}: {e}")
                    raise
        
        return created_jobs
    
    def createGlueCrawlers(self, role_arn):
        """Create Glue crawlers"""
        logger.info("Creating Glue crawlers...")
        
        crawlers = [
            {
                'name': 'credit-risk-raw-crawler',
                'description': 'Crawler for raw credit risk data',
                'role': role_arn,
                'database_name': 'credit_risk_raw',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': 's3://credit-risk-raw-data/raw/'
                        }
                    ]
                },
                'schedule': 'cron(0 2 * * ? *)',
                'table_prefix': 'raw_'
            },
            {
                'name': 'credit-risk-processed-crawler',
                'description': 'Crawler for processed credit risk data',
                'role': role_arn,
                'database_name': 'credit_risk_processed',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': 's3://credit-risk-processed-data/cleaned/'
                        }
                    ]
                },
                'schedule': 'cron(0 3 * * ? *)',
                'table_prefix': 'processed_'
            },
            {
                'name': 'credit-risk-features-crawler',
                'description': 'Crawler for ML features',
                'role': role_arn,
                'database_name': 'credit_risk_features',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': 's3://credit-risk-features/train/'
                        },
                        {
                            'Path': 's3://credit-risk-features/test/'
                        }
                    ]
                },
                'schedule': 'cron(0 4 * * ? *)',
                'table_prefix': 'features_'
            }
        ]
        
        created_crawlers = []
        
        for crawler in crawlers:
            try:
                self.glue_client.create_crawler(
                    Name=crawler['name'],
                    Description=crawler['description'],
                    Role=crawler['role'],
                    DatabaseName=crawler['database_name'],
                    Targets=crawler['targets'],
                    Schedule=crawler['schedule'],
                    TablePrefix=crawler['table_prefix'],
                    SchemaChangePolicy={
                        'UpdateBehavior': 'UPDATE_IN_DATABASE',
                        'DeleteBehavior': 'LOG'
                    }
                )
                logger.info(f"Created crawler: {crawler['name']}")
                created_crawlers.append(crawler['name'])
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'AlreadyExistsException':
                    logger.info(f"Crawler already exists: {crawler['name']}")
                    created_crawlers.append(crawler['name'])
                else:
                    logger.error(f"Error creating crawler {crawler['name']}: {e}")
                    raise
        
        return created_crawlers
    
    def createWorkflow(self):
        """Create Glue workflow"""
        logger.info("Creating Glue workflow...")
        
        workflow_name = 'credit-risk-etl-workflow'
        
        try:
            self.glue_client.create_workflow(
                Name=workflow_name,
                Description='Complete ETL workflow for credit risk data processing',
                DefaultRunProperties={
                    'S3_RAW_BUCKET': 'credit-risk-raw-data',
                    'S3_PROCESSED_BUCKET': 'credit-risk-processed-data',
                    'S3_FEATURES_BUCKET': 'credit-risk-features',
                    'QUALITY_THRESHOLD': '0.95',
                    'ALERT_THRESHOLD': '0.90'
                }
            )
            logger.info(f"Created workflow: {workflow_name}")
            return workflow_name
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AlreadyExistsException':
                logger.info(f"Workflow already exists: {workflow_name}")
                return workflow_name
            else:
                logger.error(f"Error creating workflow: {e}")
                raise
    
    def createCloudWatchLogGroup(self):
        """Create CloudWatch log group"""
        logger.info("Creating CloudWatch log group...")
        
        log_group_name = '/aws/glue/credit-risk-etl'
        
        try:
            self.cloudwatch_client.create_log_group(
                logGroupName=log_group_name,
                retentionInDays=30
            )
            logger.info(f"Created log group: {log_group_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.info(f"Log group already exists: {log_group_name}")
            else:
                logger.error(f"Error creating log group: {e}")
                raise
    
    def createSNSTopic(self):
        """Create SNS topic for notifications"""
        logger.info("Creating SNS topic...")
        
        topic_name = 'credit-risk-etl-notifications'
        
        try:
            response = self.sns_client.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            
            # Set topic attributes
            self.sns_client.set_topic_attributes(
                TopicArn=topic_arn,
                AttributeName='DisplayName',
                AttributeValue='Credit Risk ETL Notifications'
            )
            
            logger.info(f"Created SNS topic: {topic_name}")
            logger.info(f"Topic ARN: {topic_arn}")
            return topic_arn
            
        except ClientError as e:
            logger.error(f"Error creating SNS topic: {e}")
            raise
    
    def uploadGlueScripts(self):
        """Upload Glue job scripts to S3"""
        logger.info("Uploading Glue scripts to S3...")
        
        scripts = [
            'dataIngestion.py',
            'featureEngineering.py',
            'dataQualityCheck.py'
        ]
        
        uploaded_scripts = []
        
        for script in scripts:
            try:
                # Read script content
                with open(f'../glueJobs/{script}', 'r') as f:
                    script_content = f.read()
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket='credit-risk-glue-scripts',
                    Key=script,
                    Body=script_content,
                    ContentType='text/plain'
                )
                
                logger.info(f"Uploaded script: {script}")
                uploaded_scripts.append(script)
                
            except FileNotFoundError:
                logger.warning(f"Script file not found: {script}")
            except Exception as e:
                logger.error(f"Error uploading script {script}: {e}")
                raise
        
        return uploaded_scripts
    
    def runSetup(self):
        """Run complete ETL pipeline setup"""
        logger.info("Starting ETL pipeline setup...")
        
        try:
            # Step 1: Create S3 buckets
            buckets = self.createS3Buckets()
            logger.info(f"Created/verified {len(buckets)} S3 buckets")
            
            # Step 2: Create IAM role
            role_arn = self.createIAMRole()
            logger.info(f"Created/verified IAM role: {role_arn}")
            
            # Step 3: Create Glue databases
            databases = self.createGlueDatabases()
            logger.info(f"Created/verified {len(databases)} Glue databases")
            
            # Step 4: Create Glue jobs
            jobs = self.createGlueJobs(role_arn)
            logger.info(f"Created/verified {len(jobs)} Glue jobs")
            
            # Step 5: Create Glue crawlers
            crawlers = self.createGlueCrawlers(role_arn)
            logger.info(f"Created/verified {len(crawlers)} Glue crawlers")
            
            # Step 6: Create workflow
            workflow = self.createWorkflow()
            logger.info(f"Created/verified workflow: {workflow}")
            
            # Step 7: Create CloudWatch log group
            self.createCloudWatchLogGroup()
            logger.info("Created/verified CloudWatch log group")
            
            # Step 8: Create SNS topic
            topic_arn = self.createSNSTopic()
            logger.info(f"Created SNS topic: {topic_arn}")
            
            # Step 9: Upload Glue scripts
            scripts = self.uploadGlueScripts()
            logger.info(f"Uploaded {len(scripts)} Glue scripts")
            
            # Generate setup report
            setup_report = {
                "timestamp": datetime.now().isoformat(),
                "region": self.region_name,
                "buckets": buckets,
                "role_arn": role_arn,
                "databases": databases,
                "jobs": jobs,
                "crawlers": crawlers,
                "workflow": workflow,
                "topic_arn": topic_arn,
                "scripts": scripts,
                "status": "SUCCESS"
            }
            
            # Save setup report
            with open('etl_setup_report.json', 'w') as f:
                json.dump(setup_report, f, indent=2)
            
            logger.info("ETL pipeline setup completed successfully!")
            logger.info("Setup report saved to: etl_setup_report.json")
            
            return setup_report
            
        except Exception as e:
            logger.error(f"ETL pipeline setup failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup AWS Glue ETL pipeline')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run mode - no changes will be made")
        return
    
    # Initialize setup
    setup = ETLPipelineSetup(region_name=args.region)
    
    # Run setup
    result = setup.runSetup()
    
    if result["status"] == "SUCCESS":
        logger.info("ETL pipeline setup completed successfully!")
        print("\n" + "="*50)
        print("ETL PIPELINE SETUP SUMMARY")
        print("="*50)
        print(f"Region: {result['region']}")
        print(f"Buckets: {len(result['buckets'])}")
        print(f"Databases: {len(result['databases'])}")
        print(f"Jobs: {len(result['jobs'])}")
        print(f"Crawlers: {len(result['crawlers'])}")
        print(f"Workflow: {result['workflow']}")
        print(f"Topic ARN: {result['topic_arn']}")
        print("="*50)
    else:
        logger.error("ETL pipeline setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
