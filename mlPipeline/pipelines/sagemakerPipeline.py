#!/usr/bin/env python3
"""
SageMaker Pipeline Script for Credit Risk Platform
Production-ready ML pipeline orchestration with SageMaker Pipelines
"""

import os
import sys
import json
import logging
import boto3
from datetime import datetime
from typing import Dict, Any, List
import argparse

# SageMaker imports
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep, TuningStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import HyperparameterTuner
from sagemaker.inputs import TrainingInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerPipeline:
    """Production-ready SageMaker pipeline for credit risk ML"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.sagemaker_session = sagemaker.Session()
        self.role = self.get_sagemaker_role()
        self.pipeline = None
        
        logger.info("SageMaker Pipeline initialized")
    
    def load_config(self, config_path):
        """Load pipeline configuration"""
        default_config = {
            "pipeline_name": "credit-risk-ml-pipeline",
            "region": "us-east-1",
            "s3_bucket": "credit-risk-features",
            "data_prefix": "data/",
            "model_prefix": "models/",
            "output_prefix": "output/",
            "training_config": {
                "instance_type": "ml.m5.large",
                "instance_count": 1,
                "framework_version": "0.23-1",
                "python_version": "py3",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            },
            "processing_config": {
                "instance_type": "ml.m5.large",
                "instance_count": 1,
                "framework_version": "0.23-1",
                "python_version": "py3"
            },
            "tuning_config": {
                "max_jobs": 10,
                "max_parallel_jobs": 2,
                "objective_metric_name": "validation:auc",
                "objective_type": "Maximize"
            },
            "model_config": {
                "model_package_group_name": "credit-risk-models",
                "approval_status": "PendingManualApproval"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def get_sagemaker_role(self):
        """Get SageMaker execution role"""
        try:
            # Try to get role from environment or use default
            role = os.environ.get('SAGEMAKER_ROLE')
            if not role:
                # Get default SageMaker role
                iam_client = boto3.client('iam')
                response = iam_client.get_role(RoleName='SageMakerExecutionRole')
                role = response['Role']['Arn']
            
            logger.info(f"Using SageMaker role: {role}")
            return role
            
        except Exception as e:
            logger.error(f"Error getting SageMaker role: {e}")
            raise
    
    def create_parameters(self):
        """Create pipeline parameters"""
        logger.info("Creating pipeline parameters...")
        
        try:
            parameters = {
                'processing_instance_type': ParameterString(
                    name='ProcessingInstanceType',
                    default_value=self.config['processing_config']['instance_type'],
                    description='Instance type for processing jobs'
                ),
                'processing_instance_count': ParameterInteger(
                    name='ProcessingInstanceCount',
                    default_value=self.config['processing_config']['instance_count'],
                    description='Instance count for processing jobs'
                ),
                'training_instance_type': ParameterString(
                    name='TrainingInstanceType',
                    default_value=self.config['training_config']['instance_type'],
                    description='Instance type for training jobs'
                ),
                'training_instance_count': ParameterInteger(
                    name='TrainingInstanceCount',
                    default_value=self.config['training_config']['instance_count'],
                    description='Instance count for training jobs'
                ),
                'model_approval_status': ParameterString(
                    name='ModelApprovalStatus',
                    default_value=self.config['model_config']['approval_status'],
                    description='Model approval status'
                ),
                'input_data': ParameterString(
                    name='InputData',
                    default_value=f"s3://{self.config['s3_bucket']}/{self.config['data_prefix']}",
                    description='Input data S3 path'
                ),
                'output_data': ParameterString(
                    name='OutputData',
                    default_value=f"s3://{self.config['s3_bucket']}/{self.config['output_prefix']}",
                    description='Output data S3 path'
                )
            }
            
            logger.info(f"Created {len(parameters)} pipeline parameters")
            return parameters
            
        except Exception as e:
            logger.error(f"Error creating parameters: {e}")
            raise
    
    def create_data_processing_step(self, parameters):
        """Create data processing step"""
        logger.info("Creating data processing step...")
        
        try:
            # Create SKLearn processor
            processor = SKLearnProcessor(
                framework_version=self.config['processing_config']['framework_version'],
                py_version=self.config['processing_config']['python_version'],
                role=self.role,
                instance_type=parameters['processing_instance_type'],
                instance_count=parameters['processing_instance_count'],
                base_job_name='credit-risk-processing'
            )
            
            # Create processing step
            processing_step = ProcessingStep(
                name='DataProcessing',
                processor=processor,
                inputs=[
                    sagemaker.processing.ProcessingInput(
                        source=parameters['input_data'],
                        destination='/opt/ml/processing/input'
                    )
                ],
                outputs=[
                    sagemaker.processing.ProcessingOutput(
                        output_name='train_data',
                        source='/opt/ml/processing/output/train',
                        destination=f"{parameters['output_data']}/train"
                    ),
                    sagemaker.processing.ProcessingOutput(
                        output_name='test_data',
                        source='/opt/ml/processing/output/test',
                        destination=f"{parameters['output_data']}/test"
                    ),
                    sagemaker.processing.ProcessingOutput(
                        output_name='validation_data',
                        source='/opt/ml/processing/output/validation',
                        destination=f"{parameters['output_data']}/validation"
                    )
                ],
                code='processing_script.py',
                job_arguments=[
                    '--input-data', '/opt/ml/processing/input',
                    '--output-data', '/opt/ml/processing/output',
                    '--test-size', '0.2',
                    '--validation-size', '0.1'
                ]
            )
            
            logger.info("Data processing step created")
            return processing_step
            
        except Exception as e:
            logger.error(f"Error creating data processing step: {e}")
            raise
    
    def create_model_training_step(self, parameters, processing_step):
        """Create model training step"""
        logger.info("Creating model training step...")
        
        try:
            # Create SKLearn estimator
            estimator = SKLearn(
                entry_point='train_script.py',
                framework_version=self.config['training_config']['framework_version'],
                py_version=self.config['training_config']['python_version'],
                role=self.role,
                instance_type=parameters['training_instance_type'],
                instance_count=parameters['training_instance_count'],
                hyperparameters=self.config['training_config']['hyperparameters'],
                base_job_name='credit-risk-training'
            )
            
            # Create training step
            training_step = TrainingStep(
                name='ModelTraining',
                estimator=estimator,
                inputs={
                    'training': TrainingInput(
                        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri
                    ),
                    'validation': TrainingInput(
                        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['validation_data'].S3Output.S3Uri
                    )
                }
            )
            
            logger.info("Model training step created")
            return training_step
            
        except Exception as e:
            logger.error(f"Error creating model training step: {e}")
            raise
    
    def create_model_evaluation_step(self, parameters, processing_step, training_step):
        """Create model evaluation step"""
        logger.info("Creating model evaluation step...")
        
        try:
            # Create SKLearn processor for evaluation
            evaluator = SKLearnProcessor(
                framework_version=self.config['processing_config']['framework_version'],
                py_version=self.config['processing_config']['python_version'],
                role=self.role,
                instance_type=parameters['processing_instance_type'],
                instance_count=parameters['processing_instance_count'],
                base_job_name='credit-risk-evaluation'
            )
            
            # Create evaluation step
            evaluation_step = ProcessingStep(
                name='ModelEvaluation',
                processor=evaluator,
                inputs=[
                    sagemaker.processing.ProcessingInput(
                        source=processing_step.properties.ProcessingOutputConfig.Outputs['test_data'].S3Output.S3Uri,
                        destination='/opt/ml/processing/input/test'
                    ),
                    sagemaker.processing.ProcessingInput(
                        source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                        destination='/opt/ml/processing/input/model'
                    )
                ],
                outputs=[
                    sagemaker.processing.ProcessingOutput(
                        output_name='evaluation',
                        source='/opt/ml/processing/output',
                        destination=f"{parameters['output_data']}/evaluation"
                    )
                ],
                code='evaluation_script.py',
                job_arguments=[
                    '--test-data', '/opt/ml/processing/input/test',
                    '--model-data', '/opt/ml/processing/input/model',
                    '--output-data', '/opt/ml/processing/output'
                ]
            )
            
            logger.info("Model evaluation step created")
            return evaluation_step
            
        except Exception as e:
            logger.error(f"Error creating model evaluation step: {e}")
            raise
    
    def create_model_registration_step(self, parameters, training_step, evaluation_step):
        """Create model registration step"""
        logger.info("Creating model registration step...")
        
        try:
            # Create model package
            model_package = sagemaker.model.ModelPackage(
                role=self.role,
                model_package_group_name=self.config['model_config']['model_package_group_name'],
                approval_status=parameters['model_approval_status']
            )
            
            # Create registration step
            registration_step = sagemaker.workflow.steps.ModelStep(
                name='ModelRegistration',
                step_args=model_package.create(
                    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    content_types=['application/json'],
                    response_types=['application/json'],
                    inference_instances=['ml.m5.large'],
                    transform_instances=['ml.m5.large']
                )
            )
            
            logger.info("Model registration step created")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating model registration step: {e}")
            raise
    
    def create_condition_step(self, parameters, evaluation_step, registration_step):
        """Create condition step for model approval"""
        logger.info("Creating condition step...")
        
        try:
            # Create condition for model approval
            condition = ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
                    json_path='metrics.auc_score'
                ),
                right=0.8  # Minimum AUC threshold
            )
            
            # Create condition step
            condition_step = ConditionStep(
                name='ModelApprovalCondition',
                conditions=[condition],
                if_steps=[registration_step],
                else_steps=[]
            )
            
            logger.info("Condition step created")
            return condition_step
            
        except Exception as e:
            logger.error(f"Error creating condition step: {e}")
            raise
    
    def create_pipeline(self):
        """Create the complete SageMaker pipeline"""
        logger.info("Creating SageMaker pipeline...")
        
        try:
            # Create parameters
            parameters = self.create_parameters()
            
            # Create steps
            processing_step = self.create_data_processing_step(parameters)
            training_step = self.create_model_training_step(parameters, processing_step)
            evaluation_step = self.create_model_evaluation_step(parameters, processing_step, training_step)
            registration_step = self.create_model_registration_step(parameters, training_step, evaluation_step)
            condition_step = self.create_condition_step(parameters, evaluation_step, registration_step)
            
            # Create pipeline
            pipeline = Pipeline(
                name=self.config['pipeline_name'],
                parameters=list(parameters.values()),
                steps=[
                    processing_step,
                    training_step,
                    evaluation_step,
                    condition_step
                ],
                sagemaker_session=self.sagemaker_session
            )
            
            self.pipeline = pipeline
            
            logger.info(f"Pipeline '{self.config['pipeline_name']}' created successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise
    
    def deploy_pipeline(self):
        """Deploy the pipeline to SageMaker"""
        logger.info("Deploying pipeline...")
        
        try:
            if self.pipeline is None:
                self.create_pipeline()
            
            # Upsert pipeline
            self.pipeline.upsert(role_arn=self.role)
            
            logger.info("Pipeline deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying pipeline: {e}")
            raise
    
    def start_pipeline_execution(self, parameters=None):
        """Start pipeline execution"""
        logger.info("Starting pipeline execution...")
        
        try:
            if self.pipeline is None:
                self.create_pipeline()
            
            # Start execution
            execution = self.pipeline.start(
                parameters=parameters or {}
            )
            
            logger.info(f"Pipeline execution started: {execution.arn}")
            return execution
            
        except Exception as e:
            logger.error(f"Error starting pipeline execution: {e}")
            raise
    
    def list_pipeline_executions(self, max_results=10):
        """List recent pipeline executions"""
        logger.info("Listing pipeline executions...")
        
        try:
            if self.pipeline is None:
                self.create_pipeline()
            
            # List executions
            executions = self.pipeline.list_executions(
                max_results=max_results
            )
            
            logger.info(f"Found {len(executions['PipelineExecutionSummaries'])} executions")
            return executions
            
        except Exception as e:
            logger.error(f"Error listing pipeline executions: {e}")
            raise
    
    def get_pipeline_execution_status(self, execution_arn):
        """Get pipeline execution status"""
        logger.info(f"Getting execution status: {execution_arn}")
        
        try:
            # Get execution details
            execution = self.sagemaker_session.sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            status = execution['PipelineExecutionStatus']
            logger.info(f"Pipeline execution status: {status}")
            
            return {
                'status': status,
                'execution': execution
            }
            
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            raise
    
    def create_pipeline_script(self):
        """Create the pipeline script for deployment"""
        logger.info("Creating pipeline script...")
        
        try:
            script_content = f"""
import json
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor

def create_pipeline():
    # Pipeline configuration
    pipeline_name = "{self.config['pipeline_name']}"
    role = "{self.role}"
    s3_bucket = "{self.config['s3_bucket']}"
    
    # Create parameters
    processing_instance_type = ParameterString(
        name='ProcessingInstanceType',
        default_value='{self.config['processing_config']['instance_type']}'
    )
    
    training_instance_type = ParameterString(
        name='TrainingInstanceType',
        default_value='{self.config['training_config']['instance_type']}'
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[processing_instance_type, training_instance_type],
        steps=[]
    )
    
    return pipeline

if __name__ == "__main__":
    pipeline = create_pipeline()
    pipeline.upsert(role_arn="{self.role}")
    print(f"Pipeline '{self.config['pipeline_name']}' created successfully")
"""
            
            script_path = "/tmp/pipeline_script.py"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info(f"Pipeline script created: {script_path}")
            return script_path
            
        except Exception as e:
            logger.error(f"Error creating pipeline script: {e}")
            raise
    
    def run_pipeline(self):
        """Run the complete pipeline workflow"""
        logger.info("Running SageMaker pipeline workflow...")
        
        try:
            # Step 1: Create pipeline
            pipeline = self.create_pipeline()
            
            # Step 2: Deploy pipeline
            self.deploy_pipeline()
            
            # Step 3: Start execution
            execution = self.start_pipeline_execution()
            
            logger.info("SageMaker pipeline workflow completed successfully!")
            
            return {
                'status': 'SUCCESS',
                'pipeline_name': self.config['pipeline_name'],
                'execution_arn': execution.arn
            }
            
        except Exception as e:
            logger.error(f"SageMaker pipeline workflow failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SageMaker pipeline for credit risk ML')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--action', type=str, choices=['create', 'deploy', 'start', 'list'], 
                       default='create', help='Pipeline action to perform')
    parser.add_argument('--execution-arn', type=str, help='Pipeline execution ARN for status check')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SageMakerPipeline(config_path=args.config)
    
    # Perform action
    if args.action == 'create':
        result = pipeline.create_pipeline()
        print(f"Pipeline created: {result.name}")
    elif args.action == 'deploy':
        result = pipeline.deploy_pipeline()
        print(f"Pipeline deployed: {result}")
    elif args.action == 'start':
        result = pipeline.start_pipeline_execution()
        print(f"Pipeline execution started: {result.arn}")
    elif args.action == 'list':
        result = pipeline.list_pipeline_executions()
        print(f"Found {len(result['PipelineExecutionSummaries'])} executions")
    
    # Run complete workflow if no specific action
    if args.action == 'create':
        result = pipeline.run_pipeline()
        
        if result['status'] == 'SUCCESS':
            logger.info("SageMaker pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("SageMaker pipeline failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
