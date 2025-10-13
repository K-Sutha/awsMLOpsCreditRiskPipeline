# Outputs for Credit Risk Platform Infrastructure

# General Information
output "account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "region" {
  description = "AWS Region"
  value       = data.aws_region.current.name
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

# Networking Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.networking.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.networking.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.networking.private_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.networking.database_subnet_ids
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = module.networking.igw_id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = module.networking.nat_gateway_ids
}

# Storage Outputs
output "s3_bucket_ids" {
  description = "IDs of the S3 buckets"
  value       = module.storage.s3_bucket_ids
}

output "s3_bucket_arns" {
  description = "ARNs of the S3 buckets"
  value       = module.storage.s3_bucket_arns
}

output "raw_data_bucket" {
  description = "Name of the raw data S3 bucket"
  value       = "${var.project_name}-${var.environment}-raw-data"
}

output "processed_data_bucket" {
  description = "Name of the processed data S3 bucket"
  value       = "${var.project_name}-${var.environment}-processed-data"
}

output "features_bucket" {
  description = "Name of the features S3 bucket"
  value       = "${var.project_name}-${var.environment}-features"
}

output "glue_scripts_bucket" {
  description = "Name of the Glue scripts S3 bucket"
  value       = "${var.project_name}-${var.environment}-glue-scripts"
}

output "model_artifacts_bucket" {
  description = "Name of the model artifacts S3 bucket"
  value       = "${var.project_name}-${var.environment}-model-artifacts"
}

# Database Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.databases.rds_endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.databases.rds_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.databases.rds_database_name
}

output "rds_username" {
  description = "RDS master username"
  value       = module.databases.rds_username
  sensitive   = true
}

output "rds_secret_arn" {
  description = "ARN of the RDS credentials secret"
  value       = aws_secretsmanager_secret.rds_credentials.arn
}

output "dynamodb_table_names" {
  description = "Names of the DynamoDB tables"
  value       = module.databases.dynamodb_table_names
}

output "dynamodb_table_arns" {
  description = "ARNs of the DynamoDB tables"
  value       = module.databases.dynamodb_table_arns
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.databases.redis_endpoint
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = module.databases.redis_port
}

# Compute Outputs
output "ecs_cluster_id" {
  description = "ID of the ECS cluster"
  value       = module.compute.ecs_cluster_id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.compute.ecs_cluster_name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = module.compute.ecs_service_name
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = module.compute.ecs_task_definition_arn
}

output "lambda_function_names" {
  description = "Names of the Lambda functions"
  value       = module.compute.lambda_function_names
}

output "lambda_function_arns" {
  description = "ARNs of the Lambda functions"
  value       = module.compute.lambda_function_arns
}

output "lambda_risk_scoring_invoke_arn" {
  description = "Invoke ARN of the risk scoring Lambda function"
  value       = module.compute.lambda_risk_scoring_invoke_arn
}

output "sagemaker_notebook_instance_name" {
  description = "Name of the SageMaker notebook instance"
  value       = module.compute.sagemaker_notebook_instance_name
}

output "sagemaker_notebook_instance_url" {
  description = "URL of the SageMaker notebook instance"
  value       = module.compute.sagemaker_notebook_instance_url
}

# Monitoring Outputs
output "cloudwatch_log_groups" {
  description = "Names of the CloudWatch log groups"
  value       = module.monitoring.cloudwatch_log_groups
}

output "cloudwatch_alarms" {
  description = "Names of the CloudWatch alarms"
  value       = module.monitoring.cloudwatch_alarms
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic"
  value       = module.monitoring.sns_topic_arn
}

output "sns_topic_name" {
  description = "Name of the SNS topic"
  value       = module.monitoring.sns_topic_name
}

# Glue Outputs
output "glue_databases" {
  description = "Names of the Glue databases"
  value = [
    aws_glue_catalog_database.credit_risk_raw.name,
    aws_glue_catalog_database.credit_risk_processed.name,
    aws_glue_catalog_database.credit_risk_features.name,
    aws_glue_catalog_database.credit_risk_quality.name
  ]
}

output "glue_service_role_arn" {
  description = "ARN of the Glue service role"
  value       = aws_iam_role.glue_service_role.arn
}

output "glue_service_role_name" {
  description = "Name of the Glue service role"
  value       = aws_iam_role.glue_service_role.name
}

# API Gateway Outputs
output "api_gateway_id" {
  description = "ID of the API Gateway"
  value       = aws_api_gateway_rest_api.credit_risk_api.id
}

output "api_gateway_arn" {
  description = "ARN of the API Gateway"
  value       = aws_api_gateway_rest_api.credit_risk_api.arn
}

output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = "https://${aws_api_gateway_rest_api.credit_risk_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${var.environment}"
}

output "risk_scoring_endpoint" {
  description = "Risk scoring API endpoint"
  value       = "${aws_api_gateway_rest_api.credit_risk_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${var.environment}/risk-scoring"
}

# EventBridge Outputs
output "eventbridge_rule_arn" {
  description = "ARN of the EventBridge rule"
  value       = aws_cloudwatch_event_rule.etl_schedule.arn
}

output "eventbridge_rule_name" {
  description = "Name of the EventBridge rule"
  value       = aws_cloudwatch_event_rule.etl_schedule.name
}

# Security Outputs
output "security_group_ids" {
  description = "IDs of the security groups"
  value       = module.networking.security_group_ids
}

output "iam_role_arns" {
  description = "ARNs of the IAM roles"
  value = [
    aws_iam_role.glue_service_role.arn,
    module.compute.ecs_task_role_arn,
    module.compute.ecs_execution_role_arn,
    module.compute.lambda_role_arn
  ]
}

# Cost and Billing
output "cost_allocation_tags" {
  description = "Cost allocation tags"
  value = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "DataEngineering"
    CostCenter  = "FinancialServices"
  }
}

# Connection Information
output "database_connection_info" {
  description = "Database connection information"
  value = {
    endpoint = module.databases.rds_endpoint
    port     = module.databases.rds_port
    database = module.databases.rds_database_name
    username = module.databases.rds_username
    secret_arn = aws_secretsmanager_secret.rds_credentials.arn
  }
  sensitive = true
}

output "redis_connection_info" {
  description = "Redis connection information"
  value = {
    endpoint = module.databases.redis_endpoint
    port     = module.databases.redis_port
  }
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    terraform_version = ">= 1.0"
    aws_provider_version = "~> 5.0"
    region = data.aws_region.current.name
    account_id = data.aws_caller_identity.current.account_id
    deployment_timestamp = timestamp()
  }
}

# Resource Summary
output "resource_summary" {
  description = "Summary of created resources"
  value = {
    vpc_id = module.networking.vpc_id
    s3_buckets = length(module.storage.s3_bucket_ids)
    rds_instances = 1
    dynamodb_tables = length(module.databases.dynamodb_table_names)
    lambda_functions = length(module.compute.lambda_function_names)
    ecs_services = 1
    sagemaker_notebooks = 1
    glue_databases = 4
    api_gateways = 1
    cloudwatch_log_groups = length(module.monitoring.cloudwatch_log_groups)
    sns_topics = 1
  }
}

# Environment-specific outputs
output "environment_config" {
  description = "Environment-specific configuration"
  value = {
    environment = var.environment
    project_name = var.project_name
    region = data.aws_region.current.name
    vpc_cidr = var.vpc_cidr
    rds_instance_class = var.rds_instance_class
    ecs_desired_count = var.ecs_desired_count
    enable_multi_az = var.enable_multi_az
    enable_deletion_protection = var.enable_deletion_protection
  }
}

# Quick Start Commands
output "quick_start_commands" {
  description = "Quick start commands for the platform"
  value = {
    connect_to_rds = "aws rds describe-db-instances --db-instance-identifier ${var.project_name}-${var.environment}-rds"
    connect_to_redis = "aws elasticache describe-cache-clusters --cache-cluster-id ${var.project_name}-${var.environment}-redis"
    list_s3_buckets = "aws s3 ls | grep ${var.project_name}-${var.environment}"
    check_ecs_services = "aws ecs list-services --cluster ${var.project_name}-${var.environment}-cluster"
    test_api = "curl -X POST ${aws_api_gateway_rest_api.credit_risk_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${var.environment}/risk-scoring"
  }
}
