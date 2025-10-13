# Main Terraform Configuration for Credit Risk Platform
# Production-grade infrastructure for financial services

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  # Backend configuration for state management
  backend "s3" {
    bucket         = "credit-risk-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "CreditRiskPlatform"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "DataEngineering"
      CostCenter  = "FinancialServices"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# Local values for common configurations
locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  azs        = slice(data.aws_availability_zones.available.names, 0, 3)
  
  # Naming convention
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Common tags
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "DataEngineering"
    CostCenter  = "FinancialServices"
    Compliance  = "PCI-DSS"
  }
}

# Random password for RDS
resource "random_password" "rds_password" {
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true
}

# Store RDS password in Secrets Manager
resource "aws_secretsmanager_secret" "rds_credentials" {
  name                    = "${local.name_prefix}-rds-credentials"
  description             = "RDS PostgreSQL credentials for credit risk platform"
  recovery_window_in_days = 7

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "rds_credentials" {
  secret_id = aws_secretsmanager_secret.rds_credentials.id
  secret_string = jsonencode({
    username = var.rds_username
    password = random_password.rds_password.result
    engine   = "postgres"
    host     = module.rds.db_instance_endpoint
    port     = 5432
    dbname   = var.rds_database_name
  })
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  name_prefix    = local.name_prefix
  vpc_cidr       = var.vpc_cidr
  azs            = local.azs
  public_subnets = var.public_subnets
  private_subnets = var.private_subnets
  database_subnets = var.database_subnets

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = local.common_tags
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  name_prefix = local.name_prefix
  account_id  = local.account_id
  region      = local.region

  # S3 Buckets
  s3_buckets = {
    raw_data = {
      name = "${local.name_prefix}-raw-data"
      versioning = true
      encryption = true
      lifecycle_rules = [
        {
          id = "raw-data-lifecycle"
          enabled = true
          transitions = [
            {
              days = 30
              storage_class = "STANDARD_IA"
            },
            {
              days = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    processed_data = {
      name = "${local.name_prefix}-processed-data"
      versioning = true
      encryption = true
      lifecycle_rules = [
        {
          id = "processed-data-lifecycle"
          enabled = true
          transitions = [
            {
              days = 30
              storage_class = "STANDARD_IA"
            },
            {
              days = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    features = {
      name = "${local.name_prefix}-features"
      versioning = true
      encryption = true
      lifecycle_rules = [
        {
          id = "features-lifecycle"
          enabled = true
          transitions = [
            {
              days = 30
              storage_class = "STANDARD_IA"
            },
            {
              days = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    glue_scripts = {
      name = "${local.name_prefix}-glue-scripts"
      versioning = true
      encryption = true
    }
    glue_temp = {
      name = "${local.name_prefix}-glue-temp"
      versioning = false
      encryption = true
    }
    model_artifacts = {
      name = "${local.name_prefix}-model-artifacts"
      versioning = true
      encryption = true
    }
    terraform_state = {
      name = "${local.name_prefix}-terraform-state"
      versioning = true
      encryption = true
    }
  }

  tags = local.common_tags
}

# Databases Module
module "databases" {
  source = "./modules/databases"

  name_prefix = local.name_prefix
  vpc_id      = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  database_subnet_ids = module.networking.database_subnet_ids

  # RDS Configuration
  rds_config = {
    instance_class      = var.rds_instance_class
    allocated_storage   = var.rds_allocated_storage
    max_allocated_storage = var.rds_max_allocated_storage
    engine              = "postgres"
    engine_version      = var.rds_engine_version
    database_name       = var.rds_database_name
    username            = var.rds_username
    password            = random_password.rds_password.result
    backup_retention_period = 7
    backup_window       = "03:00-04:00"
    maintenance_window  = "Sun:04:00-Sun:05:00"
    multi_az           = var.environment == "prod" ? true : false
    deletion_protection = var.environment == "prod" ? true : false
    skip_final_snapshot = var.environment == "prod" ? false : true
  }

  # DynamoDB Tables
  dynamodb_tables = {
    active_applications = {
      name = "${local.name_prefix}-active-applications"
      hash_key = "applicationId"
      range_key = null
      ttl_attribute = "ttl"
      ttl_enabled = true
      billing_mode = "PAY_PER_REQUEST"
    }
    risk_cache = {
      name = "${local.name_prefix}-risk-cache"
      hash_key = "cacheKey"
      range_key = null
      ttl_attribute = "ttl"
      ttl_enabled = true
      billing_mode = "PAY_PER_REQUEST"
    }
    user_sessions = {
      name = "${local.name_prefix}-user-sessions"
      hash_key = "sessionId"
      range_key = null
      ttl_attribute = "ttl"
      ttl_enabled = true
      billing_mode = "PAY_PER_REQUEST"
    }
    api_keys = {
      name = "${local.name_prefix}-api-keys"
      hash_key = "apiKey"
      range_key = null
      ttl_attribute = null
      ttl_enabled = false
      billing_mode = "PAY_PER_REQUEST"
    }
    portfolio_metrics = {
      name = "${local.name_prefix}-portfolio-metrics"
      hash_key = "metricId"
      range_key = "timestamp"
      ttl_attribute = "ttl"
      ttl_enabled = true
      billing_mode = "PAY_PER_REQUEST"
    }
  }

  # ElastiCache Redis
  redis_config = {
    node_type = var.redis_node_type
    num_cache_nodes = 1
    engine_version = "7.0"
    port = 6379
    parameter_group_name = "default.redis7"
  }

  tags = local.common_tags
}

# Compute Module
module "compute" {
  source = "./modules/compute"

  name_prefix = local.name_prefix
  vpc_id      = module.networking.vpc_id
  public_subnet_ids = module.networking.public_subnet_ids
  private_subnet_ids = module.networking.private_subnet_ids

  # ECS Configuration
  ecs_config = {
    cluster_name = "${local.name_prefix}-cluster"
    service_name = "${local.name_prefix}-api-service"
    task_definition_name = "${local.name_prefix}-api-task"
    cpu = 512
    memory = 1024
    desired_count = var.environment == "prod" ? 3 : 1
    container_port = 8000
    container_image = "${local.account_id}.dkr.ecr.${local.region}.amazonaws.com/${local.name_prefix}-api:latest"
  }

  # Lambda Functions
  lambda_functions = {
    risk_scoring = {
      name = "${local.name_prefix}-risk-scoring"
      handler = "lambda_function.lambda_handler"
      runtime = "python3.9"
      timeout = 30
      memory_size = 512
      environment_variables = {
        RDS_SECRET_ARN = aws_secretsmanager_secret.rds_credentials.arn
        DYNAMODB_TABLE = "${local.name_prefix}-risk-cache"
      }
    }
    data_validation = {
      name = "${local.name_prefix}-data-validation"
      handler = "lambda_function.lambda_handler"
      runtime = "python3.9"
      timeout = 60
      memory_size = 1024
      environment_variables = {
        S3_BUCKET = "${local.name_prefix}-processed-data"
        QUALITY_THRESHOLD = "0.95"
      }
    }
    notification_handler = {
      name = "${local.name_prefix}-notification-handler"
      handler = "lambda_function.lambda_handler"
      runtime = "python3.9"
      timeout = 15
      memory_size = 256
      environment_variables = {
        SNS_TOPIC_ARN = module.monitoring.sns_topic_arn
      }
    }
  }

  # SageMaker Configuration
  sagemaker_config = {
    notebook_instance_name = "${local.name_prefix}-notebook"
    instance_type = "ml.t3.medium"
    volume_size = 20
    lifecycle_config_name = "${local.name_prefix}-lifecycle-config"
  }

  tags = local.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"

  name_prefix = local.name_prefix
  account_id  = local.account_id
  region      = local.region

  # CloudWatch Configuration
  cloudwatch_config = {
    log_groups = [
      {
        name = "/aws/glue/credit-risk-etl"
        retention_in_days = 30
      },
      {
        name = "/aws/ecs/credit-risk-api"
        retention_in_days = 30
      },
      {
        name = "/aws/lambda/credit-risk-risk-scoring"
        retention_in_days = 14
      },
      {
        name = "/aws/lambda/credit-risk-data-validation"
        retention_in_days = 14
      },
      {
        name = "/aws/lambda/credit-risk-notification-handler"
        retention_in_days = 14
      }
    ]
    
    alarms = [
      {
        name = "high-cpu-utilization"
        metric_name = "CPUUtilization"
        namespace = "AWS/ECS"
        threshold = 80
        comparison_operator = "GreaterThanThreshold"
        evaluation_periods = 2
        period = 300
        statistic = "Average"
      },
      {
        name = "high-memory-utilization"
        metric_name = "MemoryUtilization"
        namespace = "AWS/ECS"
        threshold = 85
        comparison_operator = "GreaterThanThreshold"
        evaluation_periods = 2
        period = 300
        statistic = "Average"
      },
      {
        name = "lambda-errors"
        metric_name = "Errors"
        namespace = "AWS/Lambda"
        threshold = 5
        comparison_operator = "GreaterThanThreshold"
        evaluation_periods = 1
        period = 300
        statistic = "Sum"
      }
    ]
  }

  # SNS Configuration
  sns_config = {
    topics = [
      {
        name = "${local.name_prefix}-notifications"
        display_name = "Credit Risk Platform Notifications"
        subscribers = [
          {
            protocol = "email"
            endpoint = var.notification_email
          }
        ]
      }
    ]
  }

  tags = local.common_tags
}

# Glue Resources
resource "aws_glue_catalog_database" "credit_risk_raw" {
  name = "credit_risk_raw"
  description = "Database for raw credit risk data"
  
  tags = local.common_tags
}

resource "aws_glue_catalog_database" "credit_risk_processed" {
  name = "credit_risk_processed"
  description = "Database for processed credit risk data"
  
  tags = local.common_tags
}

resource "aws_glue_catalog_database" "credit_risk_features" {
  name = "credit_risk_features"
  description = "Database for ML features"
  
  tags = local.common_tags
}

resource "aws_glue_catalog_database" "credit_risk_quality" {
  name = "credit_risk_quality"
  description = "Database for quality reports"
  
  tags = local.common_tags
}

# Glue Service Role
resource "aws_iam_role" "glue_service_role" {
  name = "${local.name_prefix}-glue-service-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_service_role" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
  role       = aws_iam_role.glue_service_role.name
}

resource "aws_iam_role_policy_attachment" "glue_s3_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.glue_service_role.name
}

resource "aws_iam_role_policy_attachment" "glue_cloudwatch_logs" {
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  role       = aws_iam_role.glue_service_role.name
}

# EventBridge Rules for ETL Orchestration
resource "aws_cloudwatch_event_rule" "etl_schedule" {
  name                = "${local.name_prefix}-etl-schedule"
  description         = "Daily ETL trigger"
  schedule_expression = "cron(0 1 * * ? *)"
  
  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "etl_target" {
  rule      = aws_cloudwatch_event_rule.etl_schedule.name
  target_id = "ETLTarget"
  arn       = "arn:aws:glue:${local.region}:${local.account_id}:job/data-ingestion-job"
  role_arn  = aws_iam_role.glue_service_role.arn
}

# API Gateway
resource "aws_api_gateway_rest_api" "credit_risk_api" {
  name        = "${local.name_prefix}-api"
  description = "Credit Risk Platform API"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  tags = local.common_tags
}

resource "aws_api_gateway_deployment" "credit_risk_api" {
  depends_on = [
    aws_api_gateway_method.risk_scoring,
    aws_api_gateway_integration.risk_scoring
  ]
  
  rest_api_id = aws_api_gateway_rest_api.credit_risk_api.id
  stage_name  = var.environment
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_resource" "risk_scoring" {
  rest_api_id = aws_api_gateway_rest_api.credit_risk_api.id
  parent_id   = aws_api_gateway_rest_api.credit_risk_api.root_resource_id
  path_part   = "risk-scoring"
}

resource "aws_api_gateway_method" "risk_scoring" {
  rest_api_id   = aws_api_gateway_rest_api.credit_risk_api.id
  resource_id   = aws_api_gateway_resource.risk_scoring.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "risk_scoring" {
  rest_api_id = aws_api_gateway_rest_api.credit_risk_api.id
  resource_id = aws_api_gateway_resource.risk_scoring.id
  http_method = aws_api_gateway_method.risk_scoring.http_method
  
  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = module.compute.lambda_risk_scoring_invoke_arn
}
