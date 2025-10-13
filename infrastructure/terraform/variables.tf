# Variables for Credit Risk Platform Infrastructure

# General Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "credit-risk-platform"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# Networking Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
}

variable "database_subnets" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.100.0/24", "10.0.200.0/24", "10.0.300.0/24"]
}

# RDS Configuration
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "rds_database_name" {
  description = "RDS database name"
  type        = string
  default     = "credit_risk"
}

variable "rds_username" {
  description = "RDS master username"
  type        = string
  default     = "postgres"
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

# ECS Configuration
variable "ecs_cpu" {
  description = "ECS task CPU units"
  type        = number
  default     = 512
}

variable "ecs_memory" {
  description = "ECS task memory in MB"
  type        = number
  default     = 1024
}

variable "ecs_desired_count" {
  description = "ECS service desired count"
  type        = number
  default     = 1
}

# Lambda Configuration
variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 30
}

variable "lambda_memory_size" {
  description = "Lambda function memory size in MB"
  type        = number
  default     = 512
}

# SageMaker Configuration
variable "sagemaker_instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.medium"
}

variable "sagemaker_volume_size" {
  description = "SageMaker notebook volume size in GB"
  type        = number
  default     = 20
}

# Monitoring Configuration
variable "notification_email" {
  description = "Email address for notifications"
  type        = string
  default     = "admin@company.com"
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption for all resources"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable auto scaling for ECS service"
  type        = bool
  default     = true
}

variable "min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 10
}

# Compliance Configuration
variable "compliance_standard" {
  description = "Compliance standard (PCI-DSS, SOC2, etc.)"
  type        = string
  default     = "PCI-DSS"
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 2555  # 7 years
}

variable "audit_logging" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

# Performance Configuration
variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

# Backup Configuration
variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "backup_window" {
  description = "Backup window for RDS"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "Maintenance window for RDS"
  type        = string
  default     = "Sun:04:00-Sun:05:00"
}

# Network Security
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_vpc_endpoints" {
  description = "Enable VPC endpoints for AWS services"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

# Development Configuration
variable "enable_debug_logging" {
  description = "Enable debug logging"
  type        = bool
  default     = false
}

variable "enable_cost_allocation_tags" {
  description = "Enable cost allocation tags"
  type        = bool
  default     = true
}

variable "enable_resource_tagging" {
  description = "Enable automatic resource tagging"
  type        = bool
  default     = true
}

# Feature Flags
variable "enable_ml_pipeline" {
  description = "Enable ML pipeline components"
  type        = bool
  default     = true
}

variable "enable_realtime_inference" {
  description = "Enable real-time inference"
  type        = bool
  default     = true
}

variable "enable_batch_processing" {
  description = "Enable batch processing"
  type        = bool
  default     = true
}

variable "enable_stream_processing" {
  description = "Enable stream processing"
  type        = bool
  default     = false
}

# Data Pipeline Configuration
variable "etl_schedule" {
  description = "ETL pipeline schedule (cron expression)"
  type        = string
  default     = "cron(0 1 * * ? *)"
}

variable "data_quality_threshold" {
  description = "Data quality threshold (0.0-1.0)"
  type        = number
  default     = 0.95
  
  validation {
    condition     = var.data_quality_threshold >= 0.0 && var.data_quality_threshold <= 1.0
    error_message = "Data quality threshold must be between 0.0 and 1.0."
  }
}

variable "alert_threshold" {
  description = "Alert threshold for quality breaches (0.0-1.0)"
  type        = number
  default     = 0.90
  
  validation {
    condition     = var.alert_threshold >= 0.0 && var.alert_threshold <= 1.0
    error_message = "Alert threshold must be between 0.0 and 1.0."
  }
}

# API Configuration
variable "api_rate_limit" {
  description = "API rate limit (requests per minute)"
  type        = number
  default     = 1000
}

variable "api_burst_limit" {
  description = "API burst limit (requests per minute)"
  type        = number
  default     = 2000
}

variable "enable_api_caching" {
  description = "Enable API response caching"
  type        = bool
  default     = true
}

variable "cache_ttl" {
  description = "Cache TTL in seconds"
  type        = number
  default     = 300
}

# Disaster Recovery
variable "enable_multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = false
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Backup region for cross-region backup"
  type        = string
  default     = "us-west-2"
}

# Environment-specific overrides
variable "environment_config" {
  description = "Environment-specific configuration"
  type        = map(string)
  default     = {}
}

# Custom tags
variable "custom_tags" {
  description = "Custom tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Resource naming
variable "name_prefix_override" {
  description = "Override the default name prefix"
  type        = string
  default     = ""
}

variable "use_random_suffix" {
  description = "Add random suffix to resource names"
  type        = bool
  default     = false
}
