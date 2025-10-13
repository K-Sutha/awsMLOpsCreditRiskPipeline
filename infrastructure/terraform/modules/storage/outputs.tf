# Outputs for Storage Module

output "s3_bucket_ids" {
  description = "IDs of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.id }
}

output "s3_bucket_arns" {
  description = "ARNs of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.arn }
}

output "s3_bucket_domain_names" {
  description = "Domain names of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.bucket_domain_name }
}

output "s3_bucket_regional_domain_names" {
  description = "Regional domain names of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.bucket_regional_domain_name }
}

output "s3_bucket_hosted_zone_ids" {
  description = "Hosted zone IDs of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.hosted_zone_id }
}

output "s3_bucket_website_endpoints" {
  description = "Website endpoints of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.website_endpoint }
}

output "s3_bucket_website_domains" {
  description = "Website domains of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.buckets : k => v.website_domain }
}

# Specific bucket outputs
output "raw_data_bucket" {
  description = "Raw data S3 bucket"
  value       = aws_s3_bucket.buckets["raw_data"]
}

output "processed_data_bucket" {
  description = "Processed data S3 bucket"
  value       = aws_s3_bucket.buckets["processed_data"]
}

output "features_bucket" {
  description = "Features S3 bucket"
  value       = aws_s3_bucket.buckets["features"]
}

output "glue_scripts_bucket" {
  description = "Glue scripts S3 bucket"
  value       = aws_s3_bucket.buckets["glue_scripts"]
}

output "model_artifacts_bucket" {
  description = "Model artifacts S3 bucket"
  value       = aws_s3_bucket.buckets["model_artifacts"]
}

output "terraform_state_bucket" {
  description = "Terraform state S3 bucket"
  value       = aws_s3_bucket.buckets["terraform_state"]
}

# Replication outputs
output "replication_bucket_ids" {
  description = "IDs of the replication S3 buckets"
  value       = { for k, v in aws_s3_bucket.replication_buckets : k => v.id }
}

output "replication_bucket_arns" {
  description = "ARNs of the replication S3 buckets"
  value       = { for k, v in aws_s3_bucket.replication_buckets : k => v.arn }
}

# IAM role outputs
output "replication_role_arns" {
  description = "ARNs of the S3 replication IAM roles"
  value       = { for k, v in aws_iam_role.replication : k => v.arn }
}

# Storage summary
output "storage_summary" {
  description = "Summary of storage resources"
  value = {
    total_buckets = length(aws_s3_bucket.buckets)
    buckets_with_versioning = length([for k, v in var.s3_buckets : k if v.versioning])
    buckets_with_encryption = length([for k, v in var.s3_buckets : k if v.encryption])
    buckets_with_lifecycle = length([for k, v in var.s3_buckets : k if length(v.lifecycle_rules) > 0])
    buckets_with_replication = length([for k, v in var.s3_buckets : k if v.enable_replication])
    replication_buckets = length(aws_s3_bucket.replication_buckets)
  }
}
