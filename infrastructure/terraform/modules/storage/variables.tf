# Variables for Storage Module

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "account_id" {
  description = "AWS Account ID"
  type        = string
}

variable "region" {
  description = "AWS Region"
  type        = string
}

variable "s3_buckets" {
  description = "Map of S3 buckets to create"
  type = map(object({
    name                    = string
    versioning              = bool
    encryption              = bool
    enable_cross_account_access = optional(bool, false)
    enable_cors             = optional(bool, false)
    enable_logging          = optional(bool, false)
    enable_object_lock      = optional(bool, false)
    enable_replication      = optional(bool, false)
    enable_inventory        = optional(bool, false)
    enable_analytics        = optional(bool, false)
    lifecycle_rules = list(object({
      id                    = string
      enabled               = bool
      transitions = list(object({
        days                = number
        storage_class       = string
      }))
    }))
  }))
}

variable "enable_documentation_site" {
  description = "Enable S3 static website for documentation"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Backup region for replication"
  type        = string
  default     = "us-west-2"
}

variable "lambda_function_arns" {
  description = "ARNs of Lambda functions for S3 notifications"
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
