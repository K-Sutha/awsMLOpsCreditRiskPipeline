# Storage Module for Credit Risk Platform
# S3 buckets with lifecycle policies, versioning, and encryption

# S3 Buckets
resource "aws_s3_bucket" "buckets" {
  for_each = var.s3_buckets

  bucket = each.value.name

  tags = merge(var.tags, {
    Name = each.value.name
    Type = "S3Bucket"
    Purpose = each.key
  })
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "buckets" {
  for_each = var.s3_buckets

  bucket = aws_s3_bucket.buckets[each.key].id
  versioning_configuration {
    status = each.value.versioning ? "Enabled" : "Disabled"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "buckets" {
  for_each = var.s3_buckets

  bucket = aws_s3_bucket.buckets[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "buckets" {
  for_each = var.s3_buckets

  bucket = aws_s3_bucket.buckets[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Lifecycle Configuration
resource "aws_s3_bucket_lifecycle_configuration" "buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if length(v.lifecycle_rules) > 0 }

  bucket = aws_s3_bucket.buckets[each.key].id

  dynamic "rule" {
    for_each = each.value.lifecycle_rules
    content {
      id     = rule.value.id
      status = rule.value.enabled ? "Enabled" : "Disabled"

      dynamic "transition" {
        for_each = rule.value.transitions
        content {
          days          = transition.value.days
          storage_class = transition.value.storage_class
        }
      }
    }
  }

  depends_on = [aws_s3_bucket_versioning.buckets]
}

# S3 Bucket Notification Configuration for Glue Jobs
resource "aws_s3_bucket_notification" "raw_data" {
  bucket = aws_s3_bucket.buckets["raw_data"].id

  lambda_function {
    lambda_function_arn = var.lambda_function_arns["data_ingestion"]
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
    filter_suffix       = ".csv"
  }
}

# S3 Bucket Policy for Cross-Account Access (if needed)
resource "aws_s3_bucket_policy" "buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_cross_account_access }

  bucket = aws_s3_bucket.buckets[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowGlueAccess"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.account_id}:role/${var.name_prefix}-glue-service-role"
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.buckets[each.key].arn,
          "${aws_s3_bucket.buckets[each.key].arn}/*"
        ]
      }
    ]
  })
}

# S3 Bucket CORS Configuration (for API access)
resource "aws_s3_bucket_cors_configuration" "api_buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_cors }

  bucket = aws_s3_bucket.buckets[each.key].id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# S3 Bucket Logging Configuration
resource "aws_s3_bucket_logging" "buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_logging }

  bucket = aws_s3_bucket.buckets[each.key].id

  target_bucket = aws_s3_bucket.buckets["terraform_state"].id
  target_prefix = "logs/${each.key}/"
}

# S3 Bucket Website Configuration (if needed for documentation)
resource "aws_s3_bucket_website_configuration" "documentation" {
  count = var.enable_documentation_site ? 1 : 0

  bucket = aws_s3_bucket.buckets["documentation"].id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

# S3 Bucket Object Lock Configuration (for compliance)
resource "aws_s3_bucket_object_lock_configuration" "compliance_buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_object_lock }

  bucket = aws_s3_bucket.buckets[each.key].id

  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = 2555  # 7 years for compliance
    }
  }

  object_lock_enabled = "Enabled"
}

# S3 Bucket Replication Configuration (for disaster recovery)
resource "aws_s3_bucket_replication_configuration" "replication" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_replication }

  bucket = aws_s3_bucket.buckets[each.key].id
  role   = aws_iam_role.replication[each.key].arn

  rule {
    id     = "ReplicateToBackupRegion"
    status = "Enabled"

    filter {
      prefix = ""
    }

    destination {
      bucket        = aws_s3_bucket.replication_buckets[each.key].arn
      storage_class = "STANDARD_IA"
    }
  }

  depends_on = [aws_s3_bucket_versioning.buckets]
}

# IAM Role for S3 Replication
resource "aws_iam_role" "replication" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_replication }

  name = "${var.name_prefix}-s3-replication-role-${each.key}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-s3-replication-role-${each.key}"
    Type = "IAMRole"
    Purpose = "S3Replication"
  })
}

resource "aws_iam_role_policy" "replication" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_replication }

  name = "${var.name_prefix}-s3-replication-policy-${each.key}"
  role = aws_iam_role.replication[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.buckets[each.key].arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${aws_s3_bucket.replication_buckets[each.key].arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.buckets[each.key].arn,
          aws_s3_bucket.replication_buckets[each.key].arn
        ]
      }
    ]
  })
}

# Replication Buckets (in backup region)
resource "aws_s3_bucket" "replication_buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_replication }

  bucket = "${each.value.name}-backup"
  provider = aws.backup_region

  tags = merge(var.tags, {
    Name = "${each.value.name}-backup"
    Type = "S3Bucket"
    Purpose = "${each.key}-backup"
    Replication = "true"
  })
}

# S3 Bucket Inventory Configuration (for cost optimization)
resource "aws_s3_bucket_inventory" "buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_inventory }

  bucket = aws_s3_bucket.buckets[each.key].id
  name   = "${each.key}-inventory"

  included_object_versions = "All"

  schedule {
    frequency = "Weekly"
  }

  destination {
    bucket {
      bucket_arn = aws_s3_bucket.buckets["terraform_state"].arn
      prefix     = "inventory/${each.key}/"
      format     = "CSV"
    }
  }

  optional_fields = [
    "Size",
    "LastModifiedDate",
    "StorageClass",
    "ETag",
    "IsMultipartUploaded",
    "ReplicationStatus"
  ]
}

# S3 Bucket Analytics Configuration
resource "aws_s3_bucket_analytics_configuration" "buckets" {
  for_each = { for k, v in var.s3_buckets : k => v if v.enable_analytics }

  bucket = aws_s3_bucket.buckets[each.key].id
  name   = "${each.key}-analytics"

  filter {
    prefix = ""
  }

  storage_class_analysis {
    data_export {
      destination {
        s3_bucket_destination {
          bucket_arn = aws_s3_bucket.buckets["terraform_state"].arn
          prefix     = "analytics/${each.key}/"
        }
      }
    }
  }
}
