# Outputs for Networking Module

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_arn" {
  description = "ARN of the VPC"
  value       = aws_vpc.main.arn
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "vpc_default_security_group_id" {
  description = "ID of the default security group"
  value       = aws_vpc.main.default_security_group_id
}

output "vpc_default_network_acl_id" {
  description = "ID of the default network ACL"
  value       = aws_vpc.main.default_network_acl_id
}

output "vpc_default_route_table_id" {
  description = "ID of the default route table"
  value       = aws_vpc.main.default_route_table_id
}

# Internet Gateway
output "igw_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "igw_arn" {
  description = "ARN of the Internet Gateway"
  value       = aws_internet_gateway.main.arn
}

# NAT Gateways
output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_arns" {
  description = "ARNs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].arn
}

# Elastic IPs
output "nat_eip_ids" {
  description = "IDs of the Elastic IPs"
  value       = aws_eip.nat[*].id
}

output "nat_eip_public_ips" {
  description = "Public IPs of the Elastic IPs"
  value       = aws_eip.nat[*].public_ip
}

# Public Subnets
output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "public_subnet_arns" {
  description = "ARNs of the public subnets"
  value       = aws_subnet.public[*].arn
}

output "public_subnet_cidr_blocks" {
  description = "CIDR blocks of the public subnets"
  value       = aws_subnet.public[*].cidr_block
}

# Private Subnets
output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "private_subnet_arns" {
  description = "ARNs of the private subnets"
  value       = aws_subnet.private[*].arn
}

output "private_subnet_cidr_blocks" {
  description = "CIDR blocks of the private subnets"
  value       = aws_subnet.private[*].cidr_block
}

# Database Subnets
output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database[*].id
}

output "database_subnet_arns" {
  description = "ARNs of the database subnets"
  value       = aws_subnet.database[*].arn
}

output "database_subnet_cidr_blocks" {
  description = "CIDR blocks of the database subnets"
  value       = aws_subnet.database[*].cidr_block
}

# Route Tables
output "public_route_table_id" {
  description = "ID of the public route table"
  value       = aws_route_table.public.id
}

output "private_route_table_ids" {
  description = "IDs of the private route tables"
  value       = aws_route_table.private[*].id
}

output "database_route_table_id" {
  description = "ID of the database route table"
  value       = aws_route_table.database.id
}

# Security Groups
output "security_group_ids" {
  description = "IDs of all security groups"
  value = {
    web      = aws_security_group.web.id
    app      = aws_security_group.app.id
    database = aws_security_group.database.id
    glue     = aws_security_group.glue.id
    sagemaker = aws_security_group.sagemaker.id
  }
}

output "security_group_arns" {
  description = "ARNs of all security groups"
  value = {
    web      = aws_security_group.web.arn
    app      = aws_security_group.app.arn
    database = aws_security_group.database.arn
    glue     = aws_security_group.glue.arn
    sagemaker = aws_security_group.sagemaker.arn
  }
}

# Web Security Group
output "web_security_group_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

# App Security Group
output "app_security_group_id" {
  description = "ID of the app security group"
  value       = aws_security_group.app.id
}

# Database Security Group
output "database_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}

# Glue Security Group
output "glue_security_group_id" {
  description = "ID of the Glue security group"
  value       = aws_security_group.glue.id
}

# SageMaker Security Group
output "sagemaker_security_group_id" {
  description = "ID of the SageMaker security group"
  value       = aws_security_group.sagemaker.id
}

# VPC Endpoints
output "vpc_endpoint_ids" {
  description = "IDs of the VPC endpoints"
  value = {
    s3              = var.enable_vpc_endpoints ? aws_vpc_endpoint.s3[0].id : null
    dynamodb        = var.enable_vpc_endpoints ? aws_vpc_endpoint.dynamodb[0].id : null
    secretsmanager  = var.enable_vpc_endpoints ? aws_vpc_endpoint.secretsmanager[0].id : null
  }
}

# Flow Logs
output "flow_log_id" {
  description = "ID of the VPC flow log"
  value       = var.enable_flow_logs ? aws_flow_log.main[0].id : null
}

output "flow_log_cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for flow logs"
  value       = var.enable_flow_logs ? aws_cloudwatch_log_group.flow_logs[0].name : null
}

# Availability Zones
output "availability_zones" {
  description = "List of availability zones used"
  value       = var.azs
}

# Network Summary
output "network_summary" {
  description = "Summary of network resources"
  value = {
    vpc_id = aws_vpc.main.id
    vpc_cidr = aws_vpc.main.cidr_block
    public_subnets = length(aws_subnet.public)
    private_subnets = length(aws_subnet.private)
    database_subnets = length(aws_subnet.database)
    nat_gateways = var.enable_nat_gateway ? length(aws_nat_gateway.main) : 0
    security_groups = 5
    vpc_endpoints = var.enable_vpc_endpoints ? 3 : 0
    flow_logs_enabled = var.enable_flow_logs
  }
}
