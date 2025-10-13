# Security

## Reporting Issues

Found a security vulnerability? Please email details to: **hello@rkatkam.com**

Don't open public GitHub issues for security problems.

## Security Measures

This project handles financial data with several security layers:

- Data encryption at rest and in transit
- IAM roles and policies for AWS access
- VPC with private subnets and security groups  
- AWS Secrets Manager for credentials
- Comprehensive audit logging

## Best Practices

- Never commit AWS credentials or sensitive data
- Use environment variables for configuration
- Keep dependencies updated
- Follow least privilege principle

Thanks for helping keep this secure!
