#!/bin/bash
# Infrastructure Deployment Script for Credit Risk Platform
# Production-grade deployment automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="credit-risk-platform"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    # Check Terraform version
    TERRAFORM_VERSION=$(terraform version -json | jq -r '.terraform_version')
    REQUIRED_VERSION="1.0.0"
    
    if ! terraform version -json | jq -e '.terraform_version | startswith("1.")' &> /dev/null; then
        log_error "Terraform version $TERRAFORM_VERSION is not supported. Please upgrade to version 1.0 or higher."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

setup_backend() {
    log_info "Setting up Terraform backend..."
    
    # Create S3 bucket for Terraform state (if it doesn't exist)
    STATE_BUCKET="${PROJECT_NAME}-${ENVIRONMENT}-terraform-state"
    
    if ! aws s3 ls "s3://${STATE_BUCKET}" 2>/dev/null; then
        log_info "Creating S3 bucket for Terraform state: ${STATE_BUCKET}"
        aws s3 mb "s3://${STATE_BUCKET}" --region "${AWS_REGION}"
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "${STATE_BUCKET}" \
            --versioning-configuration Status=Enabled
        
        # Enable encryption
        aws s3api put-bucket-encryption \
            --bucket "${STATE_BUCKET}" \
            --server-side-encryption-configuration '{
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            }'
        
        log_success "S3 bucket created: ${STATE_BUCKET}"
    else
        log_info "S3 bucket already exists: ${STATE_BUCKET}"
    fi
    
    # Create DynamoDB table for state locking (if it doesn't exist)
    LOCK_TABLE="${PROJECT_NAME}-${ENVIRONMENT}-terraform-lock"
    
    if ! aws dynamodb describe-table --table-name "${LOCK_TABLE}" --region "${AWS_REGION}" 2>/dev/null; then
        log_info "Creating DynamoDB table for Terraform state locking: ${LOCK_TABLE}"
        aws dynamodb create-table \
            --table-name "${LOCK_TABLE}" \
            --attribute-definitions AttributeName=LockID,AttributeType=S \
            --key-schema AttributeName=LockID,KeyType=HASH \
            --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
            --region "${AWS_REGION}"
        
        # Wait for table to be active
        aws dynamodb wait table-exists --table-name "${LOCK_TABLE}" --region "${AWS_REGION}"
        
        log_success "DynamoDB table created: ${LOCK_TABLE}"
    else
        log_info "DynamoDB table already exists: ${LOCK_TABLE}"
    fi
}

terraform_init() {
    log_info "Initializing Terraform..."
    
    cd "${TERRAFORM_DIR}"
    
    # Initialize Terraform
    terraform init \
        -backend-config="bucket=${PROJECT_NAME}-${ENVIRONMENT}-terraform-state" \
        -backend-config="key=infrastructure/terraform.tfstate" \
        -backend-config="region=${AWS_REGION}" \
        -backend-config="dynamodb_table=${PROJECT_NAME}-${ENVIRONMENT}-terraform-lock" \
        -backend-config="encrypt=true"
    
    log_success "Terraform initialized"
}

terraform_plan() {
    log_info "Planning Terraform deployment..."
    
    cd "${TERRAFORM_DIR}"
    
    # Create plan file
    terraform plan \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -var="project_name=${PROJECT_NAME}" \
        -out="terraform-${ENVIRONMENT}.plan" \
        -detailed-exitcode
    
    PLAN_EXIT_CODE=$?
    
    if [ $PLAN_EXIT_CODE -eq 0 ]; then
        log_success "No changes needed"
        return 0
    elif [ $PLAN_EXIT_CODE -eq 2 ]; then
        log_info "Changes detected. Plan saved to terraform-${ENVIRONMENT}.plan"
        return 2
    else
        log_error "Terraform plan failed"
        exit 1
    fi
}

terraform_apply() {
    log_info "Applying Terraform deployment..."
    
    cd "${TERRAFORM_DIR}"
    
    if [ -f "terraform-${ENVIRONMENT}.plan" ]; then
        terraform apply "terraform-${ENVIRONMENT}.plan"
    else
        terraform apply \
            -var="environment=${ENVIRONMENT}" \
            -var="aws_region=${AWS_REGION}" \
            -var="project_name=${PROJECT_NAME}" \
            -auto-approve
    fi
    
    log_success "Terraform deployment completed"
}

terraform_destroy() {
    log_warning "This will destroy all infrastructure resources!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Destruction cancelled"
        exit 0
    fi
    
    log_info "Destroying Terraform infrastructure..."
    
    cd "${TERRAFORM_DIR}"
    
    terraform destroy \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -var="project_name=${PROJECT_NAME}" \
        -auto-approve
    
    log_success "Terraform destruction completed"
}

show_outputs() {
    log_info "Showing Terraform outputs..."
    
    cd "${TERRAFORM_DIR}"
    
    terraform output -json | jq -r '
        "=== INFRASTRUCTURE OUTPUTS ===",
        "VPC ID: " + .vpc_id.value,
        "Public Subnets: " + (.public_subnet_ids.value | join(", ")),
        "Private Subnets: " + (.private_subnet_ids.value | join(", ")),
        "Database Subnets: " + (.database_subnet_ids.value | join(", ")),
        "",
        "=== STORAGE ===",
        "Raw Data Bucket: " + .raw_data_bucket.value,
        "Processed Data Bucket: " + .processed_data_bucket.value,
        "Features Bucket: " + .features_bucket.value,
        "Glue Scripts Bucket: " + .glue_scripts_bucket.value,
        "Model Artifacts Bucket: " + .model_artifacts_bucket.value,
        "",
        "=== DATABASES ===",
        "RDS Endpoint: " + .rds_endpoint.value,
        "RDS Port: " + (.rds_port.value | tostring),
        "Redis Endpoint: " + .redis_endpoint.value,
        "",
        "=== COMPUTE ===",
        "ECS Cluster: " + .ecs_cluster_name.value,
        "ECS Service: " + .ecs_service_name.value,
        "Lambda Functions: " + (.lambda_function_names.value | join(", ")),
        "SageMaker Notebook: " + .sagemaker_notebook_instance_name.value,
        "",
        "=== MONITORING ===",
        "SNS Topic: " + .sns_topic_name.value,
        "CloudWatch Log Groups: " + (.cloudwatch_log_groups.value | join(", ")),
        "",
        "=== API ===",
        "API Gateway URL: " + .api_gateway_url.value,
        "Risk Scoring Endpoint: " + .risk_scoring_endpoint.value,
        "",
        "=== QUICK START COMMANDS ===",
        "Connect to RDS: aws rds describe-db-instances --db-instance-identifier " + .environment_config.value.project_name + "-" + .environment_config.value.environment + "-rds",
        "List S3 Buckets: aws s3 ls | grep " + .environment_config.value.project_name + "-" + .environment_config.value.environment,
        "Check ECS Services: aws ecs list-services --cluster " + .environment_config.value.project_name + "-" + .environment_config.value.environment + "-cluster"
    '
}

validate_infrastructure() {
    log_info "Validating infrastructure deployment..."
    
    cd "${TERRAFORM_DIR}"
    
    # Get outputs
    VPC_ID=$(terraform output -raw vpc_id)
    RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
    REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
    
    # Validate VPC
    if aws ec2 describe-vpcs --vpc-ids "${VPC_ID}" --region "${AWS_REGION}" &> /dev/null; then
        log_success "VPC validation passed: ${VPC_ID}"
    else
        log_error "VPC validation failed: ${VPC_ID}"
        return 1
    fi
    
    # Validate RDS
    if aws rds describe-db-instances --db-instance-identifier "${PROJECT_NAME}-${ENVIRONMENT}-rds" --region "${AWS_REGION}" &> /dev/null; then
        log_success "RDS validation passed: ${RDS_ENDPOINT}"
    else
        log_error "RDS validation failed: ${RDS_ENDPOINT}"
        return 1
    fi
    
    # Validate ElastiCache
    if aws elasticache describe-cache-clusters --cache-cluster-id "${PROJECT_NAME}-${ENVIRONMENT}-redis" --region "${AWS_REGION}" &> /dev/null; then
        log_success "ElastiCache validation passed: ${REDIS_ENDPOINT}"
    else
        log_error "ElastiCache validation failed: ${REDIS_ENDPOINT}"
        return 1
    fi
    
    log_success "Infrastructure validation completed"
}

show_help() {
    echo "Credit Risk Platform Infrastructure Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy the infrastructure (default)"
    echo "  plan       Plan the infrastructure changes"
    echo "  destroy    Destroy the infrastructure"
    echo "  validate   Validate the deployed infrastructure"
    echo "  outputs    Show infrastructure outputs"
    echo "  help       Show this help message"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Environment (dev, staging, prod) [default: dev]"
    echo "  -r, --region         AWS region [default: us-east-1]"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT          Environment name"
    echo "  AWS_REGION           AWS region"
    echo "  AWS_PROFILE          AWS profile to use"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy dev environment"
    echo "  $0 deploy -e prod -r us-west-2  # Deploy prod environment in us-west-2"
    echo "  $0 plan -e staging           # Plan staging environment"
    echo "  $0 destroy -e dev            # Destroy dev environment"
}

# Main script logic
main() {
    local command="deploy"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy|plan|destroy|validate|outputs|help)
                command="$1"
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set AWS profile if specified
    if [ -n "${AWS_PROFILE:-}" ]; then
        export AWS_PROFILE
    fi
    
    log_info "Starting infrastructure deployment for environment: ${ENVIRONMENT}"
    log_info "AWS Region: ${AWS_REGION}"
    log_info "Project Name: ${PROJECT_NAME}"
    
    case $command in
        deploy)
            check_prerequisites
            setup_backend
            terraform_init
            terraform_plan
            if [ $? -eq 2 ]; then
                terraform_apply
                validate_infrastructure
                show_outputs
            fi
            ;;
        plan)
            check_prerequisites
            setup_backend
            terraform_init
            terraform_plan
            ;;
        destroy)
            check_prerequisites
            terraform_init
            terraform_destroy
            ;;
        validate)
            check_prerequisites
            terraform_init
            validate_infrastructure
            ;;
        outputs)
            terraform_init
            show_outputs
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Operation completed successfully"
}

# Run main function
main "$@"
