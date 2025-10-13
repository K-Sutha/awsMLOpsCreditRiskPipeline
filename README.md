# Credit Risk MLOps Pipeline

**Progress: Phase 3 Complete**

A complete MLOps platform for credit risk assessment built with AWS services. This project demonstrates end-to-end machine learning engineering from data processing to production deployment.

## What's Built So Far

**Data Layer**: Converted ARFF datasets to CSV, built comprehensive data quality analysis, and implemented production-level data cleaning with feature engineering.

**Infrastructure**: Created complete AWS infrastructure using Terraform - VPC networking, S3 data lake, RDS PostgreSQL, DynamoDB for caching, and all necessary security configurations.

**ML Pipeline**: Built multiple ML models (Random Forest, XGBoost, Neural Networks) with SageMaker integration, real-time inference capabilities, and model monitoring.

## Tech Stack
- AWS SageMaker, XGBoost, TensorFlow, Scikit-learn
- Terraform for infrastructure, PostgreSQL + DynamoDB for data storage
- AWS Glue for ETL, Lambda for serverless functions
- CloudWatch for monitoring

## Next Steps
- Event-driven microservices with Lambda functions
- FastAPI application with real-time updates  
- Complete CI/CD pipeline

## Quick Start
```bash
cd infrastructure && ./scripts/deployInfrastructure.sh deploy
cd mlPipeline && python training/trainRandomForest.py
```
