# Contributing

*This file will be updated once the entire project is deployed and testing framework is ready.*

## Current Status

The project is currently in development with core components built:
- Data processing and ML pipelines
- AWS infrastructure with Terraform
- Database schemas and connections

## Development Setup

```bash
git clone https://github.com/r0han01/awsMLOpsCreditRiskPipeline.git
cd awsMLOpsCreditRiskPipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
aws configure
```

## Project Structure

- `data/` - Data processing and analysis
- `database/` - Database schemas and connections  
- `etl/` - AWS Glue ETL scripts
- `infrastructure/` - Terraform IaC
- `mlPipeline/` - ML training and inference

## Guidelines

- Use camelCase for file names
- Follow PEP 8 for Python code
- Add type hints and docstrings
- Test your changes locally before submitting

## Questions?

Open an issue for questions or suggestions.
