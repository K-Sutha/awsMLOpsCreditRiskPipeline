import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize EventBridge client
eventbridge = boto3.client('events')

def validateApplication(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Validates incoming loan application data and routes to risk scoring.
    
    Args:
        event: API Gateway event containing application data
        context: Lambda context
    
    Returns:
        Dict containing validation result and application data
    """
    try:
        logger.info(f"Received application validation request: {event}")
        
        # Extract application data from API Gateway event
        if 'body' in event:
            application_data = json.loads(event['body'])
        else:
            application_data = event
        
        # Validate required fields
        validation_result = validate_required_fields(application_data)
        if not validation_result['valid']:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Validation failed',
                    'details': validation_result['errors']
                })
            }
        
        # Validate data types and ranges
        type_validation = validate_data_types(application_data)
        if not type_validation['valid']:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Data type validation failed',
                    'details': type_validation['errors']
                })
            }
        
        # Add validation metadata
        application_data['validation_timestamp'] = datetime.utcnow().isoformat()
        application_data['validation_status'] = 'validated'
        
        # Generate unique application ID
        application_data['application_id'] = generate_application_id()
        
        # Send to EventBridge for risk scoring
        send_to_eventbridge('application.validated', application_data)
        
        logger.info(f"Application {application_data['application_id']} validated successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Application validated successfully',
                'application_id': application_data['application_id'],
                'status': 'validated'
            })
        }
        
    except Exception as e:
        logger.error(f"Error validating application: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def validate_required_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that all required fields are present."""
    required_fields = [
        'person_age',
        'person_income',
        'person_home_ownership',
        'person_emp_length',
        'loan_intent',
        'loan_grade',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_default_on_file',
        'cb_person_cred_hist_length'
    ]
    
    errors = []
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_data_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data types and reasonable ranges."""
    errors = []
    
    # Age validation
    if 'person_age' in data:
        try:
            age = float(data['person_age'])
            if age < 18 or age > 100:
                errors.append("person_age must be between 18 and 100")
        except (ValueError, TypeError):
            errors.append("person_age must be a valid number")
    
    # Income validation
    if 'person_income' in data:
        try:
            income = float(data['person_income'])
            if income < 0 or income > 10000000:  # Max $10M
                errors.append("person_income must be between 0 and 10,000,000")
        except (ValueError, TypeError):
            errors.append("person_income must be a valid number")
    
    # Loan amount validation
    if 'loan_amnt' in data:
        try:
            loan_amnt = float(data['loan_amnt'])
            if loan_amnt < 1000 or loan_amnt > 500000:  # $1K to $500K
                errors.append("loan_amnt must be between 1,000 and 500,000")
        except (ValueError, TypeError):
            errors.append("loan_amnt must be a valid number")
    
    # Interest rate validation
    if 'loan_int_rate' in data:
        try:
            int_rate = float(data['loan_int_rate'])
            if int_rate < 0 or int_rate > 50:  # 0% to 50%
                errors.append("loan_int_rate must be between 0 and 50")
        except (ValueError, TypeError):
            errors.append("loan_int_rate must be a valid number")
    
    # Employment length validation
    if 'person_emp_length' in data:
        try:
            emp_length = float(data['person_emp_length'])
            if emp_length < 0 or emp_length > 50:  # 0 to 50 years
                errors.append("person_emp_length must be between 0 and 50")
        except (ValueError, TypeError):
            errors.append("person_emp_length must be a valid number")
    
    # Home ownership validation
    if 'person_home_ownership' in data:
        valid_ownership = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if data['person_home_ownership'] not in valid_ownership:
            errors.append(f"person_home_ownership must be one of: {valid_ownership}")
    
    # Loan intent validation
    if 'loan_intent' in data:
        valid_intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        if data['loan_intent'] not in valid_intents:
            errors.append(f"loan_intent must be one of: {valid_intents}")
    
    # Loan grade validation
    if 'loan_grade' in data:
        valid_grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        if data['loan_grade'] not in valid_grades:
            errors.append(f"loan_grade must be one of: {valid_grades}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def generate_application_id() -> str:
    """Generate a unique application ID."""
    import uuid
    return f"APP-{uuid.uuid4().hex[:8].upper()}"

def send_to_eventbridge(event_type: str, data: Dict[str, Any]) -> None:
    """Send event to EventBridge for processing."""
    try:
        response = eventbridge.put_events(
            Entries=[
                {
                    'Source': 'credit-risk-platform',
                    'DetailType': event_type,
                    'Detail': json.dumps(data),
                    'EventBusName': 'default'
                }
            ]
        )
        logger.info(f"Event sent to EventBridge: {event_type}")
    except Exception as e:
        logger.error(f"Failed to send event to EventBridge: {str(e)}")
        raise
