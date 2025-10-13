import json
import boto3
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
eventbridge = boto3.client('events')
sagemaker_runtime = boto3.client('sagemaker-runtime')

def scoreRisk(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Calls SageMaker endpoint for ML risk scoring.
    
    Args:
        event: EventBridge event containing validated application data
        context: Lambda context
    
    Returns:
        Dict containing risk score and application data
    """
    try:
        logger.info(f"Received risk scoring request: {event}")
        
        # Extract application data from EventBridge event
        if 'detail' in event:
            application_data = json.loads(event['detail'])
        else:
            application_data = event
        
        # Prepare features for ML model
        features = prepare_features(application_data)
        
        # Call SageMaker endpoint for risk scoring
        risk_score = call_sagemaker_endpoint(features)
        
        # Add risk scoring metadata
        application_data['risk_score'] = risk_score
        application_data['risk_timestamp'] = datetime.utcnow().isoformat()
        application_data['risk_status'] = 'scored'
        
        # Send to EventBridge for decision making
        await send_to_eventbridge('risk.scored', application_data)
        
        logger.info(f"Risk score calculated for application {application_data['application_id']}: {risk_score}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'application_id': application_data['application_id'],
                'risk_score': risk_score,
                'status': 'scored'
            })
        }
        
    except Exception as e:
        logger.error(f"Error scoring risk: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def prepare_features(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare and transform features for ML model."""
    
    # Extract numeric features
    features = {
        'person_age': float(application_data['person_age']),
        'person_income': float(application_data['person_income']),
        'person_emp_length': float(application_data['person_emp_length']),
        'loan_amnt': float(application_data['loan_amnt']),
        'loan_int_rate': float(application_data['loan_int_rate']),
        'loan_percent_income': float(application_data['loan_percent_income']),
        'cb_person_cred_hist_length': float(application_data['cb_person_cred_hist_length'])
    }
    
    # Encode categorical variables
    features.update(encode_categorical_features(application_data))
    
    # Create derived features
    features.update(create_derived_features(application_data))
    
    return features

def encode_categorical_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Encode categorical variables using one-hot encoding."""
    encoded = {}
    
    # Home ownership encoding
    home_ownership = data['person_home_ownership']
    encoded['home_ownership_RENT'] = 1 if home_ownership == 'RENT' else 0
    encoded['home_ownership_OWN'] = 1 if home_ownership == 'OWN' else 0
    encoded['home_ownership_MORTGAGE'] = 1 if home_ownership == 'MORTGAGE' else 0
    encoded['home_ownership_OTHER'] = 1 if home_ownership == 'OTHER' else 0
    
    # Loan intent encoding
    loan_intent = data['loan_intent']
    encoded['loan_intent_PERSONAL'] = 1 if loan_intent == 'PERSONAL' else 0
    encoded['loan_intent_EDUCATION'] = 1 if loan_intent == 'EDUCATION' else 0
    encoded['loan_intent_MEDICAL'] = 1 if loan_intent == 'MEDICAL' else 0
    encoded['loan_intent_VENTURE'] = 1 if loan_intent == 'VENTURE' else 0
    encoded['loan_intent_HOMEIMPROVEMENT'] = 1 if loan_intent == 'HOMEIMPROVEMENT' else 0
    encoded['loan_intent_DEBTCONSOLIDATION'] = 1 if loan_intent == 'DEBTCONSOLIDATION' else 0
    
    # Loan grade encoding (ordinal)
    grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    encoded['loan_grade_encoded'] = grade_mapping.get(data['loan_grade'], 4)
    
    # Default on file encoding
    encoded['cb_person_default_on_file_Y'] = 1 if data['cb_person_default_on_file'] == 'Y' else 0
    encoded['cb_person_default_on_file_N'] = 1 if data['cb_person_default_on_file'] == 'N' else 0
    
    return encoded

def create_derived_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create derived features for better model performance."""
    derived = {}
    
    # Age groups
    age = float(data['person_age'])
    derived['age_group_18_25'] = 1 if 18 <= age <= 25 else 0
    derived['age_group_26_35'] = 1 if 26 <= age <= 35 else 0
    derived['age_group_36_45'] = 1 if 36 <= age <= 45 else 0
    derived['age_group_46_plus'] = 1 if age >= 46 else 0
    
    # Income quartiles
    income = float(data['person_income'])
    derived['income_quartile_1'] = 1 if income < 30000 else 0
    derived['income_quartile_2'] = 1 if 30000 <= income < 50000 else 0
    derived['income_quartile_3'] = 1 if 50000 <= income < 80000 else 0
    derived['income_quartile_4'] = 1 if income >= 80000 else 0
    
    # Employment stability
    emp_length = float(data['person_emp_length'])
    derived['employment_stable'] = 1 if emp_length >= 5 else 0
    derived['employment_new'] = 1 if emp_length < 2 else 0
    
    # Debt-to-income ratio
    derived['debt_to_income_ratio'] = float(data['loan_percent_income'])
    derived['high_debt_ratio'] = 1 if float(data['loan_percent_income']) > 0.3 else 0
    
    # Income to loan ratio
    loan_amnt = float(data['loan_amnt'])
    income = float(data['person_income'])
    derived['income_to_loan_ratio'] = income / loan_amnt if loan_amnt > 0 else 0
    
    # Risk indicators
    derived['has_default_history'] = 1 if data['cb_person_default_on_file'] == 'Y' else 0
    derived['low_credit_history'] = 1 if float(data['cb_person_cred_hist_length']) < 2 else 0
    
    return derived

def call_sagemaker_endpoint(features: Dict[str, Any]) -> float:
    """Call SageMaker endpoint for risk prediction."""
    try:
        # Convert features to JSON
        payload = json.dumps(features)
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='credit-risk-endpoint',  # This will be configured in infrastructure
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Extract risk score (assuming model returns probability of default)
        if isinstance(result, list):
            risk_score = result[0]  # First prediction
        elif isinstance(result, dict) and 'predictions' in result:
            risk_score = result['predictions'][0]
        else:
            risk_score = float(result)
        
        # Ensure score is between 0 and 1
        risk_score = max(0.0, min(1.0, float(risk_score)))
        
        logger.info(f"SageMaker prediction: {risk_score}")
        return risk_score
        
    except Exception as e:
        logger.error(f"Error calling SageMaker endpoint: {str(e)}")
        # Fallback to rule-based scoring if SageMaker fails
        return fallback_risk_scoring(features)

def fallback_risk_scoring(features: Dict[str, Any]) -> float:
    """Fallback risk scoring using rule-based approach."""
    try:
        risk_score = 0.5  # Base risk score
        
        # Adjust based on key factors
        if features.get('has_default_history', 0) == 1:
            risk_score += 0.3
        
        if features.get('high_debt_ratio', 0) == 1:
            risk_score += 0.2
        
        if features.get('low_credit_history', 0) == 1:
            risk_score += 0.15
        
        if features.get('employment_new', 0) == 1:
            risk_score += 0.1
        
        if features.get('income_quartile_1', 0) == 1:
            risk_score += 0.1
        
        # Adjust based on loan grade
        grade_risk = {7: -0.2, 6: -0.1, 5: 0.0, 4: 0.1, 3: 0.2, 2: 0.3, 1: 0.4}
        grade = features.get('loan_grade_encoded', 4)
        risk_score += grade_risk.get(grade, 0.1)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, risk_score))
        
    except Exception as e:
        logger.error(f"Error in fallback scoring: {str(e)}")
        return 0.5  # Default moderate risk

async def send_to_eventbridge(event_type: str, data: Dict[str, Any]) -> None:
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
