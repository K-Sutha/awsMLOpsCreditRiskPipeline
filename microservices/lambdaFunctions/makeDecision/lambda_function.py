import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize EventBridge client
eventbridge = boto3.client('events')

def makeDecision(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Makes loan approval decisions based on risk scores and business rules.
    
    Args:
        event: EventBridge event containing application data with risk score
        context: Lambda context
    
    Returns:
        Dict containing decision and application data
    """
    try:
        logger.info(f"Received decision making request: {event}")
        
        # Extract application data from EventBridge event
        if 'detail' in event:
            application_data = json.loads(event['detail'])
        else:
            application_data = event
        
        # Get risk score
        risk_score = application_data.get('risk_score', 0.5)
        
        # Make decision based on business rules
        decision_result = determine_loan_decision(application_data, risk_score)
        
        # Add decision metadata
        application_data['decision'] = decision_result['decision']
        application_data['decision_reason'] = decision_result['reason']
        application_data['decision_timestamp'] = datetime.utcnow().isoformat()
        application_data['decision_status'] = 'decided'
        
        # Add additional metadata
        application_data['approval_probability'] = decision_result['approval_probability']
        application_data['risk_level'] = decision_result['risk_level']
        
        # Send to EventBridge for processing
        send_to_eventbridge('decision.made', application_data)
        
        logger.info(f"Decision made for application {application_data['application_id']}: {decision_result['decision']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'application_id': application_data['application_id'],
                'decision': decision_result['decision'],
                'risk_score': risk_score,
                'reason': decision_result['reason'],
                'status': 'decided'
            })
        }
        
    except Exception as e:
        logger.error(f"Error making decision: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def determine_loan_decision(application_data: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
    """
    Determine loan decision based on risk score and business rules.
    
    Business Rules:
    - Risk < 0.3: AUTO_APPROVE
    - Risk 0.3-0.7: MANUAL_REVIEW
    - Risk > 0.7: AUTO_REJECT
    
    Additional factors:
    - Income level
    - Employment stability
    - Credit history
    - Loan amount vs income
    """
    
    # Base decision on risk score
    if risk_score < 0.3:
        base_decision = 'AUTO_APPROVE'
        approval_probability = 0.95
        risk_level = 'LOW'
        reason = f"Low risk score ({risk_score:.3f}) indicates high creditworthiness"
        
    elif risk_score <= 0.7:
        base_decision = 'MANUAL_REVIEW'
        approval_probability = 0.6
        risk_level = 'MEDIUM'
        reason = f"Medium risk score ({risk_score:.3f}) requires manual underwriting review"
        
    else:
        base_decision = 'AUTO_REJECT'
        approval_probability = 0.05
        risk_level = 'HIGH'
        reason = f"High risk score ({risk_score:.3f}) indicates significant default risk"
    
    # Apply additional business rules that might override the base decision
    final_decision = apply_additional_rules(application_data, base_decision, risk_score)
    
    # Adjust approval probability based on final decision
    if final_decision == 'AUTO_APPROVE':
        approval_probability = 0.95
    elif final_decision == 'MANUAL_REVIEW':
        approval_probability = 0.6
    else:  # AUTO_REJECT
        approval_probability = 0.05
    
    return {
        'decision': final_decision,
        'reason': reason,
        'approval_probability': approval_probability,
        'risk_level': risk_level,
        'risk_score': risk_score
    }

def apply_additional_rules(application_data: Dict[str, Any], base_decision: str, risk_score: float) -> str:
    """Apply additional business rules that might override the base decision."""
    
    # Rule 1: Very high income applicants get more lenient treatment
    income = float(application_data.get('person_income', 0))
    if income > 150000:  # High income threshold
        if base_decision == 'AUTO_REJECT' and risk_score < 0.8:
            return 'MANUAL_REVIEW'
        elif base_decision == 'MANUAL_REVIEW' and risk_score < 0.5:
            return 'AUTO_APPROVE'
    
    # Rule 2: Very low loan amounts get more lenient treatment
    loan_amount = float(application_data.get('loan_amnt', 0))
    if loan_amount < 5000:  # Small loan threshold
        if base_decision == 'AUTO_REJECT' and risk_score < 0.8:
            return 'MANUAL_REVIEW'
    
    # Rule 3: Very high loan amounts get stricter treatment
    if loan_amount > 100000:  # Large loan threshold
        if base_decision == 'AUTO_APPROVE' and risk_score > 0.25:
            return 'MANUAL_REVIEW'
        elif base_decision == 'MANUAL_REVIEW' and risk_score > 0.6:
            return 'AUTO_REJECT'
    
    # Rule 4: Employment stability bonus
    emp_length = float(application_data.get('person_emp_length', 0))
    if emp_length > 10:  # Very stable employment
        if base_decision == 'MANUAL_REVIEW' and risk_score < 0.4:
            return 'AUTO_APPROVE'
    
    # Rule 5: Recent default history is a hard stop
    if application_data.get('cb_person_default_on_file') == 'Y':
        if base_decision == 'AUTO_APPROVE':
            return 'MANUAL_REVIEW'
        elif risk_score > 0.6:
            return 'AUTO_REJECT'
    
    # Rule 6: Very short credit history gets stricter treatment
    credit_history = float(application_data.get('cb_person_cred_hist_length', 0))
    if credit_history < 1:  # Less than 1 year credit history
        if base_decision == 'AUTO_APPROVE':
            return 'MANUAL_REVIEW'
    
    # Rule 7: High debt-to-income ratio gets stricter treatment
    debt_to_income = float(application_data.get('loan_percent_income', 0))
    if debt_to_income > 0.4:  # More than 40% debt-to-income
        if base_decision == 'AUTO_APPROVE':
            return 'MANUAL_REVIEW'
        elif base_decision == 'MANUAL_REVIEW' and risk_score > 0.5:
            return 'AUTO_REJECT'
    
    return base_decision

def calculate_risk_level(risk_score: float) -> str:
    """Calculate risk level based on score."""
    if risk_score < 0.3:
        return 'LOW'
    elif risk_score <= 0.7:
        return 'MEDIUM'
    else:
        return 'HIGH'

def get_decision_summary(decision: str, risk_score: float, application_data: Dict[str, Any]) -> str:
    """Generate a human-readable decision summary."""
    
    applicant_name = f"Applicant {application_data.get('application_id', 'Unknown')}"
    loan_amount = float(application_data.get('loan_amnt', 0))
    income = float(application_data.get('person_income', 0))
    
    if decision == 'AUTO_APPROVE':
        return f"{applicant_name} - APPROVED: Low risk ({risk_score:.3f}), Income: ${income:,.0f}, Loan: ${loan_amount:,.0f}"
    
    elif decision == 'MANUAL_REVIEW':
        return f"{applicant_name} - MANUAL REVIEW: Medium risk ({risk_score:.3f}), Income: ${income:,.0f}, Loan: ${loan_amount:,.0f}"
    
    else:  # AUTO_REJECT
        return f"{applicant_name} - REJECTED: High risk ({risk_score:.3f}), Income: ${income:,.0f}, Loan: ${loan_amount:,.0f}"

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
