import json
import boto3
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the database directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../database'))

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
eventbridge = boto3.client('events')

# Import our database repositories
try:
    from repositories.applicationsRepository import ApplicationsRepository
    from repositories.riskScoresRepository import RiskScoresRepository
    from repositories.decisionsRepository import DecisionsRepository
except ImportError as e:
    logger.warning(f"Could not import database repositories: {e}")
    # We'll handle this gracefully in the function

def processApplication(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Main orchestrator - saves application data to databases and updates status.
    
    Args:
        event: EventBridge event containing application data with decision
        context: Lambda context
    
    Returns:
        Dict containing processing result
    """
    try:
        logger.info(f"Received application processing request: {event}")
        
        # Extract application data from EventBridge event
        if 'detail' in event:
            application_data = json.loads(event['detail'])
        else:
            application_data = event
        
        application_id = application_data.get('application_id')
        if not application_id:
            raise ValueError("Application ID is required")
        
        # Save application data to PostgreSQL
        application_saved = save_application_data(application_data)
        
        # Save risk score data
        risk_score_saved = save_risk_score_data(application_data)
        
        # Save decision data
        decision_saved = save_decision_data(application_data)
        
        # Update application status
        final_status = update_application_status(application_data)
        
        # Add processing metadata
        application_data['processing_timestamp'] = datetime.utcnow().isoformat()
        application_data['processing_status'] = 'processed'
        application_data['final_status'] = final_status
        
        # Send to EventBridge for portfolio update
        await send_to_eventbridge('application.processed', application_data)
        
        logger.info(f"Application {application_id} processed successfully with status: {final_status}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'application_id': application_id,
                'status': 'processed',
                'final_status': final_status,
                'database_updates': {
                    'application_saved': application_saved,
                    'risk_score_saved': risk_score_saved,
                    'decision_saved': decision_saved
                }
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing application: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def save_application_data(application_data: Dict[str, Any]) -> bool:
    """Save application data to PostgreSQL using repository pattern."""
    try:
        # Initialize repository
        app_repo = ApplicationsRepository()
        
        # Prepare application data for database
        db_data = {
            'application_id': application_data['application_id'],
            'person_age': int(application_data['person_age']),
            'person_income': float(application_data['person_income']),
            'person_home_ownership': application_data['person_home_ownership'],
            'person_emp_length': float(application_data['person_emp_length']),
            'loan_intent': application_data['loan_intent'],
            'loan_grade': application_data['loan_grade'],
            'loan_amnt': float(application_data['loan_amnt']),
            'loan_int_rate': float(application_data['loan_int_rate']),
            'loan_percent_income': float(application_data['loan_percent_income']),
            'cb_person_default_on_file': application_data['cb_person_default_on_file'],
            'cb_person_cred_hist_length': float(application_data['cb_person_cred_hist_length']),
            'status': 'processed',
            'created_at': datetime.utcnow()
        }
        
        # Save to database
        result = app_repo.create(db_data)
        logger.info(f"Application data saved to PostgreSQL: {application_data['application_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save application data: {str(e)}")
        return False

def save_risk_score_data(application_data: Dict[str, Any]) -> bool:
    """Save risk score data to PostgreSQL."""
    try:
        # Initialize repository
        risk_repo = RiskScoresRepository()
        
        # Prepare risk score data
        risk_data = {
            'application_id': application_data['application_id'],
            'risk_score': float(application_data['risk_score']),
            'model_version': 'ensemble-v1.0',
            'features_used': json.dumps({
                'person_age': application_data['person_age'],
                'person_income': application_data['person_income'],
                'person_home_ownership': application_data['person_home_ownership'],
                'person_emp_length': application_data['person_emp_length'],
                'loan_intent': application_data['loan_intent'],
                'loan_grade': application_data['loan_grade'],
                'loan_amnt': application_data['loan_amnt'],
                'loan_int_rate': application_data['loan_int_rate'],
                'loan_percent_income': application_data['loan_percent_income'],
                'cb_person_default_on_file': application_data['cb_person_default_on_file'],
                'cb_person_cred_hist_length': application_data['cb_person_cred_hist_length']
            }),
            'calculated_at': datetime.utcnow()
        }
        
        # Save to database
        result = risk_repo.create(application_data['application_id'], risk_data)
        logger.info(f"Risk score data saved: {application_data['application_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save risk score data: {str(e)}")
        return False

def save_decision_data(application_data: Dict[str, Any]) -> bool:
    """Save decision data to PostgreSQL."""
    try:
        # Initialize repository
        decision_repo = DecisionsRepository()
        
        # Prepare decision data
        decision_data = {
            'application_id': application_data['application_id'],
            'decision': application_data['decision'],
            'decision_reason': application_data['decision_reason'],
            'risk_score': float(application_data['risk_score']),
            'approval_probability': float(application_data['approval_probability']),
            'risk_level': application_data['risk_level'],
            'decision_made_at': datetime.utcnow(),
            'decision_made_by': 'system'
        }
        
        # Save to database
        result = decision_repo.create(decision_data)
        logger.info(f"Decision data saved: {application_data['application_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save decision data: {str(e)}")
        return False

def update_application_status(application_data: Dict[str, Any]) -> str:
    """Update application status based on decision."""
    try:
        # Initialize repository
        app_repo = ApplicationsRepository()
        
        # Determine final status
        decision = application_data.get('decision', 'MANUAL_REVIEW')
        
        if decision == 'AUTO_APPROVE':
            final_status = 'APPROVED'
        elif decision == 'AUTO_REJECT':
            final_status = 'REJECTED'
        else:  # MANUAL_REVIEW
            final_status = 'PENDING_REVIEW'
        
        # Update status in database
        app_repo.updateStatus(application_data['application_id'], final_status)
        
        # Also cache in DynamoDB for fast lookup
        cache_application_status(application_data, final_status)
        
        logger.info(f"Application status updated to: {final_status}")
        return final_status
        
    except Exception as e:
        logger.error(f"Failed to update application status: {str(e)}")
        return 'ERROR'

def cache_application_status(application_data: Dict[str, Any], status: str) -> None:
    """Cache application status in DynamoDB for fast lookup."""
    try:
        # Initialize DynamoDB connection
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('activeApplications')
        
        # Prepare cache data
        cache_data = {
            'applicationId': application_data['application_id'],
            'status': status,
            'decision': application_data.get('decision', 'MANUAL_REVIEW'),
            'riskScore': float(application_data.get('risk_score', 0.5)),
            'riskLevel': application_data.get('risk_level', 'MEDIUM'),
            'updatedAt': datetime.utcnow().isoformat(),
            'ttl': int((datetime.utcnow().timestamp() + (30 * 24 * 60 * 60)))  # 30 days TTL
        }
        
        # Save to DynamoDB
        table.put_item(Item=cache_data)
        logger.info(f"Application status cached in DynamoDB: {application_data['application_id']}")
        
    except Exception as e:
        logger.error(f"Failed to cache application status: {str(e)}")

def generate_processing_summary(application_data: Dict[str, Any]) -> str:
    """Generate a summary of the processing results."""
    
    app_id = application_data.get('application_id', 'Unknown')
    decision = application_data.get('decision', 'Unknown')
    risk_score = application_data.get('risk_score', 0.5)
    final_status = application_data.get('final_status', 'Unknown')
    
    return f"Application {app_id}: Decision={decision}, Risk={risk_score:.3f}, Status={final_status}"

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
