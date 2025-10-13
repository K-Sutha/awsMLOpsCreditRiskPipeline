#!/usr/bin/env python3
"""
Test script for Lambda functions - runs locally without AWS
Tests the complete event-driven workflow
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add the microservices directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../microservices/lambdaFunctions'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock AWS services
class MockEventBridge:
    """Mock EventBridge for local testing"""
    def __init__(self):
        self.events = []
        logger.info("Mock EventBridge initialized")
    
    def put_events(self, Entries):
        """Mock EventBridge put_events"""
        for entry in Entries:
            event_data = {
                'source': entry.get('Source'),
                'detail_type': entry.get('DetailType'),
                'detail': json.loads(entry.get('Detail', '{}')),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.events.append(event_data)
            logger.info(f"📤 Event sent: {entry.get('DetailType')}")
        return {'Entries': [{'EventId': 'mock-event-id'}]}

class MockSageMaker:
    """Mock SageMaker for local testing"""
    def invoke_endpoint(self, EndpointName, ContentType, Body):
        """Mock SageMaker endpoint invocation"""
        # Parse input features
        features = json.loads(Body)
        
        # Calculate mock risk score based on features
        risk_score = calculate_mock_risk_score(features)
        
        # Return mock response
        mock_response = {
            'Body': MockBody(json.dumps([risk_score]))
        }
        logger.info(f"🤖 SageMaker mock prediction: {risk_score}")
        return mock_response

class MockBody:
    """Mock response body for SageMaker"""
    def __init__(self, content):
        self._content = content
    
    def read(self):
        return self._content.encode()
    
    def decode(self):
        return self._content

# Mock AWS clients
import boto3
boto3.client = lambda service, **kwargs: MockEventBridge() if service == 'events' else MockSageMaker() if service == 'sagemaker-runtime' else None

def calculate_mock_risk_score(features: Dict[str, Any]) -> float:
    """Calculate mock risk score for testing"""
    risk_score = 0.5  # Base risk
    
    # Adjust based on key factors
    if features.get('has_default_history', 0) == 1:
        risk_score += 0.3
    
    if features.get('high_debt_ratio', 0) == 1:
        risk_score += 0.2
    
    if features.get('low_credit_history', 0) == 1:
        risk_score += 0.15
    
    if features.get('employment_new', 0) == 1:
        risk_score += 0.1
    
    # Adjust based on loan grade
    grade = features.get('loan_grade_encoded', 4)
    grade_adjustments = {7: -0.2, 6: -0.1, 5: 0.0, 4: 0.1, 3: 0.2, 2: 0.3, 1: 0.4}
    risk_score += grade_adjustments.get(grade, 0.1)
    
    return max(0.0, min(1.0, risk_score))

# Import Lambda functions
try:
    from validateApplication.lambda_function import validateApplication
    from scoreRisk.lambda_function import scoreRisk
    from makeDecision.lambda_function import makeDecision
    from processApplication.lambda_function import processApplication
    from updatePortfolio.lambda_function import updatePortfolio
    logger.info("✅ All Lambda functions imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Lambda functions: {e}")
    sys.exit(1)

def test_validateApplication():
    """Test the validateApplication Lambda function"""
    logger.info("🧪 Testing validateApplication function...")
    
    # Test case 1: Valid application
    valid_app = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 15000,
        "loan_int_rate": 12.5,
        "loan_percent_income": 0.3,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }
    
    event = {"body": json.dumps(valid_app)}
    context = None
    
    result = validateApplication(event, context)
    
    if result['statusCode'] == 200:
        logger.info("✅ validateApplication: Valid application passed")
        response_data = json.loads(result['body'])
        return response_data.get('application_id')
    else:
        logger.error(f"❌ validateApplication failed: {result}")
        return None

def test_scoreRisk(application_data: Dict[str, Any]):
    """Test the scoreRisk Lambda function"""
    logger.info("🧪 Testing scoreRisk function...")
    
    event = {"detail": json.dumps(application_data)}
    context = None
    
    result = scoreRisk(event, context)
    
    if result['statusCode'] == 200:
        logger.info("✅ scoreRisk: Risk scoring completed")
        response_data = json.loads(result['body'])
        return response_data.get('risk_score')
    else:
        logger.error(f"❌ scoreRisk failed: {result}")
        return None

def test_makeDecision(application_data: Dict[str, Any]):
    """Test the makeDecision Lambda function"""
    logger.info("🧪 Testing makeDecision function...")
    
    event = {"detail": json.dumps(application_data)}
    context = None
    
    result = makeDecision(event, context)
    
    if result['statusCode'] == 200:
        logger.info("✅ makeDecision: Decision made successfully")
        response_data = json.loads(result['body'])
        return response_data.get('decision')
    else:
        logger.error(f"❌ makeDecision failed: {result}")
        return None

def test_processApplication(application_data: Dict[str, Any]):
    """Test the processApplication Lambda function"""
    logger.info("🧪 Testing processApplication function...")
    
    event = {"detail": json.dumps(application_data)}
    context = None
    
    result = processApplication(event, context)
    
    if result['statusCode'] == 200:
        logger.info("✅ processApplication: Application processed successfully")
        response_data = json.loads(result['body'])
        return response_data.get('final_status')
    else:
        logger.error(f"❌ processApplication failed: {result}")
        return None

def test_updatePortfolio(application_data: Dict[str, Any]):
    """Test the updatePortfolio Lambda function"""
    logger.info("🧪 Testing updatePortfolio function...")
    
    event = {"detail": json.dumps(application_data)}
    context = None
    
    result = updatePortfolio(event, context)
    
    if result['statusCode'] == 200:
        logger.info("✅ updatePortfolio: Portfolio updated successfully")
        response_data = json.loads(result['body'])
        return response_data.get('message')
    else:
        logger.error(f"❌ updatePortfolio failed: {result}")
        return None

def run_complete_workflow():
    """Run the complete event-driven workflow"""
    logger.info("🚀 Starting complete event-driven workflow test...")
    
    # Step 1: Validate Application
    app_id = test_validateApplication()
    if not app_id:
        logger.error("❌ Workflow failed at validation step")
        return False
    
    # Create application data for subsequent steps
    application_data = {
        "application_id": app_id,
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 15000,
        "loan_int_rate": 12.5,
        "loan_percent_income": 0.3,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
        "validation_timestamp": datetime.utcnow().isoformat(),
        "validation_status": "validated"
    }
    
    # Step 2: Score Risk
    risk_score = test_scoreRisk(application_data)
    if risk_score is None:
        logger.error("❌ Workflow failed at risk scoring step")
        return False
    
    application_data["risk_score"] = risk_score
    application_data["risk_timestamp"] = datetime.utcnow().isoformat()
    application_data["risk_status"] = "scored"
    
    # Step 3: Make Decision
    decision = test_makeDecision(application_data)
    if not decision:
        logger.error("❌ Workflow failed at decision making step")
        return False
    
    application_data["decision"] = decision
    application_data["decision_timestamp"] = datetime.utcnow().isoformat()
    application_data["decision_status"] = "decided"
    
    # Step 4: Process Application
    final_status = test_processApplication(application_data)
    if not final_status:
        logger.error("❌ Workflow failed at application processing step")
        return False
    
    application_data["final_status"] = final_status
    application_data["processing_timestamp"] = datetime.utcnow().isoformat()
    application_data["processing_status"] = "processed"
    
    # Step 5: Update Portfolio
    portfolio_result = test_updatePortfolio(application_data)
    if not portfolio_result:
        logger.error("❌ Workflow failed at portfolio update step")
        return False
    
    logger.info("🎉 Complete workflow test PASSED!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Application ID: {app_id}")
    logger.info(f"   Risk Score: {risk_score:.3f}")
    logger.info(f"   Decision: {decision}")
    logger.info(f"   Final Status: {final_status}")
    
    return True

def test_multiple_applications():
    """Test with multiple different applications"""
    logger.info("🧪 Testing multiple applications...")
    
    test_cases = [
        {
            "name": "Low Risk Application",
            "data": {
                "person_age": 30,
                "person_income": 80000,
                "person_home_ownership": "OWN",
                "person_emp_length": 8,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "loan_amnt": 25000,
                "loan_int_rate": 8.5,
                "loan_percent_income": 0.25,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 10
            }
        },
        {
            "name": "High Risk Application",
            "data": {
                "person_age": 22,
                "person_income": 25000,
                "person_home_ownership": "RENT",
                "person_emp_length": 1,
                "loan_intent": "PERSONAL",
                "loan_grade": "D",
                "loan_amnt": 8000,
                "loan_int_rate": 18.5,
                "loan_percent_income": 0.45,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 1
            }
        },
        {
            "name": "Medium Risk Application",
            "data": {
                "person_age": 35,
                "person_income": 60000,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 5,
                "loan_intent": "EDUCATION",
                "loan_grade": "C",
                "loan_amnt": 20000,
                "loan_int_rate": 14.2,
                "loan_percent_income": 0.35,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"🧪 Testing: {test_case['name']}")
        
        # Create event with test data
        event = {"body": json.dumps(test_case['data'])}
        context = None
        
        # Test validation
        validation_result = validateApplication(event, context)
        if validation_result['statusCode'] != 200:
            logger.error(f"❌ Validation failed for {test_case['name']}")
            continue
        
        # Extract application ID
        response_data = json.loads(validation_result['body'])
        app_id = response_data.get('application_id')
        
        # Continue with rest of workflow...
        application_data = test_case['data'].copy()
        application_data['application_id'] = app_id
        application_data['validation_timestamp'] = datetime.utcnow().isoformat()
        application_data['validation_status'] = 'validated'
        
        # Score risk
        risk_event = {"detail": json.dumps(application_data)}
        risk_result = scoreRisk(risk_event, context)
        
        if risk_result['statusCode'] == 200:
            risk_data = json.loads(risk_result['body'])
            risk_score = risk_data.get('risk_score')
            application_data['risk_score'] = risk_score
            
            # Make decision
            decision_event = {"detail": json.dumps(application_data)}
            decision_result = makeDecision(decision_event, context)
            
            if decision_result['statusCode'] == 200:
                decision_data = json.loads(decision_result['body'])
                decision = decision_data.get('decision')
                
                results.append({
                    'name': test_case['name'],
                    'application_id': app_id,
                    'risk_score': risk_score,
                    'decision': decision
                })
                
                logger.info(f"✅ {test_case['name']}: Risk={risk_score:.3f}, Decision={decision}")
    
    logger.info(f"📊 Test Results Summary:")
    for result in results:
        logger.info(f"   {result['name']}: {result['decision']} (Risk: {result['risk_score']:.3f})")
    
    return len(results) == len(test_cases)

def main():
    """Main test function"""
    logger.info("🚀 Starting Lambda Functions Local Testing")
    logger.info("=" * 60)
    
    try:
        # Test individual functions
        logger.info("📋 Testing individual Lambda functions...")
        
        # Test complete workflow
        logger.info("\n🔄 Testing complete event-driven workflow...")
        workflow_success = run_complete_workflow()
        
        if workflow_success:
            logger.info("\n🧪 Testing multiple applications...")
            multiple_success = test_multiple_applications()
            
            if multiple_success:
                logger.info("\n🎉 ALL TESTS PASSED!")
                logger.info("✅ Lambda functions are working correctly")
                logger.info("✅ Event-driven workflow is functional")
                logger.info("✅ Multiple application types handled properly")
            else:
                logger.error("\n❌ Multiple application tests failed")
        else:
            logger.error("\n❌ Complete workflow test failed")
    
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
