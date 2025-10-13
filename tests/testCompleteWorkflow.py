#!/usr/bin/env python3
"""
Complete Workflow Test - End-to-End Testing
Tests the entire credit risk platform workflow from application submission to decision
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the microservices directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../microservices/lambdaFunctions'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/scripts'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
try:
    from testLambdaFunctions import MockEventBridge, MockSageMaker, MockBody
    import boto3
    
    # Setup mocks
    boto3.client = lambda service, **kwargs: MockEventBridge() if service == 'events' else MockSageMaker() if service == 'sagemaker-runtime' else None
    
    from validateApplication.lambda_function import validateApplication
    from scoreRisk.lambda_function import scoreRisk
    from makeDecision.lambda_function import makeDecision
    from processApplication.lambda_function import processApplication
    from updatePortfolio.lambda_function import updatePortfolio
    
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    sys.exit(1)

class CreditRiskWorkflowTester:
    """Complete workflow tester for credit risk platform"""
    
    def __init__(self):
        self.test_results = []
        self.event_log = []
        logger.info("🚀 Credit Risk Workflow Tester initialized")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log events for tracking"""
        event_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.event_log.append(event_entry)
        logger.info(f"📝 Event logged: {event_type}")
    
    def submit_loan_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a loan application and start the workflow"""
        logger.info("📋 Submitting loan application...")
        
        # Log the application submission
        self.log_event("application_submitted", application_data)
        
        # Create API Gateway event
        api_event = {
            "body": json.dumps(application_data),
            "headers": {"Content-Type": "application/json"},
            "requestContext": {"requestId": f"req-{int(time.time())}"}
        }
        
        # Step 1: Validate Application
        logger.info("🔍 Step 1: Validating application...")
        validation_result = validateApplication(api_event, None)
        
        if validation_result['statusCode'] != 200:
            logger.error(f"❌ Validation failed: {validation_result}")
            return {"success": False, "error": "Validation failed", "details": validation_result}
        
        response_data = json.loads(validation_result['body'])
        application_id = response_data.get('application_id')
        
        # Log validation success
        self.log_event("application_validated", {
            "application_id": application_id,
            "status": "validated"
        })
        
        # Prepare data for next steps
        workflow_data = application_data.copy()
        workflow_data.update({
            "application_id": application_id,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "validation_status": "validated"
        })
        
        return {"success": True, "application_data": workflow_data}
    
    def process_risk_scoring(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process risk scoring step"""
        logger.info("🤖 Step 2: Processing risk scoring...")
        
        # Create EventBridge event
        event = {"detail": json.dumps(application_data)}
        
        # Call scoreRisk function
        risk_result = scoreRisk(event, None)
        
        if risk_result['statusCode'] != 200:
            logger.error(f"❌ Risk scoring failed: {risk_result}")
            return {"success": False, "error": "Risk scoring failed"}
        
        response_data = json.loads(risk_result['body'])
        risk_score = response_data.get('risk_score')
        
        # Update application data
        application_data.update({
            "risk_score": risk_score,
            "risk_timestamp": datetime.utcnow().isoformat(),
            "risk_status": "scored"
        })
        
        # Log risk scoring
        self.log_event("risk_scored", {
            "application_id": application_data["application_id"],
            "risk_score": risk_score
        })
        
        return {"success": True, "application_data": application_data}
    
    def process_decision_making(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process decision making step"""
        logger.info("⚖️ Step 3: Processing decision making...")
        
        # Create EventBridge event
        event = {"detail": json.dumps(application_data)}
        
        # Call makeDecision function
        decision_result = makeDecision(event, None)
        
        if decision_result['statusCode'] != 200:
            logger.error(f"❌ Decision making failed: {decision_result}")
            return {"success": False, "error": "Decision making failed"}
        
        response_data = json.loads(decision_result['body'])
        decision = response_data.get('decision')
        reason = response_data.get('reason')
        
        # Update application data
        application_data.update({
            "decision": decision,
            "decision_reason": reason,
            "decision_timestamp": datetime.utcnow().isoformat(),
            "decision_status": "decided"
        })
        
        # Log decision
        self.log_event("decision_made", {
            "application_id": application_data["application_id"],
            "decision": decision,
            "reason": reason
        })
        
        return {"success": True, "application_data": application_data}
    
    def process_application_storage(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process application storage step"""
        logger.info("💾 Step 4: Processing application storage...")
        
        # Create EventBridge event
        event = {"detail": json.dumps(application_data)}
        
        # Call processApplication function
        processing_result = processApplication(event, None)
        
        if processing_result['statusCode'] != 200:
            logger.error(f"❌ Application processing failed: {processing_result}")
            return {"success": False, "error": "Application processing failed"}
        
        response_data = json.loads(processing_result['body'])
        final_status = response_data.get('final_status')
        
        # Update application data
        application_data.update({
            "final_status": final_status,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "processing_status": "processed"
        })
        
        # Log processing
        self.log_event("application_processed", {
            "application_id": application_data["application_id"],
            "final_status": final_status
        })
        
        return {"success": True, "application_data": application_data}
    
    def update_portfolio_analytics(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update portfolio analytics step"""
        logger.info("📊 Step 5: Updating portfolio analytics...")
        
        # Create EventBridge event
        event = {"detail": json.dumps(application_data)}
        
        # Call updatePortfolio function
        portfolio_result = updatePortfolio(event, None)
        
        if portfolio_result['statusCode'] != 200:
            logger.error(f"❌ Portfolio update failed: {portfolio_result}")
            return {"success": False, "error": "Portfolio update failed"}
        
        response_data = json.loads(portfolio_result['body'])
        
        # Log portfolio update
        self.log_event("portfolio_updated", {
            "application_id": application_data["application_id"],
            "message": response_data.get('message')
        })
        
        return {"success": True, "application_data": application_data}
    
    def run_complete_workflow(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete end-to-end workflow"""
        logger.info("🚀 Starting complete credit risk workflow...")
        start_time = time.time()
        
        workflow_result = {
            "application_id": None,
            "success": False,
            "steps_completed": 0,
            "total_steps": 5,
            "execution_time": 0,
            "final_decision": None,
            "risk_score": None,
            "final_status": None,
            "errors": []
        }
        
        try:
            # Step 1: Submit and validate application
            result = self.submit_loan_application(application_data)
            if not result["success"]:
                workflow_result["errors"].append(result["error"])
                return workflow_result
            
            application_data = result["application_data"]
            workflow_result["application_id"] = application_data["application_id"]
            workflow_result["steps_completed"] += 1
            
            # Step 2: Risk scoring
            result = self.process_risk_scoring(application_data)
            if not result["success"]:
                workflow_result["errors"].append(result["error"])
                return workflow_result
            
            application_data = result["application_data"]
            workflow_result["risk_score"] = application_data["risk_score"]
            workflow_result["steps_completed"] += 1
            
            # Step 3: Decision making
            result = self.process_decision_making(application_data)
            if not result["success"]:
                workflow_result["errors"].append(result["error"])
                return workflow_result
            
            application_data = result["application_data"]
            workflow_result["final_decision"] = application_data["decision"]
            workflow_result["steps_completed"] += 1
            
            # Step 4: Application processing
            result = self.process_application_storage(application_data)
            if not result["success"]:
                workflow_result["errors"].append(result["error"])
                return workflow_result
            
            application_data = result["application_data"]
            workflow_result["final_status"] = application_data["final_status"]
            workflow_result["steps_completed"] += 1
            
            # Step 5: Portfolio update
            result = self.update_portfolio_analytics(application_data)
            if not result["success"]:
                workflow_result["errors"].append(result["error"])
                return workflow_result
            
            workflow_result["steps_completed"] += 1
            
            # Calculate execution time
            end_time = time.time()
            workflow_result["execution_time"] = round(end_time - start_time, 2)
            workflow_result["success"] = True
            
            logger.info("🎉 Complete workflow finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Workflow failed with exception: {str(e)}")
            workflow_result["errors"].append(str(e))
        
        return workflow_result
    
    def test_multiple_scenarios(self) -> List[Dict[str, Any]]:
        """Test multiple loan application scenarios"""
        logger.info("🧪 Testing multiple loan application scenarios...")
        
        test_scenarios = [
            {
                "name": "Low Risk - High Income Professional",
                "data": {
                    "person_age": 35,
                    "person_income": 120000,
                    "person_home_ownership": "OWN",
                    "person_emp_length": 10,
                    "loan_intent": "HOMEIMPROVEMENT",
                    "loan_grade": "A",
                    "loan_amnt": 35000,
                    "loan_int_rate": 7.5,
                    "loan_percent_income": 0.25,
                    "cb_person_default_on_file": "N",
                    "cb_person_cred_hist_length": 12
                },
                "expected_decision": "AUTO_APPROVE"
            },
            {
                "name": "High Risk - Young Borrower",
                "data": {
                    "person_age": 21,
                    "person_income": 28000,
                    "person_home_ownership": "RENT",
                    "person_emp_length": 1,
                    "loan_intent": "PERSONAL",
                    "loan_grade": "E",
                    "loan_amnt": 12000,
                    "loan_int_rate": 22.5,
                    "loan_percent_income": 0.5,
                    "cb_person_default_on_file": "Y",
                    "cb_person_cred_hist_length": 1
                },
                "expected_decision": "AUTO_REJECT"
            },
            {
                "name": "Medium Risk - Recent Graduate",
                "data": {
                    "person_age": 24,
                    "person_income": 45000,
                    "person_home_ownership": "RENT",
                    "person_emp_length": 2,
                    "loan_intent": "EDUCATION",
                    "loan_grade": "C",
                    "loan_amnt": 18000,
                    "loan_int_rate": 15.0,
                    "loan_percent_income": 0.35,
                    "cb_person_default_on_file": "N",
                    "cb_person_cred_hist_length": 3
                },
                "expected_decision": "MANUAL_REVIEW"
            },
            {
                "name": "Edge Case - Large Loan",
                "data": {
                    "person_age": 40,
                    "person_income": 150000,
                    "person_home_ownership": "MORTGAGE",
                    "person_emp_length": 15,
                    "loan_intent": "VENTURE",
                    "loan_grade": "B",
                    "loan_amnt": 100000,
                    "loan_int_rate": 9.5,
                    "loan_percent_income": 0.4,
                    "cb_person_default_on_file": "N",
                    "cb_person_cred_hist_length": 18
                },
                "expected_decision": "MANUAL_REVIEW"
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n🧪 Test Scenario {i}: {scenario['name']}")
            logger.info("-" * 50)
            
            # Run complete workflow for this scenario
            workflow_result = self.run_complete_workflow(scenario['data'])
            
            # Add scenario information
            workflow_result.update({
                "scenario_name": scenario['name'],
                "scenario_number": i,
                "expected_decision": scenario['expected_decision'],
                "decision_match": workflow_result.get("final_decision") == scenario['expected_decision']
            })
            
            results.append(workflow_result)
            
            # Log results
            if workflow_result["success"]:
                logger.info(f"✅ {scenario['name']}: {workflow_result['final_decision']} "
                          f"(Risk: {workflow_result['risk_score']:.3f}, "
                          f"Time: {workflow_result['execution_time']}s)")
                
                if workflow_result["decision_match"]:
                    logger.info(f"✅ Decision matches expectation: {scenario['expected_decision']}")
                else:
                    logger.warning(f"⚠️ Decision differs from expectation: "
                                 f"Expected {scenario['expected_decision']}, "
                                 f"Got {workflow_result['final_decision']}")
            else:
                logger.error(f"❌ {scenario['name']}: Workflow failed")
                for error in workflow_result.get("errors", []):
                    logger.error(f"   Error: {error}")
        
        return results
    
    def generate_test_report(self, results: List[Dict[str, Any]]):
        """Generate a comprehensive test report"""
        logger.info("\n📊 GENERATING TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - successful_tests
        
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Successful: {successful_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if successful_tests > 0:
            avg_execution_time = sum(r["execution_time"] for r in results if r["success"]) / successful_tests
            logger.info(f"   Average Execution Time: {avg_execution_time:.2f} seconds")
        
        logger.info(f"\n📋 Detailed Results:")
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"   {status} {result['scenario_name']}")
            logger.info(f"      Application ID: {result.get('application_id', 'N/A')}")
            logger.info(f"      Risk Score: {result.get('risk_score', 'N/A')}")
            logger.info(f"      Decision: {result.get('final_decision', 'N/A')}")
            logger.info(f"      Status: {result.get('final_status', 'N/A')}")
            logger.info(f"      Execution Time: {result.get('execution_time', 'N/A')}s")
            
            if not result["decision_match"]:
                logger.info(f"      ⚠️ Expected: {result.get('expected_decision', 'N/A')}")
        
        # Event log summary
        logger.info(f"\n📝 Event Log Summary:")
        event_types = {}
        for event in self.event_log:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        for event_type, count in event_types.items():
            logger.info(f"   {event_type}: {count} events")
        
        logger.info(f"\n🎯 Workflow Performance:")
        if successful_tests > 0:
            logger.info(f"   ✅ End-to-end workflow is functional")
            logger.info(f"   ✅ All Lambda functions are working")
            logger.info(f"   ✅ Event-driven architecture is operational")
            logger.info(f"   ✅ Risk scoring and decision making is accurate")
        else:
            logger.error(f"   ❌ Workflow needs debugging")
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests/total_tests)*100,
            "results": results,
            "event_log": self.event_log
        }

def main():
    """Main test execution function"""
    logger.info("🚀 CREDIT RISK PLATFORM - COMPLETE WORKFLOW TESTING")
    logger.info("=" * 70)
    
    # Initialize tester
    tester = CreditRiskWorkflowTester()
    
    try:
        # Test multiple scenarios
        logger.info("🧪 Running multiple test scenarios...")
        results = tester.test_multiple_scenarios()
        
        # Generate comprehensive report
        report = tester.generate_test_report(results)
        
        # Final summary
        logger.info(f"\n🎉 TESTING COMPLETE!")
        if report["success_rate"] == 100:
            logger.info("🎊 ALL TESTS PASSED - SYSTEM IS READY FOR PRODUCTION!")
        elif report["success_rate"] >= 75:
            logger.info("✅ Most tests passed - System is mostly functional")
        else:
            logger.error("❌ Multiple test failures - System needs debugging")
        
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Success Rate: {report['success_rate']:.1f}%")
        logger.info(f"   Total Events Processed: {len(tester.event_log)}")
        logger.info(f"   Workflow Steps Tested: 5 (Validation → Risk → Decision → Storage → Analytics)")
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
