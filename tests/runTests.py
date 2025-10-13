#!/usr/bin/env python3
"""
Test Runner - Runs all tests for the credit risk platform
"""

import sys
import os
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_script(script_name: str, description: str) -> bool:
    """Run a test script and return success status"""
    logger.info(f"🧪 Running {description}...")
    logger.info("-" * 50)
    
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {description} FAILED")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} TIMED OUT")
        return False
    except Exception as e:
        logger.error(f"❌ {description} ERROR: {str(e)}")
        return False

def main():
    """Main test runner"""
    logger.info("🚀 CREDIT RISK PLATFORM - TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Test execution started at: {datetime.now().isoformat()}")
    logger.info("")
    
    # Test scripts to run
    test_scripts = [
        {
            "script": "testLambdaFunctions.py",
            "description": "Lambda Functions Unit Tests"
        },
        {
            "script": "testCompleteWorkflow.py", 
            "description": "Complete End-to-End Workflow Tests"
        }
    ]
    
    results = []
    
    # Run each test script
    for test in test_scripts:
        success = run_test_script(test["script"], test["description"])
        results.append({
            "name": test["description"],
            "success": success
        })
        logger.info("")
    
    # Generate summary
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    logger.info(f"Total Test Suites: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    logger.info("")
    
    # Detailed results
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        logger.info(f"{status} {result['name']}")
    
    logger.info("")
    logger.info(f"Test execution completed at: {datetime.now().isoformat()}")
    
    # Final verdict
    if failed_tests == 0:
        logger.info("🎉 ALL TESTS PASSED! System is ready for production!")
        return True
    else:
        logger.error(f"❌ {failed_tests} test suite(s) failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
