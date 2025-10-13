#!/usr/bin/env python3
"""
Database Setup Script
Production-ready database initialization and testing
"""

import logging
import json
import boto3
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from connections.postgresqlConnection import testConnection as testPostgresConnection
from connections.dynamodbConnection import testConnection as testDynamoDbConnection
from repositories.applicationsRepository import getApplicationsRepository
from repositories.riskScoresRepository import getRiskScoresRepository
from repositories.decisionsRepository import getDecisionsRepository

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and testing class"""
    
    def __init__(self):
        self.setup_results = {
            'timestamp': datetime.now().isoformat(),
            'postgresql': {},
            'dynamodb': {},
            'repositories': {},
            'overall_status': 'UNKNOWN'
        }
    
    def testPostgreSQL(self) -> bool:
        """Test PostgreSQL connection and basic operations"""
        logger.info("Testing PostgreSQL connection...")
        
        try:
            # Test connection
            if not testPostgresConnection():
                self.setup_results['postgresql']['connection'] = 'FAILED'
                return False
            
            self.setup_results['postgresql']['connection'] = 'SUCCESS'
            
            # Test basic query
            from connections.postgresqlConnection import executeQuery
            
            result = executeQuery("SELECT version()", fetch='one')
            if result:
                self.setup_results['postgresql']['basic_query'] = 'SUCCESS'
                self.setup_results['postgresql']['version'] = result[0][:50] + '...'
            else:
                self.setup_results['postgresql']['basic_query'] = 'FAILED'
                return False
            
            logger.info("✅ PostgreSQL connection successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL test failed: {e}")
            self.setup_results['postgresql']['error'] = str(e)
            return False
    
    def testDynamoDB(self) -> bool:
        """Test DynamoDB connection and basic operations"""
        logger.info("Testing DynamoDB connection...")
        
        try:
            # Test connection
            if not testDynamoDbConnection():
                self.setup_results['dynamodb']['connection'] = 'FAILED'
                return False
            
            self.setup_results['dynamodb']['connection'] = 'SUCCESS'
            
            # Test table access
            from connections.dynamodbConnection import getDynamoDbTable
            
            try:
                table = getDynamoDbTable('activeApplications')
                self.setup_results['dynamodb']['table_access'] = 'SUCCESS'
            except Exception as e:
                self.setup_results['dynamodb']['table_access'] = 'FAILED'
                self.setup_results['dynamodb']['table_error'] = str(e)
                logger.warning(f"DynamoDB table access failed (expected if tables don't exist): {e}")
            
            logger.info("✅ DynamoDB connection successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ DynamoDB test failed: {e}")
            self.setup_results['dynamodb']['error'] = str(e)
            return False
    
    def testRepositories(self) -> bool:
        """Test repository functionality"""
        logger.info("Testing repositories...")
        
        try:
            # Test Applications Repository
            apps_repo = getApplicationsRepository()
            stats = apps_repo.getApplicationStats()
            self.setup_results['repositories']['applications'] = 'SUCCESS'
            logger.info("✅ Applications repository working")
            
            # Test Risk Scores Repository
            risk_repo = getRiskScoresRepository()
            risk_stats = risk_repo.getRiskScoreStats()
            self.setup_results['repositories']['risk_scores'] = 'SUCCESS'
            logger.info("✅ Risk scores repository working")
            
            # Test Decisions Repository
            decisions_repo = getDecisionsRepository()
            decision_stats = decisions_repo.getDecisionStats()
            self.setup_results['repositories']['decisions'] = 'SUCCESS'
            logger.info("✅ Decisions repository working")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Repository test failed: {e}")
            self.setup_results['repositories']['error'] = str(e)
            return False
    
    def runFullTest(self) -> bool:
        """Run complete database setup test"""
        logger.info("Starting comprehensive database setup test...")
        
        postgres_ok = self.testPostgreSQL()
        dynamodb_ok = self.testDynamoDB()
        repos_ok = self.testRepositories()
        
        # Determine overall status
        if postgres_ok and dynamodb_ok and repos_ok:
            self.setup_results['overall_status'] = 'SUCCESS'
            logger.info("🎉 All database components working correctly!")
            return True
        elif postgres_ok and repos_ok:
            self.setup_results['overall_status'] = 'PARTIAL_SUCCESS'
            logger.warning("⚠️ PostgreSQL and repositories working, DynamoDB issues")
            return True
        else:
            self.setup_results['overall_status'] = 'FAILED'
            logger.error("❌ Database setup failed")
            return False
    
    def generateReport(self) -> str:
        """Generate setup report"""
        report = f"""
Database Setup Report
====================
Timestamp: {self.setup_results['timestamp']}
Overall Status: {self.setup_results['overall_status']}

PostgreSQL:
- Connection: {self.setup_results['postgresql'].get('connection', 'NOT_TESTED')}
- Basic Query: {self.setup_results['postgresql'].get('basic_query', 'NOT_TESTED')}
- Version: {self.setup_results['postgresql'].get('version', 'UNKNOWN')}

DynamoDB:
- Connection: {self.setup_results['dynamodb'].get('connection', 'NOT_TESTED')}
- Table Access: {self.setup_results['dynamodb'].get('table_access', 'NOT_TESTED')}

Repositories:
- Applications: {self.setup_results['repositories'].get('applications', 'NOT_TESTED')}
- Risk Scores: {self.setup_results['repositories'].get('risk_scores', 'NOT_TESTED')}
- Decisions: {self.setup_results['repositories'].get('decisions', 'NOT_TESTED')}

Next Steps:
"""
        
        if self.setup_results['overall_status'] == 'SUCCESS':
            report += "- Database is ready for production use\n"
            report += "- You can now run the application\n"
        elif self.setup_results['overall_status'] == 'PARTIAL_SUCCESS':
            report += "- PostgreSQL is ready\n"
            report += "- Set up DynamoDB tables if needed\n"
            report += "- Application will work with reduced functionality\n"
        else:
            report += "- Fix database connection issues\n"
            report += "- Check AWS credentials and permissions\n"
            report += "- Verify network connectivity\n"
        
        return report
    
    def saveReport(self, filename: str = None):
        """Save setup report to file"""
        if not filename:
            filename = f"database_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.setup_results, f, indent=2)
        
        logger.info(f"Setup report saved to: {filepath}")
        return filepath

def main():
    """Main setup function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Credit Risk Platform - Database Setup")
    print("=" * 50)
    
    setup = DatabaseSetup()
    
    # Run tests
    success = setup.runFullTest()
    
    # Generate and display report
    report = setup.generateReport()
    print(report)
    
    # Save report
    report_file = setup.saveReport()
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if success:
        print("✅ Database setup completed successfully!")
        sys.exit(0)
    else:
        print("❌ Database setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
