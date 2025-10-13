#!/usr/bin/env python3
"""
Applications Repository
Data access layer for loan applications
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connections.postgresqlConnection import getPostgresConnection, getCursor
from connections.dynamodbConnection import getDynamoDbTable, putItem, getItem, updateItem

logger = logging.getLogger(__name__)

class ApplicationsRepository:
    """Repository for loan applications data access"""
    
    def __init__(self):
        self.db = getPostgresConnection()
        self.cache_table = getDynamoDbTable('activeApplications')
    
    def create(self, application_data: Dict[str, Any]) -> str:
        """Create new loan application"""
        try:
            application_id = str(uuid.uuid4())
            
            # Prepare data for PostgreSQL
            pg_data = {
                'application_id': application_id,
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
                'cb_person_cred_hist_length': application_data['cb_person_cred_hist_length'],
                'status': 'PENDING'
            }
            
            # Insert into PostgreSQL
            query = """
            INSERT INTO applications 
            (application_id, person_age, person_income, person_home_ownership, 
             person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate,
             loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length, status)
            VALUES (%(application_id)s, %(person_age)s, %(person_income)s, %(person_home_ownership)s,
                    %(person_emp_length)s, %(loan_intent)s, %(loan_grade)s, %(loan_amnt)s, %(loan_int_rate)s,
                    %(loan_percent_income)s, %(cb_person_default_on_file)s, %(cb_person_cred_hist_length)s, %(status)s)
            """
            
            self.db.executeQuery(query, tuple(pg_data.values()))
            
            # Cache in DynamoDB for fast access
            cache_data = {
                'applicationId': application_id,
                'status': 'PENDING',
                'personAge': application_data['person_age'],
                'personIncome': application_data['person_income'],
                'loanAmount': application_data['loan_amnt'],
                'loanGrade': application_data['loan_grade'],
                'createdAt': datetime.utcnow().isoformat(),
                'ttl': self._generateTtl(24)  # 24 hours
            }
            
            putItem('activeApplications', cache_data)
            
            logger.info(f"Created application: {application_id}")
            return application_id
            
        except Exception as e:
            logger.error(f"Error creating application: {e}")
            raise
    
    def getById(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get application by ID (check cache first, then database)"""
        try:
            # Try cache first (fast)
            cached_app = getItem('activeApplications', {'applicationId': application_id})
            if cached_app:
                logger.debug(f"Retrieved application from cache: {application_id}")
                return self._formatApplicationFromCache(cached_app)
            
            # Fallback to database
            query = """
            SELECT * FROM applications 
            WHERE application_id = %s
            """
            
            result = self.db.executeQuery(query, (application_id,), fetch='one')
            
            if result:
                logger.debug(f"Retrieved application from database: {application_id}")
                return dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting application {application_id}: {e}")
            raise
    
    def updateStatus(self, application_id: str, status: str) -> bool:
        """Update application status"""
        try:
            # Update PostgreSQL
            query = """
            UPDATE applications 
            SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE application_id = %s
            """
            
            rows_affected = self.db.executeQuery(query, (status, application_id))
            
            if rows_affected > 0:
                # Update cache
                update_expression = "SET #status = :status, updatedAt = :updatedAt"
                expression_values = {
                    ':status': status,
                    ':updatedAt': datetime.utcnow().isoformat()
                }
                expression_names = {'#status': 'status'}
                
                updateItem('activeApplications', {'applicationId': application_id},
                          update_expression, expression_values, expression_names)
                
                logger.info(f"Updated application {application_id} status to {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating application status: {e}")
            raise
    
    def getByStatus(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get applications by status"""
        try:
            query = """
            SELECT * FROM applications 
            WHERE status = %s
            ORDER BY created_at DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (status, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting applications by status {status}: {e}")
            raise
    
    def getPendingApplications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending applications for processing"""
        try:
            query = """
            SELECT * FROM applications 
            WHERE status IN ('PENDING', 'PROCESSING')
            ORDER BY created_at ASC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (limit,), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting pending applications: {e}")
            raise
    
    def getApplicationSummary(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get complete application summary with risk score and decision"""
        try:
            query = """
            SELECT 
                a.*,
                rs.risk_score,
                rs.risk_category,
                rs.model_version,
                d.decision,
                d.decision_reason,
                d.underwriter_id
            FROM applications a
            LEFT JOIN risk_scores rs ON a.application_id = rs.application_id
            LEFT JOIN decisions d ON a.application_id = d.application_id
            WHERE a.application_id = %s
            """
            
            result = self.db.executeQuery(query, (application_id,), fetch='one')
            
            if result:
                return dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting application summary: {e}")
            raise
    
    def getApplicationsByDateRange(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get applications within date range"""
        try:
            query = """
            SELECT * FROM applications 
            WHERE created_at BETWEEN %s AND %s
            ORDER BY created_at DESC
            """
            
            results = self.db.executeQuery(query, (start_date, end_date), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting applications by date range: {e}")
            raise
    
    def deleteApplication(self, application_id: str) -> bool:
        """Delete application (cascade delete from related tables)"""
        try:
            # Delete from cache first
            deleteItem('activeApplications', {'applicationId': application_id})
            
            # Delete from database (cascade will handle related records)
            query = "DELETE FROM applications WHERE application_id = %s"
            rows_affected = self.db.executeQuery(query, (application_id,))
            
            if rows_affected > 0:
                logger.info(f"Deleted application: {application_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting application: {e}")
            raise
    
    def getApplicationStats(self) -> Dict[str, Any]:
        """Get application statistics"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_applications,
                COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as approved_count,
                COUNT(CASE WHEN status = 'REJECTED' THEN 1 END) as rejected_count,
                COUNT(CASE WHEN status = 'MANUAL_REVIEW' THEN 1 END) as manual_review_count,
                COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending_count,
                AVG(loan_amnt) as avg_loan_amount,
                SUM(loan_amnt) as total_loan_amount
            FROM applications
            """
            
            result = self.db.executeQuery(query, fetch='one')
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting application stats: {e}")
            raise
    
    def _generateTtl(self, hours: int) -> int:
        """Generate TTL timestamp"""
        return int((datetime.utcnow() + timedelta(hours=hours)).timestamp())
    
    def _formatApplicationFromCache(self, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format cached application data to match database format"""
        return {
            'application_id': cached_data['applicationId'],
            'status': cached_data['status'],
            'person_age': cached_data.get('personAge'),
            'person_income': cached_data.get('personIncome'),
            'loan_amnt': cached_data.get('loanAmount'),
            'loan_grade': cached_data.get('loanGrade'),
            'created_at': cached_data.get('createdAt')
        }

# Global repository instance
applications_repo = ApplicationsRepository()

def getApplicationsRepository():
    """Get applications repository instance"""
    return applications_repo

if __name__ == "__main__":
    # Test the repository
    logging.basicConfig(level=logging.INFO)
    
    repo = ApplicationsRepository()
    
    # Test getting stats
    try:
        stats = repo.getApplicationStats()
        print(f"Application stats: {stats}")
    except Exception as e:
        print(f"Error testing repository: {e}")
