#!/usr/bin/env python3
"""
Decisions Repository
Data access layer for loan decisions and underwriting
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connections.postgresqlConnection import getPostgresConnection
from connections.dynamodbConnection import updateItem

logger = logging.getLogger(__name__)

class DecisionsRepository:
    """Repository for loan decisions data access"""
    
    def __init__(self):
        self.db = getPostgresConnection()
    
    def create(self, decision_data: Dict[str, Any]) -> str:
        """Create new loan decision"""
        try:
            decision_id = str(uuid.uuid4())
            
            # Prepare data for PostgreSQL
            pg_data = {
                'decision_id': decision_id,
                'application_id': decision_data['application_id'],
                'decision': decision_data['decision'],
                'decision_reason': decision_data.get('decision_reason'),
                'risk_score': decision_data['risk_score'],
                'underwriter_id': decision_data.get('underwriter_id'),
                'underwriter_notes': decision_data.get('underwriter_notes')
            }
            
            # Insert into PostgreSQL
            query = """
            INSERT INTO decisions 
            (decision_id, application_id, decision, decision_reason, 
             risk_score, underwriter_id, underwriter_notes)
            VALUES (%(decision_id)s, %(application_id)s, %(decision)s, %(decision_reason)s,
                    %(risk_score)s, %(underwriter_id)s, %(underwriter_notes)s)
            """
            
            self.db.executeQuery(query, tuple(pg_data.values()))
            
            logger.info(f"Created decision: {decision_id} for application: {decision_data['application_id']}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error creating decision: {e}")
            raise
    
    def getByApplicationId(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get decision by application ID"""
        try:
            query = """
            SELECT * FROM decisions 
            WHERE application_id = %s
            ORDER BY decision_timestamp DESC
            LIMIT 1
            """
            
            result = self.db.executeQuery(query, (application_id,), fetch='one')
            
            if result:
                return dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting decision for application {application_id}: {e}")
            raise
    
    def getDecisionsByStatus(self, decision: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decisions by status (APPROVE, REJECT, MANUAL_REVIEW)"""
        try:
            query = """
            SELECT 
                d.*,
                a.person_age,
                a.person_income,
                a.loan_amnt,
                a.loan_grade
            FROM decisions d
            JOIN applications a ON d.application_id = a.application_id
            WHERE d.decision = %s
            ORDER BY d.decision_timestamp DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (decision, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting decisions by status {decision}: {e}")
            raise
    
    def getDecisionsByUnderwriter(self, underwriter_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decisions by underwriter"""
        try:
            query = """
            SELECT 
                d.*,
                a.person_age,
                a.person_income,
                a.loan_amnt,
                a.loan_grade
            FROM decisions d
            JOIN applications a ON d.application_id = a.application_id
            WHERE d.underwriter_id = %s
            ORDER BY d.decision_timestamp DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (underwriter_id, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting decisions by underwriter {underwriter_id}: {e}")
            raise
    
    def getDecisionsByDateRange(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get decisions within date range"""
        try:
            query = """
            SELECT 
                d.*,
                a.person_age,
                a.person_income,
                a.loan_amnt,
                a.loan_grade
            FROM decisions d
            JOIN applications a ON d.application_id = a.application_id
            WHERE d.decision_timestamp BETWEEN %s AND %s
            ORDER BY d.decision_timestamp DESC
            """
            
            results = self.db.executeQuery(query, (start_date, end_date), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting decisions by date range: {e}")
            raise
    
    def getApprovalRate(self, date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Get approval rate statistics"""
        try:
            if date_range:
                start_date, end_date = date_range
                query = """
                SELECT 
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                    SUM(CASE WHEN decision = 'MANUAL_REVIEW' THEN 1 ELSE 0 END) as manual_review_count,
                    AVG(risk_score) as avg_risk_score,
                    ROUND(
                        (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                    ) as approval_rate
                FROM decisions 
                WHERE decision_timestamp BETWEEN %s AND %s
                """
                result = self.db.executeQuery(query, (start_date, end_date), fetch='one')
            else:
                query = """
                SELECT 
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                    SUM(CASE WHEN decision = 'MANUAL_REVIEW' THEN 1 ELSE 0 END) as manual_review_count,
                    AVG(risk_score) as avg_risk_score,
                    ROUND(
                        (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                    ) as approval_rate
                FROM decisions
                """
                result = self.db.executeQuery(query, fetch='one')
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting approval rate: {e}")
            raise
    
    def getUnderwriterPerformance(self, underwriter_id: str = None) -> Dict[str, Any]:
        """Get underwriter performance metrics"""
        try:
            if underwriter_id:
                query = """
                SELECT 
                    underwriter_id,
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                    AVG(risk_score) as avg_risk_score,
                    ROUND(
                        (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                    ) as approval_rate,
                    MIN(decision_timestamp) as first_decision,
                    MAX(decision_timestamp) as last_decision
                FROM decisions 
                WHERE underwriter_id = %s
                GROUP BY underwriter_id
                """
                result = self.db.executeQuery(query, (underwriter_id,), fetch='one')
            else:
                query = """
                SELECT 
                    underwriter_id,
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                    AVG(risk_score) as avg_risk_score,
                    ROUND(
                        (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                    ) as approval_rate
                FROM decisions 
                WHERE underwriter_id IS NOT NULL
                GROUP BY underwriter_id
                ORDER BY total_decisions DESC
                """
                results = self.db.executeQuery(query, fetch='all')
                return [dict(row) for row in results] if results else []
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting underwriter performance: {e}")
            raise
    
    def getDecisionTrends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get decision trends over time"""
        try:
            query = """
            SELECT 
                DATE(decision_timestamp) as decision_date,
                COUNT(*) as total_decisions,
                SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                SUM(CASE WHEN decision = 'MANUAL_REVIEW' THEN 1 ELSE 0 END) as manual_review_count,
                AVG(risk_score) as avg_risk_score,
                ROUND(
                    (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                ) as approval_rate
            FROM decisions 
            WHERE decision_timestamp >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY DATE(decision_timestamp)
            ORDER BY decision_date DESC
            """
            
            results = self.db.executeQuery(query, (days,), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting decision trends: {e}")
            raise
    
    def getHighRiskDecisions(self, risk_threshold: float = 0.7, limit: int = 50) -> List[Dict[str, Any]]:
        """Get high-risk decisions"""
        try:
            query = """
            SELECT 
                d.*,
                a.person_age,
                a.person_income,
                a.loan_amnt,
                a.loan_grade,
                a.person_home_ownership
            FROM decisions d
            JOIN applications a ON d.application_id = a.application_id
            WHERE d.risk_score >= %s
            ORDER BY d.risk_score DESC, d.decision_timestamp DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (risk_threshold, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting high-risk decisions: {e}")
            raise
    
    def updateDecision(self, decision_id: str, update_data: Dict[str, Any]) -> bool:
        """Update decision record"""
        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in update_data.items():
                if key in ['decision_reason', 'underwriter_notes']:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            
            if not set_clauses:
                return False
            
            query = f"""
            UPDATE decisions 
            SET {', '.join(set_clauses)}, decision_timestamp = CURRENT_TIMESTAMP
            WHERE decision_id = %s
            """
            
            values.append(decision_id)
            rows_affected = self.db.executeQuery(query, tuple(values))
            
            if rows_affected > 0:
                logger.info(f"Updated decision: {decision_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating decision: {e}")
            raise
    
    def getDecisionStats(self) -> Dict[str, Any]:
        """Get overall decision statistics"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_decisions,
                SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) as approved_count,
                SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejected_count,
                SUM(CASE WHEN decision = 'MANUAL_REVIEW' THEN 1 ELSE 0 END) as manual_review_count,
                AVG(risk_score) as avg_risk_score,
                MIN(risk_score) as min_risk_score,
                MAX(risk_score) as max_risk_score,
                COUNT(DISTINCT underwriter_id) as active_underwriters,
                ROUND(
                    (SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)) * 100, 2
                ) as overall_approval_rate
            FROM decisions
            """
            
            result = self.db.executeQuery(query, fetch='one')
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting decision stats: {e}")
            raise

# Global repository instance
decisions_repo = DecisionsRepository()

def getDecisionsRepository():
    """Get decisions repository instance"""
    return decisions_repo

if __name__ == "__main__":
    # Test the repository
    logging.basicConfig(level=logging.INFO)
    
    repo = DecisionsRepository()
    
    # Test getting stats
    try:
        stats = repo.getDecisionStats()
        print(f"Decision stats: {stats}")
    except Exception as e:
        print(f"Error testing repository: {e}")
