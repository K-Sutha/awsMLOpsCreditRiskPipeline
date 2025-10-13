#!/usr/bin/env python3
"""
Risk Scores Repository
Data access layer for ML risk scores and predictions
"""

import uuid
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connections.postgresqlConnection import getPostgresConnection
from connections.dynamodbConnection import getDynamoDbTable, putItem, getItem, updateItem

logger = logging.getLogger(__name__)

class RiskScoresRepository:
    """Repository for risk scores and ML predictions data access"""
    
    def __init__(self):
        self.db = getPostgresConnection()
        self.cache_table = getDynamoDbTable('riskCache')
    
    def create(self, application_id: str, risk_data: Dict[str, Any]) -> str:
        """Create new risk score record"""
        try:
            risk_score_id = str(uuid.uuid4())
            
            # Prepare data for PostgreSQL
            pg_data = {
                'risk_score_id': risk_score_id,
                'application_id': application_id,
                'risk_score': risk_data['risk_score'],
                'risk_category': risk_data['risk_category'],
                'model_version': risk_data['model_version'],
                'confidence_score': risk_data.get('confidence_score'),
                'feature_importance': risk_data.get('feature_importance'),
                'processing_time_ms': risk_data.get('processing_time_ms')
            }
            
            # Insert into PostgreSQL
            query = """
            INSERT INTO risk_scores 
            (risk_score_id, application_id, risk_score, risk_category, 
             model_version, confidence_score, feature_importance, processing_time_ms)
            VALUES (%(risk_score_id)s, %(application_id)s, %(risk_score)s, %(risk_category)s,
                    %(model_version)s, %(confidence_score)s, %(feature_importance)s, %(processing_time_ms)s)
            """
            
            self.db.executeQuery(query, tuple(pg_data.values()))
            
            logger.info(f"Created risk score: {risk_score_id} for application: {application_id}")
            return risk_score_id
            
        except Exception as e:
            logger.error(f"Error creating risk score: {e}")
            raise
    
    def getByApplicationId(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get risk score by application ID"""
        try:
            query = """
            SELECT * FROM risk_scores 
            WHERE application_id = %s
            ORDER BY prediction_timestamp DESC
            LIMIT 1
            """
            
            result = self.db.executeQuery(query, (application_id,), fetch='one')
            
            if result:
                return dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting risk score for application {application_id}: {e}")
            raise
    
    def getFromCache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get risk score from cache (fast lookup)"""
        try:
            cached_result = getItem('riskCache', {'cacheKey': cache_key})
            
            if cached_result:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def saveToCache(self, application_data: Dict[str, Any], risk_result: Dict[str, Any], 
                   ttl_hours: int = 168) -> bool:
        """Save risk score to cache"""
        try:
            # Generate cache key from application features
            cache_key = self._generateCacheKey(application_data)
            
            cache_data = {
                'cacheKey': cache_key,
                'riskScore': risk_result['risk_score'],
                'riskCategory': risk_result['risk_category'],
                'modelVersion': risk_result['model_version'],
                'confidenceScore': risk_result.get('confidence_score'),
                'featureImportance': risk_result.get('feature_importance'),
                'createdAt': datetime.utcnow().isoformat(),
                'ttl': self._generateTtl(ttl_hours)
            }
            
            success = putItem('riskCache', cache_data)
            
            if success:
                logger.debug(f"Saved risk score to cache with key: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def getCachedRiskScore(self, application_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get risk score from cache using application data"""
        try:
            cache_key = self._generateCacheKey(application_data)
            return self.getFromCache(cache_key)
            
        except Exception as e:
            logger.error(f"Error getting cached risk score: {e}")
            return None
    
    def getRiskScoresByModelVersion(self, model_version: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get risk scores by model version"""
        try:
            query = """
            SELECT * FROM risk_scores 
            WHERE model_version = %s
            ORDER BY prediction_timestamp DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (model_version, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting risk scores by model version: {e}")
            raise
    
    def getRiskScoreStats(self, date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Get risk score statistics"""
        try:
            if date_range:
                start_date, end_date = date_range
                query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(risk_score) as avg_risk_score,
                    MIN(risk_score) as min_risk_score,
                    MAX(risk_score) as max_risk_score,
                    AVG(processing_time_ms) as avg_processing_time,
                    COUNT(CASE WHEN risk_category = 'LOW' THEN 1 END) as low_risk_count,
                    COUNT(CASE WHEN risk_category = 'MEDIUM' THEN 1 END) as medium_risk_count,
                    COUNT(CASE WHEN risk_category = 'HIGH' THEN 1 END) as high_risk_count
                FROM risk_scores 
                WHERE prediction_timestamp BETWEEN %s AND %s
                """
                result = self.db.executeQuery(query, (start_date, end_date), fetch='one')
            else:
                query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(risk_score) as avg_risk_score,
                    MIN(risk_score) as min_risk_score,
                    MAX(risk_score) as max_risk_score,
                    AVG(processing_time_ms) as avg_processing_time,
                    COUNT(CASE WHEN risk_category = 'LOW' THEN 1 END) as low_risk_count,
                    COUNT(CASE WHEN risk_category = 'MEDIUM' THEN 1 END) as medium_risk_count,
                    COUNT(CASE WHEN risk_category = 'HIGH' THEN 1 END) as high_risk_count
                FROM risk_scores
                """
                result = self.db.executeQuery(query, fetch='one')
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting risk score stats: {e}")
            raise
    
    def getModelPerformanceMetrics(self, model_version: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            query = """
            SELECT 
                COUNT(*) as prediction_count,
                AVG(risk_score) as avg_risk_score,
                STDDEV(risk_score) as risk_score_stddev,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time_ms) as avg_processing_time
            FROM risk_scores 
            WHERE model_version = %s
            """
            
            result = self.db.executeQuery(query, (model_version,), fetch='one')
            
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting model performance metrics: {e}")
            raise
    
    def getHighRiskApplications(self, threshold: float = 0.7, limit: int = 50) -> List[Dict[str, Any]]:
        """Get applications with high risk scores"""
        try:
            query = """
            SELECT 
                rs.*,
                a.person_age,
                a.person_income,
                a.loan_amnt,
                a.loan_grade
            FROM risk_scores rs
            JOIN applications a ON rs.application_id = a.application_id
            WHERE rs.risk_score >= %s
            ORDER BY rs.risk_score DESC
            LIMIT %s
            """
            
            results = self.db.executeQuery(query, (threshold, limit), fetch='all')
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting high risk applications: {e}")
            raise
    
    def updateCacheTtl(self, cache_key: str, new_ttl_hours: int) -> bool:
        """Update cache TTL"""
        try:
            update_expression = "SET ttl = :ttl"
            expression_values = {':ttl': self._generateTtl(new_ttl_hours)}
            
            return updateItem('riskCache', {'cacheKey': cache_key},
                            update_expression, expression_values)
            
        except Exception as e:
            logger.error(f"Error updating cache TTL: {e}")
            return False
    
    def clearCache(self, cache_key_pattern: str = None) -> int:
        """Clear cache entries (be careful with this)"""
        try:
            # This is a simplified version - in production, you'd want more sophisticated cache clearing
            logger.warning("Cache clear operation requested - implement carefully")
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def _generateCacheKey(self, application_data: Dict[str, Any]) -> str:
        """Generate cache key from application features"""
        try:
            # Create a hash of key features for cache key
            key_features = [
                str(application_data.get('person_age', '')),
                str(application_data.get('person_income', '')),
                str(application_data.get('person_home_ownership', '')),
                str(application_data.get('person_emp_length', '')),
                str(application_data.get('loan_intent', '')),
                str(application_data.get('loan_grade', '')),
                str(application_data.get('loan_amnt', ''))
            ]
            
            key_string = "_".join(key_features)
            cache_key = f"risk_{hashlib.md5(key_string.encode()).hexdigest()[:16]}"
            
            return cache_key
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"risk_{uuid.uuid4().hex[:16]}"
    
    def _generateTtl(self, hours: int) -> int:
        """Generate TTL timestamp"""
        return int((datetime.utcnow() + timedelta(hours=hours)).timestamp())

# Global repository instance
risk_scores_repo = RiskScoresRepository()

def getRiskScoresRepository():
    """Get risk scores repository instance"""
    return risk_scores_repo

if __name__ == "__main__":
    # Test the repository
    logging.basicConfig(level=logging.INFO)
    
    repo = RiskScoresRepository()
    
    # Test getting stats
    try:
        stats = repo.getRiskScoreStats()
        print(f"Risk score stats: {stats}")
    except Exception as e:
        print(f"Error testing repository: {e}")
