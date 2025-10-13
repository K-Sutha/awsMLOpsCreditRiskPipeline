#!/usr/bin/env python3
"""
PostgreSQL Connection Module
Production-ready connection management for AWS RDS PostgreSQL
"""

import psycopg2
import psycopg2.extras
import boto3
import json
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PostgreSQLConnection:
    """Production-ready PostgreSQL connection manager"""
    
    def __init__(self):
        self.connection_params = self._getConnectionParams()
    
    def _getConnectionParams(self) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters from AWS Secrets Manager or environment"""
        
        # Try AWS Secrets Manager first (production)
        try:
            if os.getenv('AWS_REGION'):
                secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION'))
                secret = secrets_client.get_secret_value(SecretId='credit-risk-rds-credentials')
                credentials = json.loads(secret['SecretString'])
                
                return {
                    'host': credentials['host'],
                    'database': credentials['dbname'],
                    'user': credentials['username'],
                    'password': credentials['password'],
                    'port': credentials['port']
                }
        except Exception as e:
            logger.warning(f"Could not fetch from AWS Secrets Manager: {e}")
        
        # Fallback to environment variables (development)
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'credit_risk_platform'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    @contextmanager
    def getConnection(self):
        """Get database connection with automatic cleanup"""
        connection = None
        try:
            connection = psycopg2.connect(
                **self.connection_params,
                cursor_factory=psycopg2.extras.RealDictCursor,
                connect_timeout=10,
                application_name='credit-risk-platform'
            )
            connection.autocommit = False
            logger.debug("Database connection established")
            yield connection
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
                logger.debug("Database connection closed")
    
    @contextmanager
    def getCursor(self):
        """Get database cursor with automatic cleanup"""
        with self.getConnection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
            except psycopg2.Error as e:
                logger.error(f"Database cursor error: {e}")
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def executeQuery(self, query: str, params: tuple = None, fetch: str = 'none') -> Any:
        """Execute a SQL query with proper error handling"""
        try:
            with self.getCursor() as cursor:
                cursor.execute(query, params)
                
                if fetch == 'one':
                    return cursor.fetchone()
                elif fetch == 'all':
                    return cursor.fetchall()
                elif fetch == 'none':
                    cursor.connection.commit()
                    return cursor.rowcount
                else:
                    raise ValueError("fetch must be 'one', 'all', or 'none'")
                    
        except psycopg2.Error as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def executeBatch(self, queries: list) -> list:
        """Execute multiple queries in a transaction"""
        results = []
        with self.getConnection() as connection:
            cursor = connection.cursor()
            try:
                for query, params in queries:
                    cursor.execute(query, params)
                    results.append(cursor.rowcount)
                connection.commit()
                logger.info(f"Executed {len(queries)} queries successfully")
                return results
            except psycopg2.Error as e:
                logger.error(f"Batch execution error: {e}")
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def testConnection(self) -> bool:
        """Test database connection"""
        try:
            with self.getCursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("Database connection test successful")
                return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def getConnectionInfo(self) -> Dict[str, Any]:
        """Get connection information (without sensitive data)"""
        return {
            'host': self.connection_params['host'],
            'database': self.connection_params['database'],
            'port': self.connection_params['port'],
            'user': self.connection_params['user']
        }

# Global connection instance
db_connection = PostgreSQLConnection()

def getPostgresConnection():
    """Get PostgreSQL connection instance"""
    return db_connection

def getConnection():
    """Get database connection context manager"""
    return db_connection.getConnection()

def getCursor():
    """Get database cursor context manager"""
    return db_connection.getCursor()

def executeQuery(query: str, params: tuple = None, fetch: str = 'none'):
    """Execute a SQL query"""
    return db_connection.executeQuery(query, params, fetch)

def testConnection() -> bool:
    """Test database connection"""
    return db_connection.testConnection()

if __name__ == "__main__":
    # Test the connection
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PostgreSQL connection...")
    if testConnection():
        print("✅ Connection successful!")
        print(f"Connection info: {db_connection.getConnectionInfo()}")
    else:
        print("❌ Connection failed!")
