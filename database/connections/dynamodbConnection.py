#!/usr/bin/env python3
"""
DynamoDB Connection Module
Production-ready connection management for AWS DynamoDB
"""

import boto3
import json
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, Any, Optional, List
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DynamoDBConnection:
    """Production-ready DynamoDB connection manager"""
    
    def __init__(self, region_name: str = None):
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.dynamodb = self._createDynamoDbClient()
        self.client = self._createDynamoDbClient()
    
    def _createDynamoDbClient(self):
        """Create DynamoDB client with proper configuration"""
        try:
            # Use IAM role when running on AWS (EC2, Lambda, ECS)
            # Falls back to AWS credentials when running locally
            dynamodb = boto3.resource(
                'dynamodb',
                region_name=self.region_name,
                config=boto3.session.Config(
                    max_pool_connections=50,
                    retries={'max_attempts': 3}
                )
            )
            return dynamodb
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to create DynamoDB client: {e}")
            raise
    
    def getTable(self, table_name: str):
        """Get DynamoDB table reference"""
        try:
            table = self.dynamodb.Table(table_name)
            # Test table access
            table.load()
            logger.debug(f"Successfully accessed table: {table_name}")
            return table
        except ClientError as e:
            logger.error(f"Error accessing table {table_name}: {e}")
            raise
    
    def putItem(self, table_name: str, item: Dict[str, Any]) -> bool:
        """Put item into DynamoDB table"""
        try:
            table = self.getTable(table_name)
            table.put_item(Item=item)
            logger.debug(f"Successfully put item into {table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error putting item into {table_name}: {e}")
            return False
    
    def getItem(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get item from DynamoDB table"""
        try:
            table = self.getTable(table_name)
            response = table.get_item(Key=key)
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Error getting item from {table_name}: {e}")
            return None
    
    def updateItem(self, table_name: str, key: Dict[str, Any], 
                   update_expression: str, expression_values: Dict[str, Any],
                   expression_names: Dict[str, str] = None) -> bool:
        """Update item in DynamoDB table"""
        try:
            table = self.getTable(table_name)
            
            update_params = {
                'Key': key,
                'UpdateExpression': update_expression,
                'ExpressionAttributeValues': expression_values
            }
            
            if expression_names:
                update_params['ExpressionAttributeNames'] = expression_names
            
            table.update_item(**update_params)
            logger.debug(f"Successfully updated item in {table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error updating item in {table_name}: {e}")
            return False
    
    def deleteItem(self, table_name: str, key: Dict[str, Any]) -> bool:
        """Delete item from DynamoDB table"""
        try:
            table = self.getTable(table_name)
            table.delete_item(Key=key)
            logger.debug(f"Successfully deleted item from {table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting item from {table_name}: {e}")
            return False
    
    def queryTable(self, table_name: str, key_condition_expression: str,
                   expression_values: Dict[str, Any], 
                   index_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Query DynamoDB table"""
        try:
            table = self.getTable(table_name)
            
            query_params = {
                'KeyConditionExpression': key_condition_expression,
                'ExpressionAttributeValues': expression_values
            }
            
            if index_name:
                query_params['IndexName'] = index_name
            if limit:
                query_params['Limit'] = limit
            
            response = table.query(**query_params)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error querying table {table_name}: {e}")
            return []
    
    def scanTable(self, table_name: str, filter_expression: str = None,
                  expression_values: Dict[str, Any] = None, limit: int = None) -> List[Dict[str, Any]]:
        """Scan DynamoDB table"""
        try:
            table = self.getTable(table_name)
            
            scan_params = {}
            if filter_expression:
                scan_params['FilterExpression'] = filter_expression
            if expression_values:
                scan_params['ExpressionAttributeValues'] = expression_values
            if limit:
                scan_params['Limit'] = limit
            
            response = table.scan(**scan_params)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error scanning table {table_name}: {e}")
            return []
    
    def batchWriteItems(self, table_name: str, items: List[Dict[str, Any]]) -> bool:
        """Batch write items to DynamoDB table"""
        try:
            table = self.getTable(table_name)
            
            with table.batch_writer() as batch:
                for item in items:
                    batch.put_item(Item=item)
            
            logger.info(f"Successfully batch wrote {len(items)} items to {table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error batch writing to {table_name}: {e}")
            return False
    
    def generateTtl(self, hours: int = 24) -> int:
        """Generate TTL timestamp (hours from now)"""
        return int((datetime.utcnow() + timedelta(hours=hours)).timestamp())
    
    def createCacheKey(self, *args) -> str:
        """Create cache key from arguments"""
        return "_".join(str(arg) for arg in args)
    
    def testConnection(self) -> bool:
        """Test DynamoDB connection"""
        try:
            # Try to list tables to test connection
            response = self.client.list_tables()
            logger.info("DynamoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"DynamoDB connection test failed: {e}")
            return False
    
    def getConnectionInfo(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'region': self.region_name,
            'endpoint': self.client._endpoint.host
        }

# Global connection instance
dynamodb_connection = DynamoDBConnection()

def getDynamoDbClient():
    """Get DynamoDB client instance"""
    return dynamodb_connection.client

def getDynamoDbTable(table_name: str):
    """Get DynamoDB table reference"""
    return dynamodb_connection.getTable(table_name)

def putItem(table_name: str, item: Dict[str, Any]) -> bool:
    """Put item into DynamoDB table"""
    return dynamodb_connection.putItem(table_name, item)

def getItem(table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get item from DynamoDB table"""
    return dynamodb_connection.getItem(table_name, key)

def updateItem(table_name: str, key: Dict[str, Any], 
               update_expression: str, expression_values: Dict[str, Any],
               expression_names: Dict[str, str] = None) -> bool:
    """Update item in DynamoDB table"""
    return dynamodb_connection.updateItem(table_name, key, update_expression, 
                                         expression_values, expression_names)

def deleteItem(table_name: str, key: Dict[str, Any]) -> bool:
    """Delete item from DynamoDB table"""
    return dynamodb_connection.deleteItem(table_name, key)

def queryTable(table_name: str, key_condition_expression: str,
               expression_values: Dict[str, Any], 
               index_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
    """Query DynamoDB table"""
    return dynamodb_connection.queryTable(table_name, key_condition_expression,
                                         expression_values, index_name, limit)

def testConnection() -> bool:
    """Test DynamoDB connection"""
    return dynamodb_connection.testConnection()

if __name__ == "__main__":
    # Test the connection
    logging.basicConfig(level=logging.INFO)
    
    print("Testing DynamoDB connection...")
    if testConnection():
        print("✅ Connection successful!")
        print(f"Connection info: {dynamodb_connection.getConnectionInfo()}")
    else:
        print("❌ Connection failed!")
