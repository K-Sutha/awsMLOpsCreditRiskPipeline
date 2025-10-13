import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
eventbridge = boto3.client('events')
dynamodb = boto3.resource('dynamodb')

def updatePortfolio(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Updates portfolio analytics and metrics based on processed applications.
    
    Args:
        event: EventBridge event containing processed application data
        context: Lambda context
    
    Returns:
        Dict containing portfolio update result
    """
    try:
        logger.info(f"Received portfolio update request: {event}")
        
        # Extract application data from EventBridge event
        if 'detail' in event:
            application_data = json.loads(event['detail'])
        else:
            application_data = event
        
        # Update real-time portfolio metrics
        portfolio_updated = update_realtime_metrics(application_data)
        
        # Calculate aggregate statistics
        aggregate_stats = calculate_aggregate_statistics(application_data)
        
        # Update portfolio analytics table
        analytics_updated = update_portfolio_analytics(application_data, aggregate_stats)
        
        # Generate portfolio summary
        portfolio_summary = generate_portfolio_summary(application_data, aggregate_stats)
        
        logger.info(f"Portfolio updated for application {application_data.get('application_id', 'Unknown')}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Portfolio updated successfully',
                'portfolio_updated': portfolio_updated,
                'analytics_updated': analytics_updated,
                'summary': portfolio_summary
            })
        }
        
    except Exception as e:
        logger.error(f"Error updating portfolio: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def update_realtime_metrics(application_data: Dict[str, Any]) -> bool:
    """Update real-time portfolio metrics in DynamoDB."""
    try:
        table = dynamodb.Table('portfolioMetrics')
        
        # Get current metrics
        response = table.get_item(
            Key={'metricType': 'realtime', 'metricId': 'current'}
        )
        
        current_metrics = response.get('Item', {
            'totalApplications': 0,
            'approvedApplications': 0,
            'rejectedApplications': 0,
            'pendingReview': 0,
            'totalLoanAmount': 0,
            'averageRiskScore': 0.5,
            'lastUpdated': datetime.utcnow().isoformat()
        })
        
        # Update metrics based on new application
        decision = application_data.get('decision', 'MANUAL_REVIEW')
        loan_amount = float(application_data.get('loan_amnt', 0))
        risk_score = float(application_data.get('risk_score', 0.5))
        
        # Increment counters
        current_metrics['totalApplications'] = current_metrics.get('totalApplications', 0) + 1
        
        if decision == 'AUTO_APPROVE':
            current_metrics['approvedApplications'] = current_metrics.get('approvedApplications', 0) + 1
        elif decision == 'AUTO_REJECT':
            current_metrics['rejectedApplications'] = current_metrics.get('rejectedApplications', 0) + 1
        else:  # MANUAL_REVIEW
            current_metrics['pendingReview'] = current_metrics.get('pendingReview', 0) + 1
        
        # Update loan amounts and risk scores
        current_metrics['totalLoanAmount'] = current_metrics.get('totalLoanAmount', 0) + loan_amount
        
        # Update average risk score (simple moving average)
        total_apps = current_metrics['totalApplications']
        current_avg_risk = current_metrics.get('averageRiskScore', 0.5)
        current_metrics['averageRiskScore'] = ((current_avg_risk * (total_apps - 1)) + risk_score) / total_apps
        
        current_metrics['lastUpdated'] = datetime.utcnow().isoformat()
        current_metrics['ttl'] = int((datetime.utcnow() + timedelta(days=30)).timestamp())
        
        # Save updated metrics
        table.put_item(Item=current_metrics)
        
        logger.info("Real-time portfolio metrics updated")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update real-time metrics: {str(e)}")
        return False

def calculate_aggregate_statistics(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate statistics for the portfolio."""
    try:
        # This would typically query the database for historical data
        # For now, we'll calculate based on current application
        
        loan_amount = float(application_data.get('loan_amnt', 0))
        risk_score = float(application_data.get('risk_score', 0.5))
        decision = application_data.get('decision', 'MANUAL_REVIEW')
        
        # Calculate approval rate (simplified - would normally be based on historical data)
        approval_rate = 0.7 if decision == 'AUTO_APPROVE' else 0.3 if decision == 'AUTO_REJECT' else 0.5
        
        # Calculate risk distribution
        risk_distribution = {
            'low_risk': 1 if risk_score < 0.3 else 0,
            'medium_risk': 1 if 0.3 <= risk_score <= 0.7 else 0,
            'high_risk': 1 if risk_score > 0.7 else 0
        }
        
        # Calculate loan amount distribution
        loan_distribution = {
            'small_loans': 1 if loan_amount < 10000 else 0,
            'medium_loans': 1 if 10000 <= loan_amount <= 50000 else 0,
            'large_loans': 1 if loan_amount > 50000 else 0
        }
        
        return {
            'approval_rate': approval_rate,
            'average_loan_amount': loan_amount,
            'average_risk_score': risk_score,
            'risk_distribution': risk_distribution,
            'loan_distribution': loan_distribution,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate aggregate statistics: {str(e)}")
        return {}

def update_portfolio_analytics(application_data: Dict[str, Any], aggregate_stats: Dict[str, Any]) -> bool:
    """Update portfolio analytics in DynamoDB."""
    try:
        table = dynamodb.Table('portfolioAnalytics')
        
        # Create analytics record
        analytics_record = {
            'applicationId': application_data.get('application_id', 'Unknown'),
            'processedAt': datetime.utcnow().isoformat(),
            'decision': application_data.get('decision', 'MANUAL_REVIEW'),
            'riskScore': float(application_data.get('risk_score', 0.5)),
            'riskLevel': application_data.get('risk_level', 'MEDIUM'),
            'loanAmount': float(application_data.get('loan_amnt', 0)),
            'personIncome': float(application_data.get('person_income', 0)),
            'loanGrade': application_data.get('loan_grade', 'C'),
            'aggregateStats': aggregate_stats,
            'ttl': int((datetime.utcnow() + timedelta(days=90)).timestamp())  # 90 days retention
        }
        
        # Save analytics record
        table.put_item(Item=analytics_record)
        
        logger.info("Portfolio analytics updated")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update portfolio analytics: {str(e)}")
        return False

def generate_portfolio_summary(application_data: Dict[str, Any], aggregate_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of portfolio metrics."""
    
    decision = application_data.get('decision', 'MANUAL_REVIEW')
    risk_score = float(application_data.get('risk_score', 0.5))
    loan_amount = float(application_data.get('loan_amnt', 0))
    
    # Generate insights
    insights = []
    
    if decision == 'AUTO_APPROVE':
        insights.append("Low-risk application approved automatically")
    elif decision == 'AUTO_REJECT':
        insights.append("High-risk application rejected automatically")
    else:
        insights.append("Medium-risk application requires manual review")
    
    if risk_score < 0.3:
        insights.append("Excellent credit profile")
    elif risk_score > 0.7:
        insights.append("High default risk detected")
    
    if loan_amount > 50000:
        insights.append("Large loan amount - increased scrutiny applied")
    
    return {
        'application_id': application_data.get('application_id', 'Unknown'),
        'decision': decision,
        'risk_score': risk_score,
        'loan_amount': loan_amount,
        'approval_rate': aggregate_stats.get('approval_rate', 0.5),
        'insights': insights,
        'timestamp': datetime.utcnow().isoformat()
    }

def calculate_portfolio_health_score(aggregate_stats: Dict[str, Any]) -> float:
    """Calculate overall portfolio health score."""
    try:
        approval_rate = aggregate_stats.get('approval_rate', 0.5)
        avg_risk_score = aggregate_stats.get('average_risk_score', 0.5)
        
        # Portfolio health is higher with higher approval rates and lower risk scores
        health_score = (approval_rate * 0.6) + ((1 - avg_risk_score) * 0.4)
        
        return min(1.0, max(0.0, health_score))
        
    except Exception as e:
        logger.error(f"Failed to calculate portfolio health score: {str(e)}")
        return 0.5

def get_portfolio_trends() -> Dict[str, Any]:
    """Get portfolio trends over time (simplified implementation)."""
    try:
        # This would typically query historical data
        # For now, return mock trends
        
        return {
            'approval_rate_trend': 'stable',
            'risk_score_trend': 'stable',
            'loan_volume_trend': 'increasing',
            'default_rate_trend': 'decreasing',
            'calculated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio trends: {str(e)}")
        return {}

def send_to_eventbridge(event_type: str, data: Dict[str, Any]) -> None:
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
