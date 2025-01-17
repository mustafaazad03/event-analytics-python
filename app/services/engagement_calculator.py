from typing import List, Dict, Union
import numpy as np
from datetime import datetime, timedelta
from app.core.logger import setup_logger

logger = setup_logger("engagement_calculator")

class EngagementCalculator:
    def __init__(self):
        self.engagement_weights = {
            'chat': 1.0,
            'question': 2.0,
            'poll_response': 1.5,
            'reaction': 0.5
        }
        
    def calculate_user_engagement(
        self,
        interactions: List[Dict],
        session_duration: timedelta,
        total_participants: int
    ) -> Dict[str, float]:
        """
        Calculate engagement metrics for a session
        """
        try:
            if not interactions or total_participants == 0:
                return {
                    'engagement_score': 0.0,
                    'interaction_rate': 0.0,
                    'sentiment_score': 0.0
                }

            # Calculate base metrics
            total_interactions = len(interactions)
            interaction_rate = total_interactions / total_participants
            
            # Calculate weighted engagement score
            weighted_sum = sum(
                self.engagement_weights.get(interaction['type'], 0)
                for interaction in interactions
            )
            
            # Normalize by session duration (in hours)
            duration_hours = session_duration.total_seconds() / 3600
            if duration_hours > 0:
                engagement_score = (weighted_sum / duration_hours) / total_participants
            else:
                engagement_score = 0
                
            # Calculate average sentiment
            sentiments = [
                interaction.get('sentiment', 0)
                for interaction in interactions
                if interaction.get('sentiment') is not None
            ]
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'engagement_score': round(engagement_score, 2),
                'interaction_rate': round(interaction_rate, 2),
                'sentiment_score': round(avg_sentiment, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating engagement: {str(e)}")
            raise
            
    def calculate_time_based_metrics(
        self,
        interactions: List[Dict],
        session_start: datetime,
        session_end: datetime,
        interval_minutes: int = 5
    ) -> List[Dict[str, Union[datetime, float]]]:
        """
        Calculate engagement metrics over time intervals
        """
        try:
            if not interactions:
                return []
                
            # Create time intervals
            current_time = session_start
            intervals = []
            
            while current_time < session_end:
                interval_end = min(
                    current_time + timedelta(minutes=interval_minutes),
                    session_end
                )
                
                # Filter interactions for current interval
                interval_interactions = [
                    interaction for interaction in interactions
                    if current_time <= interaction['timestamp'] < interval_end
                ]
                
                # Calculate metrics for interval
                metrics = {
                    'timestamp': current_time,
                    'interaction_count': len(interval_interactions),
                    'unique_users': len(set(
                        interaction['user_id']
                        for interaction in interval_interactions
                    )),
                    'avg_sentiment': np.mean([
                        interaction.get('sentiment', 0)
                        for interaction in interval_interactions
                        if interaction.get('sentiment') is not None
                    ]) if interval_interactions else 0
                }
                
                intervals.append(metrics)
                current_time = interval_end
                
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating time-based metrics: {str(e)}")
            raise