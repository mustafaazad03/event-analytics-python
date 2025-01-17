from fastapi import APIRouter, HTTPException, Depends
from typing import List
from models.sentiment import (
    SentimentRequest,
    SentimentResponse,
    EngagementRequest,
    EngagementResponse
)
from services.sentiment_analyzer import SentimentAnalyzer
from services.engagement_calculator import EngagementCalculator
from core.logger import setup_logger

router = APIRouter()
logger = setup_logger("sentiment_endpoint")

# Dependencies
def get_sentiment_analyzer():
    return SentimentAnalyzer()

def get_engagement_calculator():
    return EngagementCalculator()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    try:
        texts = [interaction.text for interaction in request.interactions]
        results = analyzer.analyze_batch(texts)
        
        # Combine results with interaction metadata
        for result, interaction in zip(results, request.interactions):
            result.update({
                "user_id": interaction.user_id,
                "session_id": interaction.session_id,
                "type": interaction.type,
                "timestamp": interaction.timestamp
            })
        
        return SentimentResponse(results=results)
    except Exception as e:
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/engagement", response_model=EngagementResponse)
async def calculate_engagement(
    request: EngagementRequest,
    calculator: EngagementCalculator = Depends(get_engagement_calculator)
):
    try:
        # Calculate session duration
        duration = request.end_time - request.start_time
        
        # Calculate overall engagement metrics
        engagement_metrics = calculator.calculate_user_engagement(
            request.interactions,
            duration,
            request.total_participants
        )
        
        # Calculate time-based metrics
        time_series = calculator.calculate_time_based_metrics(
            request.interactions,
            request.start_time,
            request.end_time
        )
        
        return EngagementResponse(
            **engagement_metrics,
            time_series=time_series
        )
    except Exception as e:
        logger.error(f"Error in engagement calculation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
