from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class InteractionBase(BaseModel):
    text: str
    type: str
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SentimentRequest(BaseModel):
    interactions: List[InteractionBase]

class SentimentResponse(BaseModel):
    results: List[Dict]
    
class EngagementRequest(BaseModel):
    session_id: str
    start_time: datetime
    end_time: datetime
    total_participants: int
    interactions: List[Dict]

class EngagementResponse(BaseModel):
    engagement_score: float
    interaction_rate: float
    sentiment_score: float
    time_series: Optional[List[Dict]] = None