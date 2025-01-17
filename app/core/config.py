from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Event Analytics Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Sentiment Analysis Settings
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    SENTIMENT_BATCH_SIZE: int = 32
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()