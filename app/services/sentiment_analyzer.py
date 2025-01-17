from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import torch
import numpy as np
from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("sentiment_analyzer")
settings = get_settings()

class SentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.SENTIMENT_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.SENTIMENT_MODEL
            ).to(self.device)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a batch of texts
        """
        try:
            results = self.sentiment_pipeline(texts, batch_size=settings.SENTIMENT_BATCH_SIZE)
            processed_results = []
            
            for text, result in zip(texts, results):
                score = result['score']
                label = result['label']
                
                # Convert score to a -1 to 1 range if negative sentiment
                normalized_score = score if label == 'POSITIVE' else -score
                
                processed_results.append({
                    'text': text,
                    'sentiment_score': normalized_score,
                    'confidence': score,
                    'label': label
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            raise

    def analyze_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment for a single text
        """
        try:
            return self.analyze_batch([text])[0]
        except Exception as e:
            logger.error(f"Error in single text sentiment analysis: {str(e)}")
            raise