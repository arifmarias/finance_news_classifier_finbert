# sentiment_analyzer.py
import logging
import time
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import config
from .models import SentimentType

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self._init_finbert()
        logger.info("Initialized sentiment analyzer with FinBERT")

    def _init_finbert(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(config.FINBERT_TOKENIZER_PATH)
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(config.FINBERT_MODEL_PATH)
            )
            
            # Move model to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"FinBERT initialized successfully (using {self.device})")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT: {str(e)}")
            raise

    def analyze_sentiment(self, text: str) -> Tuple[str, float, str]:
        """Analyze sentiment of text"""
        try:
            text = text.strip()
            if not text:
                return SentimentType.NEUTRAL.value, 0.0, "Empty text"

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.FINBERT_MAX_LENGTH,
                padding=True
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(scores, dim=1)
                confidence = scores[0][prediction].item()
            
            # Map prediction to sentiment (FinBERT: 0=neutral, 1=positive, 2=negative)
            sentiment_map = {
                0: SentimentType.NEUTRAL.value,
                1: SentimentType.POSITIVE.value,
                2: SentimentType.NEGATIVE.value
            }
            
            sentiment = sentiment_map[prediction.item()]
            
            # Create detailed response
            all_scores = scores[0].tolist()
            raw_response = (
                f"Sentiment scores - "
                f"Neutral: {all_scores[0]:.3f}, "
                f"Positive: {all_scores[1]:.3f}, "
                f"Negative: {all_scores[2]:.3f}"
            )
            
            return sentiment, confidence, raw_response

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return SentimentType.NEUTRAL.value, 0.0, str(e)