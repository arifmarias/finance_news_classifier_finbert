# news_analyzer.py
import logging
import time
from typing import Optional

from .category_classifier import CategoryClassifier
from .sentiment_analyzer import SentimentAnalyzer
from .models import NewsAnalysis, NewsCategory, SentimentType

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        self.category_classifier = CategoryClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("Initialized news analyzer with category and sentiment classifiers")

    def analyze_news(self, news_text: str) -> NewsAnalysis:
        """Analyze news for category and sentiment"""
        start_time = time.time()
        try:
            # Preprocess text
            news_text = news_text.strip()
            if not news_text:
                return NewsAnalysis(
                    category=NewsCategory.OTHERS.value,
                    sentiment=SentimentType.NEUTRAL.value,
                    success=False,
                    raw_response="Empty text",
                    processing_time=0.0,
                    confidence_score=0.0,
                    sentiment_confidence=0.0
                )

            # Get category
            category, cat_confidence, cat_raw = self.category_classifier.classify(news_text)
            
            # Get sentiment
            sentiment, sent_confidence, sent_raw = self.sentiment_analyzer.analyze_sentiment(news_text)

            # Combine results
            raw_response = (
                f"Category Analysis:\n{cat_raw}\n"
                f"Category Confidence: {cat_confidence:.2f}\n\n"
                f"Sentiment Analysis:\n{sent_raw}\n"
                f"Sentiment Confidence: {sent_confidence:.2f}"
            )

            processing_time = time.time() - start_time
            
            return NewsAnalysis(
                category=category,
                sentiment=sentiment,
                success=True,
                raw_response=raw_response,
                processing_time=processing_time,
                confidence_score=cat_confidence,
                sentiment_confidence=sent_confidence
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {str(e)}")
            return NewsAnalysis(
                category=NewsCategory.OTHERS.value,
                sentiment=SentimentType.NEUTRAL.value,
                success=False,
                raw_response=str(e),
                processing_time=processing_time,
                confidence_score=0.0,
                sentiment_confidence=0.0
            )