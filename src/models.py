# models.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class NewsCategory(str, Enum):
    OIL_AND_GAS = "oil_and_gas"
    AGRICULTURE = "agriculture"
    HOUSING = "housing"
    BANKING = "banking"
    STOCK_MARKET = "stock_market"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    COMMODITIES = "commodities"
    OTHERS = "others"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsAnalysis(BaseModel):
    category: NewsCategory
    sentiment: SentimentType
    success: bool
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None