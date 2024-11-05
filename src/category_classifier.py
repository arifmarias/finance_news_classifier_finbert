# category_classifier.py
import logging
import time
from typing import Optional, Dict, Any, Tuple
import requests
import json

from .config import config
from .models import NewsCategory

logger = logging.getLogger(__name__)

class OllamaConnectionError(Exception):
    """Raised when there are issues connecting to Ollama"""
    pass

class CategoryClassifier:
    def __init__(self):
        self.api_url = config.OLLAMA_URL
        self.model_name = config.MODEL_NAME
        self._verify_ollama_connection()
        logger.info(f"Initialized category classifier with model: {self.model_name}")

    def _verify_ollama_connection(self) -> None:
        """Verify that Ollama is running and accessible"""
        try:
            response = requests.get(
                self.api_url.replace("/generate", "/version"),
                timeout=5
            )
            response.raise_for_status()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            raise OllamaConnectionError(
                "Could not connect to Ollama. Ensure Llama2 is installed with 'ollama pull llama2'. "
                f"Error: {str(e)}"
            )

    def _generate_classification_prompt(self, text: str) -> str:
        """Generate a prompt for category classification using Llama2's format"""
        categories = [
            f"{i+1}. {cat.value}" 
            for i, cat in enumerate(NewsCategory)
        ]
        categories_list = "\n".join(categories)
        
        return f"""<s>[INST] You are a financial news classifier. Analyze this article and classify it into exactly ONE category.

Available categories:
{categories_list}

Rules:
1. Read the article carefully and consider the main topic
2. Choose the MOST relevant category
3. Provide your response in JSON format:
   {{"category_number": X, "confidence": Y}}
   where X is the category number (1-9) and Y is your confidence (0-1)

Article:
{text}

Classify this article: [/INST]</s>"""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback"""
        try:
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, response)
            if matches:
                return json.loads(matches[0])
        except Exception as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
        return {}

    def _normalize_category(self, response: str) -> Tuple[str, float]:
        """Normalize category response and extract confidence"""
        try:
            # Parse JSON response
            result = self._parse_json_response(response)
            category_num = result.get('category_number')
            confidence = float(result.get('confidence', 0))
            
            # Number-based category selection
            if category_num:
                categories = list(NewsCategory)
                if 1 <= category_num <= len(categories):
                    return categories[category_num-1].value, confidence

            # Fallback to text analysis
            response = response.lower().strip()
            category_mapping = {
                'stock': ('stock_market', 0.8),
                'equity': ('stock_market', 0.8),
                'oil': ('oil_and_gas', 0.8),
                'gas': ('oil_and_gas', 0.8),
                'bank': ('banking', 0.8),
                'crypto': ('cryptocurrency', 0.8),
                'forex': ('forex', 0.8),
                'commodity': ('commodities', 0.8),
                'agriculture': ('agriculture', 0.8),
                'housing': ('housing', 0.8)
            }

            for key, (category, conf) in category_mapping.items():
                if key in response:
                    return category, conf

            return NewsCategory.OTHERS.value, 0.5

        except Exception as e:
            logger.warning(f"Category normalization error: {str(e)}")
            return NewsCategory.OTHERS.value, 0.0

    def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                    "max_tokens": config.MAX_TOKENS
                }
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == config.MAX_RETRIES - 1:
                    logger.error(f"Failed after {config.MAX_RETRIES} attempts")
                    return None
                time.sleep(config.RETRY_DELAY * (2 ** attempt))
        
        return None

    def classify(self, text: str) -> Tuple[str, float, str]:
        """Classify text into category"""
        try:
            text = text.strip()
            if not text:
                return NewsCategory.OTHERS.value, 0.0, "Empty text"

            # Get category
            category_prompt = self._generate_classification_prompt(text)
            category_response = self._call_ollama(category_prompt)

            if category_response:
                category_raw = category_response.get('response', '').strip()
                category, confidence = self._normalize_category(category_raw)
                return category, confidence, category_raw
            
            return NewsCategory.OTHERS.value, 0.0, "Failed to get response"

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return NewsCategory.OTHERS.value, 0.0, str(e)