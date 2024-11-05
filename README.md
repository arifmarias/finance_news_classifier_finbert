# Financial News Analysis System
A comprehensive system for analyzing financial news articles using Llama2 for category classification and FinBERT for sentiment analysis.

[X] Author: Mohammed Arif

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation Guide](#installation-guide)
4. [Detailed Component Guide](#detailed-component-guide)
5. [Usage Guide](#usage-guide)
6. [Training Guide](#training-guide)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## Overview

### Purpose
This system analyzes financial news articles to:
- Classify articles into specific financial categories
- Determine the sentiment (positive/negative/neutral)
- Provide confidence scores for predictions
- Generate detailed analysis reports

### Features
- Multi-model approach using Llama2 and FinBERT
- Batch processing capability
- Detailed logging and statistics
- Confidence scoring
- Error handling and recovery
- Progress tracking

## Project Structure
```
financial_news_classifier/
├── models/                  # Model storage directory
│   └── finbert/            # FinBERT model files
│       ├── model/          # Main model files
│       │   ├── config.json
│       │   ├── pytorch_model.bin
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer_config.json
│       │   └── vocab.txt
│       └── tokenizer/      # Tokenizer files
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── src/                    # Source code directory
│   ├── __init__.py
│   ├── models.py          # Data models
│   ├── config.py          # Configuration
│   ├── category_classifier.py  # Llama2 classifier
│   ├── sentiment_analyzer.py   # FinBERT analyzer
│   ├── news_analyzer.py        # Combined analyzer
│   └── processor.py            # Batch processor
├── data/                   # Data directory
│   └── news_articles.csv   # Input data
├── logs/                   # Log files
├── tests/                  # Test files
├── requirements.txt        # Dependencies
├── setup.py               # Setup script
└── main.py                # Main script
```

## Installation Guide

### Prerequisites
1. Python 3.8 or higher
2. CUDA-capable GPU (optional but recommended)
3. 8GB RAM minimum (16GB recommended)
4. 2GB free disk space for models

### Step-by-Step Installation

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama**
- Visit [Ollama's website](https://ollama.ai)
- Follow installation instructions
- Run: `ollama pull llama2`

4. **Setup FinBERT**
- Create directories:
```bash
mkdir -p models/finbert/{model,tokenizer}
```
- Download model files from Hugging Face (see Model Setup section)

## Detailed Component Guide

### 1. Configuration (config.py)
**Purpose**: Centralizes all configuration settings
```python
# Key configurations:
OLLAMA_URL: str = "http://localhost:11434/api/generate"  # Ollama API endpoint
MODEL_NAME: str = "llama2"                               # Model for classification
FINBERT_MODEL_PATH: Path = Path("models/finbert/model")  # FinBERT path
```
**Usage**: Controls system behavior, paths, and model parameters

### 2. Data Models (models.py)
**Purpose**: Defines core data structures
```python
class NewsCategory(str, Enum):
    # Financial categories
class SentimentType(str, Enum):
    # Sentiment types
class NewsAnalysis(BaseModel):
    # Analysis result structure
```
**Usage**: Ensures type safety and data validation

### 3. Category Classifier (category_classifier.py)
**Purpose**: Classifies articles using Llama2
**Key Features**:
- Connects to Ollama API
- Generates structured prompts
- Normalizes categories
- Provides confidence scores

### 4. Sentiment Analyzer (sentiment_analyzer.py)
**Purpose**: Analyzes sentiment using FinBERT
**Key Features**:
- Local model inference
- GPU acceleration
- Confidence scoring
- Error handling

### 5. News Analyzer (news_analyzer.py)
**Purpose**: Combines category and sentiment analysis
**Key Features**:
- Orchestrates both analyzers
- Combines results
- Provides unified interface

### 6. Processor (processor.py)
**Purpose**: Handles batch processing
**Key Features**:
- CSV validation
- Progress tracking
- Statistics generation
- Error recovery

## Usage Guide

### Basic Usage
```bash
python main.py
```

### Custom Input/Output
```bash
python main.py input.csv output.csv
```

### Sample Input CSV Format
```csv
Headline,Date,Article
"Company X Reports Growth",2024-01-01,"Article text here..."
```

## Training Guide

### Step 1: Understanding the Pipeline
1. Article input
2. Category classification (Llama2)
3. Sentiment analysis (FinBERT)
4. Result combination
5. Output generation

### Step 2: Model Setup
1. Llama2 Setup:
```bash
ollama pull llama2
ollama run llama2  # Test model
```

2. FinBERT Setup:
- Download required files
- Place in correct directories
- Run verification:
```bash
python test_setup.py
```

### Step 3: Testing Components
1. Test category classification:
```python
from src.category_classifier import CategoryClassifier
classifier = CategoryClassifier()
result = classifier.classify("Sample article text")
```

2. Test sentiment analysis:
```python
from src.sentiment_analyzer import SentimentAnalyzer
analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment("Sample article text")
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
```
Solution: Verify Ollama is running:
ollama list
```

2. **FinBERT Loading Error**
```
Solution: Check model files:
python test_setup.py
```

3. **CUDA/GPU Issues**
```python
# Check GPU availability:
import torch
print(torch.cuda.is_available())
```

## Advanced Topics

### 1. Custom Categories
Modify `NewsCategory` in models.py:
```python
class NewsCategory(str, Enum):
    # Add custom categories
```

### 2. Performance Tuning
Adjust in config.py:
```python
BATCH_SIZE: int = 10        # Processing batch size
MAX_TOKENS: int = 2048      # Token limit
FINBERT_MAX_LENGTH: int = 512  # Max input length
```

### 3. Custom Logging
Modify logging configuration in main.py:
```python
def setup_logging():
    # Customize logging
```

## Contributing
1. Fork the repository
2. Create feature branch
3. Submit pull request