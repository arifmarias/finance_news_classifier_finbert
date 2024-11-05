# main.py
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import config
from src.processor import NewsProcessor

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"processing_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def verify_environment():
    """Verify the environment setup"""
    logger = logging.getLogger(__name__)
    
    # Check directories
    logger.info("Verifying environment setup...")
    
    # Check Ollama
    try:
        from src.category_classifier import CategoryClassifier
        classifier = CategoryClassifier()
        classifier._verify_ollama_connection()
        logger.info("Ollama connection verified")
    except Exception as e:
        logger.error(f"Ollama verification failed: {str(e)}")
        return False
    
    # Check FinBERT files
    model_path = config.FINBERT_MODEL_PATH
    tokenizer_path = config.FINBERT_TOKENIZER_PATH
    
    required_files = {
        model_path: ['config.json', 'pytorch_model.bin', 'special_tokens_map.json'],
        tokenizer_path: ['special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt']
    }
    
    for path, files in required_files.items():
        for file in files:
            file_path = path / file
            if not file_path.exists():
                logger.error(f"Missing required file: {file_path}")
                return False
    
    logger.info("FinBERT files verified")
    return True

def process_articles(input_file: Path = None, output_file: Path = None):
    """Process articles from input CSV file"""
    logger = logging.getLogger(__name__)
    
    try:
        processor = NewsProcessor()
        
        if input_file:
            logger.info(f"Using custom input file: {input_file}")
        if output_file:
            logger.info(f"Using custom output file: {output_file}")
        
        success = processor.process_csv_file(input_file, output_file)
        
        if success:
            logger.info("Processing completed successfully")
        else:
            logger.error("Processing completed with errors")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        # Create necessary directories
        config.create_directories()
        
        # Setup logging
        log_file = setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting financial news analysis process")
        logger.info(f"Log file: {log_file}")
        
        # Verify environment
        if not verify_environment():
            logger.error("Environment verification failed")
            return
        
        # Process command line arguments
        input_file = None
        output_file = None
        
        if len(sys.argv) > 1:
            input_file = Path(sys.argv[1])
        if len(sys.argv) > 2:
            output_file = Path(sys.argv[2])
        
        # Process articles
        process_articles(input_file, output_file)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()