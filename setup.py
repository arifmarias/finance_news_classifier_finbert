# test_setup.py
import logging
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

def test_ollama():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        response.raise_for_status()
        print("✅ Ollama connection successful")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        return False

def test_finbert():
    """Test FinBERT setup"""
    print("\nTesting FinBERT setup...")
    try:
        model_path = Path("models/finbert/model")
        tokenizer_path = Path("models/finbert/tokenizer")
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print("✅ Tokenizer loaded successfully")
        
        # Test model
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        print("✅ Model loaded successfully")
        
        # Test GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {device}")
        
        # Test inference
        test_text = "The company reported positive earnings growth."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
        
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        print("✅ Test inference successful")
        return True
        
    except Exception as e:
        print(f"❌ FinBERT setup failed: {str(e)}")
        return False

def test_directories():
    """Test required directories"""
    print("\nTesting directory structure...")
    required_dirs = ['data', 'logs', 'models/finbert/model', 'models/finbert/tokenizer']
    all_good = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✅ Found directory: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("Starting setup verification...\n")
    
    tests = {
        "Directories": test_directories,
        "Ollama": test_ollama,
        "FinBERT": test_finbert
    }
    
    results = {}
    for name, test_func in tests.items():
        results[name] = test_func()
    
    print("\nTest Results Summary:")
    print("=" * 50)
    all_passed = True
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and result
    
    if all_passed:
        print("\n✨ All tests passed! System is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())