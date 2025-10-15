#!/usr/bin/env python3
"""
Download Model Files Script
Run this script to download the trained model files for the Urdu Chatbot.
"""

import os
import requests
from pathlib import Path
import gdown

def download_models():
    """Download model files from cloud storage"""
    
    files_dir = Path("files")
    files_dir.mkdir(exist_ok=True)
    
    print("📥 Downloading Urdu Chatbot model files...")
    
    # Example using Google Drive (you can upload your models there)
    model_urls = {
        "best_model.pth": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
        "best_model.pkl": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
        # Small files can stay in repo
        "tokenizer.model": None,
        "tokenizer.vocab": None,
        "vocab_mapping.pkl": None
    }
    
    for filename, file_id in model_urls.items():
        file_path = files_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            continue
            
        if file_id:
            print(f"📥 Downloading {filename}...")
            try:
                # Download from Google Drive
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(file_path), quiet=False)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"ℹ️ {filename} should be included in repository")
    
    print("🎉 Model download completed!")

if __name__ == "__main__":
    download_models()