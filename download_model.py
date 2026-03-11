"""
Pre-download the wav2vec2 SER model from Hugging Face.
Run this ONCE before starting the server to avoid download stalls during inference.

Usage:
    python download_model.py
"""

import os
import sys

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

def download_model():
    print(f"Downloading model: {MODEL_NAME}")
    print("This is ~1.2 GB and may take several minutes...\n")

    # Method 1: Use huggingface_hub with explicit progress
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
            resume_download=True,       # Resume if partially downloaded
            max_workers=1,              # Single thread = more stable on slow connections
        )
        print(f"\n[SUCCESS] Model downloaded to cache: {cache_dir}")
        return True
    except ImportError:
        print("[INFO] huggingface_hub not installed, trying transformers directly...")
    except Exception as e:
        print(f"[WARNING] snapshot_download failed: {e}")
        print("Trying alternative method...\n")

    # Method 2: Use transformers directly
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        
        print("Downloading feature extractor...")
        AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        print("[OK] Feature extractor downloaded.")
        
        print("Downloading model weights (this is the large file)...")
        AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        print("[OK] Model weights downloaded.")
        
        print(f"\n[SUCCESS] Model '{MODEL_NAME}' is now cached locally.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try a VPN if Hugging Face is slow/blocked in your region")
        print("  3. Install huggingface_hub: pip install huggingface_hub")
        print("  4. Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads:")
        print("     pip install hf_transfer")
        print("     set HF_HUB_ENABLE_HF_TRANSFER=1")
        return False


def verify_model():
    """Verify the model is cached and loadable."""
    print("\nVerifying model can be loaded from cache...")
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "audio-classification",
            model=MODEL_NAME,
            device=-1  # CPU for verification
        )
        print("[SUCCESS] Model loads correctly from cache!")
        print("You can now start the server without download delays.")
        del pipe
        return True
    except Exception as e:
        print(f"[WARNING] Model verification failed: {e}")
        return False


if __name__ == "__main__":
    success = download_model()
    if success:
        verify_model()
    sys.exit(0 if success else 1)
