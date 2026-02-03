"""
Inference module for PHQ depression score prediction.
Loads pretrained model and performs inference only (no training).

DEMO MODE: If no model file exists, returns emotion features + mock score.
"""

import os
import pickle
import numpy as np
from audio_utils import extract_features_from_file

# Path to the pretrained model file
MODEL_PATH = os.environ.get("PHQ_MODEL_PATH", "phq_model.pkl")

# Global model (loaded once, None if demo mode)
_model = None

# Demo mode flag - set to True when model is not available
_demo_mode = False


def load_model(model_path: str = None):
    """
    Load the pretrained XGBoost model from disk.
    If model not found, enables demo mode.
    """
    global _model, _demo_mode
    
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        _demo_mode = True
        print(f"⚠️  Model not found at {model_path}")
        print("   Running in DEMO MODE - will return mock scores")
        print("   To enable real predictions, add phq_model.pkl file")
        return None
    
    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    
    _demo_mode = False
    return _model


def get_model():
    """Get the loaded model. Returns None in demo mode."""
    global _model
    return _model


def is_demo_mode():
    """Check if running in demo mode (no model)."""
    return _demo_mode


def predict_phq(audio_path: str) -> dict:
    """
    Predict PHQ-8 depression score from an audio file.
    
    Args:
        audio_path: Path to the audio file (wav, webm, ogg, etc.)
    
    Returns:
        dict with:
            - phq_score: Predicted score (0-24) or mock value in demo mode
            - demo_mode: True if using mock score
            - emotions: Extracted emotion features (always available)
    """
    # Extract features using the same logic as the notebook
    feat = extract_features_from_file(audio_path)
    
    # If we have a model, use it
    if _model is not None:
        prediction = _model.predict(feat.reshape(1, -1))[0]
        prediction = float(np.clip(prediction, 0, 24))
        return {
            "phq_score": prediction,
            "demo_mode": False,
            "features_extracted": True
        }
    
    # DEMO MODE: Return a mock score based on emotion features
    # This allows frontend development to continue
    # Mock score: use mean of features scaled to 0-24 range
    mock_score = float(np.clip(np.mean(feat) * 24, 0, 24))
    
    return {
        "phq_score": round(mock_score, 2),
        "demo_mode": True,
        "features_extracted": True,
        "message": "Demo mode - using mock PHQ score. Real predictions require phq_model.pkl"
    }


def predict_phq_from_features(features: np.ndarray) -> float:
    """
    Predict PHQ-8 score from pre-extracted features.
    
    Args:
        features: numpy array of shape (28,) containing emotion features
    
    Returns:
        Predicted PHQ-8 score (0-24)
    """
    if _model is None:
        raise ValueError("Model not loaded. Call load_model() first or use demo mode.")
    
    prediction = _model.predict(features.reshape(1, -1))[0]
    return float(np.clip(prediction, 0, 24))
