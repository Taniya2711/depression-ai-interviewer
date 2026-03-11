"""
Audio utilities for speech emotion recognition.
Matches the feature extraction logic from Pretrained SER.ipynb
"""

import os
import tempfile
import subprocess
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')

# Try to import torch and transformers, but allow graceful degradation
try:
    import torch
    from transformers.pipelines import pipeline
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Torch/Transformers import failed: {e}")
    print("[INFO] Running in degraded mode - audio analysis will use fallback features")
    TORCH_AVAILABLE = False
    torch = None
    pipeline = None

# Constants matching the notebook
SAMPLE_RATE = 16000
WINDOW_SEC = 4
HOP_SEC = 2

# Global SER pipeline (loaded once)
_ser_pipeline = None
_ser_pipeline_error = None

# FFmpeg paths to try
FFMPEG_PATHS = [
    "ffmpeg",  # In PATH
    r"C:\Users\SOUREN~1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]


def _is_model_cached(model_name: str) -> bool:
    """Check if the HuggingFace model is already downloaded in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if the key file exists in cache
        result = try_to_load_from_cache(model_name, "model.safetensors")
        if result is not None and not isinstance(result, str):
            return False
        return result is not None
    except ImportError:
        pass

    # Fallback: check default HF cache directory
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_dir_name = "models--" + model_name.replace("/", "--")
        model_dir = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(model_dir):
            # Check if snapshots folder has content (model was fully downloaded)
            snapshots = os.path.join(model_dir, "snapshots")
            if os.path.isdir(snapshots) and os.listdir(snapshots):
                return True
    except Exception:
        pass
    return False


SER_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

# Timeout for model loading (seconds). Set to 0 to disable.
SER_LOAD_TIMEOUT = int(os.environ.get("SER_LOAD_TIMEOUT", "120"))


def get_ser_pipeline():
    """Load the pretrained wav2vec2 emotion classification pipeline.
    
    If the model is not cached locally, skips download to avoid stalling
    the server. Run `python download_model.py` first to pre-download.
    """
    global _ser_pipeline, _ser_pipeline_error
    
    if _ser_pipeline_error:
        return None
    
    if _ser_pipeline is None:
        if not TORCH_AVAILABLE or pipeline is None:
            _ser_pipeline_error = "Torch not available"
            return None
        
        # Check if model is already cached before attempting to load
        if not _is_model_cached(SER_MODEL_NAME):
            print(f"[WARNING] Model '{SER_MODEL_NAME}' is NOT cached locally.")
            print("[WARNING] Downloading ~1.2 GB during inference causes stalls.")
            print("[WARNING] Run 'python download_model.py' first to pre-download.")
            print("[INFO] Falling back to mock emotion features for now.")
            _ser_pipeline_error = "Model not cached. Run: python download_model.py"
            return None
        
        try:
            print(f"[INFO] Loading SER model from local cache...")
            # Set environment variable to prevent re-downloading
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            device = 0 if torch.cuda.is_available() else -1
            _ser_pipeline = pipeline(
                "audio-classification",
                model=SER_MODEL_NAME,
                device=device
            )
            print("[SUCCESS] SER pipeline loaded from cache.")
            
            # Reset offline mode so other things can download if needed
            os.environ.pop("HF_HUB_OFFLINE", None)
        except Exception as e:
            os.environ.pop("HF_HUB_OFFLINE", None)
            print(f"[ERROR] Failed to load SER pipeline: {e}")
            print("[INFO] Run 'python download_model.py' to download the model.")
            _ser_pipeline_error = str(e)
            return None
    
    return _ser_pipeline


def find_ffmpeg() -> str:
    """Find ffmpeg executable on the system."""
    for path in FFMPEG_PATHS:
        try:
            if os.path.isfile(path):
                return path
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                print(f"[INFO] Found FFmpeg at: {path}")
                return path
        except:
            continue
    return None


def convert_webm_to_wav_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Convert webm/ogg to WAV using ffmpeg.
    Returns True if successful, False otherwise.
    """
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print("[WARNING] FFmpeg not found")
        return False
    
    try:
        cmd = [
            ffmpeg_path, '-y', '-i', input_path,
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-f', 'wav',
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"[SUCCESS] FFmpeg converted: {output_path} ({file_size} bytes)")
            return True
        else:
            print(f"[WARNING] FFmpeg failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[WARNING] FFmpeg conversion timed out")
        return False
    except Exception as e:
        print(f"[WARNING] FFmpeg error: {e}")
        return False


def convert_webm_to_wav_pydub(input_path: str, output_path: str) -> bool:
    """
    Fallback conversion using pydub (requires ffmpeg/ffprobe).
    Returns True if successful, False otherwise.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(output_path, format="wav")
        print(f"[SUCCESS] Pydub converted: {output_path}")
        return True
    except Exception as e:
        print(f"[WARNING] Pydub conversion failed: {e}")
        return False


def ensure_wav(audio_path: str) -> str:
    """
    Ensure audio is in WAV format at 16kHz mono.
    Returns path to WAV file (either original or converted).
    """
    ext = os.path.splitext(audio_path)[1].lower()
    
    # If already WAV, check sample rate
    if ext == '.wav':
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            if sr == SAMPLE_RATE:
                print(f"[INFO] Audio already at {SAMPLE_RATE}Hz: {audio_path}")
                return audio_path
        except Exception as e:
            print(f"[WARNING] Failed to check WAV format: {e}")
    
    # Need to convert
    print(f"[INFO] Converting {ext} to WAV...")
    
    # Try FFmpeg first (faster and more reliable)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name
    
    if convert_webm_to_wav_ffmpeg(audio_path, tmp_wav):
        return tmp_wav
    
    # Fallback to pydub
    if convert_webm_to_wav_pydub(audio_path, tmp_wav):
        return tmp_wav
    
    # Last resort: try librosa's native loading
    try:
        print("[INFO] Attempting direct librosa load...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        import soundfile as sf
        sf.write(tmp_wav, y, SAMPLE_RATE)
        print(f"[SUCCESS] Librosa converted: {tmp_wav}")
        return tmp_wav
    except Exception as e:
        print(f"[ERROR] All conversion methods failed: {e}")
        raise ValueError(f"Could not convert audio to WAV format: {e}")


def extract_emotion_features(audio_path: str) -> np.ndarray:
    """
    Extract emotion features from audio using wav2vec2 SER model.
    Returns matrix of shape (num_chunks, num_labels) with scores in pipeline
    output order — this EXACTLY matches the training notebook's
    extract_emotion_features() which does [x["score"] for x in p].
    If model unavailable, returns mock feature matrix.
    """
    pipeline_obj = get_ser_pipeline()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # If pipeline unavailable, return mock feature matrix
    if pipeline_obj is None:
        print("[WARNING] SER pipeline unavailable - using mock features")
        # Return mock feature matrix matching expected shape (1 chunk, 2 labels)
        return np.array([[0.6, 0.4]])
    
    try:
        # Sliding window approach (matches training notebook's chunk_audio + loop)
        window_samples = int(WINDOW_SEC * SAMPLE_RATE)
        hop_samples = int(HOP_SEC * SAMPLE_RATE)
        
        all_scores = []
        
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            chunk = y[start:start + window_samples]
            
            # Run through SER pipeline
            results = pipeline_obj(chunk, sampling_rate=SAMPLE_RATE)
            
            # Take scores in pipeline return order (sorted by score desc)
            # This matches the training notebook: [x["score"] for x in p]
            scores = [r['score'] for r in results]
            all_scores.append(scores)
        
        if not all_scores:
            # Audio too short - use the whole thing
            results = pipeline_obj(y, sampling_rate=SAMPLE_RATE)
            scores = [r['score'] for r in results]
            all_scores.append(scores)
        
        mat = np.array(all_scores)
        num_labels = mat.shape[1] if mat.ndim > 1 else len(all_scores[0])
        print(f"[DEBUG] SER returned {num_labels} labels across {len(all_scores)} chunks")
        return mat
    except Exception as e:
        print(f"[WARNING] SER feature extraction failed: {e} - using mock features")
        # Return mock feature matrix on error
        return np.array([[0.6, 0.4]])


def extract_features_from_file(audio_path: str) -> np.ndarray:
    """
    Extract feature vector from audio file.
    EXACTLY matches the training notebook's aggregate_features():
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        maxv = mat.max(axis=0)
        return np.concatenate([mean, std, maxv])
    
    Returns:
        numpy array of shape (3 * num_labels,) — e.g. 6 for 2 labels
    """
    # Ensure audio is WAV format
    wav_path = ensure_wav(audio_path)
    
    try:
        # Extract emotion feature matrix: shape (num_chunks, num_labels)
        # Scores are in pipeline output order (same as training notebook)
        mat = extract_emotion_features(wav_path)
        
        # Aggregate exactly like the training notebook's aggregate_features()
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        maxv = mat.max(axis=0)
        feature_vec = np.concatenate([mean, std, maxv])
        
        num_labels = mat.shape[1] if mat.ndim > 1 else mat.shape[0]
        print(f"[DEBUG] Feature vector length: {len(feature_vec)} (from {num_labels} labels × 3 stats)")
        
        return feature_vec.astype(np.float32)
    
    finally:
        # Cleanup temporary WAV file if created
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass
