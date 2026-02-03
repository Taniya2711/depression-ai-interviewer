"""
Audio utilities for speech emotion recognition.
Matches the feature extraction logic from Pretrained SER.ipynb
"""

import os
import tempfile
import subprocess
import numpy as np
import librosa
import torch
from transformers import pipeline

# Constants matching the notebook
SAMPLE_RATE = 16000
WINDOW_SEC = 4
HOP_SEC = 2

# Global SER pipeline (loaded once)
_ser_pipeline = None

# FFmpeg paths to try
FFMPEG_PATHS = [
    "ffmpeg",  # In PATH
    r"C:\Users\SOUREN~1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]


def get_ser_pipeline():
    """Load the pretrained wav2vec2 emotion classification pipeline."""
    global _ser_pipeline
    if _ser_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        _ser_pipeline = pipeline(
            "audio-classification",
            model="jonatasgrosman/wav2vec2-large-xlsr-53-english",
            device=device
        )
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


def extract_emotion_features(audio_path: str) -> dict:
    """
    Extract emotion features from audio using wav2vec2 SER model.
    Returns dict mapping emotion names to scores (0-1).
    """
    pipeline_obj = get_ser_pipeline()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Sliding window approach
    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    
    all_emotions = []
    
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        chunk = y[start:start + window_samples]
        
        # Run through SER pipeline
        results = pipeline_obj(chunk, sampling_rate=SAMPLE_RATE)
        
        # Convert to dict
        emotion_dict = {r['label']: r['score'] for r in results}
        all_emotions.append(emotion_dict)
    
    if not all_emotions:
        # Audio too short - use the whole thing
        results = pipeline_obj(y, sampling_rate=SAMPLE_RATE)
        emotion_dict = {r['label']: r['score'] for r in results}
        all_emotions.append(emotion_dict)
    
    # Aggregate emotions across all windows
    emotion_names = list(all_emotions[0].keys())
    aggregated = {}
    
    for emotion in emotion_names:
        values = [window[emotion] for window in all_emotions]
        aggregated[emotion] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    return aggregated


def extract_features_from_file(audio_path: str) -> np.ndarray:
    """
    Extract feature vector from audio file.
    This matches the feature extraction in the training notebook.
    
    Returns:
        numpy array of shape (28,) - 7 emotions × 4 statistics each
    """
    # Ensure audio is WAV format
    wav_path = ensure_wav(audio_path)
    
    try:
        # Extract emotions
        emotions = extract_emotion_features(wav_path)
        
        # Flatten to feature vector
        # Order: for each emotion, output [mean, std, min, max]
        feature_vec = []
        for emotion_name in sorted(emotions.keys()):  # Sort for consistency
            stats = emotions[emotion_name]
            feature_vec.extend([
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max']
            ])
        
        return np.array(feature_vec, dtype=np.float32)
    
    finally:
        # Cleanup temporary WAV file if created
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass
