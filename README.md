# Depression AI Interviewer

An AI-powered depression screening system that conducts conversational interviews and analyzes speech patterns to predict PHQ-8 (Patient Health Questionnaire-8) depression scores. The system combines speech emotion recognition (SER) with machine learning to assess depression severity through voice analysis.

---

## 📁 Project Structure

```
depression-ai-interviewer/
├── app.py                      # FastAPI backend server
├── inference.py                # PHQ prediction and model loading
├── audio_utils.py              # Audio processing and feature extraction
├── index.html                  # Frontend web interface
├── train_ser_model.ipynb       # Training notebook for SER model
├── README.md                   # This file
└── __pycache__/               # Python cache files

../
├── Pretrained SER.ipynb        # Pretrained model exploration notebook
└── Detailed_PHQ8_Labels.csv    # PHQ-8 labeled dataset
```

---

## 📋 File Descriptions

### **app.py**
FastAPI backend server that orchestrates the entire depression screening system.

**Features:**
- 8-question conversational interview interface
- Audio recording and upload handling
- Speech analysis endpoint using pretrained models
- Session management for tracking user responses
- PHQ-8 score calculation and severity classification
- CORS-enabled for frontend integration
- Demo mode support when trained model is unavailable

**Key Endpoints:**
- `GET /` - Serves the frontend HTML interface
- `GET /health` - Health check and model status
- `GET /start` - Initialize new interview session
- `POST /next_question` - Get next interview question
- `POST /analyze_speech` - Analyze audio and predict PHQ score
- `GET /results` - Get final interview results with average PHQ score

### **inference.py**
Handles PHQ-8 score prediction from audio files.

**Features:**
- Loads pretrained XGBoost model from `phq_model.pkl`
- Extracts audio features and performs inference
- Demo mode fallback when model file is missing
- Returns PHQ scores in 0-24 range
- Mock scoring based on emotion features when in demo mode

**Functions:**
- `load_model()` - Load pretrained model or enable demo mode
- `predict_phq(audio_path)` - Predict PHQ-8 score from audio file
- `is_demo_mode()` - Check if running in demo mode
- `predict_phq_from_features()` - Predict from pre-extracted features

### **audio_utils.py**
Audio processing utilities for speech emotion recognition.

**Features:**
- Audio format conversion (WebM, OGG, MP3 → WAV)
- FFmpeg integration for audio conversion
- Feature extraction using wav2vec2 pretrained model
- Sliding window analysis (4-second windows, 2-second hop)
- Emotion classification (7 emotions: sad, angry, happy, fearful, neutral, disgusted, surprised)
- Statistical feature aggregation (mean, std, min, max)

**Functions:**
- `extract_features_from_file(audio_path)` - Extract emotion features from audio
- `convert_webm_to_wav_ffmpeg()` - Convert audio formats using FFmpeg
- `get_ser_pipeline()` - Load pretrained wav2vec2 emotion classifier
- `find_ffmpeg()` - Locate FFmpeg executable on system

### **index.html**
Frontend web interface for the depression screening system.

**Features:**
- Interactive chat interface with AI interviewer
- Real-time audio recording using MediaRecorder API
- Automatic audio upload and PHQ score display
- Progress tracking through interview questions
- Final results with severity classification
- Responsive design with modern UI

### **train_ser_model.ipynb**
Jupyter notebook for training custom Speech Emotion Recognition models.

**Purpose:**
- Train XGBoost/Random Forest models on PHQ-8 labeled data
- Feature extraction from audio recordings
- Model evaluation and validation
- Export trained model as `phq_model.pkl`

**Workflow:**
1. Load audio dataset with PHQ-8 labels
2. Extract emotion features using wav2vec2
3. Train regression model (XGBoost)
4. Evaluate model performance
5. Save trained model for inference

### **Pretrained SER.ipynb**
Notebook demonstrating pretrained Speech Emotion Recognition model usage.

**Purpose:**
- Explore pretrained wav2vec2 model capabilities
- Test emotion classification on sample audio
- Feature extraction pipeline demonstration
- Baseline for comparison with custom models

---

## 🚀 How to Run

### **Prerequisites**

1. **Python 3.8+** installed
2. **FFmpeg** installed on your system
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via `winget install Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

### **Installation**

1. **Install Required Packages**

```bash
pip install fastapi uvicorn python-multipart numpy librosa torch transformers soundfile pydub xgboost scikit-learn pandas matplotlib seaborn jupyter
```

2. **Install PyTorch** (if not already installed)

For CPU:
```bash
pip install torch torchvision torchaudio
```

For GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Running the Application**

#### **Option 1: Run FastAPI Server (Main Application)**

```bash
cd depression-ai-interviewer
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at:
- **Frontend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

**Usage:**
1. Open http://localhost:8000 in your browser
2. Click "Start Interview" to begin
3. Answer each question by recording your voice
4. The system analyzes your speech and predicts PHQ-8 scores
5. View final results with depression severity assessment

**Note:** The application runs in **demo mode** if `phq_model.pkl` is not present. Demo mode uses mock scores based on emotion features but demonstrates full functionality.

#### **Option 2: Train Your Own Model**

To create a real `phq_model.pkl` file:

```bash
jupyter notebook train_ser_model.ipynb
```

**Steps:**
1. Open the notebook in Jupyter
2. Ensure you have the PHQ-8 labeled dataset (`Detailed_PHQ8_Labels.csv`)
3. Run all cells sequentially
4. The notebook will:
   - Load and preprocess audio data
   - Extract emotion features
   - Train XGBoost regression model
   - Save `phq_model.pkl` in the project directory
5. Restart the FastAPI server to use the trained model

#### **Option 3: Explore Pretrained Models**

```bash
jupyter notebook "Pretrained SER.ipynb"
```

**Purpose:**
- Understand how the pretrained wav2vec2 model works
- Test emotion classification on your own audio files
- Experiment with different feature extraction techniques

---

## 📊 Model Information

### **Pretrained Model**
- **Base Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- **Task:** Audio classification (emotion recognition)
- **Emotions Detected:** sad, angry, happy, fearful, neutral, disgusted, surprised
- **Framework:** HuggingFace Transformers

### **PHQ-8 Prediction Model**
- **Algorithm:** XGBoost Regressor
- **Input:** 28 statistical features from emotion scores (7 emotions × 4 statistics)
- **Output:** PHQ-8 score (0-24 range)
- **Training Data:** Audio recordings labeled with PHQ-8 scores

### **PHQ-8 Score Interpretation**
- **0-4:** Minimal depression
- **5-9:** Mild depression
- **10-14:** Moderate depression
- **15-19:** Moderately severe depression
- **20-24:** Severe depression

---

## 🔧 Configuration

### **Environment Variables**
- `PHQ_MODEL_PATH`: Path to trained model file (default: `phq_model.pkl`)

### **Audio Settings**
- **Sample Rate:** 16,000 Hz
- **Window Size:** 4 seconds
- **Hop Length:** 2 seconds
- **Supported Formats:** WAV, WebM, OGG, MP3

---

## 🐛 Troubleshooting

### **"FFmpeg not found" Error**
- Install FFmpeg and ensure it's in your system PATH
- Or specify FFmpeg path in `audio_utils.py` FFMPEG_PATHS list

### **Model Not Found (Demo Mode)**
- Train a model using `train_ser_model.ipynb` to generate `phq_model.pkl`
- Or continue using demo mode for testing (uses mock scores)

### **Audio Recording Not Working**
- Ensure browser has microphone permissions
- Use HTTPS or localhost (browser security requirement)
- Check browser console for MediaRecorder API errors

### **Torch/CUDA Errors**
- Install PyTorch with appropriate CUDA version for GPU
- Or use CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 📝 Development Notes

### **Demo Mode**
When `phq_model.pkl` is not found, the system runs in demo mode:
- Audio processing and emotion extraction work normally
- PHQ scores are calculated as mock values based on emotion features
- All functionality is preserved for development/testing
- Warning message displayed to indicate demo mode

### **Production Deployment**
For production use:
1. Train and deploy a real PHQ prediction model
2. Use secure session management (Redis/database)
3. Configure CORS to allow only your frontend domain
4. Add user authentication and data privacy measures
5. Implement proper logging and monitoring
6. Use HTTPS for all communications

---

## 🎯 Quick Start Summary

**To run the web application:**
```bash
cd depression-ai-interviewer
pip install fastapi uvicorn python-multipart numpy librosa torch transformers soundfile
python app.py
# Open http://localhost:8000
```

**To train a model:**
```bash
jupyter notebook train_ser_model.ipynb
# Run all cells, then restart app.py
```

**To explore pretrained models:**
```bash
jupyter notebook "Pretrained SER.ipynb"
# Experiment with emotion recognition
```

---

## 📄 License

This project is for educational and research purposes. Ensure compliance with relevant regulations when handling health-related data.

---

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for mental health concerns.