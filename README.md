# MindSpace - Depression AI Interviewer 🧠

An advanced AI-powered depression screening system that combines personalized conversational interviews with speech emotion recognition to predict PHQ-8 (Patient Health Questionnaire-8) depression scores. The system uses Google's Gemini LLM for personalized questions and machine learning for speech-based depression assessment.

---

## ✨ Key Features

- **🤖 Personalized Interviews**: LLM-generated questions tailored to user demographics
- **🎯 Speech Analysis**: Real-time emotion recognition from voice recordings
- **📊 PHQ-8 Assessment**: Clinically validated depression severity scoring (0-24)
- **📱 Interactive Web Interface**: Modern, responsive UI with real-time progress tracking
- **🔒 Privacy-First**: No data storage, local processing, optional demographic collection
- **⚡ Demo Mode**: Fully functional without trained models for development/testing
- **🌍 Production Ready**: FastAPI backend with CORS, health checks, and error handling

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Demographics  │───▶│    LLM Engine    │───▶│   Interview     │
│   Collection    │    │  (Gemini API)    │    │   Questions     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PHQ Scores    │◄───│  Audio Analysis  │◄───│   Voice Input   │
│ & Recommendations│    │(wav2vec2 + XGB) │    │   Recording     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 📁 Project Structure

```
IS_project/
├── depression-ai-interviewer/           # Main application
│   ├── main.py                         # FastAPI server & orchestration
│   ├── llm_utils.py                    # Gemini LLM integration
│   ├── inference.py                    # PHQ prediction engine
│   ├── audio_utils.py                  # Speech processing & feature extraction
│   ├── demographic.html                # User info collection form
│   ├── demographic.css                 # Styling for demographics form
│   ├── demographic.js                  # Frontend logic for demographics
│   ├── index.html                      # Main interview interface
│   ├── phq_xgb.pkl                    # Trained XGBoost model (generated)
│   ├── user_demographics.json          # Stored user info (auto-generated)
│   ├── .env                           # Environment variables (API keys)
│   └── .gitignore                     # Git ignore rules
├── train_ser_model.ipynb              # Model training notebook
├── Pretrained SER.ipynb               # Pre-trained model exploration
└── Detailed_PHQ8_Labels.csv           # Training dataset
```

---

## 🔧 File Descriptions

### **Core Application**

#### **main.py** - FastAPI Application Server
- **Role**: Central orchestrator coordinating all system components
- **Features**:
  - Demographic data collection and persistence
  - LLM integration for personalized questioning
  - Audio upload and analysis pipeline coordination
  - PHQ score calculation and recommendation generation
  - Session management and progress tracking
  - Health monitoring and status endpoints

#### **llm_utils.py** - Gemini LLM Integration
- **Role**: Generates personalized interview questions and recommendations
- **Features**:
  - Environment variable + .env file API key loading
  - Dynamic question generation based on user demographics
  - Context-aware follow-up questions
  - Personalized recommendation generation
  - Fallback to hardcoded questions/recommendations
  - Support for both batch and real-time question generation

#### **inference.py** - PHQ Prediction Engine
- **Role**: Converts speech features to PHQ-8 depression scores
- **Features**:
  - XGBoost model loading and inference
  - Demo mode with mock predictions for development
  - Feature validation and preprocessing
  - Score normalization (0-24 range)
  - Error handling and logging

#### **audio_utils.py** - Speech Processing Pipeline
- **Role**: Extracts emotion features from audio recordings
- **Features**:
  - Multi-format audio conversion (WebM, OGG, MP3 → WAV)
  - wav2vec2-based emotion classification (7 emotions)
  - Sliding window analysis (4-second windows, 2-second hop)
  - Statistical feature aggregation (mean, std, min, max)
  - FFmpeg integration for format conversion

### **Frontend Interface**

#### **demographic.html** - Initial User Information Collection
- **Role**: Collects user context for question personalization
- **Features**:
  - Optional demographic form (age, role, stressors, etc.)
  - Modern responsive design
  - Data validation and submission handling
  - Seamless redirect to interview interface

#### **index.html** - Main Interview Interface
- **Role**: Interactive chat interface for interview and audio recording
- **Features**:
  - Real-time audio recording with MediaRecorder API
  - Progressive question display with typing animation
  - Live PHQ score tracking and visualization
  - Final results with severity classification
  - Mobile-responsive design

### **Model Training & Data**

#### **train_ser_model.ipynb** - Model Training Pipeline
- **Role**: Trains custom XGBoost regression model for PHQ prediction
- **Process**:
  1. Load PHQ-8 labeled audio dataset
  2. Extract emotion features using wav2vec2
  3. Train XGBoost regression model
  4. Evaluate model performance (MAE, R²)
  5. Export `phq_xgb.pkl` for production use

#### **Pretrained SER.ipynb** - Speech Emotion Research
- **Role**: Explores pretrained emotion recognition capabilities
- **Features**:
  - wav2vec2 model testing and validation
  - Feature extraction pipeline demonstration
  - Emotion classification accuracy assessment

#### **Detailed_PHQ8_Labels.csv** - Training Dataset
- **Content**: Audio file paths with corresponding PHQ-8 labels
- **Usage**: Ground truth data for model training and validation

---

## 🚀 Quick Start Guide

### **Prerequisites**

1. **Python 3.8+** with pip
2. **FFmpeg** installed and in PATH
   - Windows: `winget install Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`  
   - macOS: `brew install ffmpeg`
3. **Gemini API Key** (optional, for personalized questions)

### **Installation**

1. **Clone and navigate to project**
```bash
cd depression-ai-interviewer
```

2. **Install Python dependencies**
```bash
pip install fastapi uvicorn python-multipart numpy librosa torch transformers soundfile pydub xgboost scikit-learn pandas
```

3. **Configure Gemini API (Optional)**

Option A: Environment variable
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Option B: .env file
```bash
# Edit .env file and add:
GEMINI_API_KEY=your_gemini_api_key_here
```

### **Running the Application**

#### **Option 1: Quick Start (Demo Mode)**
```bash
python main.py
```
- Opens at http://localhost:8000
- Uses hardcoded questions and mock PHQ scores
- Full functionality for testing and development

#### **Option 2: With Trained Model**
```bash
# First, train the model
jupyter notebook ../train_ser_model.ipynb
# Run all cells to generate phq_xgb.pkl

# Then start the application
python main.py
```
- Real PHQ predictions based on speech analysis
- Trained model provides accurate depression scoring

#### **Option 3: Production Mode (with LLM)**
```bash
# Configure Gemini API key (see Installation step 3)
python main.py
```
- Personalized questions based on user demographics
- Enhanced recommendations tailored to user context
- Professional-grade interview experience

---

## 📊 System Workflow

### **1. Demographic Collection**
```
User visits / → demographic.html → Collects optional user info → Stores in user_demographics.json
```

### **2. Interview Generation**
```
LLM available? → Generate personalized questions → Fallback to hardcoded questions
                     ↓
            Store session state → Present questions sequentially
```

### **3. Audio Processing**
```
User records voice → Upload to /analyze_speech → Convert to WAV using FFmpeg
                           ↓
                Extract emotion features → Predict PHQ score → Update session
```

### **4. Results & Recommendations**
```
All questions complete → Calculate average PHQ → Generate recommendations
                                ↓
                    Display severity classification → Provide resources
```

---

## 🔬 Technical Specifications

### **Machine Learning Pipeline**
- **Emotion Model**: `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- **Emotions Detected**: sad, angry, happy, fearful, neutral, disgusted, surprised
- **Feature Extraction**: 7 emotions × 4 statistics = 28 features per audio
- **Prediction Model**: XGBoost Regressor (28 features → PHQ score 0-24)
- **Audio Processing**: 16kHz sampling, 4-second windows, 2-second overlap

### **PHQ-8 Score Interpretation**
| Score Range | Severity Level | Recommended Action |
|-------------|----------------|-------------------|
| 0-4 | Minimal | Continue healthy habits |
| 5-9 | Mild | Self-care and monitoring |
| 10-14 | Moderate | Consider professional consultation |
| 15-19 | Moderately Severe | Seek professional help |
| 20-24 | Severe | Immediate professional attention |

### **API Endpoints**
```
GET  /                    # Demographic collection form
GET  /interview          # Main interview interface  
POST /submit_demographics # Save user information
GET  /start              # Initialize interview session
GET  /next_question      # Get next personalized question
POST /analyze_speech     # Process audio and predict PHQ
GET  /results           # Final interview results
GET  /health           # System health and model status
GET  /llm_status      # LLM availability check
```

---

## 🔧 Configuration Options

### **Environment Variables**
```bash
GEMINI_API_KEY=your_api_key         # For personalized questions
PHQ_MODEL_PATH=./phq_xgb.pkl       # Custom model path (optional)
```

### **Audio Settings** (modify in audio_utils.py)
```python
SAMPLE_RATE = 16000      # Audio sampling rate
WINDOW_SIZE = 4.0        # Analysis window (seconds)
HOP_LENGTH = 2.0         # Window overlap (seconds)
```

### **LLM Settings** (modify in llm_utils.py)
```python
MODEL_NAME = "gemini-2.5-flash"     # Gemini model version
MAX_QUESTIONS = 5                   # Interview length
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### **"FFmpeg not found" Error**
```bash
# Windows
winget install Gyan.FFmpeg
# Add to PATH if needed

# Linux
sudo apt update && sudo apt install ffmpeg

# macOS  
brew install ffmpeg
```

#### **"GEMINI_API_KEY not set" Warning**
- System uses hardcoded questions in fallback mode
- Set API key in environment or .env file for personalized experience
- Application remains fully functional without API key

#### **"Model not found" (Demo Mode)**
- Run `../train_ser_model.ipynb` to generate `phq_xgb.pkl`
- Or continue in demo mode for development/testing
- Mock scores maintain full functionality

#### **Audio Recording Issues**
- Ensure browser microphone permissions are granted
- Use HTTPS or localhost (browser security requirement)
- Check browser console for MediaRecorder API errors

#### **PyTorch/CUDA Warnings**
```bash
# CPU-only installation (recommended for most users)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU installation (if CUDA available)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Development Guide

### **Running in Development Mode**
```bash
# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Debug mode with verbose logging
python main.py --debug
```

### **Testing Components**
```bash
# Test LLM integration
curl http://localhost:8000/llm_status

# Test model loading
curl http://localhost:8000/health

# Test audio processing (with audio file)
curl -X POST -F "audio=@test.wav" http://localhost:8000/analyze_speech
```

### **Adding New Features**
1. **New Question Types**: Modify `llm_utils.py` prompt templates
2. **Additional Audio Features**: Extend `audio_utils.py` feature extraction
3. **Custom Models**: Replace XGBoost in `inference.py` with your model
4. **UI Enhancements**: Edit `index.html` and `demographic.html`

---

## 📈 Production Deployment

### **Security Recommendations**
- [ ] Use HTTPS for all communications
- [ ] Implement proper session management (Redis/database)
- [ ] Configure CORS to allow only your frontend domain
- [ ] Add rate limiting and request validation
- [ ] Implement user authentication if needed
- [ ] Use secure API key storage (not .env in production)

### **Scalability Considerations**
- [ ] Use cloud-based speech processing for high volume
- [ ] Implement model serving (TensorFlow Serving, MLflow)
- [ ] Add caching for LLM responses
- [ ] Use CDN for static assets
- [ ] Monitor API quotas (Gemini API limits)

### **Health Monitoring**
```bash
# Regular health checks
GET /health              # Overall system status
GET /llm_status         # LLM availability
```

---

## 📄 License & Ethics

**License**: Educational and research use only
**Privacy**: No audio or personal data is permanently stored
**Medical Disclaimer**: This tool is for research purposes only and should not replace professional medical diagnosis
**Ethics**: Always encourage users to seek professional help for mental health concerns

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Resources

**Crisis Hotlines:**
- **US**: National Suicide Prevention Lifeline: 988
- **India**: iCall: 9152987821 | NIMHANS: 080-46110007
- **Emergency**: 911 (US) | 100 (India) | 112 (EU)

**Technical Issues**: Check troubleshooting section or create a GitHub issue

---

*Remember: This tool provides screening insights, not clinical diagnosis. Always consult qualified healthcare professionals for mental health concerns.*