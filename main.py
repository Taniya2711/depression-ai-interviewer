"""
FastAPI backend for speech-based depression analysis.
Unified chatbot + audio analysis system.
"""

import os
import json
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from inference import predict_phq, load_model, is_demo_mode

# File to store demographic data for future LLM use
DEMOGRAPHIC_DATA_FILE = "user_demographics.json"

# Initialize FastAPI app
app = FastAPI(
    title="Depression AI Interviewer",
    description="AI-powered depression screening through conversational interview + speech analysis",
    version="1.0.0"
)

# Chatbot questions
QUESTIONS = [
    "Hello! I'm here to chat with you today. How have you been feeling lately?",
    "Have you felt little interest or pleasure in doing things you usually enjoy?",
    "Have you been feeling down, depressed, or hopeless?",
    #"Have you had trouble falling or staying asleep, or sleeping too much?",
    #"Have you been feeling tired or having little energy?",
    #"Tell me about your appetite - has it changed recently?",
    "Have you had trouble concentrating on things?",
    "How do you feel about yourself lately?"
]

# Session state storage (in production, use Redis or database)
sessions = {}

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    """Response model for speech analysis."""
    phq_score: float
    demo_mode: bool = False
    message: str = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    demo_mode: bool


class ChatResponse(BaseModel):
    """Response model for chatbot."""
    question: str
    question_number: int
    total_questions: int
    completed: bool = False


class FinalResultResponse(BaseModel):
    """Response model for final interview results."""
    completed: bool
    average_phq_score: float
    num_responses: int
    demo_mode: bool
    message: str


class DemographicData(BaseModel):
    """Data model for demographic information."""
    age: Optional[str] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    role: Optional[str] = None
    stage: Optional[str] = None
    focus: Optional[str] = None
    sleep_duration: Optional[str] = None
    workload: Optional[str] = None
    screen_time: Optional[str] = None
    living_situation: Optional[str] = None
    support_system: Optional[str] = None
    stressors: Optional[List[str]] = []


@app.on_event("startup")
async def startup_event():
    """Load the model on startup (or enter demo mode)."""
    load_model()
    if is_demo_mode():
        print("=" * 50)
        print("🔶 RUNNING IN DEMO MODE")
        print("   Audio processing works, but PHQ scores are mocked")
        print("   Add phq_model.pkl to enable real predictions")
        print("=" * 50)
    else:
        print("✓ PHQ prediction model loaded successfully")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the demographic form first."""
    html_path = os.path.join(os.path.dirname(__file__), "demographic.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Depression AI Interviewer</h1><p>demographic.html not found</p>"


@app.get("/interview", response_class=HTMLResponse)
async def interview():
    """Serve the interview page."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Depression AI Interviewer</h1><p>index.html not found</p>"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and model is loaded."""
    from inference import _model
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        demo_mode=is_demo_mode()
    )


@app.post("/submit_demographics")
async def submit_demographics(data: DemographicData):
    """Store demographic information and return success."""
    session_id = "default"
    
    # Store demographics in session
    if session_id not in sessions:
        sessions[session_id] = {"current_question": 0, "phq_scores": []}
    
    sessions[session_id]["demographics"] = data.dict()
    
    # Save to file for future LLM use
    try:
        data_file = os.path.join(os.path.dirname(__file__), DEMOGRAPHIC_DATA_FILE)
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data.dict(), f, indent=2)
        print(f"✓ Demographic data saved to {DEMOGRAPHIC_DATA_FILE}")
    except Exception as e:
        print(f"Warning: Could not save demographic data to file: {e}")
    
    return {
        "status": "success",
        "redirect_url": "/interview",
        "message": "Demographic data saved successfully"
    }


@app.get("/demographic.css")
async def get_css():
    """Serve the demographic CSS file."""
    css_path = os.path.join(os.path.dirname(__file__), "demographic.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/demographic.js")
async def get_js():
    """Serve the demographic JS file."""
    js_path = os.path.join(os.path.dirname(__file__), "demographic.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")


@app.get("/start", response_model=ChatResponse)
async def start_interview():
    """Start a new interview session."""
    session_id = "default"  # In production, generate unique session IDs
    sessions[session_id] = {
        "current_question": 0,
        "phq_scores": []
    }
    
    return ChatResponse(
        question=QUESTIONS[0],
        question_number=1,
        total_questions=len(QUESTIONS),
        completed=False
    )


@app.post("/next_question", response_model=ChatResponse)
async def next_question():
    """Get the next question in the interview."""
    session_id = "default"
    
    if session_id not in sessions:
        sessions[session_id] = {"current_question": 0, "phq_scores": []}
    
    sessions[session_id]["current_question"] += 1
    current = sessions[session_id]["current_question"]
    
    if current >= len(QUESTIONS):
        return ChatResponse(
            question="Thank you for completing the interview.",
            question_number=current + 1,
            total_questions=len(QUESTIONS),
            completed=True
        )
    
    return ChatResponse(
        question=QUESTIONS[current],
        question_number=current + 1,
        total_questions=len(QUESTIONS),
        completed=False
    )


@app.get("/results", response_model=FinalResultResponse)
async def get_results():
    """Get final interview results with average PHQ score."""
    session_id = "default"
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="No active session found")
    
    scores = sessions[session_id]["phq_scores"]
    
    if len(scores) == 0:
        return FinalResultResponse(
            completed=True,
            average_phq_score=0.0,
            num_responses=0,
            demo_mode=is_demo_mode(),
            message="No audio responses were analyzed"
        )
    
    avg_score = sum(scores) / len(scores)
    
    # Interpretation
    if avg_score < 5:
        severity = "minimal"
    elif avg_score < 10:
        severity = "mild"
    elif avg_score < 15:
        severity = "moderate"
    elif avg_score < 20:
        severity = "moderately severe"
    else:
        severity = "severe"
    
    demo_msg = " (Demo Mode - Mock Scores)" if is_demo_mode() else ""
    
    return FinalResultResponse(
        completed=True,
        average_phq_score=round(avg_score, 2),
        num_responses=len(scores),
        demo_mode=is_demo_mode(),
        message=f"Interview completed. Average PHQ-8 score: {avg_score:.1f} ({severity} depression){demo_msg}"
    )


@app.post("/analyze_speech", response_model=AnalysisResponse)
async def analyze_speech(audio: UploadFile = File(...), force_demo: bool = False):
    """
    Analyze speech audio to predict PHQ-8 depression score.
    Stores the score in the current session.
    
    Args:
        audio: Uploaded audio file
        force_demo: If True, skip real inference and return mock scores (Quick Test Mode)
    """
    session_id = "default"
    
    # Quick Test Mode - skip all inference, return mock score immediately
    if force_demo:
        import random
        mock_score = round(random.uniform(3, 12), 1)
        print(f"[QUICK TEST MODE] Returning mock score: {mock_score}")
        
        if session_id not in sessions:
            sessions[session_id] = {"current_question": 0, "phq_scores": []}
        sessions[session_id]["phq_scores"].append(mock_score)
        
        return AnalysisResponse(
            phq_score=mock_score,
            demo_mode=True,
            message="Quick Test Mode - Pipeline validated, no ML inference performed"
        )
    
    if session_id not in sessions:
        sessions[session_id] = {"current_question": 0, "phq_scores": []}
    
    # Determine file extension
    ext_map = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
    }
    
    content_type = audio.content_type or "application/octet-stream"
    print(f"[DEBUG] Received audio: content_type={content_type}, filename={audio.filename}")
    
    if audio.filename:
        ext = os.path.splitext(audio.filename)[1].lower()
        if not ext:
            ext = ext_map.get(content_type, ".webm")
    else:
        ext = ext_map.get(content_type, ".webm")
    
    # Save uploaded audio to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
            print(f"[DEBUG] Saved audio to: {tmp_path}, size: {len(content)} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to save audio: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to save audio: {str(e)}")
    
    # Perform prediction
    result = None
    try:
        print("[DEBUG] Starting prediction...")
        result = predict_phq(tmp_path)
        print(f"[DEBUG] Prediction result: {result}")
        
        # Store score in session
        sessions[session_id]["phq_scores"].append(result["phq_score"])
        
    except ValueError as e:
        print(f"[ERROR] Audio processing error: {e}")
        # Return a fallback response instead of crashing
        result = {
            "phq_score": 5.0,
            "demo_mode": True,
            "message": f"Audio processing failed: {str(e)}. Using fallback score."
        }
        sessions[session_id]["phq_scores"].append(result["phq_score"])
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Return a fallback response instead of crashing
        result = {
            "phq_score": 5.0,
            "demo_mode": True,
            "message": f"Prediction failed: {str(e)}. Using fallback score."
        }
        sessions[session_id]["phq_scores"].append(result["phq_score"])
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return AnalysisResponse(
        phq_score=result["phq_score"],
        demo_mode=result.get("demo_mode", False),
        message=result.get("message")
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
