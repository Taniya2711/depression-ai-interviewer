"""
LLM Utilities for generating personalized interview questions and recommendations.
Uses Google's Gemini API for LLM capabilities.
"""

import os
import json
from typing import List, Dict, Optional

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("[WARNING] google-generativeai not installed. Install with: pip install google-generativeai")


def load_api_key_from_env_file():
    """Load GEMINI_API_KEY from .env file if it exists."""
    env_file_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file_path):
        try:
            with open(env_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "GEMINI_API_KEY":
                            return value.strip().strip('"\'')
        except Exception as e:
            print(f"[WARNING] Error reading .env file: {e}")
    return None


# Configuration
# Check environment variable first, then .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or load_api_key_from_env_file() or ""
MODEL_NAME = "gemini-2.5-flash"

# Hardcoded fallback questions (PHQ-8 based)
HARDCODED_QUESTIONS = [
    "Hello! I'm here to chat with you today. How have you been feeling lately?",
    "Over the past two weeks, have you felt little interest or pleasure in doing things you usually enjoy?",
    #"Have you been feeling down, depressed, or hopeless recently?",
    "How has your sleep been? Have you had trouble falling asleep, staying asleep, or sleeping too much?",
    "Have you been feeling tired or having little energy lately?",
    #"How has your appetite been? Have you noticed any changes in your eating habits?",
    #"Have you had trouble concentrating on things, like reading or watching TV?",
    "How do you feel about yourself lately? Have you been feeling bad about yourself?"
]

# Hardcoded recommendations based on severity
HARDCODED_RECOMMENDATIONS = {
    "minimal": {
        "severity": "Minimal Depression",
        "score_range": "0-4",
        "summary": "Your responses suggest minimal or no significant symptoms of depression.",
        "recommendations": [
            "Continue maintaining your current healthy lifestyle habits",
            "Practice regular self-care activities that bring you joy",
            "Stay connected with friends and family",
            "Maintain a regular sleep schedule",
            "Consider keeping a gratitude journal"
        ],
        "resources": [
            "Mindfulness apps like Headspace or Calm for stress management",
            "Regular physical exercise (30 mins, 3-5 times per week)"
        ]
    },
    "mild": {
        "severity": "Mild Depression",
        "score_range": "5-9",
        "summary": "Your responses suggest some mild symptoms that may benefit from attention.",
        "recommendations": [
            "Consider talking to a trusted friend or family member about how you feel",
            "Establish a regular daily routine with consistent sleep and wake times",
            "Engage in physical activities you enjoy, even light walking helps",
            "Limit alcohol and caffeine consumption",
            "Practice relaxation techniques like deep breathing or meditation"
        ],
        "resources": [
            "Self-help books on cognitive behavioral techniques",
            "Online mental wellness programs",
            "Support groups or peer counseling services"
        ]
    },
    "moderate": {
        "severity": "Moderate Depression",
        "score_range": "10-14",
        "summary": "Your responses indicate moderate symptoms. Professional support is recommended.",
        "recommendations": [
            "Schedule an appointment with a mental health professional",
            "Consider speaking with your primary care physician",
            "Maintain social connections even when it feels difficult",
            "Set small, achievable daily goals",
            "Avoid making major life decisions while feeling this way"
        ],
        "resources": [
            "Licensed therapists or counselors",
            "Employee Assistance Programs (EAP) if available",
            "Your campus counseling center (for students)"
        ]
    },
    "moderately_severe": {
        "severity": "Moderately Severe Depression",
        "score_range": "15-19",
        "summary": "Your responses suggest significant symptoms. Please seek professional help.",
        "recommendations": [
            "Contact a mental health professional as soon as possible",
            "Speak with your doctor about treatment options including therapy",
            "Reach out to a crisis helpline if you feel overwhelmed",
            "Avoid being alone for long periods - stay connected with others",
            "Remove access to harmful substances or items"
        ],
        "resources": [
            "Crisis Text Line: Text HOME to 741741",
            "National Suicide Prevention Lifeline: 988 (US)",
            "iCall: 9152987821 (India)",
            "Urgent care or emergency services if needed"
        ]
    },
    "severe": {
        "severity": "Severe Depression",
        "score_range": "20-24",
        "summary": "Your responses indicate severe symptoms requiring immediate professional attention.",
        "recommendations": [
            "Please seek immediate professional help today",
            "Contact a crisis helpline or go to your nearest emergency room",
            "Do not be alone - stay with a trusted person",
            "Tell someone you trust how you're feeling right now",
            "Remember: this is treatable and help is available"
        ],
        "resources": [
            "Emergency Services: 911 (US) / 112 (EU) / 100 (India)",
            "National Suicide Prevention Lifeline: 988",
            "iCall: 9152987821 (India)",
            "NIMHANS Helpline: 080-46110007 (India)",
            "Vandrevala Foundation: 1860-2662-345 (India)"
        ]
    }
}


def configure_genai():
    """Configure the Gemini API with the API key."""
    if not GENAI_AVAILABLE:
        return False
    
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY not set. Set it with: export GEMINI_API_KEY='your-key'")
        return False
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini API: {e}")
        return False


def get_llm_model():
    """Get the configured Gemini model."""
    if not configure_genai():
        return None
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini model: {e}")
        return None


def generate_personalized_questions(demographics: Dict) -> List[str]:
    """
    Generate personalized interview questions based on demographic information.
    Falls back to hardcoded questions if LLM is unavailable.
    
    Args:
        demographics: Dictionary containing user demographic information
        
    Returns:
        List of personalized interview questions
    """
    model = get_llm_model()
    
    if not model:
        print("[INFO] Using hardcoded questions (LLM unavailable)")
        return HARDCODED_QUESTIONS
    
    # Build context from demographics
    demo_context = _format_demographics(demographics)
    
    prompt = f"""You are a compassionate mental health screening assistant. Generate 5 personalized interview questions to assess depression symptoms based on PHQ-8 criteria.

User Profile:
{demo_context}

Requirements:
1. Questions should be empathetic, non-judgmental, and conversational
2. Do NOT include any placeholder entities like [name] [school] etc — assume or reason when ambiguous
3. Tailor questions to the user's context (student, professional, etc.)
4. Cover key PHQ-8 domains: mood, interest/pleasure, sleep, energy, appetite, self-worth, concentration
5. Start with a warm greeting that acknowledges their situation
6. NAME RULES (CRITICAL):
   - If a name IS in the profile: use their EXACT first name in ONLY 2 of the 5 questions (e.g. questions 1 and 3). The other 3 should NOT use the name at all — this keeps it natural.
   - If NO name is in the profile: NEVER guess or invent a name. Use "you" or conversational phrasing.
7. Questions should feel natural, not clinical

Output Format: Return ONLY a JSON array of 5 questions, nothing else.
Example: ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse JSON response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        questions = json.loads(text)
        
        if isinstance(questions, list) and len(questions) >= 3:
            print(f"[INFO] LLM generated {len(questions)} personalized questions")
            return questions
        else:
            raise ValueError("Invalid response format")
            
    except Exception as e:
        print(f"[ERROR] LLM question generation failed: {e}")
        return HARDCODED_QUESTIONS


def generate_next_question(demographics: Dict, conversation_history: list, question_number: int, total_questions: int = 5) -> str:
    """
    Generate the next personalized question based on demographics and conversation history.
    
    Args:
        demographics: Dictionary containing user demographic information
        conversation_history: List of dicts with 'question' and 'answer' keys
        question_number: Current question number (1-indexed)
        total_questions: Total number of questions to ask
        
    Returns:
        A single personalized question string
    """
    model = get_llm_model()
    
    if not model:
        print("[INFO] LLM unavailable, using hardcoded question")
        idx = min(question_number - 1, len(HARDCODED_QUESTIONS) - 1)
        return HARDCODED_QUESTIONS[idx]
    
    demo_context = _format_demographics(demographics)
    
    # Format conversation history
    conversation_text = ""
    if conversation_history:
        conversation_text = "\n\nConversation so far:\n"
        for i, entry in enumerate(conversation_history, 1):
            conversation_text += f"Q{i}: {entry.get('question', 'N/A')}\n"
            conversation_text += f"A{i}: {entry.get('answer', 'No response')}\n\n"
    
    # Determine question focus based on number
    focus_areas = [
        "their current emotional state and daily mood",
        "sleep patterns, energy levels, or fatigue",
        "social connections, loneliness, or support systems",
        "work/study stress, motivation, or concentration",
        "self-worth, future outlook, or things they enjoy"
    ]
    current_focus = focus_areas[min(question_number - 1, len(focus_areas) - 1)]
    
    prompt = f"""You are a compassionate mental health interviewer conducting a depression screening.

User Profile:
{demo_context}
{conversation_text}

This is question {question_number} of {total_questions}. Focus on: {current_focus}

Generate ONE short, highly personalized follow-up question.

CRITICAL REQUIREMENTS:
1. Be VERY SHORT - 10-20 words maximum
2. NAME USAGE RULES (VERY IMPORTANT):
   - If a name IS provided in the profile: use their EXACT first name in ONLY 2 out of every 5 questions (questions 1 and 3, or 2 and 4). The rest should NOT use the name — this feels more natural.
   - If NO name is provided: NEVER guess or invent a name. Just use "you" or be conversational.
   - Current question is #{question_number} — decide accordingly whether to include the name or not.
3. Reference SPECIFIC details from their profile or previous answers (role, stressors, living situation, etc.)
4. DO NOT ask generic questions - make it feel like you know them personally
5. Build on what they've shared before (if any conversation history exists)
6. Be warm and conversational, like a caring friend checking in
7. Avoid clinical language - keep it natural

Examples of GOOD personalized questions:
- With name (use sparingly): "Hey Sarah, how's hostel life been affecting your sleep?"
- Without name (most questions): "You mentioned academic pressure — what's been weighing on you most?"
- Without name: "With your mid-stage studies, do you still find joy in learning?"

Examples of BAD questions (avoid these):
- "How have you been feeling lately?" (too generic)
- "Can you tell me about your sleep?" (too clinical)
- "Dear friend, how are you?" (never invent names or use generic terms of address)

Return ONLY the question text, nothing else. No quotes, no explanation."""
    
    try:
        response = model.generate_content(prompt)
        question = response.text.strip()
        
        # Clean up any quotes or extra formatting
        question = question.strip('"\'')
        
        if len(question) > 10:  # Basic validation
            print(f"[INFO] LLM generated question {question_number}: {question[:50]}...")
            return question
        else:
            raise ValueError("Question too short")
            
    except Exception as e:
        print(f"[ERROR] LLM question generation failed: {e}")
        idx = min(question_number - 1, len(HARDCODED_QUESTIONS) - 1)
        return HARDCODED_QUESTIONS[idx]


def generate_recommendation(phq_score: float, demographics: Dict) -> Dict:
    """
    Generate personalized recommendation based on PHQ score and demographics.
    Falls back to hardcoded recommendations if LLM is unavailable.
    
    Args:
        phq_score: Average PHQ-8 score (0-24)
        demographics: Dictionary containing user demographic information
        
    Returns:
        Dictionary with recommendation details
    """
    severity = _get_severity_level(phq_score)
    
    model = get_llm_model()
    
    if not model:
        print("[INFO] Using hardcoded recommendations (LLM unavailable)")
        return _get_hardcoded_recommendation(severity, phq_score)
    
    demo_context = _format_demographics(demographics)
    
    prompt = f"""You are a compassionate, evidence-informed mental health advisor writing personalized recommendations after a depression screening.

User Profile:
{demo_context}

Screening Result:
- PHQ-8 Score: {phq_score}/24
- Severity Level: {severity.replace('_', ' ').title()}

IMPORTANT NAME RULES:
- If a name IS in the profile: use their first name ONCE in the summary or encouragement, not everywhere.
- If NO name is in the profile: NEVER guess or invent a name. Use "you" or general phrasing.

Return a JSON object with this exact structure:
{{
    "severity": "{severity.replace('_', ' ').title()} Depression",
    "summary": "<2 sentences addressing the person by context (e.g. their role, stressors). Acknowledge the screening result warmly. Be concise.>",
    "recommendations": [
        "<action 1: short, specific, doable this week — relate to their profile>",
        "<action 2: short & specific>",
        "<action 3: short & specific>",
        "<action 4: short & specific>"
    ],
    "resources": [
        "<resource 1: name + brief description — relevant to their location or role>",
        "<resource 2>"
    ],
    "encouragement": "<1 warm sentence — personal, not generic>"
}}

Style guide:
- Each recommendation must be 1 sentence, under 20 words, starting with a verb
- No bullet symbols, no numbering inside the strings
- Use their specific context (role, stressors, living situation) — never be vague
- Keep the summary grounded and supportive, never dramatic
- For scores >= 15: lead recommendations with professional help
- For scores < 5: keep tone light and affirming
- Resources should be concrete (apps, hotlines, campus services) not generic categories

Return ONLY valid JSON — no markdown, no code fences, no extra text."""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse JSON response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        recommendation = json.loads(text)
        recommendation["score"] = phq_score
        recommendation["llm_generated"] = True
        
        print("[INFO] LLM generated personalized recommendation")
        return recommendation
            
    except Exception as e:
        print(f"[ERROR] LLM recommendation generation failed: {e}")
        return _get_hardcoded_recommendation(severity, phq_score)


def _get_severity_level(score: float) -> str:
    """Get severity level from PHQ score."""
    if score < 5:
        return "minimal"
    elif score < 10:
        return "mild"
    elif score < 15:
        return "moderate"
    elif score < 20:
        return "moderately_severe"
    else:
        return "severe"


def _get_hardcoded_recommendation(severity: str, score: float) -> Dict:
    """Get hardcoded recommendation for a severity level."""
    rec = HARDCODED_RECOMMENDATIONS.get(severity, HARDCODED_RECOMMENDATIONS["minimal"]).copy()
    rec["score"] = score
    rec["llm_generated"] = False
    rec["encouragement"] = "Remember, seeking help is a sign of strength, not weakness. You've taken an important first step today."
    return rec


def _format_demographics(demographics: Dict) -> str:
    """Format demographics dictionary into readable context."""
    if not demographics:
        return "No demographic information provided."
    
    parts = []
    
    # Include name first for LLM personalization
    if demographics.get("name"):
        parts.append(f"- Name: {demographics['name']}")
    if demographics.get("age"):
        parts.append(f"- Age: {demographics['age']}")
    if demographics.get("gender"):
        parts.append(f"- Gender: {demographics['gender']}")
    if demographics.get("country"):
        parts.append(f"- Location: {demographics['country']}")
    if demographics.get("role"):
        parts.append(f"- Role: {demographics['role']}")
    if demographics.get("stage"):
        parts.append(f"- Current Stage: {demographics['stage']}")
    if demographics.get("focus"):
        parts.append(f"- Primary Focus: {demographics['focus']}")
    if demographics.get("sleep_duration"):
        parts.append(f"- Sleep: {demographics['sleep_duration']}")
    if demographics.get("workload"):
        parts.append(f"- Workload: {demographics['workload']}")
    if demographics.get("screen_time"):
        parts.append(f"- Screen Time: {demographics['screen_time']}")
    if demographics.get("living_situation"):
        parts.append(f"- Living Situation: {demographics['living_situation']}")
    if demographics.get("support_system"):
        parts.append(f"- Support System: {demographics['support_system']}")
    if demographics.get("stressors"):
        stressors = demographics['stressors']
        if isinstance(stressors, list) and stressors:
            parts.append(f"- Current Stressors: {', '.join(stressors)}")
    
    return "\n".join(parts) if parts else "No demographic information provided."


def is_llm_available() -> bool:
    """Check if LLM is available and configured."""
    return GENAI_AVAILABLE and bool(GEMINI_API_KEY)
