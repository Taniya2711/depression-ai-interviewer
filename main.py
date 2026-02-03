from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

questions = [
    "How have you been feeling lately?",
    "Have you felt little interest or pleasure in doing things?",
    "Have you been feeling down or hopeless?",
    "Have you had trouble sleeping?"
]

state = {"i": 0}

@app.get("/")
def home():
    return HTMLResponse(open("index.html").read())

@app.get("/start")
def start():
    state["i"] = 0
    return {"q": questions[0]}

@app.get("/reply")
def reply(text: str):
    state["i"] += 1
    if state["i"] < len(questions):
        return {"q": questions[state["i"]]}
    return {"q": "Interview completed. Thank you."}
