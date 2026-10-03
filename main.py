import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="ChatGPT Clone Backend")


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


@app.get("/")
def root():
    return {"status": "API Chatbot Active"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Build contents array containing chat history + new message
        contents = []

        for msg in request.history:
            contents.append(
                types.Content(
                    role=msg.role,
                    parts=[types.Part.from_text(text=msg.content)]
                )
            )

        # Append latest message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.message)]
            )
        )

        # Send full chat history context to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))