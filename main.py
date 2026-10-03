import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Enable CORS for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(model="gemini-3.8-flash")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "API Chatbot Active"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Retry up to 3 times if Google's servers return a 503 error
    for attempt in range(3):
        try:
            response = chat.send_message(request.message)
            return {"reply": response.text}
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                await asyncio.sleep(1)
                continue
            raise HTTPException(status_code=500, detail=str(e))