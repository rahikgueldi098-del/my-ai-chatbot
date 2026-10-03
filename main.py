import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="ChatGPT Clone")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []



HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #212121; color: #ececec; display: flex; flex-direction: column; height: 100vh; }
        header { padding: 15px 20px; border-bottom: 1px solid #333; text-align: center; font-weight: 600; font-size: 1.1rem; background: #171717; }
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }
        .message { display: flex; gap: 12px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; }
        .user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }
        .model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; }
        #input-container { padding: 20px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; gap: 10px; }
        #user-input { flex: 1; padding: 14px; border-radius: 24px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; }
        #user-input:focus { border-color: #666; }
        button { padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <header>AI Assistant</header>
    <div id="chat-box"></div>
    <div id="input-container">
        <input type="text" id="user-input" placeholder="Message AI Assistant..." onkeydown="if(event.key==='Enter') sendMessage()">
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        let conversationHistory = [];

        async function sendMessage() {
            const inputEl = document.getElementById("user-input");
            const btnEl = document.getElementById("send-btn");
            const text = inputEl.value.trim();
            if (!text) return;

            // Render user message
            appendMessage("user", text);
            inputEl.value = "";
            inputEl.disabled = true;
            btnEl.disabled = true;

            // Render temporary bot message loading indicator
            const botMessageEl = appendMessage("model", "Thinking...");

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text, history: conversationHistory })
                });

                const data = await response.json();

                if (data.response) {
                    botMessageEl.innerText = data.response;
                    // Keep track of memory context locally
                    conversationHistory.push({ role: "user", content: text });
                    conversationHistory.push({ role: "model", content: data.response });
                } else {
                    botMessageEl.innerText = "Error: " + (data.detail || "Something went wrong.");
                }
            } catch (err) {
                botMessageEl.innerText = "Error connecting to server.";
            } finally {
                inputEl.disabled = false;
                btnEl.disabled = false;
                inputEl.focus();
            }
        }

        function appendMessage(role, text) {
            const chatBox = document.getElementById("chat-box");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${role}`;
            msgDiv.innerText = text;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return msgDiv;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_CONTENT


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        contents = []
        for msg in request.history:
            contents.append(
                types.Content(
                    role=msg.role,
                    parts=[types.Part.from_text(text=msg.content)]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.message)]
            )
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents
        )

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))