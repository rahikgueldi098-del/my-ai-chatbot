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
    role: str  # "user" or "model"
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
    <!-- Marked for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js for Code Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #212121; color: #ececec; display: flex; height: 100vh; overflow: hidden; }

        /* Sidebar Layout */
        #sidebar { width: 260px; background-color: #171717; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; }
        #new-chat-btn { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 8px; padding: 10px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: 0.2s; }
        #new-chat-btn:hover { background: #383838; }

        /* Main Chat Area */
        #main-container { flex: 1; display: flex; flex-direction: column; height: 100vh; }
        header { padding: 15px 20px; border-bottom: 1px solid #333; text-align: center; font-weight: 600; font-size: 1.1rem; background: #171717; }
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }
        .message { display: flex; flex-direction: column; gap: 6px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; }
        .user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }
        .model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; width: 100%; }

        /* Markdown formatting styles */
        .message p { margin-bottom: 8px; }
        .message p:last-child { margin-bottom: 0; }
        .message ul, .message ol { margin-left: 20px; margin-bottom: 8px; }
        .message code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }

        /* Code Block Container with Copy Button */
        .code-container { position: relative; margin: 10px 0; }
        .message pre { background: #0d1117; padding: 14px; border-radius: 8px; overflow-x: auto; border: 1px solid #30363d; margin: 0; }
        .message pre code { background: transparent; padding: 0; }
        .copy-btn { position: absolute; top: 8px; right: 8px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 0.75rem; cursor: pointer; transition: 0.2s; }
        .copy-btn:hover { background: #30363d; color: #fff; }

        /* Multi-line Auto-Expanding Textarea */
        #input-container { padding: 20px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
        #user-input { flex: 1; padding: 12px 16px; border-radius: 18px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; resize: none; max-height: 150px; min-height: 48px; line-height: 1.4; }
        #user-input:focus { border-color: #666; }
        #send-btn { height: 48px; padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
        #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <div id="sidebar">
        <button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
    </div>

    <div id="main-container">
        <header>AI Assistant</header>
        <div id="chat-box"></div>
        <div id="input-container">
            <textarea id="user-input" placeholder="Message AI Assistant... (Shift + Enter for new line)" rows="1" onkeydown="handleKeyDown(event)" oninput="autoExpand(this)"></textarea>
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let conversationHistory = [];

        function startNewChat() {
            conversationHistory = [];
            document.getElementById("chat-box").innerHTML = "";
            const inputEl = document.getElementById("user-input");
            inputEl.value = "";
            inputEl.style.height = "48px";
            inputEl.focus();
        }

        function autoExpand(field) {
            field.style.height = 'inherit';
            const computed = window.getComputedStyle(field);
            const height = parseInt(computed.getPropertyValue('border-top-width'), 10)
                         + parseInt(computed.getPropertyValue('padding-top'), 10)
                         + field.scrollHeight
                         + parseInt(computed.getPropertyValue('padding-bottom'), 10)
                         + parseInt(computed.getPropertyValue('border-bottom-width'), 10);
            field.style.height = Math.min(height, 150) + 'px';
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const inputEl = document.getElementById("user-input");
            const btnEl = document.getElementById("send-btn");
            const text = inputEl.value.trim();
            if (!text) return;

            appendMessage("user", text);
            inputEl.value = "";
            inputEl.style.height = "48px";
            inputEl.disabled = true;
            btnEl.disabled = true;

            const botMessageEl = appendMessage("model", "Thinking...");

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text, history: conversationHistory })
                });

                const data = await response.json();

                if (!response.ok) {
                    botMessageEl.innerText = "Error: " + (data.detail || "Server returned an error");
                    return;
                }

                if (data.response) {
                    botMessageEl.innerHTML = marked.parse(data.response);
                    hljs.highlightAll();
                    addCopyButtons(botMessageEl);

                    conversationHistory.push({ role: "user", content: text });
                    conversationHistory.push({ role: "model", content: data.response });
                }
            } catch (err) {
                botMessageEl.innerText = "Error connecting to server.";
            } finally {
                inputEl.disabled = false;
                btnEl.disabled = false;
                inputEl.focus();
            }
        }

        function addCopyButtons(container) {
            const preBlocks = container.querySelectorAll("pre");
            preBlocks.forEach((pre) => {
                const wrapper = document.createElement("div");
                wrapper.className = "code-container";
                pre.parentNode.insertBefore(wrapper, pre);
                wrapper.appendChild(pre);

                const btn = document.createElement("button");
                btn.className = "copy-btn";
                btn.innerText = "Copy";
                btn.onclick = () => {
                    const code = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
                    navigator.clipboard.writeText(code);
                    btn.innerText = "Copied!";
                    setTimeout(() => { btn.innerText = "Copy"; }, 2000);
                };
                wrapper.appendChild(btn);
            });
        }

        function appendMessage(role, text) {
            const chatBox = document.getElementById("chat-box");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${role}`;

            if (role === "user") {
                msgDiv.innerText = text;
            } else {
                msgDiv.innerHTML = text === "Thinking..." ? text : marked.parse(text);
            }

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

        # Automatic model fallback list
        models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return {"response": response.text}
            except Exception as model_err:
                err_str = str(model_err)
                # Catch 503 (Server Busy), 429 (Quota Exceeded), and NOT_FOUND
                if any(code in err_str for code in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND"]):
                    continue
                raise model_err

        raise HTTPException(status_code=429,
                            detail="All free tier model quotas have been reached for today. Please try again later.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))