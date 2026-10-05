import os
import json
import base64
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="AI Assistant SaaS")

# Load multiple API keys for automatic key-rotation on 429 rate limits
raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or ""
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
clients = [genai.Client(api_key=key) for key in API_KEYS] if API_KEYS else []


# Pydantic Schemas
class FileData(BaseModel):
    mime_type: str
    data: str  # Base64 string


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    system_instruction: Optional[str] = "You are a helpful, smart, and precise AI assistant."
    file: Optional[FileData] = None
    history: Optional[List[ChatMessage]] = []
    enable_search: Optional[bool] = True
    model_name: Optional[str] = "gemini-2.0-flash"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


@app.post("/chat")
def chat(request: ChatRequest):
    if not clients:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")

    # Reconstruct history into GenAI Content structures
    contents = []
    for msg in request.history or []:
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.text)]))

    # Prepare current prompt parts
    current_parts = []
    if request.file:
        file_bytes = base64.b64decode(request.file.data)
        current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.file.mime_type))

    current_parts.append(types.Part.from_text(text=request.message))
    contents.append(types.Content(role="user", parts=current_parts))

    # Configure tools & persona
    tools = [types.Tool(google_search=types.GoogleSearch())] if request.enable_search else []
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=tools
    )

    requested_model = request.model_name or "gemini-2.0-flash"

    def generate_stream():
        # Try each configured API Key in sequence if a 429 quota error occurs
        for key_idx, client in enumerate(clients):
            try:
                response_stream = client.models.generate_content_stream(
                    model=requested_model,
                    contents=contents,
                    config=config
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return  # Stream completed successfully
            except Exception as err:
                err_str = str(err)
                is_quota_error = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                # If quota exhausted and more keys exist, failover to the next key
                if is_quota_error and key_idx < len(clients) - 1:
                    continue
                elif is_quota_error:
                    yield "⚠️ **Quota Exceeded**: All configured Google API keys have hit their daily limit. Please generate a new key in Google AI Studio or add billing to continue."
                    return
                else:
                    yield f"⚠️ **API Error**: {err_str}"
                    return

    return StreamingResponse(generate_stream(), media_type="text/plain")


# Frontend App UI
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Workspace Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <style>
        :root {
            --bg-primary: #121316;
            --bg-secondary: #1a1b1e;
            --bg-card: #25262b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #334155;
            --user-bubble: #2563eb;
            --ai-bubble: #1e293b;
            --code-bg: #0d1117;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { display: flex; height: 100vh; background: var(--bg-primary); color: var(--text-primary); overflow: hidden; }

        #sidebar { width: 280px; background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 16px; gap: 12px; }
        .new-chat-btn { background: var(--accent); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .new-chat-btn:hover { background: var(--accent-hover); }
        .search-box { background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 8px 12px; border-radius: 6px; width: 100%; font-size: 14px; outline: none; }
        #chat-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
        .chat-item { padding: 10px 12px; border-radius: 6px; cursor: pointer; background: transparent; color: var(--text-secondary); display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
        .chat-item:hover, .chat-item.active { background: var(--bg-card); color: var(--text-primary); }
        .delete-btn { opacity: 0.6; padding: 2px 6px; border-radius: 4px; }
        .delete-btn:hover { opacity: 1; background: rgba(239, 68, 68, 0.2); color: #ef4444; }

        #main-container { flex: 1; display: flex; flex-direction: column; }
        header { padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); display: flex; justify-content: space-between; align-items: center; }
        .header-title { font-size: 18px; font-weight: 600; }
        .header-controls { display: flex; gap: 10px; align-items: center; }
        select, .toggle-btn { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 8px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; outline: none; }
        .toggle-btn.active { border-color: #10b981; color: #10b981; background: rgba(16, 185, 129, 0.1); }

        #chat-window { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
        .msg-row { display: flex; flex-direction: column; max-width: 85%; gap: 6px; }
        .msg-row.user { align-self: flex-end; }
        .msg-row.assistant { align-self: flex-start; }
        .bubble { padding: 14px 18px; border-radius: 12px; font-size: 15px; line-height: 1.6; word-break: break-word; }
        .msg-row.user .bubble { background: var(--user-bubble); color: white; border-bottom-right-radius: 2px; }
        .msg-row.assistant .bubble { background: var(--ai-bubble); border: 1px solid var(--border); color: var(--text-primary); border-bottom-left-radius: 2px; }

        .bubble p { margin-bottom: 10px; }
        .bubble p:last-child { margin-bottom: 0; }
        .bubble code { font-family: monospace; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        .bubble pre { background: var(--code-bg); padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); margin: 10px 0; position: relative; }
        .bubble pre code { background: transparent; padding: 0; }
        .copy-code-btn { position: absolute; top: 8px; right: 8px; background: var(--bg-card); border: 1px solid var(--border); color: var(--text-secondary); padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; }

        .msg-actions { display: flex; gap: 12px; font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
        .action-btn { background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 12px; }
        .action-btn:hover { color: var(--text-primary); text-decoration: underline; }

        #file-preview { display: none; padding: 8px 16px; background: var(--bg-card); border-top: 1px solid var(--border); font-size: 13px; align-items: center; justify-content: space-between; }

        #input-container { padding: 16px 24px; background: var(--bg-secondary); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }
        #message-input { flex: 1; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 12px 16px; border-radius: 8px; font-size: 15px; resize: none; height: 48px; outline: none; }
        .icon-btn { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); width: 44px; height: 44px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; }
        .send-btn { background: var(--accent); color: white; border: none; padding: 0 20px; height: 44px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    </style>
</head>
<body>

    <div id="sidebar">
        <button class="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
        <input type="text" class="search-box" id="search-history" placeholder="Search chats..." oninput="filterHistory()">
        <div id="chat-list"></div>
    </div>

    <div id="main-container">
        <header>
            <div class="header-title">✨ AI Assistant Workspace</div>
            <div class="header-controls">
                <select id="model-select">
                    <option value="gemini-2.0-flash">⚡ Gemini 2.0 Flash</option>
                    <option value="gemini-1.5-pro">🧠 Gemini 1.5 Pro</option>
                </select>
                <button id="search-toggle" class="toggle-btn active" onclick="toggleSearch()">🌐 Web Search: ON</button>
                <select id="persona-select">
                    <option value="You are a helpful, smart, and precise AI assistant.">🤖 Default Assistant</option>
                    <option value="You are an expert senior full-stack developer. Write clean code.">💻 Code Specialist</option>
                </select>
            </div>
        </header>

        <div id="chat-window"></div>

        <div id="file-preview">
            <span id="file-name-text"></span>
            <button class="action-btn" onclick="removeFile()">✕ Remove</button>
        </div>

        <div id="input-container">
            <input type="file" id="file-input" style="display:none;" onchange="handleFileSelect(event)">
            <button class="icon-btn" onclick="document.getElementById('file-input').click()">📎</button>
            <button class="icon-btn" id="mic-btn" onclick="toggleSpeechToText()">🎙️</button>
            <textarea id="message-input" placeholder="Type a message..." onkeydown="handleKeyDown(event)"></textarea>
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let conversations = JSON.parse(localStorage.getItem('ai_conversations') || '{}');
        let currentChatId = localStorage.getItem('ai_current_chat_id') || createChatId();
        let webSearchEnabled = true;
        let activeFile = null;

        function createChatId() { return 'chat_' + Date.now(); }

        function saveToStorage() {
            localStorage.setItem('ai_conversations', JSON.stringify(conversations));
            localStorage.setItem('ai_current_chat_id', currentChatId);
        }

        function init() {
            if (!conversations[currentChatId]) {
                conversations[currentChatId] = { title: 'New Conversation', messages: [] };
            }
            renderSidebar();
            renderMessages();
        }

        function renderSidebar() {
            const list = document.getElementById('chat-list');
            list.innerHTML = '';
            Object.keys(conversations).reverse().forEach(id => {
                const chat = conversations[id];
                const item = document.createElement('div');
                item.className = `chat-item ${id === currentChatId ? 'active' : ''}`;
                item.onclick = () => switchChat(id);
                item.innerHTML = `<span>💬 ${chat.title}</span><span class="delete-btn" onclick="deleteChat(event, '${id}')">🗑️</span>`;
                list.appendChild(item);
            });
        }

        function filterHistory() {
            const query = document.getElementById('search-history').value.toLowerCase();
            document.querySelectorAll('.chat-item').forEach(item => {
                item.style.display = item.innerText.toLowerCase().includes(query) ? 'flex' : 'none';
            });
        }

        function switchChat(id) {
            currentChatId = id;
            saveToStorage();
            renderSidebar();
            renderMessages();
        }

        function startNewChat() {
            currentChatId = createChatId();
            conversations[currentChatId] = { title: 'New Conversation', messages: [] };
            saveToStorage();
            renderSidebar();
            renderMessages();
        }

        function deleteChat(e, id) {
            e.stopPropagation();
            delete conversations[id];
            if (currentChatId === id) {
                const remaining = Object.keys(conversations);
                currentChatId = remaining.length ? remaining[0] : createChatId();
                if (!conversations[currentChatId]) conversations[currentChatId] = { title: 'New Conversation', messages: [] };
            }
            saveToStorage();
            renderSidebar();
            renderMessages();
        }

        function toggleSearch() {
            webSearchEnabled = !webSearchEnabled;
            const btn = document.getElementById('search-toggle');
            btn.className = `toggle-btn ${webSearchEnabled ? 'active' : ''}`;
            btn.innerText = `🌐 Web Search: ${webSearchEnabled ? 'ON' : 'OFF'}`;
        }

        function renderMessages() {
            const win = document.getElementById('chat-window');
            win.innerHTML = '';
            const msgs = conversations[currentChatId].messages;
            msgs.forEach((m) => {
                const row = document.createElement('div');
                row.className = `msg-row ${m.role}`;
                const contentHtml = m.role === 'assistant' ? marked.parse(m.text || '') : escapeHtml(m.text);
                row.innerHTML = `<div class="bubble">${contentHtml}</div>`;
                win.appendChild(row);
            });

            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });

            win.scrollTop = win.scrollHeight;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        async function sendMessage() {
            const input = document.getElementById('message-input');
            const text = input.value.trim();
            if (!text && !activeFile) return;

            const chat = conversations[currentChatId];
            if (chat.messages.length === 0) chat.title = text.slice(0, 24) || 'File Upload';

            chat.messages.push({ role: 'user', text: text });
            input.value = '';
            renderMessages();

            const historyPayload = chat.messages.slice(0, -1);
            const persona = document.getElementById('persona-select').value;
            const selectedModel = document.getElementById('model-select').value;

            chat.messages.push({ role: 'assistant', text: '' });
            const assistantIndex = chat.messages.length - 1;
            renderMessages();

            const win = document.getElementById('chat-window');
            const bubbles = win.querySelectorAll('.msg-row.assistant .bubble');
            const targetBubble = bubbles[bubbles.length - 1];

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        system_instruction: persona,
                        model_name: selectedModel,
                        file: activeFile,
                        history: historyPayload,
                        enable_search: webSearchEnabled
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    fullText += decoder.decode(value);
                    targetBubble.innerHTML = marked.parse(fullText);
                    win.scrollTop = win.scrollHeight;
                }

                chat.messages[assistantIndex].text = fullText;
                removeFile();
                saveToStorage();
            } catch (err) {
                targetBubble.innerText = "Error communicating with server.";
            }
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(evt) {
                activeFile = { mime_type: file.type, data: evt.target.result.split(',')[1] };
                document.getElementById('file-name-text').innerText = `📄 ${file.name}`;
                document.getElementById('file-preview').style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }

        function removeFile() {
            activeFile = null;
            document.getElementById('file-preview').style.display = 'none';
        }

        init();
    </script>
</body>
</html>
"""