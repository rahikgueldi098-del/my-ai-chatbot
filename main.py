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

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


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
    model_name: Optional[str] = "gemini-3.8-flash"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


@app.post("/chat")
def chat(request: ChatRequest):
    if not client:
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

    selected_model = request.model_name or "gemini-3.8-flash"

    def generate_stream():
        try:
            response_stream = client.models.generate_content_stream(
                model=selected_model,
                contents=contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as err:
            err_str = str(err)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                yield "⚠️ **Quota Exceeded**: Your Google API key has reached its free tier rate limit.\n\n*Please switch back to **Gemini 3.8 Flash** or attach billing in your Google AI Studio account to continue using Pro.*"
            else:
                yield f"⚠️ **Service Error**: {err_str}"

    return StreamingResponse(generate_stream(), media_type="text/plain")


# Frontend App UI (HTML / CSS / JS)
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Workspace Assistant</title>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js for Syntax Highlighting -->
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

        /* Sidebar */
        #sidebar { width: 280px; background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 16px; gap: 12px; }
        .new-chat-btn { background: var(--accent); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: background 0.2s; }
        .new-chat-btn:hover { background: var(--accent-hover); }
        .search-box { background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 8px 12px; border-radius: 6px; width: 100%; font-size: 14px; outline: none; }
        #chat-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
        .chat-item { padding: 10px 12px; border-radius: 6px; cursor: pointer; background: transparent; color: var(--text-secondary); transition: all 0.2s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
        .chat-item:hover, .chat-item.active { background: var(--bg-card); color: var(--text-primary); }
        .delete-btn { opacity: 0.6; padding: 2px 6px; border-radius: 4px; }
        .delete-btn:hover { opacity: 1; background: rgba(239, 68, 68, 0.2); color: #ef4444; }

        /* Main Workspace */
        #main-container { flex: 1; display: flex; flex-direction: column; }
        header { padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); display: flex; justify-content: space-between; align-items: center; }
        .header-title { font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .header-controls { display: flex; gap: 10px; align-items: center; }
        select, .toggle-btn { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 8px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; outline: none; }
        .toggle-btn.active { border-color: #10b981; color: #10b981; background: rgba(16, 185, 129, 0.1); }

        /* Chat Scroll Area */
        #chat-window { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
        .msg-row { display: flex; flex-direction: column; max-width: 85%; gap: 6px; }
        .msg-row.user { align-self: flex-end; }
        .msg-row.assistant { align-self: flex-start; }
        .bubble { padding: 14px 18px; border-radius: 12px; font-size: 15px; line-height: 1.6; word-break: break-word; }
        .msg-row.user .bubble { background: var(--user-bubble); color: white; border-bottom-right-radius: 2px; }
        .msg-row.assistant .bubble { background: var(--ai-bubble); border: 1px solid var(--border); color: var(--text-primary); border-bottom-left-radius: 2px; }

        /* Code Block & Formatting Styles */
        .bubble p { margin-bottom: 10px; }
        .bubble p:last-child { margin-bottom: 0; }
        .bubble code { font-family: "Fira Code", Consolas, Monaco, monospace; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        .bubble pre { background: var(--code-bg); padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); margin: 10px 0; position: relative; }
        .bubble pre code { background: transparent; padding: 0; }
        .copy-code-btn { position: absolute; top: 8px; right: 8px; background: var(--bg-card); border: 1px solid var(--border); color: var(--text-secondary); padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; }
        .copy-code-btn:hover { color: var(--text-primary); background: var(--border); }

        .msg-actions { display: flex; gap: 12px; font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
        .action-btn { background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 4px; }
        .action-btn:hover { color: var(--text-primary); text-decoration: underline; }

        /* File Preview Area */
        #file-preview { display: none; padding: 8px 16px; background: var(--bg-card); border-top: 1px solid var(--border); font-size: 13px; align-items: center; justify-content: space-between; }

        /* Input Controls */
        #input-container { padding: 16px 24px; background: var(--bg-secondary); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }
        #message-input { flex: 1; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 12px 16px; border-radius: 8px; font-size: 15px; resize: none; height: 48px; outline: none; line-height: 1.4; }
        .icon-btn { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); width: 44px; height: 44px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; transition: border-color 0.2s; }
        .icon-btn:hover { border-color: var(--accent); }
        .send-btn { background: var(--accent); color: white; border: none; padding: 0 20px; height: 44px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        .send-btn:hover { background: var(--accent-hover); }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div id="sidebar">
        <button class="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
        <input type="text" class="search-box" id="search-history" placeholder="Search chats..." oninput="filterHistory()">
        <div id="chat-list"></div>
    </div>

    <!-- Main Workspace -->
    <div id="main-container">
        <header>
            <div class="header-title">✨ AI Assistant Workspace</div>
            <div class="header-controls">
                <select id="model-select">
                    <option value="gemini-3.8-flash">⚡ Gemini 3.8 Flash (Fast)</option>
                    <option value="gemini-3.1-pro-preview">🧠 Gemini 3.1 Pro (Reasoning)</option>
                </select>
                <button id="search-toggle" class="toggle-btn active" onclick="toggleSearch()">🌐 Web Search: ON</button>
                <select id="persona-select">
                    <option value="You are a helpful, smart, and precise AI assistant.">🤖 Default Assistant</option>
                    <option value="You are an expert senior full-stack developer. Write clean, modern, efficient code with explanations.">💻 Code Specialist</option>
                    <option value="You are an executive strategy consultant. Provide concise, high-impact business advice.">📊 Business Strategist</option>
                    <option value="You are a creative writer. Craft rich, engaging, and expressive prose.">✍️ Creative Writer</option>
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
            <button class="icon-btn" onclick="document.getElementById('file-input').click()" title="Attach File">📎</button>
            <button class="icon-btn" id="mic-btn" onclick="toggleSpeechToText()" title="Voice Input">🎙️</button>
            <textarea id="message-input" placeholder="Type a message or command..." onkeydown="handleKeyDown(event)"></textarea>
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let conversations = JSON.parse(localStorage.getItem('ai_conversations') || '{}');
        let currentChatId = localStorage.getItem('ai_current_chat_id') || createChatId();
        let webSearchEnabled = true;
        let activeFile = null;
        let recognition = null;
        let isListening = false;

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
            initSTT();
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
            const items = document.querySelectorAll('.chat-item');
            items.forEach(item => {
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
            msgs.forEach((m, idx) => {
                const row = document.createElement('div');
                row.className = `msg-row ${m.role}`;

                let actions = '';
                let contentHtml = '';

                if (m.role === 'assistant') {
                    contentHtml = marked.parse(m.text || '');
                    actions = `<div class="msg-actions">
                        <button class="action-btn" onclick="speakText(${idx})">🔊 Read Aloud</button>
                        <button class="action-btn" onclick="regenerateFrom(${idx})">🔄 Regenerate</button>
                    </div>`;
                } else {
                    contentHtml = escapeHtml(m.text);
                    actions = `<div class="msg-actions">
                        <button class="action-btn" onclick="editPrompt(${idx})">✏️ Edit</button>
                    </div>`;
                }

                row.innerHTML = `<div class="bubble">${contentHtml}</div>${actions}`;
                win.appendChild(row);
            });

            // Highlight Code Blocks & Attach Copy Buttons
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
                const pre = block.parentElement;
                if (!pre.querySelector('.copy-code-btn')) {
                    const btn = document.createElement('button');
                    btn.className = 'copy-code-btn';
                    btn.innerText = 'Copy';
                    btn.onclick = () => {
                        navigator.clipboard.writeText(block.innerText);
                        btn.innerText = 'Copied!';
                        setTimeout(() => btn.innerText = 'Copy', 2000);
                    };
                    pre.appendChild(btn);
                }
            });

            win.scrollTop = win.scrollHeight;
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        async function sendMessage(customText = null) {
            const input = document.getElementById('message-input');
            const text = customText || input.value.trim();
            if (!text && !activeFile) return;

            const chat = conversations[currentChatId];
            if (chat.messages.length === 0) chat.title = text.slice(0, 24) || 'File Upload';

            chat.messages.push({ role: 'user', text: text });
            if (!customText) input.value = '';

            renderMessages();

            // Prepare API Payload
            const historyPayload = chat.messages.slice(0, -1);
            const persona = document.getElementById('persona-select').value;
            const selectedModel = document.getElementById('model-select').value;

            // Prepare Assistant Placeholder
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
                    const chunk = decoder.decode(value);
                    fullText += chunk;
                    targetBubble.innerHTML = marked.parse(fullText);
                    win.scrollTop = win.scrollHeight;
                }

                chat.messages[assistantIndex].text = fullText;
                removeFile();
                saveToStorage();
                renderSidebar();
                renderMessages();
            } catch (err) {
                targetBubble.innerText = "Error streaming response. Please try again.";
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
                const base64Data = evt.target.result.split(',')[1];
                activeFile = { mime_type: file.type, data: base64Data };
                document.getElementById('file-name-text').innerText = `📄 ${file.name}`;
                document.getElementById('file-preview').style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }

        function removeFile() {
            activeFile = null;
            document.getElementById('file-preview').style.display = 'none';
            document.getElementById('file-input').value = '';
        }

        function editPrompt(idx) {
            const chat = conversations[currentChatId];
            const oldText = chat.messages[idx].text;
            const newText = prompt("Edit your prompt:", oldText);
            if (newText !== null && newText.trim() !== "") {
                chat.messages = chat.messages.slice(0, idx);
                saveToStorage();
                sendMessage(newText.trim());
            }
        }

        function regenerateFrom(idx) {
            const chat = conversations[currentChatId];
            const lastUserMsgIndex = idx - 1;
            if (lastUserMsgIndex >= 0 && chat.messages[lastUserMsgIndex].role === 'user') {
                const textToResend = chat.messages[lastUserMsgIndex].text;
                chat.messages = chat.messages.slice(0, lastUserMsgIndex);
                saveToStorage();
                sendMessage(textToResend);
            }
        }

        function speakText(idx) {
            const chat = conversations[currentChatId];
            const text = chat.messages[idx].text;
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }

        function initSTT() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;

                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('message-input').value += ' ' + transcript;
                    stopSTT();
                };

                recognition.onerror = stopSTT;
                recognition.onend = stopSTT;
            }
        }

        function toggleSpeechToText() {
            if (!recognition) return alert("Speech recognition is not supported in this browser.");
            if (isListening) stopSTT(); else startSTT();
        }

        function startSTT() {
            isListening = true;
            document.getElementById('mic-btn').style.borderColor = '#ef4444';
            recognition.start();
        }

        function stopSTT() {
            isListening = false;
            document.getElementById('mic-btn').style.borderColor = 'var(--border)';
            if (recognition) recognition.stop();
        }

        init();
    </script>
</body>
</html>
"""