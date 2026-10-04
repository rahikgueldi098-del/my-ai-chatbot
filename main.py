import os
import base64
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

app = FastAPI(title="ChatGPT Clone")


class FileData(BaseModel):
    mime_type: str
    data_base64: str
    name: Optional[str] = "file"


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str
    file: Optional[FileData] = None


class ChatRequest(BaseModel):
    message: str
    system_instruction: Optional[str] = "You are a helpful, smart, and precise AI assistant."
    file: Optional[FileData] = None
    history: Optional[List[ChatMessage]] = []
    enable_search: Optional[bool] = True


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <!-- Marked for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js for Code Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for Math Equations -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script>

    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #212121; color: #ececec; display: flex; height: 100vh; overflow: hidden; }

        /* Sidebar Layout */
        #sidebar { width: 260px; background-color: #171717; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; gap: 15px; }
        #new-chat-btn { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 8px; padding: 10px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: 0.2s; }
        #new-chat-btn:hover { background: #383838; }

        #history-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
        .history-item { padding: 10px 12px; border-radius: 6px; font-size: 0.88rem; color: #b4b4b4; cursor: pointer; transition: 0.2s; display: flex; justify-content: space-between; align-items: center; }
        .history-item:hover { background: #2f2f2f; color: #fff; }
        .history-item.active { background: #212121; color: #fff; font-weight: 500; }
        .history-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }

        .delete-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.9rem; padding: 2px 6px; border-radius: 4px; display: none; }
        .history-item:hover .delete-btn { display: block; }
        .delete-btn:hover { color: #ff5555; background: #3a2222; }

        /* Sidebar Export Tools */
        .export-box { border-top: 1px solid #333; padding-top: 12px; display: flex; flex-direction: column; gap: 8px; }
        .export-title { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
        .export-buttons { display: flex; gap: 8px; }
        .export-btn { flex: 1; background: #2f2f2f; color: #ccc; border: 1px solid #424242; border-radius: 6px; padding: 6px; font-size: 0.8rem; cursor: pointer; text-align: center; transition: 0.2s; }
        .export-btn:hover { background: #383838; color: #fff; }

        /* Main Chat Area */
        #main-container { flex: 1; display: flex; flex-direction: column; height: 100vh; }
        header { padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; background: #171717; font-weight: 600; }
        .header-controls { display: flex; gap: 12px; align-items: center; }
        #persona-select { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.88rem; outline: none; cursor: pointer; }
        .toggle-btn { background: #2f2f2f; color: #888; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: 0.2s; }
        .toggle-btn.active { background: #1b3a2b; color: #4ade80; border-color: #22c55e; }

        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }
        .message { display: flex; flex-direction: column; gap: 6px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; position: relative; }
        .user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }
        .model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; width: 100%; }

        .message img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
        .doc-badge { display: inline-flex; align-items: center; gap: 8px; background: #1e293b; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-size: 0.88rem; color: #38bdf8; word-break: break-all; }
        .doc-icon { font-size: 1.1rem; }

        .message p { margin-bottom: 8px; }
        .message p:last-child { margin-bottom: 0; }
        .message ul, .message ol { margin-left: 20px; margin-bottom: 8px; }
        .message code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }

        .code-container { position: relative; margin: 10px 0; }
        .message pre { background: #0d1117; padding: 14px; border-radius: 8px; overflow-x: auto; border: 1px solid #30363d; margin: 0; }
        .message pre code { background: transparent; padding: 0; }
        .copy-btn { position: absolute; top: 8px; right: 8px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 0.75rem; cursor: pointer; transition: 0.2s; }
        .copy-btn:hover { background: #30363d; color: #fff; }

        /* Action bar for messages */
        .msg-actions { display: flex; gap: 8px; margin-top: 8px; padding-top: 6px; border-top: 1px solid #2a2a2a; }
        .action-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.85rem; padding: 2px 6px; border-radius: 4px; transition: 0.2s; display: flex; align-items: center; gap: 4px; }
        .action-btn:hover { color: #fff; background: #2f2f2f; }

        /* Typing Dots Animation */
        .typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 4px 0; }
        .typing-dot { width: 6px; height: 6px; background: #aaa; border-radius: 50%; animation: blink 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes blink { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }

        /* Input Area & Attachment Preview */
        #input-wrapper { padding: 20px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; }
        #file-preview { display: none; align-items: center; gap: 10px; background: #2f2f2f; padding: 8px 12px; border-radius: 8px; border: 1px solid #424242; width: fit-content; font-size: 0.85rem; }
        #preview-img { height: 40px; width: 40px; object-fit: cover; border-radius: 4px; display: none; }
        #preview-icon { font-size: 1.5rem; display: none; }
        #remove-file-btn { background: transparent; border: none; color: #ff5555; cursor: pointer; font-size: 1rem; margin-left: 6px; }

        #input-container { display: flex; gap: 10px; align-items: flex-end; }
        #file-input { display: none; }
        .icon-btn { height: 48px; width: 48px; border-radius: 24px; border: 1px solid #424242; background: #2f2f2f; color: #fff; font-size: 1.2rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: 0.2s; flex-shrink: 0; }
        .icon-btn:hover { background: #383838; }
        .icon-btn.recording { background: #3a1c1c; border-color: #ef4444; color: #ef4444; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

        #user-input { flex: 1; padding: 12px 16px; border-radius: 18px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; resize: none; max-height: 150px; min-height: 48px; line-height: 1.4; }
        #user-input:focus { border-color: #666; }

        #send-btn { height: 48px; min-width: 70px; padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; transition: 0.2s; flex-shrink: 0; }
        #send-btn.stop-btn { background: #ef4444; color: #fff; }
        #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <div id="sidebar">
        <button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
        <div id="history-list"></div>

        <div class="export-box">
            <span class="export-title">Export Discussion</span>
            <div class="export-buttons">
                <button class="export-btn" onclick="exportChat('md')">📄 Markdown</button>
                <button class="export-btn" onclick="exportChat('json')">📦 JSON</button>
            </div>
        </div>
    </div>

    <div id="main-container">
        <header>
            <span>AI Assistant</span>
            <div class="header-controls">
                <button id="search-toggle" class="toggle-btn active" onclick="toggleSearch()">🌐 Web Search: ON</button>
                <select id="persona-select">
                    <option value="You are a helpful, smart, and precise AI assistant.">🤖 Default Assistant</option>
                    <option value="You are a Senior Full-Stack Software Engineer. Provide clean, efficient code and explain tech concepts concisely.">💻 Senior Engineer</option>
                    <option value="You are a strict, ultra-concise assistant. Answer using minimal words and direct bullet points only. No fluff.">⚡ Ultra-Concise Mode</option>
                    <option value="You are a creative writer and storytelling assistant with a rich, expressive vocabulary.">✍️ Creative Writer</option>
                </select>
            </div>
        </header>
        <div id="chat-box"></div>
        <div id="input-wrapper">
            <div id="file-preview">
                <img id="preview-img" src="" alt="preview">
                <span id="preview-icon">📄</span>
                <span id="file-name"></span>
                <button id="remove-file-btn" onclick="clearFile()">✕</button>
            </div>
            <div id="input-container">
                <input type="file" id="file-input" accept="image/*,.pdf,.txt,.csv,.md,.json,.py,.js,.html,.css" onchange="handleFileSelect(event)">
                <button id="attach-btn" class="icon-btn" onclick="document.getElementById('file-input').click()" title="Joindre un fichier (Image, PDF, TXT, CSV...)">📎</button>
                <button id="mic-btn" class="icon-btn" onclick="toggleSpeechRecognition()" title="Dictée vocale">🎙</button>
                <textarea id="user-input" placeholder="Message AI Assistant... (Shift + Enter pour ligne suivante)" rows="1" onkeydown="handleKeyDown(event)" oninput="autoExpand(this)"></textarea>
                <button id="send-btn" onclick="handleSendOrStop()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let currentChatId = null;
        let chats = JSON.parse(localStorage.getItem('ai_chats') || '{}');
        let currentFile = null;
        let webSearchEnabled = true;
        let recognition = null;
        let isRecording = false;
        let baseTranscript = '';
        let finalizedTranscript = '';

        let isGenerating = false;
        let abortController = null;

        window.onload = () => {
            renderSidebar();
            const keys = Object.keys(chats);
            if (keys.length > 0) {
                loadChat(keys[0]);
            } else {
                startNewChat();
            }
            initSpeechRecognition();
        };

        function initSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) return;

            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onresult = (event) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalizedTranscript += transcript + ' ';
                    } else {
                        interimTranscript += transcript;
                    }
                }

                const inputEl = document.getElementById("user-input");
                inputEl.value = baseTranscript + finalizedTranscript + interimTranscript;
                autoExpand(inputEl);
            };

            recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                stopRecording();
            };

            recognition.onend = () => {
                stopRecording();
            };
        }

        function toggleSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert("Votre navigateur ne supporte pas la dictée vocale.");
                return;
            }

            if (isRecording) {
                if (recognition) recognition.stop();
                stopRecording();
            } else {
                try {
                    if (!recognition) initSpeechRecognition();

                    const inputEl = document.getElementById("user-input");
                    baseTranscript = inputEl.value ? inputEl.value.trim() + " " : "";
                    finalizedTranscript = '';

                    recognition.start();
                    isRecording = true;
                    const micBtn = document.getElementById('mic-btn');
                    micBtn.classList.add('recording');
                    micBtn.title = "Arrêter l'enregistrement";
                } catch (e) {
                    console.error("Erreur lancement dictée :", e);
                }
            }
        }

        function stopRecording() {
            isRecording = false;
            const micBtn = document.getElementById('mic-btn');
            if (micBtn) {
                micBtn.classList.remove('recording');
                micBtn.title = "Dictée vocale";
            }
        }

        function toggleSearch() {
            webSearchEnabled = !webSearchEnabled;
            const btn = document.getElementById('search-toggle');
            if (webSearchEnabled) {
                btn.classList.add('active');
                btn.innerText = '🌐 Web Search: ON';
            } else {
                btn.classList.remove('active');
                btn.innerText = '🌐 Web Search: OFF';
            }
        }

        function saveChats() {
            localStorage.setItem('ai_chats', JSON.stringify(chats));
        }

        function startNewChat() {
            currentChatId = Date.now().toString();
            chats[currentChatId] = { title: "Nouvelle discussion", history: [] };
            saveChats();
            renderSidebar();
            loadChat(currentChatId);
        }

        function deleteChat(id, event) {
            event.stopPropagation();
            delete chats[id];
            saveChats();

            const keys = Object.keys(chats).sort((a, b) => b - a);
            if (id === currentChatId) {
                if (keys.length > 0) {
                    loadChat(keys[0]);
                } else {
                    startNewChat();
                }
            } else {
                renderSidebar();
            }
        }

        function renderSidebar() {
            const listEl = document.getElementById("history-list");
            listEl.innerHTML = "";
            const keys = Object.keys(chats).sort((a, b) => b - a);

            keys.forEach(id => {
                const item = document.createElement("div");
                item.className = `history-item ${id === currentChatId ? 'active' : ''}`;

                const titleSpan = document.createElement("span");
                titleSpan.className = "history-title";
                titleSpan.innerText = chats[id].title || "Discussion";

                const delBtn = document.createElement("button");
                delBtn.className = "delete-btn";
                delBtn.innerHTML = "🗑";
                delBtn.title = "Supprimer la discussion";
                delBtn.onclick = (e) => deleteChat(id, e);

                item.appendChild(titleSpan);
                item.appendChild(delBtn);
                item.onclick = () => loadChat(id);
                listEl.appendChild(item);
            });
        }

        function renderMath(element) {
            if (window.renderMathInElement) {
                renderMathInElement(element, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\\\(', right: '\\\\)', display: false},
                        {left: '\\\\[', right: '\\\\]', display: true}
                    ],
                    throwOnError: false
                });
            }
        }

        function renderChatHistory() {
            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML = "";

            const history = chats[currentChatId]?.history || [];
            history.forEach((msg, index) => {
                appendMessage(msg.role, msg.content, msg.file, index);
            });
            hljs.highlightAll();
        }

        function loadChat(id) {
            currentChatId = id;
            renderSidebar();
            renderChatHistory();
        }

        function exportChat(format) {
            if (!currentChatId || !chats[currentChatId]) return;
            const chat = chats[currentChatId];
            const title = (chat.title || "chat").replace(/[^a-z0-9]/gi, '_').toLowerCase();

            let content = "";
            let mimeType = "";
            let extension = "";

            if (format === 'md') {
                content = `# ${chat.title || 'Discussion'}\n\n`;
                chat.history.forEach(msg => {
                    const roleName = msg.role === 'user' ? 'User' : 'AI Assistant';
                    content += `### ${roleName}\n${msg.content}\n\n---\n\n`;
                });
                mimeType = 'text/markdown';
                extension = 'md';
            } else if (format === 'json') {
                content = JSON.stringify(chat, null, 2);
                mimeType = 'application/json';
                extension = 'json';
            }

            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${title}.${extension}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function speakText(btn, text) {
            if (!('speechSynthesis' in window)) {
                alert("La synthèse vocale n'est pas supportée par votre navigateur.");
                return;
            }

            if (window.speechSynthesis.speaking) {
                window.speechSynthesis.cancel();
                btn.innerText = "🔊 Read Aloud";
                return;
            }

            const cleanText = text.replace(/[*_#`$]/g, '');
            const utterance = new SpeechSynthesisUtterance(cleanText);

            utterance.onstart = () => { btn.innerText = "⏹ Stop"; };
            utterance.onend = () => { btn.innerText = "🔊 Read Aloud"; };
            utterance.onerror = () => { btn.innerText = "🔊 Read Aloud"; };

            window.speechSynthesis.speak(utterance);
        }

        function getFileIcon(mimeType, filename) {
            if (mimeType.startsWith('image/')) return '📷';
            if (mimeType === 'application/pdf' || filename.endsWith('.pdf')) return '📕';
            if (mimeType.includes('csv') || filename.endsWith('.csv')) return '📊';
            if (mimeType.includes('json') || filename.endsWith('.json')) return '📦';
            return '📄';
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                const base64Data = e.target.result.split(',')[1];
                let mimeType = file.type || 'text/plain';

                // Fallback mime types based on extension
                if (file.name.endsWith('.csv')) mimeType = 'text/csv';
                else if (file.name.endsWith('.pdf')) mimeType = 'application/pdf';
                else if (file.name.endsWith('.md')) mimeType = 'text/markdown';
                else if (file.name.endsWith('.py') || file.name.endsWith('.js') || file.name.endsWith('.html') || file.name.endsWith('.css')) mimeType = 'text/plain';

                currentFile = {
                    mime_type: mimeType,
                    data_base64: base64Data,
                    name: file.name
                };

                const previewImg = document.getElementById("preview-img");
                const previewIcon = document.getElementById("preview-icon");

                if (mimeType.startsWith('image/')) {
                    previewImg.src = e.target.result;
                    previewImg.style.display = "block";
                    previewIcon.style.display = "none";
                } else {
                    previewImg.style.display = "none";
                    previewIcon.innerText = getFileIcon(mimeType, file.name);
                    previewIcon.style.display = "block";
                }

                document.getElementById("file-name").innerText = file.name;
                document.getElementById("file-preview").style.display = "flex";
            };
            reader.readAsDataURL(file);
        }

        function clearFile() {
            currentFile = null;
            document.getElementById("file-input").value = "";
            document.getElementById("file-preview").style.display = "none";
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
                handleSendOrStop();
            }
        }

        function handleSendOrStop() {
            if (isGenerating) {
                if (abortController) abortController.abort();
            } else {
                sendMessage();
            }
        }

        function updateSendButton(generating) {
            isGenerating = generating;
            const btnEl = document.getElementById("send-btn");
            if (generating) {
                btnEl.innerText = "⏹ Stop";
                btnEl.classList.add("stop-btn");
            } else {
                btnEl.innerText = "Send";
                btnEl.classList.remove("stop-btn");
            }
        }

        async function sendMessage(overrideText = null, overrideFile = null) {
            if (isRecording) {
                toggleSpeechRecognition();
            }

            const inputEl = document.getElementById("user-input");
            const personaEl = document.getElementById("persona-select");

            const text = overrideText !== null ? overrideText : inputEl.value.trim();
            const activeFile = overrideFile !== null ? overrideFile : currentFile;

            if (!text && !activeFile) return;

            if (!currentChatId || !chats[currentChatId]) {
                startNewChat();
            }

            const currentHistory = chats[currentChatId].history;

            if (currentHistory.length === 0) {
                const titleText = text || (activeFile ? activeFile.name : "Fichier envoyé");
                chats[currentChatId].title = titleText.length > 25 ? titleText.substring(0, 25) + "..." : titleText;
                renderSidebar();
            }

            // Append user turn
            currentHistory.push({ role: "user", content: text, file: activeFile });
            saveChats();

            inputEl.value = "";
            inputEl.style.height = "48px";
            clearFile();
            inputEl.disabled = true;

            renderChatHistory();

            const botMessageEl = appendMessage("model", "", null, currentHistory.length);
            botMessageEl.querySelector('.text-content').innerHTML = '<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

            let accumulatedText = "";
            abortController = new AbortController();
            updateSendButton(true);

            // Extract prompt history before current turn
            const historyForBackend = currentHistory.slice(0, -1);

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    signal: abortController.signal,
                    body: JSON.stringify({ 
                        message: text, 
                        system_instruction: personaEl.value,
                        file: activeFile, 
                        history: historyForBackend,
                        enable_search: webSearchEnabled
                    })
                });

                if (!response.ok) {
                    botMessageEl.querySelector('.text-content').innerText = "Erreur de communication avec le serveur.";
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    accumulatedText += chunk;

                    botMessageEl.querySelector('.text-content').innerHTML = marked.parse(accumulatedText);
                    document.getElementById("chat-box").scrollTop = document.getElementById("chat-box").scrollHeight;
                }

                hljs.highlightAll();
                renderMath(botMessageEl.querySelector('.text-content'));
                addCopyButtons(botMessageEl);

                currentHistory.push({ role: "model", content: accumulatedText });
                saveChats();
                renderChatHistory();

            } catch (err) {
                if (err.name === 'AbortError') {
                    if (accumulatedText.trim()) {
                        accumulatedText += " *(Interrompu)*";
                        currentHistory.push({ role: "model", content: accumulatedText });
                        saveChats();
                        renderChatHistory();
                    } else {
                        currentHistory.pop();
                        saveChats();
                        renderChatHistory();
                    }
                } else {
                    botMessageEl.querySelector('.text-content').innerText = "Erreur de connexion au serveur.";
                }
            } finally {
                abortController = null;
                updateSendButton(false);
                inputEl.disabled = false;
                inputEl.focus();
            }
        }

        function regenerateResponse(aiMsgIndex) {
            if (isGenerating) return;

            const currentHistory = chats[currentChatId].history;
            currentHistory.splice(aiMsgIndex);

            let lastUserIndex = currentHistory.length - 1;
            while (lastUserIndex >= 0 && currentHistory[lastUserIndex].role !== "user") {
                lastUserIndex--;
            }

            if (lastUserIndex < 0) return;

            const lastUserMsg = currentHistory[lastUserIndex];
            currentHistory.splice(lastUserIndex, 1);
            saveChats();

            sendMessage(lastUserMsg.content, lastUserMsg.file);
        }

        function editUserMessage(msgIndex) {
            if (isGenerating) return;

            const currentHistory = chats[currentChatId].history;
            const msgToEdit = currentHistory[msgIndex];

            const inputEl = document.getElementById("user-input");
            inputEl.value = msgToEdit.content;
            autoExpand(inputEl);

            currentHistory.splice(msgIndex);
            saveChats();
            renderChatHistory();

            inputEl.focus();
        }

        function addCopyButtons(container) {
            const preBlocks = container.querySelectorAll("pre");
            preBlocks.forEach((pre) => {
                if (pre.parentNode.classList.contains("code-container")) return;
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

        function attachActions(msgDiv, text, role, msgIndex) {
            if (msgDiv.querySelector('.msg-actions')) return;
            const actionsDiv = document.createElement("div");
            actionsDiv.className = "msg-actions";

            if (role === "model") {
                const speakBtn = document.createElement("button");
                speakBtn.className = "action-btn";
                speakBtn.innerText = "🔊 Read Aloud";
                speakBtn.onclick = () => speakText(speakBtn, text);
                actionsDiv.appendChild(speakBtn);

                const regenBtn = document.createElement("button");
                regenBtn.className = "action-btn";
                regenBtn.innerText = "🔄 Regenerate";
                regenBtn.onclick = () => regenerateResponse(msgIndex);
                actionsDiv.appendChild(regenBtn);
            } else if (role === "user") {
                const editBtn = document.createElement("button");
                editBtn.className = "action-btn";
                editBtn.innerText = "✏ Edit";
                editBtn.onclick = () => editUserMessage(msgIndex);
                actionsDiv.appendChild(editBtn);
            }

            msgDiv.appendChild(actionsDiv);
        }

        function appendMessage(role, text, file = null, msgIndex = -1) {
            const chatBox = document.getElementById("chat-box");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${role}`;

            if (file) {
                if (file.mime_type.startsWith('image/')) {
                    const img = document.createElement("img");
                    img.src = `data:${file.mime_type};base64,${file.data_base64}`;
                    msgDiv.appendChild(img);
                } else {
                    const badge = document.createElement("div");
                    badge.className = "doc-badge";
                    const icon = getFileIcon(file.mime_type, file.name || "");
                    badge.innerHTML = `<span class="doc-icon">${icon}</span> <span>${file.name || "Attachment"}</span>`;
                    msgDiv.appendChild(badge);
                }
            }

            const textSpan = document.createElement("div");
            textSpan.className = "text-content";

            if (role === "user") {
                textSpan.innerText = text;
            } else {
                textSpan.innerHTML = text ? marked.parse(text) : "...";
            }
            msgDiv.appendChild(textSpan);

            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            if (text) {
                if (role === "model") {
                    renderMath(textSpan);
                    addCopyButtons(msgDiv);
                }
                attachActions(msgDiv, text, role, msgIndex);
            }
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
    if not client:
        def err_gen():
            yield "GEMINI_API_KEY non configurée dans l'environnement."

        return StreamingResponse(err_gen(), media_type="text/plain")

    contents = []

    for msg in request.history:
        parts = []
        if msg.file:
            file_bytes = base64.b64decode(msg.file.data_base64)
            parts.append(types.Part.from_bytes(data=file_bytes, mime_type=msg.file.mime_type))
        if msg.content:
            parts.append(types.Part.from_text(text=msg.content))

        contents.append(types.Content(role=msg.role, parts=parts))

    current_parts = []
    if request.file:
        file_bytes = base64.b64decode(request.file.data_base64)
        current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=request.file.mime_type))
    if request.message:
        current_parts.append(types.Part.from_text(text=request.message))

    contents.append(types.Content(role="user", parts=current_parts))

    tools = [{"google_search": {}}] if request.enable_search else []

    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=tools
    )

    model_name = "gemini-3.8-flash"

    def generate_stream():
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as err:
            if request.enable_search:
                fallback_config = types.GenerateContentConfig(
                    system_instruction=request.system_instruction,
                    tools=[]
                )
                try:
                    fallback_stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=fallback_config
                    )
                    for chunk in fallback_stream:
                        if chunk.text:
                            yield chunk.text
                except Exception as fb_err:
                    yield f"Erreur API : {str(fb_err)}"
            else:
                yield f"Erreur API : {str(err)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")