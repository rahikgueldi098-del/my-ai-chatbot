import base64
import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from supabase import Client, create_client

load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Supabase Client
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY
    else None
)

# Initialize Gemini Client
client = (
    genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=4,
                initial_delay=1.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            )
        ),
    )
    if api_key
    else None
)

app = FastAPI(title="Gemini AI Studio Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
]


# --- Models ---
class FileData(BaseModel):
    mime_type: str
    data_base64: str
    name: Optional[str] = "file"


class ChatMessage(BaseModel):
    role: str
    content: str
    file: Optional[FileData] = None


class ChatRequest(BaseModel):
    message: str
    system_instruction: Optional[str] = (
        "You are a helpful, smart, and precise AI assistant."
    )
    file: Optional[FileData] = None
    history: Optional[List[ChatMessage]] = []
    enable_search: Optional[bool] = True
    conversation_id: Optional[str] = None


class EnhancePromptRequest(BaseModel):
    prompt: str


# --- Authentication Dependency ---
async def get_optional_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    if not supabase or not token or token in ["null", "undefined", ""]:
        return None
    try:
        user_response = supabase.auth.get_user(token)
        return user_response.user if user_response else None
    except Exception:
        return None


# --- Supabase Database Helpers ---
def save_message_to_db(conversation_id: str, user_id: str, role: str, content: str):
    if not supabase or not conversation_id or not user_id:
        return
    try:
        supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        print(f"Error saving message to Supabase: {e}")


def create_conversation_in_db(user_id: str, initial_title: str = "New Chat") -> Optional[str]:
    if not supabase or not user_id:
        return None
    try:
        res = supabase.table("conversations").insert({
            "user_id": user_id,
            "title": initial_title[:50]
        }).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["id"]
    except Exception as e:
        print(f"Error creating conversation in Supabase: {e}")
    return None


HTML_CONTENT = """<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>AI Assistant Studio Pro</title><script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"><script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css"><script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script><script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script><style>* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }body { background-color: #212121; color: #ececec; display: flex; height: 100vh; overflow: hidden; }#sidebar { width: 260px; background-color: #171717; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; gap: 12px; }#new-chat-btn { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 8px; padding: 10px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: 0.2s; }#new-chat-btn:hover { background: #383838; }#chat-search { width: 100%; padding: 8px 12px; border-radius: 6px; border: 1px solid #333; background: #212121; color: #fff; font-size: 0.85rem; outline: none; }#history-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }.history-item { padding: 10px 12px; border-radius: 6px; font-size: 0.88rem; color: #b4b4b4; cursor: pointer; transition: 0.2s; display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }.history-item:hover { background: #2f2f2f; color: #fff; }.history-item.active { background: #212121; color: #fff; font-weight: 500; }.history-header { display: flex; justify-content: space-between; align-items: center; width: 100%; }.history-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }.history-actions { display: none; gap: 4px; align-items: center; }.history-item:hover .history-actions { display: flex; }.item-action-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.85rem; padding: 2px 4px; border-radius: 4px; }.item-action-btn:hover { color: #fff; background: #3a3a3a; }.export-box { border-top: 1px solid #333; padding-top: 12px; display: flex; flex-direction: column; gap: 8px; }.export-title { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }.export-buttons { display: flex; gap: 8px; }.export-btn { flex: 1; background: #2f2f2f; color: #ccc; border: 1px solid #424242; border-radius: 6px; padding: 6px; font-size: 0.8rem; cursor: pointer; text-align: center; }.export-btn:hover { background: #383838; color: #fff; }#main-container { flex: 1; display: flex; flex-direction: column; height: 100vh; }header { padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; background: #171717; font-weight: 600; }.header-controls { display: flex; gap: 12px; align-items: center; }#persona-select { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.88rem; outline: none; cursor: pointer; max-width: 200px; }.toggle-btn { background: #2f2f2f; color: #888; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; transition: 0.2s; }.toggle-btn.active { background: #1b3a2b; color: #4ade80; border-color: #22c55e; }#chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }.message { display: flex; flex-direction: column; gap: 6px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; position: relative; }.user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }.model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; width: 100%; }.message img { max-width: 100%; border-radius: 8px; margin-top: 8px; }.doc-badge { display: inline-flex; align-items: center; gap: 8px; background: #1e293b; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-size: 0.88rem; color: #38bdf8; }.message p { margin-bottom: 8px; }.message p:last-child { margin-bottom: 0; }.message code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: monospace; }.code-container { position: relative; margin: 10px 0; }.message pre { background: #0d1117; padding: 14px; border-radius: 8px; overflow-x: auto; border: 1px solid #30363d; margin: 0; }.copy-btn { position: absolute; top: 8px; right: 8px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 0.75rem; cursor: pointer; }.msg-actions { display: flex; gap: 8px; margin-top: 8px; padding-top: 6px; border-top: 1px solid #2a2a2a; align-items: center; }.action-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.85rem; padding: 2px 6px; border-radius: 4px; display: flex; align-items: center; gap: 4px; }.action-btn:hover { color: #fff; background: #2f2f2f; }.msg-meta { font-size: 0.75rem; color: #666; margin-left: auto; }.typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 4px 0; }.typing-dot { width: 6px; height: 6px; background: #aaa; border-radius: 50%; animation: blink 1.4s infinite ease-in-out both; }.typing-dot:nth-child(1) { animation-delay: -0.32s; }.typing-dot:nth-child(2) { animation-delay: -0.16s; }@keyframes blink { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }/* Action Chips & Input Wrapper */#input-wrapper { padding: 15px 20px 20px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }.chips-row { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }.chip { background: #2a2a2a; border: 1px solid #3a3a3a; color: #ccc; padding: 5px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; white-space: nowrap; transition: 0.2s; }.chip:hover { background: #383838; color: #fff; border-color: #555; }#file-preview { display: none; align-items: center; gap: 10px; background: #2f2f2f; padding: 8px 12px; border-radius: 8px; border: 1px solid #424242; width: fit-content; font-size: 0.85rem; }#preview-img { height: 40px; width: 40px; object-fit: cover; border-radius: 4px; display: none; }#remove-file-btn { background: transparent; border: none; color: #ff5555; cursor: pointer; font-size: 1rem; margin-left: 6px; }#input-container { display: flex; gap: 8px; align-items: flex-end; }#file-input { display: none; }.icon-btn { height: 48px; width: 48px; border-radius: 24px; border: 1px solid #424242; background: #2f2f2f; color: #fff; font-size: 1.1rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: 0.2s; flex-shrink: 0; }.icon-btn:hover { background: #383838; }.icon-btn.recording { background: #3a1c1c; border-color: #ef4444; color: #ef4444; animation: pulse 1.5s infinite; }@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }#user-input { flex: 1; padding: 12px 16px; border-radius: 18px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; resize: none; max-height: 150px; min-height: 48px; line-height: 1.4; }#send-btn { height: 48px; min-width: 70px; padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; flex-shrink: 0; }#send-btn.stop-btn { background: #ef4444; color: #fff; }</style></head><body><div id="sidebar">  <button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>  <input type="text" id="chat-search" placeholder="Search chats..." oninput="renderSidebar()">  <div id="history-list"></div>  <div class="export-box">    <span class="export-title">Export Chat</span>    <div class="export-buttons">      <button class="export-btn" onclick="exportChat('md')">Markdown</button>      <button class="export-btn" onclick="exportChat('json')">JSON</button>    </div>  </div></div><div id="main-container">  <header>    <span>AI Assistant Studio Pro</span>    <div class="header-controls">      <button id="search-toggle" class="toggle-btn active" onclick="toggleSearch()">Web Search: ON</button>      <select id="persona-select" onchange="handlePersonaChange(this)">        <option value="You are a helpful, smart, and precise AI assistant.">Default Assistant</option>        <option value="You are a Senior Full-Stack Software Engineer. Provide clean, efficient code and explain tech concepts concisely.">Senior Engineer</option>        <option value="You are a strict, ultra-concise assistant. Answer using minimal words and direct bullet points only. No fluff.">Ultra-Concise Mode</option>        <option value="You are a creative writer and storytelling assistant with a rich, expressive vocabulary.">Creative Writer</option>        <option value="__NEW__">➕ Add Custom Persona...</option>      </select>    </div>  </header>  <div id="chat-box"></div>  <div id="input-wrapper">    <!-- Quick Action Chips -->    <div class="chips-row">      <button class="chip" onclick="applyChip('📝 Summarize')">📝 Summarize</button>      <button class="chip" onclick="applyChip('🐛 Fix Code')">🐛 Fix Code</button>      <button class="chip" onclick="applyChip('💡 Brainstorm')">💡 Brainstorm</button>      <button class="chip" onclick="applyChip('🌐 Translate to English')">🌐 Translate</button>      <button class="chip" onclick="applyChip('🔍 Explain Simply')">🔍 Explain Simply</button>    </div>    <div id="file-preview">      <img id="preview-img" src="" alt="preview">      <span id="preview-icon"></span>      <span id="file-name"></span>      <button id="remove-file-btn" onclick="clearFile()">✕</button>    </div>    <div id="input-container">      <input type="file" id="file-input" accept="image/*,.pdf,.txt,.csv,.md,.json,.py,.js,.html,.css" onchange="handleFileSelect(event)">      <button id="attach-btn" class="icon-btn" onclick="document.getElementById('file-input').click()" title="Attach File">📎</button>      <button id="mic-btn" class="icon-btn" onclick="toggleSpeechRecognition()" title="Voice Dictation">🎤</button>      <button id="enhance-btn" class="icon-btn" onclick="enhanceCurrentPrompt()" title="Magic Wand: Enhance Prompt with AI">🪄</button>      <textarea id="user-input" placeholder="Ask AI Assistant... (Shift + Enter for new line)" rows="1" onkeydown="handleKeyDown(event)" oninput="autoExpand(this)"></textarea>      <button id="send-btn" onclick="handleSendOrStop()">Send</button>    </div>  </div></div><script>let currentChatId = null;let chats = JSON.parse(localStorage.getItem('ai_chats') || '{}');let customPersonas = JSON.parse(localStorage.getItem('ai_custom_personas') || '[]');let currentFile = null;let webSearchEnabled = true;let recognition = null;let isRecording = false;let baseTranscript = '';let finalizedTranscript = '';let isGenerating = false;let abortController = null;window.onload = () => {  loadStoredPersonas();  renderSidebar();  const keys = Object.keys(chats);  if (keys.length > 0) loadChat(keys[0]);  else startNewChat();  initSpeechRecognition();};function loadStoredPersonas() {  const select = document.getElementById("persona-select");  customPersonas.forEach(p => {    const opt = document.createElement("option");    opt.value = p.instruction;    opt.innerText = p.name;    select.insertBefore(opt, select.lastElementChild);  });}function handlePersonaChange(select) {  if (select.value === "__NEW__") {    const name = prompt("Enter Persona Name (e.g. Math Tutor):");    if (!name) { select.selectedIndex = 0; return; }    const instruction = prompt(`Enter System Instructions for "${name}":`);    if (!instruction) { select.selectedIndex = 0; return; }    customPersonas.push({ name, instruction });    localStorage.setItem('ai_custom_personas', JSON.stringify(customPersonas));    const opt = document.createElement("option");    opt.value = instruction;    opt.innerText = name;    select.insertBefore(opt, select.lastElementChild);    select.value = instruction;  }}function applyChip(actionText) {  const input = document.getElementById("user-input");  if (input.value.trim()) input.value = `${actionText}: ${input.value}`;  else input.value = `${actionText}: `;  input.focus();  autoExpand(input);}async function enhanceCurrentPrompt() {  const input = document.getElementById("user-input");  const currentVal = input.value.trim();  if (!currentVal) { alert("Please type a draft prompt first!"); return; }  const btn = document.getElementById("enhance-btn");  btn.innerText = "⏳";  btn.disabled = true;  try {    const res = await fetch("/api/enhance-prompt", {      method: "POST",      headers: { "Content-Type": "application/json" },      body: JSON.stringify({ prompt: currentVal })    });    const data = await res.json();    if (data.enhanced_prompt) {      input.value = data.enhanced_prompt;      autoExpand(input);    }  } catch (e) {    console.error("Failed to enhance prompt:", e);  } finally {    btn.innerText = "🪄";    btn.disabled = false;  }}function initSpeechRecognition() {  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;  if (!SpeechRecognition) return;  recognition = new SpeechRecognition();  recognition.continuous = true;  recognition.interimResults = true;  recognition.onresult = (event) => {    let interim = '';    for (let i = event.resultIndex; i < event.results.length; i++) {      if (event.results[i].isFinal) finalizedTranscript += event.results[i][0].transcript + ' ';      else interim += event.results[i][0].transcript;    }    const inputEl = document.getElementById("user-input");    inputEl.value = baseTranscript + finalizedTranscript + interim;    autoExpand(inputEl);  };  recognition.onerror = () => stopRecording();  recognition.onend = () => stopRecording();}function toggleSpeechRecognition() {  if (isRecording) { if (recognition) recognition.stop(); stopRecording(); }  else {    try {      if (!recognition) initSpeechRecognition();      const inputEl = document.getElementById("user-input");      baseTranscript = inputEl.value ? inputEl.value.trim() + " " : "";      finalizedTranscript = "";      recognition.start();      isRecording = true;      const micBtn = document.getElementById('mic-btn');      micBtn.classList.add('recording');    } catch (e) { console.error(e); }  }}function stopRecording() {  isRecording = false;  const micBtn = document.getElementById('mic-btn');  if (micBtn) micBtn.classList.remove('recording');}function toggleSearch() {  webSearchEnabled = !webSearchEnabled;  const btn = document.getElementById('search-toggle');  btn.classList.toggle('active', webSearchEnabled);  btn.innerText = webSearchEnabled ? 'Web Search: ON' : 'Web Search: OFF';}function saveChats() { localStorage.setItem('ai_chats', JSON.stringify(chats)); }function startNewChat() {  currentChatId = Date.now().toString();  chats[currentChatId] = { title: "New Chat", history: [] };  saveChats();  renderSidebar();  loadChat(currentChatId);}function renameChat(id, event) {  if (event) event.stopPropagation();  const currentTitle = chats[id]?.title || "Chat";  const newTitle = prompt("Rename Chat:", currentTitle);  if (newTitle && newTitle.trim()) {    chats[id].title = newTitle.trim();    saveChats();    renderSidebar();  }}function deleteChat(id, event) {  if (event) event.stopPropagation();  delete chats[id];  saveChats();  const keys = Object.keys(chats).sort((a, b) => b - a);  if (id === currentChatId) {    if (keys.length > 0) loadChat(keys[0]);    else startNewChat();  } else renderSidebar();}function renderSidebar() {  const listEl = document.getElementById("history-list");  const query = (document.getElementById("chat-search")?.value || "").toLowerCase().trim();  listEl.innerHTML = "";  Object.keys(chats).sort((a, b) => b - a).forEach(id => {    const chat = chats[id];    const title = chat.title || "Chat";    if (query && !title.toLowerCase().includes(query)) return;    const item = document.createElement("div");    item.className = `history-item ${id === currentChatId ? 'active' : ''}`;    item.innerHTML = `<div class="history-header"><span class="history-title">${title}</span><div class="history-actions"><button class="item-action-btn" onclick="renameChat('${id}', event)">✏️</button><button class="item-action-btn" onclick="deleteChat('${id}', event)">🗑️</button></div></div>`;    item.onclick = () => loadChat(id);    listEl.appendChild(item);  });}function renderMath(element) {  if (window.renderMathInElement) {    renderMathInElement(element, {      delimiters: [        {left: '$$', right: '$$', display: true},        {left: '$', right: '$', display: false}      ],      throwOnError: false    });  }}function renderChatHistory() {  const chatBox = document.getElementById("chat-box");  chatBox.innerHTML = "";  (chats[currentChatId]?.history || []).forEach((msg, idx) => {    appendMessage(msg.role, msg.content, msg.file, idx);  });  hljs.highlightAll();}function loadChat(id) { currentChatId = id; renderSidebar(); renderChatHistory(); }function exportChat(format) {  if (!currentChatId || !chats[currentChatId]) return;  const chat = chats[currentChatId];  let content = format === 'json' ? JSON.stringify(chat, null, 2) : `# ${chat.title}\\n\\n` + chat.history.map(m => `### ${m.role.toUpperCase()}\\n${m.content}\\n`).join('\n---\n');  const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/markdown' });  const a = document.createElement('a');  a.href = URL.createObjectURL(blob);  a.download = `chat_${currentChatId}.${format}`;  a.click();}function getFileIcon(mime, name) {  if (mime.startsWith('image/')) return '🖼️';  if (name.endsWith('.pdf')) return '📄';  if (name.endsWith('.csv')) return '📊';  return '📁';}function handleFileSelect(event) {  const file = event.target.files[0];  if (!file) return;  const reader = new FileReader();  reader.onload = (e) => {    const base64Data = e.target.result.split(',')[1];    currentFile = { mime_type: file.type || 'text/plain', data_base64: base64Data, name: file.name };    document.getElementById("preview-img").style.display = file.type.startsWith('image/') ? 'block' : 'none';    if (file.type.startsWith('image/')) document.getElementById("preview-img").src = e.target.result;    document.getElementById("file-name").innerText = file.name;    document.getElementById("file-preview").style.display = "flex";  };  reader.readAsDataURL(file);}function clearFile() {  currentFile = null;  document.getElementById("file-input").value = "";  document.getElementById("file-preview").style.display = "none";}function autoExpand(field) {  field.style.height = 'inherit';  field.style.height = Math.min(field.scrollHeight, 150) + 'px';}function handleKeyDown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendOrStop(); } }function handleSendOrStop() { if (isGenerating) { if (abortController) abortController.abort(); } else sendMessage(); }function updateSendButton(generating) {  isGenerating = generating;  const btn = document.getElementById("send-btn");  btn.innerText = generating ? "⏹ Stop" : "Send";  btn.classList.toggle("stop-btn", generating);}async function sendMessage(overrideText = null, overrideFile = null) {  if (isRecording) toggleSpeechRecognition();  const input = document.getElementById("user-input");  const persona = document.getElementById("persona-select");  const text = overrideText !== null ? overrideText : input.value.trim();  const file = overrideFile !== null ? overrideFile : currentFile;  if (!text && !file) return;  if (!currentChatId || !chats[currentChatId]) startNewChat();  const history = chats[currentChatId].history;  if (history.length === 0) {    chats[currentChatId].title = text.length > 25 ? text.substring(0, 25) + "..." : text;    renderSidebar();  }  history.push({ role: "user", content: text, file });  saveChats();  input.value = ""; input.style.height = "48px"; clearFile(); input.disabled = true;  renderChatHistory();  const botMsgEl = appendMessage("model", "", null, history.length);  botMsgEl.querySelector('.text-content').innerHTML = '<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';  let accText = "";  abortController = new AbortController();  updateSendButton(true);  try {    const res = await fetch("/chat", {      method: "POST",      headers: { "Content-Type": "application/json" },      signal: abortController.signal,      body: JSON.stringify({        message: text,        system_instruction: persona.value,        file: file,        history: history.slice(0, -1),        enable_search: webSearchEnabled      })    });    const reader = res.body.getReader();    const decoder = new TextDecoder();    while (true) {      const { done, value } = await reader.read();      if (done) break;      accText += decoder.decode(value, { stream: true });      botMsgEl.querySelector('.text-content').innerHTML = marked.parse(accText);      document.getElementById("chat-box").scrollTop = document.getElementById("chat-box").scrollHeight;    }    hljs.highlightAll();    renderMath(botMsgEl.querySelector('.text-content'));    history.push({ role: "model", content: accText });    saveChats();    renderChatHistory();  } catch (err) {    if (err.name === 'AbortError') {      if (accText.trim()) { accText += " *(Stopped)*"; history.push({ role: "model", content: accText }); saveChats(); }      else history.pop();      renderChatHistory();    }  } finally {    abortController = null;    updateSendButton(false);    input.disabled = false;    input.focus();  }}function appendMessage(role, text, file = null, msgIndex = null) {  const chatBox = document.getElementById("chat-box");  const msgDiv = document.createElement("div");  msgDiv.className = `message ${role}`;  if (file) {    if (file.mime_type.startsWith('image/')) {      msgDiv.innerHTML += `<img src="data:${file.mime_type};base64,${file.data_base64}">`;    } else {      msgDiv.innerHTML += `<div class="doc-badge"><span>📁</span><span>${file.name}</span></div>`;    }  }  const textDiv = document.createElement("div");  textDiv.className = "text-content";  textDiv.innerHTML = role === "user" ? text : (text ? marked.parse(text) : "...");  msgDiv.appendChild(textDiv);  if (text && role === "model") {    const wordCount = text.trim().split(/\s+/).length;    const readTime = Math.max(1, Math.ceil(wordCount / 200));    const metaSpan = document.createElement("div");    metaSpan.className = "msg-actions";    metaSpan.innerHTML = `<button class="action-btn" onclick="navigator.clipboard.writeText(\`${text.replace(/`/g, '\\`')}\`)">📋 Copy</button><span class="msg-meta">${wordCount} words • ~${readTime} min read</span>`;    msgDiv.appendChild(metaSpan);  }  chatBox.appendChild(msgDiv);  chatBox.scrollTop = chatBox.scrollHeight;  return msgDiv;}</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_CONTENT


@app.post("/api/enhance-prompt")
async def enhance_prompt(req: EnhancePromptRequest):
    """Uses Gemini to transform a simple prompt into an enhanced, detailed prompt."""
    if not client:
        return {"enhanced_prompt": req.prompt}
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Rewrite and enhance the following user prompt to make it clear, detailed, and structured for an AI response. Return ONLY the enhanced prompt text directly without intro/outro commentary:\n\nPrompt: '{req.prompt}'",
        )
        return {"enhanced_prompt": response.text.strip()}
    except Exception as e:
        return {"enhanced_prompt": req.prompt}


@app.post("/chat")
@app.post("/api/chat")
async def chat(request: ChatRequest, user=Depends(get_optional_user)):
    if not client:
        def err_gen():
            yield "GEMINI_API_KEY is not configured."
        return StreamingResponse(err_gen(), media_type="text/plain")

    user_id = user.id if user else None
    conversation_id = request.conversation_id

    if user_id and not conversation_id:
        conversation_id = create_conversation_in_db(user_id, request.message or "New Chat")

    if user_id and conversation_id and request.message:
        save_message_to_db(conversation_id, user_id, "user", request.message)

    contents = []
    for msg in request.history:
        parts = []
        if msg.file:
            parts.append(types.Part.from_bytes(data=base64.b64decode(msg.file.data_base64), mime_type=msg.file.mime_type))
        if msg.content:
            parts.append(types.Part.from_text(text=msg.content))
        contents.append(types.Content(role=msg.role, parts=parts))

    current_parts = []
    if request.file:
        current_parts.append(types.Part.from_bytes(data=base64.b64decode(request.file.data_base64), mime_type=request.file.mime_type))
    if request.message:
        current_parts.append(types.Part.from_text(text=request.message))
    contents.append(types.Content(role="user", parts=current_parts))

    tools = [{"google_search": {}}] if request.enable_search else []

    def generate_stream():
        full_response = ""
        for model_name in CANDIDATE_MODELS:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=request.system_instruction,
                    tools=tools,
                )
                response_stream = client.models.generate_content_stream(
                    model=model_name, contents=contents, config=config
                )
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        yield chunk.text

                if user_id and conversation_id and full_response:
                    save_message_to_db(conversation_id, user_id, "assistant", full_response)
                return
            except Exception:
                continue

        yield "⚠️ Service temporarily unavailable. Please try again in a few moments."

    return StreamingResponse(generate_stream(), media_type="text/plain")