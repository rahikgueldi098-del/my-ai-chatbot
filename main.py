import base64
import collections
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from supabase import Client, create_client

# Try loading new Google GenAI SDK, fallback to legacy if not installed
try:
    from google import genai
    from google.genai import types

    SDK_MODE = "NEW"
except ImportError:
    import google.generativeai as genai_legacy

    SDK_MODE = "LEGACY"

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# 0 = no "thinking" (fastest). Raise it (e.g. 1024) if you want deeper answers.
THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
# Only the last N messages of the history are sent to Gemini (faster + cheaper).
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "12"))
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 1.0

SUPABASE_URL = (
    os.environ.get("SUPABASE_URL", "").replace("/rest/v1/", "").strip("/")
)
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Erreur d'initialisation Supabase: {e}")

app = FastAPI(title="AI Assistant Studio Pro")

# Thread pool used to save to Supabase WITHOUT blocking the AI response.
_executor = ThreadPoolExecutor(max_workers=4)

# Reuse the Gemini client between requests (avoids re-creating it every time).
_gemini_client = None
_gemini_client_key = None


def get_gemini_client(api_key: str):
    global _gemini_client, _gemini_client_key
    if _gemini_client is None or _gemini_client_key != api_key:
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_client_key = api_key
    return _gemini_client


# --- Authentication (Supabase Auth) ---
# Every /api/* route (except /api/auth/*) requires a logged-in user.
# Each request gets its own Supabase client carrying the USER's token, so
# Row-Level Security (RLS) is enforced by the database itself.
# NOTE: SUPABASE_KEY must be the "anon" (public) key, NOT the service_role key,
# otherwise RLS is bypassed.
AUTH_CACHE_TTL_SEC = 300
_auth_cache: Dict[str, Any] = {}
_auth_lock = threading.Lock()
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
}


class AuthUser:
    def __init__(self, user: Any, token: str):
        self.id = str(user.id)
        self.email = getattr(user, "email", None)
        self.token = token
        # Client acting as this user -> RLS applies to every query.
        self.db: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.db.postgrest.auth(token)


def require_user(authorization: Optional[str] = Header(None)) -> AuthUser:
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured (SUPABASE_URL / SUPABASE_KEY).",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    now = time.time()
    with _auth_lock:
        cached = _auth_cache.get(token)
        if cached and now - cached[0] < AUTH_CACHE_TTL_SEC:
            return cached[1]

    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired session.")
        auth_user = AuthUser(user_response.user, token)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    with _auth_lock:
        if len(_auth_cache) > 500:
            _auth_cache.clear()
        _auth_cache[token] = (now, auth_user)
    return auth_user


# --- Request Models ---
class AuthRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class EnhanceRequest(BaseModel):
    prompt: str = ""


class CreateChatRequest(BaseModel):
    title: Optional[str] = "New Discussion"


class ChatRequest(BaseModel):
    chat_id: Optional[str] = "default_chat"
    message: str = ""
    history: List[Dict[str, Any]] = []
    file: Optional[Dict[str, Any]] = None
    system_instruction: str = "You are a helpful assistant."
    # Web search makes every answer slower, so it is OFF unless requested.
    web_search: bool = False


# --- Rate Limiting & Analytics ---
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 30
ip_request_history = collections.defaultdict(list)

INPUT_TOKEN_COST_USD = 0.075 / 1_000_000
OUTPUT_TOKEN_COST_USD = 0.30 / 1_000_000

analytics_store = {
    "total_requests": 0,
    "total_successful": 0,
    "total_failed": 0,
    "total_rate_limited": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_cost_usd": 0.0,
    "latency_ms_history": [],
    "recent_logs": [],
}


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    timestamps = ip_request_history[client_ip]
    valid_timestamps = [
        ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW_SEC
    ]
    ip_request_history[client_ip] = valid_timestamps
    if len(valid_timestamps) >= MAX_REQUESTS_PER_WINDOW:
        return False
    ip_request_history[client_ip].append(now)
    return True


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def log_analytics_entry(
    endpoint: str,
    status_code: int,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    client_ip: str,
):
    cost = (prompt_tokens * INPUT_TOKEN_COST_USD) + (
        completion_tokens * OUTPUT_TOKEN_COST_USD
    )
    analytics_store["total_requests"] += 1
    if status_code == 200:
        analytics_store["total_successful"] += 1
    elif status_code == 429:
        analytics_store["total_rate_limited"] += 1
        analytics_store["total_failed"] += 1
    else:
        analytics_store["total_failed"] += 1

    analytics_store["total_prompt_tokens"] += prompt_tokens
    analytics_store["total_completion_tokens"] += completion_tokens
    analytics_store["total_cost_usd"] += cost
    analytics_store["latency_ms_history"].append(latency_ms)
    if len(analytics_store["latency_ms_history"]) > 200:
        analytics_store["latency_ms_history"].pop(0)

    log_entry = {
        "time": time.strftime("%H:%M:%S"),
        "endpoint": endpoint,
        "ip": client_ip,
        "status": status_code,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 6),
    }
    analytics_store["recent_logs"].insert(0, log_entry)
    if len(analytics_store["recent_logs"]) > 50:
        analytics_store["recent_logs"].pop()


def decode_file(file_payload: dict):
    _, b64 = file_payload["data"].split(",", 1)
    file_bytes = base64.b64decode(b64)
    mime = file_payload.get("type", "application/octet-stream")
    name = file_payload.get("name", "file")
    return file_bytes, mime, name


# --- Database helpers (run in background threads) ---
def save_message(
    db: Client,
    chat_id: str,
    role: str,
    content: str,
    file_payload: Optional[dict] = None,
    user_id: Optional[str] = None,
):
    payload = {"chat_id": chat_id, "role": role, "content": content}
    if file_payload is not None:
        payload["file_payload"] = file_payload
    try:
        row = dict(payload)
        if user_id:
            row["user_id"] = user_id
        db.table("messages").insert(row).execute()
    except Exception:
        try:
            db.table("messages").insert(payload).execute()
        except Exception as ex:
            print(f"Failed to save {role} message:", ex)


def save_user_turn(
    db: Client,
    chat_id: str,
    message: str,
    file_payload: Optional[dict],
    user_id: str,
):
    save_message(db, chat_id, "user", message, file_payload, user_id)
    try:
        title_snippet = (
            message[:30] + "..." if len(message) > 30 else message
        ) or "New Discussion"
        db.table("chats").update(
            {"title": title_snippet, "updated_at": "now()"}
        ).eq("id", chat_id).eq("user_id", user_id).execute()
    except Exception as ex:
        print("Failed to update chat title:", ex)


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Assistant Studio Pro</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { background-color: #212121; color: #ececec; display: flex; height: 100vh; height: 100dvh; overflow: hidden; }
#sidebar { width: 260px; background-color: #171717; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; gap: 12px; flex-shrink: 0; }
#new-chat-btn { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 8px; padding: 10px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: 0.2s; }
#new-chat-btn:hover { background: #383838; }
#chat-search { width: 100%; padding: 8px 12px; border-radius: 6px; border: 1px solid #333; background: #212121; color: #fff; font-size: 0.85rem; outline: none; }
#history-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.history-item { padding: 10px 12px; border-radius: 6px; font-size: 0.88rem; color: #b4b4b4; cursor: pointer; transition: 0.2s; display: flex; justify-content: space-between; align-items: center; }
.history-item:hover { background: #2f2f2f; color: #fff; }
.history-item.active { background: #212121; color: #fff; font-weight: 500; }
.history-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
.item-action-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.85rem; padding: 2px 4px; border-radius: 4px; display: none; }
.history-item:hover .item-action-btn { display: inline-block; }
.item-action-btn:hover { color: #fff; }
.export-box { border-top: 1px solid #333; padding-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.export-title { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.export-buttons { display: flex; gap: 8px; }
.export-btn { flex: 1; background: #2f2f2f; color: #ccc; border: 1px solid #424242; border-radius: 6px; padding: 6px; font-size: 0.8rem; cursor: pointer; text-align: center; }
.export-btn:hover { background: #383838; color: #fff; }
#main-container { flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100vh; height: 100dvh; }
header { padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; background: #171717; font-weight: 600; gap: 8px; }
.header-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
#menu-btn { display: none; background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 6px 10px; font-size: 1rem; cursor: pointer; }
.header-controls { display: flex; gap: 10px; align-items: center; }
#persona-select { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.88rem; outline: none; cursor: pointer; max-width: 180px; }
.toggle-btn { background: #2f2f2f; color: #888; border: 1px solid #424242; border-radius: 6px; padding: 6px 10px; font-size: 0.85rem; cursor: pointer; transition: 0.2s; white-space: nowrap; }
.toggle-btn.active { background: #1b3a2b; color: #4ade80; border-color: #22c55e; }
.admin-btn { background: #1e293b; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; transition: 0.2s; font-weight: 600; }
.admin-btn:hover { background: #334155; color: #fff; }
#chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }
.message { display: flex; flex-direction: column; gap: 6px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; position: relative; word-break: break-word; }
.user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }
.model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; width: 100%; }
.message img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
.message pre { overflow-x: auto; background: #171717; padding: 10px; border-radius: 8px; margin: 8px 0; }
.doc-badge { display: inline-flex; align-items: center; gap: 8px; background: #1e293b; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-size: 0.88rem; color: #38bdf8; }
.message p { margin-bottom: 8px; }
.message p:last-child { margin-bottom: 0; }
.message code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
.msg-actions { display: flex; gap: 8px; margin-top: 8px; padding-top: 6px; border-top: 1px solid #2a2a2a; align-items: center; flex-wrap: wrap; }
.action-btn { background: transparent; border: none; color: #888; cursor: pointer; font-size: 0.85rem; padding: 2px 6px; border-radius: 4px; display: flex; align-items: center; gap: 4px; }
.action-btn:hover { color: #fff; background: #2f2f2f; }
.msg-meta { font-size: 0.75rem; color: #666; margin-left: auto; }
.typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 4px 0; }
.typing-dot { width: 6px; height: 6px; background: #aaa; border-radius: 50%; animation: blink 1.4s infinite ease-in-out both; }
.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }
#input-wrapper { padding: 15px 20px 20px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
.chips-row { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }
.chip { background: #2a2a2a; border: 1px solid #3a3a3a; color: #ccc; padding: 5px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; white-space: nowrap; transition: 0.2s; flex-shrink: 0; }
.chip:hover { background: #383838; color: #fff; border-color: #555; }
#file-preview { display: none; align-items: center; gap: 10px; background: #2f2f2f; padding: 8px 12px; border-radius: 8px; border: 1px solid #424242; width: fit-content; max-width: 100%; font-size: 0.85rem; }
#preview-img { height: 40px; width: 40px; object-fit: cover; border-radius: 4px; display: none; }
#remove-file-btn { background: transparent; border: none; color: #ff5555; cursor: pointer; font-size: 1rem; margin-left: 6px; }
#input-container { display: flex; gap: 8px; align-items: flex-end; }
#file-input { display: none; }
.icon-btn { height: 48px; width: 48px; border-radius: 24px; border: 1px solid #424242; background: #2f2f2f; color: #fff; font-size: 1.1rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: 0.2s; flex-shrink: 0; }
.icon-btn:hover { background: #383838; }
.icon-btn.recording { background: #3a1c1c; border-color: #ef4444; color: #ef4444; animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
#user-input { flex: 1; min-width: 0; padding: 12px 16px; border-radius: 18px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; resize: none; max-height: 150px; min-height: 48px; line-height: 1.4; }
#send-btn { height: 48px; min-width: 70px; padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; flex-shrink: 0; }
#send-btn.stop-btn { background: #ef4444; color: #fff; }
.error-box { color: #f87171; background: #450a0a; padding: 10px 14px; border-radius: 8px; border: 1px solid #991b1b; font-size: 0.9rem; }
#sidebar-backdrop { display: none; }
.modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 100; align-items: center; justify-content: center; padding: 20px; }
.modal-overlay.open { display: flex; }
.modal-content { background: #171717; border: 1px solid #333; border-radius: 12px; max-width: 700px; width: 100%; max-height: 85vh; overflow-y: auto; padding: 20px; color: #ececec; display: flex; flex-direction: column; gap: 16px; }
.modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 12px; }
.modal-title { font-size: 1.2rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
.close-btn { background: transparent; border: none; color: #888; font-size: 1.2rem; cursor: pointer; }
.close-btn:hover { color: #fff; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
.stat-card { background: #212121; border: 1px solid #333; padding: 12px; border-radius: 8px; display: flex; flex-direction: column; gap: 4px; }
.stat-value { font-size: 1.2rem; font-weight: 700; color: #4ade80; }
.stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; }
.log-table-container { overflow-x: auto; border: 1px solid #333; border-radius: 8px; background: #212121; max-height: 250px; }
.log-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left; }
.log-table th, .log-table td { padding: 8px 10px; border-bottom: 1px solid #2a2a2a; }
.log-table th { background: #1a1a1a; color: #888; position: sticky; top: 0; }
.status-200 { color: #4ade80; font-weight: 600; }
.status-429 { color: #f87171; font-weight: 600; }
@media (max-width: 768px) {
#menu-btn { display: inline-block; }
#sidebar { position: fixed; top: 0; left: 0; bottom: 0; z-index: 30; transform: translateX(-100%); transition: transform 0.25s ease; }
#sidebar.open { transform: translateX(0); }
#sidebar-backdrop.show { display: block; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 20; }
header { padding: 10px; flex-wrap: wrap; }
.header-controls { gap: 6px; }
#persona-select { max-width: 110px; padding: 6px 8px; font-size: 0.8rem; }
#chat-box { padding: 12px; }
.message { max-width: 95%; }
#input-wrapper { padding: 10px; }
.icon-btn { height: 40px; width: 40px; font-size: 1rem; }
#send-btn { height: 40px; min-width: 60px; padding: 0 14px; }
#user-input { min-height: 40px; padding: 9px 14px; }
}
.auth-input { width: 100%; padding: 12px 14px; border-radius: 8px; border: 1px solid #424242; background: #2f2f2f; color: #fff; font-size: 1rem; outline: none; }
#auth-msg { font-size: 0.85rem; min-height: 1.2em; }
#auth-msg.err { color: #f87171; }
#auth-msg.ok { color: #4ade80; }
#auth-submit { background: #fff; color: #000; border: none; border-radius: 8px; padding: 12px; font-weight: 600; cursor: pointer; font-size: 1rem; }
#auth-switch { color: #38bdf8; font-size: 0.85rem; text-align: center; cursor: pointer; text-decoration: none; }
#user-email { text-transform: none; word-break: break-all; }
</style>
</head>
<body>
<div id="sidebar-backdrop" onclick="toggleSidebar()"></div>
<div id="sidebar">
<button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
<input type="text" id="chat-search" placeholder="Search chats..." oninput="renderSidebar()">
<div id="history-list"></div>
<div class="export-box">
<span class="export-title" id="user-email"></span>
<button class="export-btn" onclick="logout()">Log out</button>
</div>
<div class="export-box">
<span class="export-title">Export Chat</span>
<div class="export-buttons">
<button class="export-btn" onclick="exportChat('md')">Markdown</button>
<button class="export-btn" onclick="exportChat('json')">JSON</button>
</div>
</div>
</div>
<div id="main-container">
<header>
<div class="header-left">
<button id="menu-btn" onclick="toggleSidebar()">☰</button>
<span>AI Assistant Studio Pro</span>
</div>
<div class="header-controls">
<button class="admin-btn" onclick="openAnalyticsModal()">📊 Analytics</button>
<button id="search-toggle" class="toggle-btn" onclick="toggleSearch()">Web Search: OFF</button>
<select id="persona-select" onchange="handlePersonaChange(this)">
<option value="You are a helpful, smart, and precise AI assistant.">Default Assistant</option>
<option value="You are a Senior Full-Stack Software Engineer. Provide clean, efficient code and explain tech concepts concisely.">Senior Engineer</option>
<option value="You are a strict, ultra-concise assistant. Answer using minimal words and direct bullet points only. No fluff.">Ultra-Concise Mode</option>
<option value="You are a creative writer and storytelling assistant with a rich, expressive vocabulary.">Creative Writer</option>
<option value="__NEW__">➕ Add Custom Persona...</option>
</select>
</div>
</header>
<div id="chat-box"></div>
<div id="input-wrapper">
<div class="chips-row">
<button class="chip" onclick="applyChip('Summarize this context clearly:')">📝 Summarize</button>
<button class="chip" onclick="applyChip('Find bugs and fix this code:')">🐛 Fix Code</button>
<button class="chip" onclick="applyChip('Brainstorm creative ideas for:')">💡 Brainstorm</button>
<button class="chip" onclick="applyChip('Translate the following text to English:')">🌐 Translate</button>
<button class="chip" onclick="applyChip('Explain this concept in simple terms suitable for a beginner:')">🔍 Explain Simply</button>
</div>
<div id="file-preview">
<img id="preview-img" src="" alt="preview">
<span id="preview-icon"></span>
<span id="file-name"></span>
<button id="remove-file-btn" onclick="clearFile()">✕</button>
</div>
<div id="input-container">
<input type="file" id="file-input" accept="image/*,.pdf,.txt,.csv,.md,.json,.py,.js,.html,.css" onchange="handleFileSelect(event)">
<button id="attach-btn" class="icon-btn" onclick="document.getElementById('file-input').click()" title="Attach File">📎</button>
<button id="mic-btn" class="icon-btn" onclick="toggleSpeechRecognition()" title="Voice Dictation">🎤</button>
<button id="enhance-btn" class="icon-btn" onclick="enhanceCurrentPrompt()" title="Magic Wand: Enhance Prompt with AI">🪄</button>
<textarea id="user-input" placeholder="Ask AI Assistant... (Shift + Enter for new line)" rows="1" onkeydown="handleKeyDown(event)" oninput="autoExpand(this)"></textarea>
<button id="send-btn" onclick="handleSendOrStop()">Send</button>
</div>
</div>
</div>
<div id="auth-modal" class="modal-overlay" style="z-index:200;">
<div class="modal-content" style="max-width:380px;">
<div class="modal-title" id="auth-title">🔐 Log in</div>
<input id="auth-email" class="auth-input" type="email" placeholder="Email" autocomplete="email">
<input id="auth-password" class="auth-input" type="password" placeholder="Password (min 6 characters)" autocomplete="current-password" onkeydown="if(event.key==='Enter') submitAuth()">
<div id="auth-msg"></div>
<button id="auth-submit" onclick="submitAuth()">Log in</button>
<a id="auth-switch" href="#" onclick="toggleAuthMode(); return false;">No account? Sign up</a>
</div>
</div>
<div id="analytics-modal" class="modal-overlay" onclick="if(event.target===this) closeAnalyticsModal()">
<div class="modal-content">
<div class="modal-header">
<div class="modal-title">📊 Operations & Cost Analytics</div>
<button class="close-btn" onclick="closeAnalyticsModal()">✕</button>
</div>
<div class="stats-grid">
<div class="stat-card">
<span class="stat-value" id="stat-reqs">0</span>
<span class="stat-label">Total Requests</span>
</div>
<div class="stat-card">
<span class="stat-value" id="stat-cost" style="color: #38bdf8;">$0.0000</span>
<span class="stat-label">Est. Cost (USD)</span>
</div>
<div class="stat-card">
<span class="stat-value" id="stat-tokens" style="color: #a78bfa;">0</span>
<span class="stat-label">Total Tokens</span>
</div>
<div class="stat-card">
<span class="stat-value" id="stat-latency" style="color: #facc15;">0ms</span>
<span class="stat-label">Avg Latency</span>
</div>
</div>
<div style="font-size: 0.85rem; font-weight: 600; color: #aaa; margin-top: 4px;">Recent API Logs (Last 50)</div>
<div class="log-table-container">
<table class="log-table">
<thead>
<tr>
<th>Time</th>
<th>Status</th>
<th>IP</th>
<th>Latency</th>
<th>Tokens (P/C)</th>
<th>Cost ($)</th>
</tr>
</thead>
<tbody id="log-table-body">
<tr><td colspan="6" style="text-align: center; color: #666;">No logs recorded yet.</td></tr>
</tbody>
</table>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.75rem; color: #666;">Rate Limit: 30 requests / min / IP</span>
<button class="export-btn" style="max-width: 120px;" onclick="fetchAnalyticsStats()">🔄 Refresh</button>
</div>
</div>
</div>
<script>
let chats = [];
let currentChatId = null;
let currentHistory = [];
let webSearchEnabled = false;
let selectedFile = null;
let recognition = null;
let isRecording = false;
let isStreaming = false;
let activeAbortController = null;
if (window.marked) {
window.marked.setOptions({
highlight: function(code, lang) {
if (lang && window.hljs && window.hljs.getLanguage(lang)) {
try { return window.hljs.highlight(code, { language: lang }).value; } catch (e) {}
}
return window.hljs ? window.hljs.highlightAuto(code).value : code;
},
breaks: true
});
}
const _origFetch = window.fetch.bind(window);
let authMode = 'login';
let _refreshing = null;
function saveSession(d) {
  try {
    localStorage.setItem('supabase_token', d.access_token || '');
    if (d.refresh_token) localStorage.setItem('supabase_refresh', d.refresh_token);
    if (d.email) localStorage.setItem('supabase_email', d.email);
  } catch (e) {}
  updateUserBox();
}
function updateUserBox() {
  const el = document.getElementById('user-email');
  if (el) el.textContent = localStorage.getItem('supabase_email') || '';
}
function showAuth(msg) {
  const m = document.getElementById('auth-modal');
  if (m) m.classList.add('open');
  if (msg) setAuthMsg(msg, 'err');
}
function hideAuth() {
  const m = document.getElementById('auth-modal');
  if (m) m.classList.remove('open');
}
function setAuthMsg(text, kind) {
  const el = document.getElementById('auth-msg');
  el.textContent = text || '';
  el.className = kind || '';
}
function setAuthMode(mode) {
  authMode = mode;
  const login = mode === 'login';
  document.getElementById('auth-title').textContent = login ? '🔐 Log in' : '📝 Create account';
  document.getElementById('auth-submit').textContent = login ? 'Log in' : 'Sign up';
  document.getElementById('auth-switch').textContent = login ? 'No account? Sign up' : 'Already have an account? Log in';
  document.getElementById('auth-password').autocomplete = login ? 'current-password' : 'new-password';
}
function toggleAuthMode() {
  setAuthMode(authMode === 'login' ? 'signup' : 'login');
  setAuthMsg('', '');
}
async function submitAuth() {
  const email = document.getElementById('auth-email').value.trim();
  const password = document.getElementById('auth-password').value;
  if (!email || !password) { setAuthMsg('Enter your email and password.', 'err'); return; }
  const btn = document.getElementById('auth-submit');
  btn.disabled = true;
  setAuthMsg('Please wait...', '');
  try {
    const r = await _origFetch('/api/auth/' + (authMode === 'login' ? 'login' : 'signup'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) { setAuthMsg(d.detail || ('Error ' + r.status), 'err'); return; }
    if (d.needs_confirmation) {
      setAuthMode('login');
      setAuthMsg('Account created. Check your email to confirm it, then log in.', 'ok');
      return;
    }
    saveSession(d);
    document.getElementById('auth-password').value = '';
    setAuthMsg('', '');
    hideAuth();
    chats = [];
    currentHistory = [];
    currentChatId = null;
    await initStorage();
  } catch (e) {
    setAuthMsg('Network error. Try again.', 'err');
  } finally {
    btn.disabled = false;
  }
}
function logout() {
  try {
    localStorage.removeItem('supabase_token');
    localStorage.removeItem('supabase_refresh');
    localStorage.removeItem('supabase_email');
  } catch (e) {}
  location.reload();
}
async function tryRefresh() {
  const rt = localStorage.getItem('supabase_refresh');
  if (!rt) return false;
  if (!_refreshing) {
    _refreshing = (async () => {
      try {
        const r = await _origFetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: rt })
        });
        if (!r.ok) return false;
        saveSession(await r.json());
        return true;
      } catch (e) { return false; }
      finally { setTimeout(() => { _refreshing = null; }, 0); }
    })();
  }
  return _refreshing;
}
// Every /api call: if the server says 401, try to refresh the session once,
// otherwise show the login screen.
window.fetch = async function(url, opts) {
  opts = opts || {};
  let res = await _origFetch(url, opts);
  const u = String(url);
  if (res.status === 401 && u.startsWith('/api/') && !u.startsWith('/api/auth/')) {
    if (await tryRefresh()) {
      opts = Object.assign({}, opts, {
        headers: Object.assign({}, opts.headers, { 'Authorization': 'Bearer ' + localStorage.getItem('supabase_token') })
      });
      res = await _origFetch(url, opts);
    }
    if (res.status === 401) showAuth('Please log in to continue.');
  }
  return res;
};
function getAuthHeaders() {
const headers = { 'Content-Type': 'application/json' };
const token = localStorage.getItem('supabase_token');
if (token) {
headers['Authorization'] = `Bearer ${token}`;
}
return headers;
}
async function initStorage() {
if (!localStorage.getItem('supabase_token')) { showAuth(); return; }
try {
const res = await fetch('/api/chats', { headers: getAuthHeaders() });
if (res.status === 401) return;
if (res.ok) {
chats = await res.json();
} else {
chats = [];
}
} catch (e) {
chats = [];
}
if (!Array.isArray(chats) || chats.length === 0) {
await startNewChat();
} else {
await loadChat(chats[0].id);
}
}
function safeParseMarkdown(str) {
if (window.marked && typeof window.marked.parse === 'function') {
try { return window.marked.parse(str); } catch (e) {}
}
return escapeHtml(str).replace(/\n/g, '<br>');
}
function escapeHtml(text) {
if (!text) return '';
return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function toggleSidebar() {
document.getElementById('sidebar').classList.toggle('open');
document.getElementById('sidebar-backdrop').classList.toggle('show');
}
function closeSidebarOnMobile() {
document.getElementById('sidebar').classList.remove('open');
document.getElementById('sidebar-backdrop').classList.remove('show');
}
async function startNewChat() {
try {
const res = await fetch('/api/chats', {
method: 'POST',
headers: getAuthHeaders(),
body: JSON.stringify({ title: 'New Discussion' })
});
const newChat = await res.json();
chats.unshift(newChat);
currentChatId = newChat.id;
currentHistory = [];
renderSidebar();
renderChatBox();
closeSidebarOnMobile();
} catch (e) {
console.error("Failed to start new chat:", e);
}
}
async function loadChat(id) {
currentChatId = id;
try {
const res = await fetch(`/api/chats/${id}/messages`, { headers: getAuthHeaders() });
if (res.ok) {
currentHistory = await res.json();
} else {
currentHistory = [];
}
} catch (e) {
currentHistory = [];
}
renderSidebar();
renderChatBox();
closeSidebarOnMobile();
}
async function deleteChat(id, event) {
if (event) event.stopPropagation();
try {
await fetch(`/api/chats/${id}`, { method: 'DELETE', headers: getAuthHeaders() });
chats = chats.filter(c => c.id !== id);
if (chats.length === 0) {
await startNewChat();
} else {
await loadChat(chats[0].id);
}
} catch (e) {
console.error("Failed to delete chat:", e);
}
}
function renderSidebar() {
const list = document.getElementById('history-list');
const searchInput = document.getElementById('chat-search');
const search = searchInput ? searchInput.value.toLowerCase() : '';
if (!list) return;
list.innerHTML = '';
if (!Array.isArray(chats)) chats = [];
chats.forEach(chat => {
if (search && !chat.title.toLowerCase().includes(search)) return;
const item = document.createElement('div');
item.className = 'history-item' + (chat.id === currentChatId ? ' active' : '');
item.onclick = () => loadChat(chat.id);
const title = document.createElement('span');
title.className = 'history-title';
title.innerText = chat.title || 'New Discussion';
const delBtn = document.createElement('button');
delBtn.className = 'item-action-btn';
delBtn.innerText = '🗑️';
delBtn.onclick = (e) => deleteChat(chat.id, e);
item.appendChild(title);
item.appendChild(delBtn);
list.appendChild(item);
});
}
function renderChatBox() {
const box = document.getElementById('chat-box');
if (!box) return;
box.innerHTML = '';
if (!Array.isArray(currentHistory)) return;
currentHistory.forEach((msg) => {
if (!msg) return;
const role = msg.role || 'user';
const content = msg.content || '';
appendMessageUI(role, content, msg.file_payload || msg.file, msg.meta);
});
box.scrollTop = box.scrollHeight;
}
function appendMessageUI(role, text, fileObj, metaObj) {
const box = document.getElementById('chat-box');
if (!box) return;
const msgDiv = document.createElement('div');
msgDiv.className = `message ${role}`;
if (fileObj) {
if (fileObj.type && fileObj.type.startsWith('image/')) {
const img = document.createElement('img');
img.src = fileObj.data;
msgDiv.appendChild(img);
} else {
const badge = document.createElement('div');
badge.className = 'doc-badge';
badge.innerHTML = `📄 <strong>${escapeHtml(fileObj.name)}</strong>`;
msgDiv.appendChild(badge);
}
}
const contentDiv = document.createElement('div');
contentDiv.className = 'text-content';
if (role === 'model') {
contentDiv.innerHTML = text ? safeParseMarkdown(text) : '<div class="typing-dots"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
} else {
contentDiv.innerText = text;
}
msgDiv.appendChild(contentDiv);
if (text && role === 'model') {
addMessageActions(msgDiv, text, metaObj);
}
box.appendChild(msgDiv);
box.scrollTop = box.scrollHeight;
return msgDiv;
}
function addMessageActions(msgDiv, text, metaObj) {
const words = text.trim().split(/\s+/).filter(Boolean).length;
const readTime = Math.max(1, Math.ceil(words / 200));
const estTokens = metaObj && metaObj.total_tokens ? metaObj.total_tokens : Math.max(1, Math.ceil(text.length / 4));
const estCost = metaObj && metaObj.cost_usd !== undefined ? metaObj.cost_usd : (estTokens * 0.0000003);
const metaSpan = document.createElement('div');
metaSpan.className = 'msg-actions';
const copyBtn = document.createElement('button');
copyBtn.className = 'action-btn';
copyBtn.innerText = '📋 Copy';
copyBtn.onclick = () => {
try { navigator.clipboard.writeText(text); } catch (e) {}
copyBtn.innerText = '✅ Copied';
setTimeout(() => copyBtn.innerText = '📋 Copy', 2000);
};
const metaInfo = document.createElement('span');
metaInfo.className = 'msg-meta';
metaInfo.innerText = `${words} words • ~${estTokens} tokens ($${estCost.toFixed(5)}) • ~${readTime} min read`;
metaSpan.appendChild(copyBtn);
metaSpan.appendChild(metaInfo);
msgDiv.appendChild(metaSpan);
}
function toggleSearch() {
webSearchEnabled = !webSearchEnabled;
const btn = document.getElementById('search-toggle');
btn.innerText = webSearchEnabled ? "Web Search: ON" : "Web Search: OFF";
btn.className = webSearchEnabled ? "toggle-btn active" : "toggle-btn";
}
function applyChip(prefix) {
const input = document.getElementById('user-input');
input.value = input.value.trim() ? prefix + " " + input.value : prefix + " ";
input.focus();
autoExpand(input);
}
function handlePersonaChange(select) {
if (select.value === '__NEW__') {
const name = prompt("Enter Custom Persona Name:");
if (!name) { select.selectedIndex = 0; return; }
const promptText = prompt("Enter Persona System Instructions:");
if (!promptText) { select.selectedIndex = 0; return; }
const opt = document.createElement('option');
opt.value = promptText;
opt.innerText = name;
select.insertBefore(opt, select.lastElementChild);
select.value = promptText;
}
}
function autoExpand(textarea) {
textarea.style.height = 'auto';
textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}
function handleKeyDown(event) {
if (event.key === 'Enter' && !event.shiftKey) {
event.preventDefault();
handleSendOrStop();
}
}
function handleFileSelect(e) {
const file = e.target.files[0];
if (!file) return;
const reader = new FileReader();
reader.onload = function(evt) {
selectedFile = {
name: file.name,
type: file.type || 'text/plain',
size: file.size,
data: evt.target.result
};
const preview = document.getElementById('file-preview');
const previewImg = document.getElementById('preview-img');
const previewIcon = document.getElementById('preview-icon');
const fileName = document.getElementById('file-name');
if (file.type && file.type.startsWith('image/')) {
previewImg.src = evt.target.result;
previewImg.style.display = 'block';
previewIcon.innerText = '';
} else {
previewImg.style.display = 'none';
previewIcon.innerText = '📄';
}
fileName.innerText = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
preview.style.display = 'flex';
};
reader.readAsDataURL(file);
}
function clearFile() {
selectedFile = null;
document.getElementById('file-input').value = '';
document.getElementById('file-preview').style.display = 'none';
}
function toggleSpeechRecognition() {
const micBtn = document.getElementById('mic-btn');
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) {
alert("Speech Recognition is not supported in this browser.");
return;
}
if (isRecording) {
if (recognition) recognition.stop();
return;
}
try {
recognition = new SpeechRecognition();
recognition.continuous = false;
recognition.interimResults = false;
recognition.lang = 'en-US';
recognition.onstart = () => { isRecording = true; micBtn.classList.add('recording'); };
recognition.onresult = (e) => {
const transcript = e.results[0][0].transcript;
const input = document.getElementById('user-input');
input.value = input.value ? input.value + ' ' + transcript : transcript;
autoExpand(input);
};
recognition.onerror = () => { isRecording = false; micBtn.classList.remove('recording'); };
recognition.onend = () => { isRecording = false; micBtn.classList.remove('recording'); };
recognition.start();
} catch (err) {
isRecording = false;
micBtn.classList.remove('recording');
}
}
async function enhanceCurrentPrompt() {
const input = document.getElementById('user-input');
const text = input.value.trim();
if (!text) { alert("Please enter a prompt first!"); return; }
const enhanceBtn = document.getElementById('enhance-btn');
enhanceBtn.innerText = '⏳';
try {
const res = await fetch('/api/enhance-prompt', {
method: 'POST',
headers: getAuthHeaders(),
body: JSON.stringify({ prompt: text })
});
const data = await res.json();
if (data.enhanced_prompt) {
input.value = data.enhanced_prompt;
autoExpand(input);
} else if (data.error || data.detail) {
alert("Enhance failed: " + (data.error || data.detail));
}
} catch (e) {
alert("Enhance failed: " + e.message);
} finally {
enhanceBtn.innerText = '🪄';
}
}
function handleSendOrStop() {
if (isStreaming) {
if (activeAbortController) activeAbortController.abort();
isStreaming = false;
updateSendBtnUI(false);
} else {
sendMessage();
}
}
function updateSendBtnUI(streaming) {
const btn = document.getElementById('send-btn');
if (!btn) return;
if (streaming) {
btn.innerText = 'Stop ⏹';
btn.classList.add('stop-btn');
} else {
btn.innerText = 'Send';
btn.classList.remove('stop-btn');
}
}
function buildHistoryForApi() {
if (!Array.isArray(currentHistory)) return [];
return currentHistory
.filter(m => m && m.content && String(m.content).trim())
.map(m => ({ role: m.role || 'user', content: String(m.content) }));
}
async function sendMessage() {
const input = document.getElementById('user-input');
const text = input ? input.value.trim() : '';
if (!text && !selectedFile) return;
if (!currentChatId) {
await startNewChat();
}
const filePayload = selectedFile;
currentHistory.push({ role: 'user', content: text, file_payload: filePayload });
appendMessageUI('user', text, filePayload);
if (input) {
input.value = '';
input.style.height = 'auto';
}
clearFile();
isStreaming = true;
updateSendBtnUI(true);
activeAbortController = new AbortController();
const botMsgDiv = appendMessageUI('model', '', null);
const contentDiv = botMsgDiv.querySelector('.text-content');
const systemPrompt = document.getElementById('persona-select').value;
let fullText = '';
try {
const res = await fetch('/api/chat', {
method: 'POST',
headers: getAuthHeaders(),
signal: activeAbortController.signal,
body: JSON.stringify({
chat_id: currentChatId,
history: buildHistoryForApi(),
message: text,
file: filePayload,
web_search: webSearchEnabled,
system_instruction: systemPrompt === '__NEW__' ? 'You are a helpful assistant.' : systemPrompt
})
});
if (!res.ok) {
const err = await res.json().catch(() => ({ detail: "HTTP " + res.status }));
contentDiv.innerHTML = `<div class="error-box">Server Error (${res.status}): ${escapeHtml(err.detail || 'Failed')}</div>`;
currentHistory.pop();
return;
}
const reader = res.body.getReader();
const decoder = new TextDecoder();
while (true) {
const { done, value } = await reader.read();
if (done) break;
fullText += decoder.decode(value, { stream: true });
contentDiv.innerHTML = safeParseMarkdown(fullText);
const box = document.getElementById('chat-box');
box.scrollTop = box.scrollHeight;
}
if (fullText.trim()) {
currentHistory.push({ role: 'model', content: fullText });
addMessageActions(botMsgDiv, fullText);
const activeChat = chats.find(c => c.id === currentChatId);
if (activeChat && (activeChat.title === 'New Discussion' || !activeChat.title)) {
activeChat.title = text ? (text.slice(0, 30) + (text.length > 30 ? '...' : '')) : 'New Discussion';
renderSidebar();
}
} else {
contentDiv.innerHTML = '<div class="error-box">Empty response from model.</div>';
currentHistory.pop();
}
} catch (err) {
if (err.name !== 'AbortError') {
contentDiv.innerHTML = `<div class="error-box">Error: ${escapeHtml(err.message)}</div>`;
currentHistory.pop();
} else if (fullText.trim()) {
currentHistory.push({ role: 'model', content: fullText });
}
} finally {
isStreaming = false;
activeAbortController = null;
updateSendBtnUI(false);
}
}
function exportChat(format) {
if (!currentHistory || currentHistory.length === 0) return alert("Nothing to export!");
let dataStr = '';
let filename = `chat_${Date.now()}.${format}`;
if (format === 'json') {
dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentHistory, null, 2));
} else {
let md = `# Chat Export\n\n`;
currentHistory.forEach(m => md += `### ${(m.role||'user').toUpperCase()}\n${m.content||''}\n\n`);
dataStr = "data:text/markdown;charset=utf-8," + encodeURIComponent(md);
}
const a = document.createElement('a');
a.href = dataStr;
a.download = filename;
a.click();
}
function openAnalyticsModal() {
document.getElementById('analytics-modal').classList.add('open');
fetchAnalyticsStats();
}
function closeAnalyticsModal() {
document.getElementById('analytics-modal').classList.remove('open');
}
async function fetchAnalyticsStats() {
try {
const res = await fetch('/api/admin/stats', { headers: getAuthHeaders() });
if (!res.ok) return;
const data = await res.json();
document.getElementById('stat-reqs').innerText = `${data.total_successful}/${data.total_requests}`;
document.getElementById('stat-cost').innerText = `$${data.total_cost_usd.toFixed(5)}`;
document.getElementById('stat-tokens').innerText = (data.total_prompt_tokens + data.total_completion_tokens).toLocaleString();
document.getElementById('stat-latency').innerText = `${data.avg_latency_ms.toFixed(0)}ms`;
const tableBody = document.getElementById('log-table-body');
if (!data.recent_logs || data.recent_logs.length === 0) {
tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No logs recorded yet.</td></tr>';
return;
}
let html = '';
data.recent_logs.forEach(log => {
const statusClass = log.status === 200 ? 'status-200' : 'status-429';
html += `<tr>
<td>${log.time}</td>
<td class="${statusClass}">${log.status}</td>
<td>${log.ip}</td>
<td>${log.latency_ms}ms</td>
<td>${log.prompt_tokens}/${log.completion_tokens}</td>
<td>$${log.cost_usd.toFixed(6)}</td>
</tr>`;
});
tableBody.innerHTML = html;
} catch (e) {}
}
window.addEventListener('DOMContentLoaded', () => { updateUserBox(); initStorage(); });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def serve_gui():
    return HTML_CONTENT


# --- Auth routes (no token needed) ---
def _auth_client() -> Client:
    # A fresh client per call so one user's session never leaks into another's.
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _session_payload(res: Any) -> Dict[str, Any]:
    session = getattr(res, "session", None)
    user = getattr(res, "user", None)
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "email": getattr(user, "email", None),
    }


@app.post("/api/auth/signup")
def auth_signup(body: AuthRequest, request: Request):
    if not supabase:
        return JSONResponse({"detail": "Supabase is not configured."}, status_code=503)
    if not check_rate_limit(get_client_ip(request)):
        return JSONResponse({"detail": "Too many attempts. Wait a minute."}, status_code=429)
    email = body.email.strip().lower()
    if "@" not in email or len(body.password) < 6:
        return JSONResponse(
            {"detail": "Enter a valid email and a password of at least 6 characters."},
            status_code=400,
        )
    try:
        res = _auth_client().auth.sign_up({"email": email, "password": body.password})
        if not getattr(res, "session", None):
            # Email confirmation is enabled in Supabase: user must confirm first.
            return JSONResponse({"needs_confirmation": True, "email": email})
        return JSONResponse(_session_payload(res))
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=400)


@app.post("/api/auth/login")
def auth_login(body: AuthRequest, request: Request):
    if not supabase:
        return JSONResponse({"detail": "Supabase is not configured."}, status_code=503)
    if not check_rate_limit(get_client_ip(request)):
        return JSONResponse({"detail": "Too many attempts. Wait a minute."}, status_code=429)
    try:
        res = _auth_client().auth.sign_in_with_password(
            {"email": body.email.strip().lower(), "password": body.password}
        )
        return JSONResponse(_session_payload(res))
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=400)


@app.post("/api/auth/refresh")
def auth_refresh(body: RefreshRequest):
    if not supabase:
        return JSONResponse({"detail": "Supabase is not configured."}, status_code=503)
    try:
        res = _auth_client().auth.refresh_session(body.refresh_token)
        return JSONResponse(_session_payload(res))
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=401)


# --- Chat routes (login required, filtered per user) ---
@app.get("/api/chats")
def get_chats(user: AuthUser = Depends(require_user)):
    try:
        res = (
            user.db.table("chats")
            .select("*")
            .eq("user_id", user.id)
            .order("updated_at", desc=True)
            .execute()
        )
        return JSONResponse(res.data or [])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chats")
def create_chat(body: CreateChatRequest, user: AuthUser = Depends(require_user)):
    chat_id = f"chat_{int(time.time() * 1000)}"
    title = body.title or "New Discussion"
    try:
        res = (
            user.db.table("chats")
            .insert({"id": chat_id, "title": title, "user_id": user.id})
            .execute()
        )
        return JSONResponse(
            res.data[0] if res.data else {"id": chat_id, "title": title}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _owns_chat(user: AuthUser, chat_id: str) -> bool:
    res = (
        user.db.table("chats")
        .select("id")
        .eq("id", chat_id)
        .eq("user_id", user.id)
        .execute()
    )
    return bool(res.data)


@app.get("/api/chats/{chat_id}/messages")
def get_messages(chat_id: str, user: AuthUser = Depends(require_user)):
    try:
        if not _owns_chat(user, chat_id):
            return JSONResponse({"detail": "Chat not found."}, status_code=404)
        res = (
            user.db.table("messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )
        return JSONResponse(res.data or [])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user: AuthUser = Depends(require_user)):
    try:
        if not _owns_chat(user, chat_id):
            return JSONResponse({"detail": "Chat not found."}, status_code=404)
        user.db.table("messages").delete().eq("chat_id", chat_id).execute()
        user.db.table("chats").delete().eq("id", chat_id).eq(
            "user_id", user.id
        ).execute()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"status": "deleted"})


@app.get("/api/admin/stats")
def get_admin_stats(request: Request, user: AuthUser = Depends(require_user)):
    # If ADMIN_EMAILS is set (comma-separated), only those users may see stats.
    if ADMIN_EMAILS and (user.email or "").lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admins only.")
    latencies = analytics_store["latency_ms_history"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return JSONResponse({
        "total_requests": analytics_store["total_requests"],
        "total_successful": analytics_store["total_successful"],
        "total_failed": analytics_store["total_failed"],
        "total_rate_limited": analytics_store["total_rate_limited"],
        "total_prompt_tokens": analytics_store["total_prompt_tokens"],
        "total_completion_tokens": analytics_store["total_completion_tokens"],
        "total_cost_usd": round(analytics_store["total_cost_usd"], 6),
        "avg_latency_ms": round(avg_latency, 1),
        "recent_logs": analytics_store["recent_logs"],
    })


@app.post("/api/enhance-prompt")
def enhance_prompt(
    body: EnhanceRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
):
    start_time = time.time()
    client_ip = get_client_ip(request)

    if not check_rate_limit(client_ip):
        log_analytics_entry("/api/enhance-prompt", 429, 0.0, 0, 0, client_ip)
        raise HTTPException(
            status_code=429, detail="Too Many Requests. Rate limit exceeded."
        )

    try:
        raw_prompt = body.prompt.strip()
        if not raw_prompt:
            return JSONResponse({"enhanced_prompt": ""})

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({"error": "GEMINI_API_KEY missing"}, status_code=500)

        sys_inst = (
            "Improve this user prompt into a clear, structured prompt. Output ONLY"
            " the improved text."
        )
        p_tokens = estimate_tokens(raw_prompt) + estimate_tokens(sys_inst)

        if SDK_MODE == "NEW":
            client = get_gemini_client(api_key)
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Enhance: {raw_prompt}",
                config=build_config(sys_inst, None, True),
            )
            enhanced = (res.text or "").strip()
        else:
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(
                MODEL_NAME, system_instruction=sys_inst
            )
            res = model.generate_content(f"Enhance: {raw_prompt}")
            enhanced = (res.text or "").strip()

        c_tokens = estimate_tokens(enhanced)
        latency_ms = (time.time() - start_time) * 1000
        log_analytics_entry(
            "/api/enhance-prompt", 200, latency_ms, p_tokens, c_tokens, client_ip
        )
        return JSONResponse({"enhanced_prompt": enhanced})
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_analytics_entry(
            "/api/enhance-prompt", 500, latency_ms, 0, 0, client_ip
        )
        return JSONResponse({"error": str(e)}, status_code=500)


def build_config(system_instruction: str, tools, use_thinking: bool):
    """Gemini config. Thinking is limited so the first word arrives faster."""
    kwargs = {"system_instruction": system_instruction}
    if tools:
        kwargs["tools"] = tools
    if use_thinking:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=THINKING_BUDGET
            )
        except Exception:
            pass
    return types.GenerateContentConfig(**kwargs)


@app.post("/api/chat")
def chat_endpoint(
    body: ChatRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
):
    start_time = time.time()
    client_ip = get_client_ip(request)

    if not check_rate_limit(client_ip):
        log_analytics_entry("/api/chat", 429, 0.0, 0, 0, client_ip)
        raise HTTPException(
            status_code=429,
            detail=(
                "Too Many Requests. Rate limit exceeded (30 reqs/min). Please wait"
                " a moment."
            ),
        )

    try:
        chat_id = body.chat_id
        message = body.message
        # Only keep the most recent messages -> much faster on long chats.
        history = (body.history or [])[-MAX_HISTORY_MESSAGES:]
        file_payload = body.file
        system_instruction = body.system_instruction or "You are a helpful assistant."
        web_search = body.web_search

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY environment variable is not configured.",
            )

        user_id = user.id

        # Save the user message in the background: Gemini starts immediately.
        save_future = None
        if chat_id and message:
            save_future = _executor.submit(
                save_user_turn, user.db, chat_id, message, file_payload, user_id
            )

        prompt_tokens_est = estimate_tokens(message) + estimate_tokens(
            system_instruction
        )
        for h in history:
            prompt_tokens_est += estimate_tokens(h.get("content", ""))

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }

        # `state` lets the retry loop switch thinking off if the model rejects it.
        state = {"use_thinking": True}

        if SDK_MODE == "NEW":
            client = get_gemini_client(api_key)
            contents = []
            for m in history:
                text = (m.get("content") or "").strip()
                role = m.get("role")
                if not text or role not in ("user", "model"):
                    continue
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=text)])
                )

            parts = []
            if message:
                parts.append(types.Part.from_text(text=message))
            if file_payload and "data" in file_payload:
                file_bytes, mime, name = decode_file(file_payload)
                if mime.startswith("image/") or mime == "application/pdf":
                    parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))
                else:
                    try:
                        text_str = file_bytes.decode("utf-8")
                        parts.append(
                            types.Part.from_text(
                                text=f"\n[Attached file: {name}]\n{text_str}"
                            )
                        )
                    except Exception:
                        pass

            if not parts:
                raise HTTPException(status_code=400, detail="Empty message.")
            contents.append(types.Content(role="user", parts=parts))

            tools = (
                [types.Tool(google_search=types.GoogleSearch())]
                if web_search
                else None
            )

            def make_stream():
                config = build_config(
                    system_instruction, tools, state["use_thinking"]
                )
                response = client.models.generate_content_stream(
                    model=MODEL_NAME, contents=contents, config=config
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

        else:
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(
                MODEL_NAME, system_instruction=system_instruction
            )
            legacy_history = []
            for m in history:
                text = (m.get("content") or "").strip()
                role = m.get("role")
                if not text or role not in ("user", "model"):
                    continue
                legacy_history.append({"role": role, "parts": [text]})

            prompt_content = [message] if message else []
            if file_payload and "data" in file_payload:
                file_bytes, mime, name = decode_file(file_payload)
                if mime.startswith("image/") or mime == "application/pdf":
                    prompt_content.append({"mime_type": mime, "data": file_bytes})
                else:
                    try:
                        text_str = file_bytes.decode("utf-8")
                        prompt_content.append(f"\n[Attached file: {name}]\n{text_str}")
                    except Exception:
                        pass

            if not prompt_content:
                raise HTTPException(status_code=400, detail="Empty message.")

            chat_session = model.start_chat(history=legacy_history)

            def make_stream():
                res = chat_session.send_message(prompt_content, stream=True)
                for chunk in res:
                    if chunk.text:
                        yield chunk.text

        def generate():
            total_output_text = ""
            for attempt in range(MAX_RETRIES):
                try:
                    for text in make_stream():
                        total_output_text += text
                        yield text

                    # Success
                    latency_ms = (time.time() - start_time) * 1000
                    log_analytics_entry(
                        "/api/chat",
                        200,
                        latency_ms,
                        prompt_tokens_est,
                        estimate_tokens(total_output_text),
                        client_ip,
                    )
                    if chat_id and total_output_text:
                        # Keep order: user message first, then the answer.
                        if save_future is not None:
                            try:
                                save_future.result(timeout=10)
                            except Exception:
                                pass
                        save_message(
                            user.db, chat_id, "model", total_output_text, None, user_id
                        )
                    return
                except Exception as ex:
                    err_msg = str(ex)

                    # Model refused the "thinking" option -> retry without it.
                    if (
                        state["use_thinking"]
                        and "thinking" in err_msg.lower()
                        and not total_output_text
                    ):
                        state["use_thinking"] = False
                        continue

                    # Retry only if nothing was sent yet (no duplicated text).
                    if (
                        ("429" in err_msg or "503" in err_msg)
                        and not total_output_text
                        and attempt < (MAX_RETRIES - 1)
                    ):
                        time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                        continue

                    latency_ms = (time.time() - start_time) * 1000
                    status_code = 429 if "429" in err_msg else 500
                    log_analytics_entry(
                        "/api/chat", status_code, latency_ms, 0, 0, client_ip
                    )
                    yield f"\n\n⚠️ Error: {err_msg}"
                    return

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
