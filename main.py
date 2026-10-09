import base64
import collections
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
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
# Limits (change them in Vercel -> Environment Variables, no code change needed)
USER_RATE_LIMIT_PER_MIN = int(os.environ.get("USER_RATE_LIMIT_PER_MIN", "20"))
DAILY_REQUEST_LIMIT = int(os.environ.get("DAILY_REQUEST_LIMIT", "200"))  # 0 = unlimited
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


# Added to every chat so the answer can be shown in the live preview panel.
CANVAS_HINT = (
    " When the user asks for a web page, app, UI, game, chart, diagram or"
    " visualization, reply with ONE complete, self-contained code block so it can"
    " be previewed live: ```html with all CSS and JavaScript inline in a single"
    " file (for charts use Chart.js or D3 loaded from cdnjs.cloudflare.com or"
    " cdn.jsdelivr.net), ```mermaid for diagrams and flowcharts, or ```svg for"
    " vector graphics. Never split a page into several files."
    " For mermaid: start with flowchart TD, use short letter-only node ids,"
    " put every node label in double quotes, write edge labels as -->|Yes|,"
    " use at most 15 nodes and at most one loop back, and never use <br/>,"
    " ::: classes or parentheses inside labels."
)


# --- Request Models ---
class AuthRequest(BaseModel):
    email: str
    password: str


class RenameChatRequest(BaseModel):
    title: str


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

# Price per 1M tokens (USD). Check your model's price and set these in Vercel.
INPUT_TOKEN_COST_USD = float(os.environ.get("GEMINI_INPUT_USD_PER_M", "0.075")) / 1_000_000
OUTPUT_TOKEN_COST_USD = float(os.environ.get("GEMINI_OUTPUT_USD_PER_M", "0.30")) / 1_000_000

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


def check_rate_limit(client_ip: str, limit: int = MAX_REQUESTS_PER_WINDOW) -> bool:
    now = time.time()
    timestamps = ip_request_history[client_ip]
    valid_timestamps = [
        ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW_SEC
    ]
    ip_request_history[client_ip] = valid_timestamps
    if len(valid_timestamps) >= limit:
        return False
    ip_request_history[client_ip].append(now)
    return True


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _capture_usage(chunk: Any, usage: dict):
    """Read the REAL token counts Gemini reports (instead of guessing)."""
    um = getattr(chunk, "usage_metadata", None)
    if not um:
        return
    p = getattr(um, "prompt_token_count", None)
    c = getattr(um, "candidates_token_count", None)
    t = getattr(um, "thoughts_token_count", None)
    if p:
        usage["p"] = p
    if c is not None or t is not None:
        usage["c"] = (c or 0) + (t or 0)


def log_analytics_entry(
    endpoint: str,
    status_code: int,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    client_ip: str,
    user: Any = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
):
    cost = (prompt_tokens * INPUT_TOKEN_COST_USD) + (
        completion_tokens * OUTPUT_TOKEN_COST_USD
    )

    # Saved in Supabase (table api_logs) so the numbers survive restarts.
    if user is not None:
        try:
            user.db.table("api_logs").insert({
                "user_id": user.id,
                "email": user.email,
                "endpoint": endpoint,
                "status": status_code,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "cost_usd": round(cost, 6),
                "model": model,
                "ip": client_ip,
                "error": (error or None) and str(error)[:500],
            }).execute()
        except Exception as ex:
            print("Failed to save api_logs row:", ex)

    # In-memory copy: only used as a fallback if the database cannot be read.
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


# --- Daily quota per user (counted from the api_logs table) ---
_quota_cache: Dict[str, List[float]] = {}  # user_id -> [requests today, fetched_at]
_quota_lock = threading.Lock()


def check_daily_quota(user: Any) -> bool:
    if not DAILY_REQUEST_LIMIT:
        return True
    if ADMIN_EMAILS and (user.email or "").lower() in ADMIN_EMAILS:
        return True  # admins are not limited
    now = time.time()
    with _quota_lock:
        entry = _quota_cache.get(user.id)
    if entry is None or now - entry[1] > 60:
        try:
            day_start = (
                datetime.now(timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )
            res = (
                user.db.table("api_logs")
                .select("id", count="exact")
                .eq("user_id", user.id)
                .eq("endpoint", "/api/chat")
                .eq("status", 200)
                .gte("created_at", day_start)
                .limit(1)
                .execute()
            )
            entry = [float(res.count or 0), now]
        except Exception as ex:
            print("Quota check failed (allowing request):", ex)
            return True
        with _quota_lock:
            _quota_cache[user.id] = entry
    if entry[0] >= DAILY_REQUEST_LIMIT:
        return False
    entry[0] += 1
    return True


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
        # Only name the chat automatically while it still has the default
        # title, so a title you renamed yourself is never overwritten.
        db.table("chats").update({"title": title_snippet}).eq("id", chat_id).eq(
            "user_id", user_id
        ).eq("title", "New Discussion").execute()
        db.table("chats").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
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
.history-item.active .item-action-btn { display: inline-block; }
.rename-input { flex: 1; min-width: 0; background: #2f2f2f; color: #fff; border: 1px solid #38bdf8; border-radius: 4px; padding: 3px 6px; font-size: 0.88rem; outline: none; }
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

#canvas-panel { display: none; width: 46%; min-width: 320px; max-width: 920px; flex-shrink: 0; background: #171717; border-left: 1px solid #333; flex-direction: column; height: 100vh; height: 100dvh; }
#canvas-panel.open { display: flex; }
#canvas-panel.fullscreen { position: fixed; inset: 0; width: 100%; max-width: none; z-index: 90; }
.canvas-head { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid #333; flex-wrap: wrap; }
.canvas-title { font-weight: 600; font-size: 0.9rem; flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#canvas-select { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 4px 8px; font-size: 0.8rem; max-width: 160px; }
.canvas-tabs { display: flex; border: 1px solid #424242; border-radius: 6px; overflow: hidden; }
.canvas-tab { background: #2f2f2f; color: #aaa; border: none; padding: 5px 12px; font-size: 0.8rem; cursor: pointer; }
.canvas-tab.active { background: #fff; color: #000; font-weight: 600; }
.canvas-btn { background: #2f2f2f; color: #ddd; border: 1px solid #424242; border-radius: 6px; padding: 5px 9px; font-size: 0.8rem; cursor: pointer; }
.canvas-btn:hover { background: #383838; color: #fff; }
#canvas-error { display: none; background: #450a0a; color: #fca5a5; font-size: 0.8rem; padding: 6px 12px; border-bottom: 1px solid #7f1d1d; }
#canvas-body { flex: 1; min-height: 0; position: relative; background: #fff; }
#canvas-frame { width: 100%; height: 100%; border: 0; background: #fff; display: block; }
#canvas-code { display: none; position: absolute; inset: 0; overflow: auto; margin: 0; padding: 14px; background: #0d1117; color: #e6edf3; font-size: 0.82rem; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Consolas, Menlo, monospace; white-space: pre; tab-size: 2; }
@media (max-width: 768px) {
#canvas-panel.open { position: fixed; inset: 0; width: 100%; max-width: none; z-index: 60; }
}
</style>
</head>
<body>
<div id="sidebar-backdrop" onclick="toggleSidebar()"></div>
<div id="sidebar">
<button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
<input type="text" id="chat-search" name="q-chat-filter-x" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" readonly onfocus="this.removeAttribute('readonly')" data-lpignore="true" data-1p-ignore data-form-type="other" placeholder="Search chats..." oninput="onChatSearchInput(event)">
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
<button id="analytics-btn" class="admin-btn" style="display:none;" onclick="openAnalyticsModal()">📊 Analytics</button>
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
<div id="canvas-panel">
<div class="canvas-head">
<span class="canvas-title" id="canvas-title">Preview</span>
<select id="canvas-select" onchange="selectCanvasArtifact(this.value)" style="display:none;"></select>
<div class="canvas-tabs">
<button class="canvas-tab active" id="tab-preview" onclick="setCanvasView('preview')">Preview</button>
<button class="canvas-tab" id="tab-code" onclick="setCanvasView('code')">Code</button>
</div>
<button class="canvas-btn" id="canvas-copy" onclick="copyCanvasCode()" title="Copy code">📋</button>
<button class="canvas-btn" onclick="downloadCanvas()" title="Download file">⬇</button>
<button class="canvas-btn" onclick="toggleCanvasFullscreen()" title="Full screen">⛶</button>
<button class="canvas-btn" onclick="closeCanvas()" title="Close">✕</button>
</div>
<div id="canvas-error"></div>
<div id="canvas-body">
<iframe id="canvas-frame" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referrer" title="Live preview"></iframe>
<pre id="canvas-code"></pre>
</div>
</div>
<div id="auth-modal" class="modal-overlay" style="z-index:200;">
<div class="modal-content" style="max-width:380px;">
<div class="modal-title" id="auth-title">🔐 Log in</div>
<input id="auth-email" class="auth-input" type="email" name="email" placeholder="Email" autocomplete="username">
<input id="auth-password" class="auth-input" type="password" name="password" placeholder="Password (min 6 characters)" autocomplete="current-password" onkeydown="if(event.key==='Enter') submitAuth()">
<div id="auth-msg"></div>
<button id="auth-submit" onclick="submitAuth()">Log in</button>
<a id="auth-switch" href="#" onclick="toggleAuthMode(); return false;">No account? Sign up</a>
</div>
</div>
<div id="analytics-modal" class="modal-overlay" onclick="if(event.target===this) closeAnalyticsModal()">
<div class="modal-content">
<div class="modal-header">
<div class="modal-title">📊 Operations & Cost Analytics</div>
<div style="display:flex; gap:10px; align-items:center;">
<select id="analytics-hours" onchange="fetchAnalyticsStats()" style="background:#2f2f2f; color:#fff; border:1px solid #424242; border-radius:6px; padding:4px 8px; font-size:0.8rem;">
<option value="1">Last hour</option>
<option value="24" selected>Last 24 hours</option>
<option value="168">Last 7 days</option>
<option value="720">Last 30 days</option>
</select>
<button class="close-btn" onclick="closeAnalyticsModal()">✕</button>
</div>
</div>
<div class="stats-grid">
<div class="stat-card"><span class="stat-value" id="stat-reqs">0</span><span class="stat-label">Success / Total</span></div>
<div class="stat-card"><span class="stat-value" id="stat-errrate" style="color: #f87171;">0%</span><span class="stat-label">Error Rate</span></div>
<div class="stat-card"><span class="stat-value" id="stat-cost" style="color: #38bdf8;">$0.0000</span><span class="stat-label">Est. Cost (USD)</span></div>
<div class="stat-card"><span class="stat-value" id="stat-tokens" style="color: #a78bfa;">0</span><span class="stat-label">Total Tokens</span></div>
<div class="stat-card"><span class="stat-value" id="stat-latency" style="color: #facc15;">0ms</span><span class="stat-label">Avg Latency</span></div>
<div class="stat-card"><span class="stat-value" id="stat-p95" style="color: #facc15;">0ms</span><span class="stat-label">P95 Latency</span></div>
<div class="stat-card"><span class="stat-value" id="stat-users">0</span><span class="stat-label">Users (period)</span></div>
<div class="stat-card"><span class="stat-value" id="stat-active">0</span><span class="stat-label">Active (15 min)</span></div>
</div>
<div style="font-size: 0.85rem; font-weight: 600; color: #aaa; margin-top: 4px;">Top Users by Cost</div>
<div class="log-table-container" style="max-height: 160px;">
<table class="log-table">
<thead><tr><th>User</th><th>Requests</th><th>Tokens</th><th>Cost ($)</th></tr></thead>
<tbody id="top-users-body"><tr><td colspan="4" style="text-align: center; color: #666;">No data yet.</td></tr></tbody>
</table>
</div>
<div id="model-stats" style="font-size: 0.8rem; color: #888;"></div>
<div style="font-size: 0.85rem; font-weight: 600; color: #aaa; margin-top: 4px;">Recent API Logs (Last 50)</div>
<div class="log-table-container">
<table class="log-table">
<thead>
<tr>
<th>Time</th>
<th>User</th>
<th>Status</th>
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
<div style="display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap;">
<span id="limits-note" style="font-size: 0.75rem; color: #666;"></span>
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
checkAdmin();
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
function startRename(chat, titleEl, event) {
if (event) event.stopPropagation();
const input = document.createElement('input');
input.className = 'rename-input';
input.value = chat.title || '';
input.maxLength = 80;
input.onclick = (e) => e.stopPropagation();
let finished = false;
const finish = async (save) => {
if (finished) return;
finished = true;
const newTitle = input.value.trim();
if (save && newTitle && newTitle !== chat.title) {
const oldTitle = chat.title;
chat.title = newTitle;
renderSidebar();
try {
const res = await fetch('/api/chats/' + encodeURIComponent(chat.id), {
method: 'PATCH',
headers: getAuthHeaders(),
body: JSON.stringify({ title: newTitle })
});
if (!res.ok) { chat.title = oldTitle; renderSidebar(); }
} catch (e) { chat.title = oldTitle; renderSidebar(); }
} else {
renderSidebar();
}
};
input.onkeydown = (e) => {
if (e.key === 'Enter') finish(true);
else if (e.key === 'Escape') finish(false);
};
input.onblur = () => finish(true);
titleEl.replaceWith(input);
input.focus();
input.select();
}
// Autofill fires a plain "input" event with no inputType; real typing/pasting always has one.
function onChatSearchInput(e) {
if (!e.inputType || e.inputType === 'insertReplacementText') {
e.target.value = '';
}
renderSidebar();
}
// Safety net: wipe any email Chrome slips into the search box after load.
window.addEventListener('load', () => {
const s = document.getElementById('chat-search');
if (!s) return;
[0, 300, 1000, 2500].forEach(t => setTimeout(() => {
if (document.activeElement !== s && s.value.includes('@')) {
s.value = '';
renderSidebar();
}
}, t));
});
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
title.title = 'Double-click to rename';
title.ondblclick = (e) => startRename(chat, title, e);
const renameBtn = document.createElement('button');
renameBtn.className = 'item-action-btn';
renameBtn.innerText = '✏️';
renameBtn.title = 'Rename';
renameBtn.onclick = (e) => startRename(chat, title, e);
const delBtn = document.createElement('button');
delBtn.className = 'item-action-btn';
delBtn.innerText = '🗑️';
delBtn.onclick = (e) => deleteChat(chat.id, e);
item.appendChild(title);
item.appendChild(renameBtn);
item.appendChild(delBtn);
list.appendChild(item);
});
}
function renderChatBox() {
const box = document.getElementById('chat-box');
if (!box) return;
closeCanvas();
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
// ===== Live preview panel (Canvas) =====
let canvasArtifacts = [];
let canvasIndex = 0;
let canvasView = 'preview';
const CANVAS_LABELS = { html: 'HTML page', svg: 'SVG', mermaid: 'Diagram', markdown: 'Document' };
function extractArtifacts(text) {
  const blocks = [];
  const re = /```([^\n`]*)\n([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(text || '')) !== null) {
    blocks.push({ lang: (m[1] || '').trim().split(/\s+/)[0].toLowerCase(), code: m[2].replace(/\s+$/, '') });
  }
  const css = blocks.filter(b => b.lang === 'css').map(b => b.code);
  const js = blocks.filter(b => ['javascript', 'js'].includes(b.lang)).map(b => b.code);
  const arts = [];
  let first = true;
  blocks.forEach(b => {
    const c = b.code.trim();
    let type = null;
    if (b.lang === 'mermaid') type = 'mermaid';
    else if (b.lang === 'svg' || ((b.lang === 'html' || b.lang === 'xml') && /^<svg[\s>]/i.test(c))) type = 'svg';
    else if (['html', 'htm', 'xhtml'].includes(b.lang) || (!b.lang && /^<!doctype html|^<html[\s>]/i.test(c))) type = 'html';
    else if (b.lang === 'markdown' || b.lang === 'md') type = 'markdown';
    if (!type || !c) return;
    let code = b.code;
    if (type === 'html' && first) { code = mergeAssets(code, css, js); }
    if (type === 'html') first = false;
    arts.push({ type: type, code: code, label: CANVAS_LABELS[type] });
  });
  return arts;
}
function mergeAssets(code, cssBlocks, jsBlocks) {
  let out = code;
  const isLocal = u => !/^(https?:)?\/\//i.test(u) && !/^data:/i.test(u);
  let removedCss = false, removedJs = false;
  if (cssBlocks.length) {
    out = out.replace(/<link\b[^>]*>/gi, tag => {
      const h = tag.match(/href=["']([^"']+)["']/i);
      if (/rel=["']?stylesheet/i.test(tag) && h && isLocal(h[1])) { removedCss = true; return ''; }
      return tag;
    });
  }
  if (jsBlocks.length) {
    out = out.replace(/<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>\s*<\/script>/gi, (tag, u) => {
      if (isLocal(u)) { removedJs = true; return ''; }
      return tag;
    });
  }
  if (cssBlocks.length && (removedCss || !/<style[\s>]/i.test(out))) {
    const tag = '<style>\n' + cssBlocks.join('\n') + '\n</style>';
    out = /<\/head>/i.test(out) ? out.replace(/<\/head>/i, tag + '\n</head>') : tag + '\n' + out;
  }
  if (jsBlocks.length && (removedJs || !/<script\b(?![^>]*\bsrc=)[^>]*>/i.test(out))) {
    const tag = '<script>\n' + jsBlocks.join('\n') + '\n<\/script>';
    out = /<\/body>/i.test(out) ? out.replace(/<\/body>/i, tag + '\n</body>') : out + '\n' + tag;
  }
  return out;
}
function canvasEsc(t) { return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
// Runs INSIDE the preview iframe: loads Mermaid (newest first, older as fallback),
// tries several layout/cleanup variants, and shows a clear message if all of them fail.
function mermaidRunner(CODE) {
  var out = document.getElementById('out');
  var finished = false;
  var lastErr = null;
  function cleanup() {
    Array.prototype.slice.call(document.body.children).forEach(function (el) {
      if (el.id !== 'out' && el.tagName !== 'SCRIPT') el.remove();
    });
  }
  function fail(msg) {
    finished = true;
    cleanup();
    out.innerHTML = '';
    var d = document.createElement('div');
    d.style.cssText = 'color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;padding:12px;border-radius:8px;font:14px system-ui,sans-serif;white-space:pre-wrap';
    d.textContent = msg;
    out.appendChild(d);
    try { parent.postMessage({ canvasError: msg }, '*'); } catch (e) {}
  }
  var versions = ['11.4.1', '10.9.1'];
  var hosts = [
    function (v) { return 'https://cdn.jsdelivr.net/npm/mermaid@' + v + '/dist/mermaid.min.js'; },
    function (v) { return 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/' + v + '/mermaid.min.js'; },
    function (v) { return 'https://unpkg.com/mermaid@' + v + '/dist/mermaid.min.js'; }
  ];
  function stripBr(c) { return c.replace(/<br\s*\/?>/gi, ' '); }
  function stripClasses(c) { return c.replace(/:::[\w-]+/g, ''); }
  // Turns  A -- Yes --> B  and  A -->|Yes| B  into  A --> lbl1(["Yes"]) --> B
  // (Mermaid sometimes crashes while positioning labels on edges of cyclic graphs.)
  function labelsToNodes(c) {
    var n = 0;
    return c.split('\n').map(function (line) {
      if (/^\s*%%/.test(line)) return line;
      var m = line.match(/^(\s*)(.+?)\s+--\s+(.+?)\s+-->\s+(.+?)\s*$/) ||
              line.match(/^(\s*)(.+?)\s*-->\s*\|([^|]+)\|\s*(.+?)\s*$/);
      if (!m) return line;
      n++;
      var label = m[3].replace(/^"|"$/g, '').replace(/"/g, "'");
      return m[1] + m[2] + ' --> lbl' + n + '(["' + label + '"]) --> ' + m[4];
    }).join('\n');
  }
  function flipDir(c) {
    return c.replace(/^(\s*(?:flowchart|graph)\s+)(TD|TB)\b/m, '$1LR');
  }

  var fixes = [
    function (c) { return c.replace(/<br\s*\/?>/gi, '<br/>'); },
    function (c) { return stripClasses(stripBr(c)); },
    function (c) { return stripClasses(stripBr(c)); },
    function (c) { return labelsToNodes(stripClasses(stripBr(c))); },
    function (c) { return flipDir(labelsToNodes(stripClasses(stripBr(c)))); }
  ];
  var curves = ['basis', 'basis', 'linear', 'linear', 'linear'];
  var attempts = fixes.map(function (f, i) { return { curve: curves[i], fix: f }; });
  function run(vi, ai) {
    if (ai >= attempts.length) { loadVersion(vi + 1); return; }
    var a = attempts[ai];
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        suppressErrorRendering: true,
        securityLevel: 'loose',
        theme: 'default',
        flowchart: { curve: a.curve, htmlLabels: true }
      });
      window.mermaid.render('dg' + Date.now() + '_' + vi + '_' + ai, a.fix(CODE).trim()).then(function (r) {
        finished = true;
        cleanup();
        out.innerHTML = r.svg;
      }).catch(function (e) {
        lastErr = e;
        cleanup();
        run(vi, ai + 1);
      });
    } catch (e) {
      lastErr = e;
      cleanup();
      run(vi, ai + 1);
    }
  }
  function loadScript(v, hi, done) {
    if (hi >= hosts.length) { done(false); return; }
    var s = document.createElement('script');
    s.src = hosts[hi](v);
    s.onload = function () { done(true); };
    s.onerror = function () { s.remove(); loadScript(v, hi + 1, done); };
    document.head.appendChild(s);
  }
  function loadVersion(vi) {
    if (vi >= versions.length) {
      if (lastErr) {
        fail('Diagram error:\n' + (lastErr.message ? lastErr.message : lastErr) + '\n\nThe diagram code itself needs simplifying: ask the AI to redraw it with simpler labels and fewer loops.');
      } else {
        fail('Could not load the Mermaid library (CDN blocked or no connection).');
      }
      return;
    }
    try { window.mermaid = undefined; } catch (e) {}
    loadScript(versions[vi], 0, function (ok) {
      if (ok && window.mermaid) run(vi, 0); else loadVersion(vi + 1);
    });
  }
  setTimeout(function () { if (!finished) fail('The diagram took too long to render. Try again or check your connection.'); }, 30000);
  loadVersion(0);
}
function buildSrcdoc(art) {
  const storageShim = '(function(){function mk(){var d={};return{getItem:function(k){return Object.prototype.hasOwnProperty.call(d,k)?d[k]:null},setItem:function(k,v){d[k]=String(v)},removeItem:function(k){delete d[k]},clear:function(){d={}},key:function(i){return Object.keys(d)[i]||null},get length(){return Object.keys(d).length}}}try{Object.defineProperty(window,"localStorage",{value:mk(),configurable:true});Object.defineProperty(window,"sessionStorage",{value:mk(),configurable:true})}catch(e){}})();';
  const errHook = '<script>' + storageShim + 'window.addEventListener("error",function(e){parent.postMessage({canvasError:String(e.message||"Script error")},"*")});<\/script>';
  const head = '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">';
  if (art.type === 'html') {
    let doc = art.code;
    if (!/<html[\s>]/i.test(doc) && !/<!doctype/i.test(doc)) {
      doc = '<!doctype html><html><head>' + head + '</head><body>' + doc + '</body></html>';
    }
    if (/<head[^>]*>/i.test(doc)) return doc.replace(/<head[^>]*>/i, m => m + errHook);
    if (/<html[^>]*>/i.test(doc)) return doc.replace(/<html[^>]*>/i, m => m + '<head>' + errHook + '</head>');
    return doc;
  }
  if (art.type === 'svg') {
    return '<!doctype html><html><head>' + head + '<style>html,body{margin:0;height:100%}body{display:flex;align-items:center;justify-content:center;background:#fff}svg{max-width:100%;max-height:100%;height:auto}</style></head><body>' + art.code + '</body></html>';
  }
  if (art.type === 'mermaid') {
    const codeJson = JSON.stringify(art.code).replace(/</g, '\\u003c');
    return '<!doctype html><html><head>' + head + '<style>html,body{margin:0;background:#fff;font-family:system-ui,sans-serif}body{padding:16px;overflow:auto}#out{min-width:min-content}#out svg{max-width:100%;height:auto}</style>' + errHook + '</head><body><div id="out" style="color:#666;font-size:14px">Rendering diagram...</div><script>(' + mermaidRunner.toString() + ')(' + codeJson + ');<\/script></body></html>';
  }
  const body = safeParseMarkdown(art.code);
  return '<!doctype html><html><head>' + head + '<style>body{max-width:760px;margin:0 auto;padding:24px 20px;font-family:system-ui,sans-serif;line-height:1.65;color:#1f2328;background:#fff}pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}code{background:#f6f8fa;padding:2px 5px;border-radius:4px}table{border-collapse:collapse}td,th{border:1px solid #d0d7de;padding:6px 10px}blockquote{border-left:4px solid #d0d7de;margin:0;padding-left:14px;color:#57606a}img{max-width:100%}</style></head><body>' + body + '</body></html>';
}
function openCanvas(arts, idx) {
  canvasArtifacts = arts;
  canvasIndex = idx || 0;
  const sel = document.getElementById('canvas-select');
  sel.innerHTML = arts.map((a, i) => `<option value="${i}">${i + 1}. ${canvasEsc(a.label)}</option>`).join('');
  sel.value = String(canvasIndex);
  sel.style.display = arts.length > 1 ? '' : 'none';
  document.getElementById('canvas-panel').classList.add('open');
  renderCanvas();
}
function selectCanvasArtifact(i) {
  canvasIndex = parseInt(i, 10) || 0;
  renderCanvas();
}
function renderCanvas() {
  const art = canvasArtifacts[canvasIndex];
  if (!art) return;
  document.getElementById('canvas-title').textContent = art.label + ' preview';
  const err = document.getElementById('canvas-error');
  err.style.display = 'none';
  err.textContent = '';
  document.getElementById('canvas-frame').srcdoc = buildSrcdoc(art);
  const codeEl = document.getElementById('canvas-code');
  const lang = art.type === 'markdown' ? 'markdown' : (art.type === 'mermaid' ? 'plaintext' : 'xml');
  try { codeEl.innerHTML = window.hljs ? window.hljs.highlight(art.code, { language: lang }).value : canvasEsc(art.code); }
  catch (e) { codeEl.textContent = art.code; }
  setCanvasView(canvasView);
}
function setCanvasView(v) {
  canvasView = v;
  document.getElementById('canvas-frame').style.display = v === 'preview' ? 'block' : 'none';
  document.getElementById('canvas-code').style.display = v === 'code' ? 'block' : 'none';
  document.getElementById('tab-preview').classList.toggle('active', v === 'preview');
  document.getElementById('tab-code').classList.toggle('active', v === 'code');
}
function closeCanvas() {
  const p = document.getElementById('canvas-panel');
  if (!p) return;
  p.classList.remove('open');
  p.classList.remove('fullscreen');
  document.getElementById('canvas-frame').srcdoc = '';
}
function toggleCanvasFullscreen() {
  document.getElementById('canvas-panel').classList.toggle('fullscreen');
}
function copyCanvasCode() {
  const art = canvasArtifacts[canvasIndex];
  if (!art) return;
  const btn = document.getElementById('canvas-copy');
  try { navigator.clipboard.writeText(art.code); } catch (e) {}
  btn.textContent = '✅';
  setTimeout(() => { btn.textContent = '📋'; }, 1500);
}
function downloadCanvas() {
  const art = canvasArtifacts[canvasIndex];
  if (!art) return;
  const isDoc = art.type === 'markdown';
  const ext = art.type === 'svg' ? 'svg' : (art.type === 'mermaid' ? 'mmd' : (isDoc ? 'md' : 'html'));
  const content = art.type === 'html' ? buildSrcdoc(art) : art.code;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: 'text/plain;charset=utf-8' }));
  a.download = 'preview_' + Date.now() + '.' + ext;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
function autoOpenCanvas(text) {
  const arts = extractArtifacts(text);
  const idx = arts.findIndex(a =>
    a.type === 'mermaid' || a.type === 'svg' ||
    (a.type === 'html' && (/<!doctype|<html[\s>]|<script|<canvas|<style/i.test(a.code) || a.code.length > 600)));
  if (idx >= 0) openCanvas(arts, idx);
}
window.addEventListener('message', (ev) => {
  const frame = document.getElementById('canvas-frame');
  if (!frame || ev.source !== frame.contentWindow) return;
  if (ev.data && ev.data.canvasError) {
    const err = document.getElementById('canvas-error');
    err.textContent = '⚠ Error in the preview: ' + ev.data.canvasError + '  (ask the AI to fix it)';
    err.style.display = 'block';
  }
});
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
const arts = extractArtifacts(text);
arts.forEach((a, i) => {
const pb = document.createElement('button');
pb.className = 'action-btn';
pb.innerText = '▶ ' + (arts.length > 1 ? (i + 1) + '. ' + a.label : 'Preview ' + a.label);
pb.onclick = () => openCanvas(arts, i);
metaSpan.appendChild(pb);
});
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
autoOpenCanvas(fullText);
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
async function checkAdmin() {
try {
const res = await fetch('/api/me', { headers: getAuthHeaders() });
if (!res.ok) return;
const d = await res.json();
const b = document.getElementById('analytics-btn');
if (b) b.style.display = d.is_admin ? '' : 'none';
} catch (e) {}
}
function fmtInt(n) { return Number(n || 0).toLocaleString(); }
async function fetchAnalyticsStats() {
const body = document.getElementById('log-table-body');
const hoursEl = document.getElementById('analytics-hours');
const hours = hoursEl ? hoursEl.value : 24;
try {
const res = await fetch('/api/admin/stats?hours=' + encodeURIComponent(hours), { headers: getAuthHeaders() });
if (res.status === 403) {
body.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #f87171;">Analytics is restricted to administrators.</td></tr>';
return;
}
const data = await res.json().catch(() => ({}));
if (!res.ok) {
body.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #f87171;">' + escapeHtml(data.detail || ('Error ' + res.status)) + '</td></tr>';
return;
}
const set = (id, v) => { const el = document.getElementById(id); if (el) el.innerText = v; };
set('stat-reqs', `${data.total_successful}/${data.total_requests}`);
set('stat-errrate', `${data.error_rate}%`);
set('stat-cost', `$${data.total_cost_usd.toFixed(5)}`);
set('stat-tokens', fmtInt(data.total_prompt_tokens + data.total_completion_tokens));
set('stat-latency', `${Math.round(data.avg_latency_ms)}ms`);
set('stat-p95', `${Math.round(data.p95_latency_ms)}ms`);
set('stat-users', data.active_users);
set('stat-active', data.active_now);
const lim = data.limits || {};
set('limits-note', `Limits: ${lim.user_per_min}/min per user, ${lim.ip_per_min}/min per IP, ${lim.daily || 'no'} messages/day per user` + (data.truncated ? ' · showing the latest 1000 requests' : ''));
const ms = (data.by_model || []).map(m => `${escapeHtml(m.model)}: ${Math.round(m.avg_latency_ms)}ms avg (${m.requests} requests)`).join(' · ');
const msEl = document.getElementById('model-stats');
if (msEl) msEl.innerHTML = ms ? 'Model latency: ' + ms : '';
const topBody = document.getElementById('top-users-body');
if (topBody) {
if (!data.top_users || data.top_users.length === 0) {
topBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #666;">No data yet.</td></tr>';
} else {
topBody.innerHTML = data.top_users.map(u => `<tr><td>${escapeHtml(u.email)}</td><td>${u.requests}</td><td>${fmtInt(u.tokens)}</td><td>$${u.cost.toFixed(5)}</td></tr>`).join('');
}
}
if (!data.recent_logs || data.recent_logs.length === 0) {
body.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No logs recorded yet.</td></tr>';
return;
}
body.innerHTML = data.recent_logs.map(log => {
const statusClass = log.status === 200 ? 'status-200' : 'status-429';
const tip = log.error ? ` title="${escapeHtml(log.error)}"` : '';
return `<tr>
<td>${escapeHtml(log.time)}</td>
<td>${escapeHtml(log.email)}</td>
<td class="${statusClass}"${tip}>${log.status}</td>
<td>${Math.round(log.latency_ms)}ms</td>
<td>${log.prompt_tokens}/${log.completion_tokens}</td>
<td>$${log.cost_usd.toFixed(6)}</td>
</tr>`;
}).join('');
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


@app.patch("/api/chats/{chat_id}")
def rename_chat(
    chat_id: str, body: RenameChatRequest, user: AuthUser = Depends(require_user)
):
    title = (body.title or "").strip()[:80]
    if not title:
        return JSONResponse({"detail": "Title cannot be empty."}, status_code=400)
    try:
        res = (
            user.db.table("chats")
            .update({"title": title})
            .eq("id", chat_id)
            .eq("user_id", user.id)
            .execute()
        )
        if not res.data:
            return JSONResponse({"detail": "Chat not found."}, status_code=404)
        return JSONResponse({"id": chat_id, "title": title})
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


def is_admin(user: AuthUser) -> bool:
    # If ADMIN_EMAILS is not set, every logged-in user counts as admin.
    return not ADMIN_EMAILS or (user.email or "").lower() in ADMIN_EMAILS


@app.get("/api/me")
def get_me(user: AuthUser = Depends(require_user)):
    return JSONResponse({"email": user.email, "is_admin": is_admin(user)})


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@app.get("/api/admin/stats")
def get_admin_stats(
    request: Request, hours: int = 24, user: AuthUser = Depends(require_user)
):
    # If ADMIN_EMAILS is set (comma-separated), only those users may see stats.
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Admins only.")

    hours = max(1, min(int(hours), 24 * 30))
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).isoformat()
    try:
        res = (
            user.db.table("api_logs")
            .select(
                "created_at,user_id,email,endpoint,status,latency_ms,"
                "prompt_tokens,completion_tokens,cost_usd,model,ip,error"
            )
            .gte("created_at", since)
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        return JSONResponse(
            {"detail": f"Could not read the api_logs table: {e}"}, status_code=500
        )

    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == 200)
    limited = sum(1 for r in rows if r["status"] == 429)
    failed = total - ok
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in rows)
    cost = sum(float(r.get("cost_usd") or 0) for r in rows)

    lats = sorted(
        float(r["latency_ms"])
        for r in rows
        if r["status"] == 200 and r.get("latency_ms")
    )
    avg_latency = sum(lats) / len(lats) if lats else 0.0
    p95_latency = lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else 0.0

    recent_cut = now - timedelta(minutes=15)
    users_all = set()
    users_now = set()
    by_user: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, List[float]] = {}
    for r in rows:
        uid = r.get("user_id") or "anonymous"
        users_all.add(uid)
        try:
            if _parse_ts(r["created_at"]) >= recent_cut:
                users_now.add(uid)
        except Exception:
            pass
        u = by_user.setdefault(
            uid,
            {"email": r.get("email") or "unknown", "requests": 0, "tokens": 0, "cost": 0.0},
        )
        u["requests"] += 1
        u["tokens"] += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
        u["cost"] += float(r.get("cost_usd") or 0)
        if r["status"] == 200 and r.get("model") and r.get("latency_ms"):
            m = by_model.setdefault(r["model"], [0.0, 0.0])
            m[0] += 1
            m[1] += float(r["latency_ms"])

    top_users = sorted(by_user.values(), key=lambda x: x["cost"], reverse=True)[:10]
    for u in top_users:
        u["cost"] = round(u["cost"], 6)

    recent_logs = []
    for r in rows[:50]:
        try:
            when = _parse_ts(r["created_at"]).strftime("%d/%m %H:%M:%S")
        except Exception:
            when = ""
        recent_logs.append({
            "time": when,
            "email": r.get("email") or "",
            "endpoint": r.get("endpoint"),
            "ip": r.get("ip"),
            "status": r["status"],
            "latency_ms": r.get("latency_ms") or 0,
            "prompt_tokens": r.get("prompt_tokens") or 0,
            "completion_tokens": r.get("completion_tokens") or 0,
            "cost_usd": float(r.get("cost_usd") or 0),
            "error": r.get("error"),
        })

    return JSONResponse({
        "hours": hours,
        "truncated": total >= 1000,
        "total_requests": total,
        "total_successful": ok,
        "total_failed": failed,
        "total_rate_limited": limited,
        "error_rate": round((failed - limited) / total * 100, 1) if total else 0.0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_cost_usd": round(cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "active_users": len(users_all),
        "active_now": len(users_now),
        "top_users": top_users,
        "by_model": [
            {"model": k, "requests": int(v[0]), "avg_latency_ms": round(v[1] / v[0], 1)}
            for k, v in by_model.items()
        ],
        "limits": {
            "user_per_min": USER_RATE_LIMIT_PER_MIN,
            "ip_per_min": MAX_REQUESTS_PER_WINDOW,
            "daily": DAILY_REQUEST_LIMIT,
        },
        "recent_logs": recent_logs,
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
        log_analytics_entry(
            "/api/enhance-prompt", 429, 0.0, 0, 0, client_ip, user=user, model=MODEL_NAME
        )
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
            "/api/enhance-prompt", 200, latency_ms, p_tokens, c_tokens, client_ip,
            user=user, model=MODEL_NAME,
        )
        return JSONResponse({"enhanced_prompt": enhanced})
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_analytics_entry(
            "/api/enhance-prompt", 500, latency_ms, 0, 0, client_ip,
            user=user, model=MODEL_NAME, error=str(e),
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
        log_analytics_entry(
            "/api/chat", 429, 0.0, 0, 0, client_ip, user=user, model=MODEL_NAME,
            error="IP rate limit",
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "Too Many Requests. Rate limit exceeded (30 reqs/min). Please wait"
                " a moment."
            ),
        )

    if not check_rate_limit(f"user:{user.id}", USER_RATE_LIMIT_PER_MIN):
        log_analytics_entry(
            "/api/chat", 429, 0.0, 0, 0, client_ip, user=user, model=MODEL_NAME,
            error="User rate limit",
        )
        raise HTTPException(
            status_code=429,
            detail=f"Too many messages ({USER_RATE_LIMIT_PER_MIN} per minute). Please wait a moment.",
        )

    if not check_daily_quota(user):
        log_analytics_entry(
            "/api/chat", 429, 0.0, 0, 0, client_ip, user=user, model=MODEL_NAME,
            error="Daily quota",
        )
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({DAILY_REQUEST_LIMIT} messages per day). Try again tomorrow.",
        )

    try:
        chat_id = body.chat_id
        message = body.message
        # Only keep the most recent messages -> much faster on long chats.
        history = (body.history or [])[-MAX_HISTORY_MESSAGES:]
        file_payload = body.file
        system_instruction = (
            body.system_instruction or "You are a helpful assistant."
        ) + CANVAS_HINT
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
        usage: Dict[str, int] = {}

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
                usage.clear()
                config = build_config(
                    system_instruction, tools, state["use_thinking"]
                )
                response = client.models.generate_content_stream(
                    model=MODEL_NAME, contents=contents, config=config
                )
                for chunk in response:
                    _capture_usage(chunk, usage)
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
                usage.clear()
                res = chat_session.send_message(prompt_content, stream=True)
                for chunk in res:
                    _capture_usage(chunk, usage)
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
                        usage.get("p") or prompt_tokens_est,
                        usage.get("c") or estimate_tokens(total_output_text),
                        client_ip,
                        user=user,
                        model=MODEL_NAME,
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
                        "/api/chat", status_code, latency_ms, 0, 0, client_ip,
                        user=user, model=MODEL_NAME, error=err_msg,
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