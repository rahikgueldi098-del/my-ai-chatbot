import os
import io
import base64
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from google import genai
from google.genai import types

app = FastAPI(title="AI Assistant Studio Pro")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=GEMINI_API_KEY)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Assistant Studio Pro</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #212121; color: #ececec; display: flex; height: 100vh; overflow: hidden; }
    #sidebar { width: 260px; background-color: #171717; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; gap: 12px; }
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
    #main-container { flex: 1; display: flex; flex-direction: column; height: 100vh; }
    header { padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; background: #171717; font-weight: 600; }
    .header-controls { display: flex; gap: 12px; align-items: center; }
    #persona-select { background: #2f2f2f; color: #fff; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.88rem; outline: none; cursor: pointer; max-width: 200px; }
    .toggle-btn { background: #2f2f2f; color: #888; border: 1px solid #424242; border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; transition: 0.2s; }
    .toggle-btn.active { background: #1b3a2b; color: #4ade80; border-color: #22c55e; }
    #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; width: 100%; margin: 0 auto; }
    .message { display: flex; flex-direction: column; gap: 6px; max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; position: relative; word-break: break-word; }
    .user { align-self: flex-end; background-color: #303030; color: #fff; border-bottom-right-radius: 2px; }
    .model { align-self: flex-start; background-color: #212121; color: #ececec; border-bottom-left-radius: 2px; border: 1px solid #333; width: 100%; }
    .message img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
    .doc-badge { display: inline-flex; align-items: center; gap: 8px; background: #1e293b; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; margin-bottom: 6px; font-size: 0.88rem; color: #38bdf8; }
    .message p { margin-bottom: 8px; }
    .message p:last-child { margin-bottom: 0; }
    .message code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
    .copy-btn { position: absolute; top: 8px; right: 8px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 0.75rem; cursor: pointer; }
    .msg-actions { display: flex; gap: 8px; margin-top: 8px; padding-top: 6px; border-top: 1px solid #2a2a2a; align-items: center; }
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
    .chip { background: #2a2a2a; border: 1px solid #3a3a3a; color: #ccc; padding: 5px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; white-space: nowrap; transition: 0.2s; }
    .chip:hover { background: #383838; color: #fff; border-color: #555; }
    #file-preview { display: none; align-items: center; gap: 10px; background: #2f2f2f; padding: 8px 12px; border-radius: 8px; border: 1px solid #424242; width: fit-content; font-size: 0.85rem; }
    #preview-img { height: 40px; width: 40px; object-fit: cover; border-radius: 4px; display: none; }
    #remove-file-btn { background: transparent; border: none; color: #ff5555; cursor: pointer; font-size: 1rem; margin-left: 6px; }
    #input-container { display: flex; gap: 8px; align-items: flex-end; }
    #file-input { display: none; }
    .icon-btn { height: 48px; width: 48px; border-radius: 24px; border: 1px solid #424242; background: #2f2f2f; color: #fff; font-size: 1.1rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: 0.2s; flex-shrink: 0; }
    .icon-btn:hover { background: #383838; }
    .icon-btn.recording { background: #3a1c1c; border-color: #ef4444; color: #ef4444; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    #user-input { flex: 1; padding: 12px 16px; border-radius: 18px; border: 1px solid #424242; background: #2f2f2f; color: #fff; outline: none; font-size: 1rem; resize: none; max-height: 150px; min-height: 48px; line-height: 1.4; }
    #send-btn { height: 48px; min-width: 70px; padding: 0 20px; border-radius: 24px; border: none; background: #fff; color: #000; font-weight: 600; cursor: pointer; flex-shrink: 0; }
    #send-btn.stop-btn { background: #ef4444; color: #fff; }
  </style>
</head>
<body>
  <div id="sidebar">
    <button id="new-chat-btn" onclick="startNewChat()">+ New Chat</button>
    <input type="text" id="chat-search" placeholder="Search chats..." oninput="renderSidebar()">
    <div id="history-list"></div>
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
      <span>AI Assistant Studio Pro</span>
      <div class="header-controls">
        <button id="search-toggle" class="toggle-btn active" onclick="toggleSearch()">Web Search: ON</button>
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
  <script>
    let chats = JSON.parse(localStorage.getItem('ai_chats') || '[]');
    let currentChatId = localStorage.getItem('ai_current_chat_id') || null;
    let webSearchEnabled = true;
    let selectedFile = null;
    let recognition = null;
    let isRecording = false;
    let isStreaming = false;
    let activeAbortController = null;
    if (chats.length === 0) {
      startNewChat();
    } else {
      if (!currentChatId || !chats.find(c => c.id === currentChatId)) {
        currentChatId = chats[0].id;
      }
      loadChat(currentChatId);
    }
    function saveChats() {
      localStorage.setItem('ai_chats', JSON.stringify(chats));
      localStorage.setItem('ai_current_chat_id', currentChatId);
    }
    function getCurrentChat() {
      return chats.find(c => c.id === currentChatId);
    }
    function startNewChat() {
      const newId = 'chat_' + Date.now();
      const newChat = { id: newId, title: 'New Discussion', history: [] };
      chats.unshift(newChat);
      currentChatId = newId;
      saveChats();
      renderSidebar();
      renderChatBox();
    }
    function loadChat(id) {
      currentChatId = id;
      saveChats();
      renderSidebar();
      renderChatBox();
    }
    function deleteChat(id, event) {
      if (event) event.stopPropagation();
      chats = chats.filter(c => c.id !== id);
      if (chats.length === 0) {
        startNewChat();
      } else {
        currentChatId = chats[0].id;
        saveChats();
        renderSidebar();
        renderChatBox();
      }
    }
    function renderSidebar() {
      const list = document.getElementById('history-list');
      const search = document.getElementById('chat-search').value.toLowerCase();
      list.innerHTML = '';
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
      box.innerHTML = '';
      const chat = getCurrentChat();
      if (!chat) return;
      chat.history.forEach((msg) => {
        appendMessageUI(msg.role, msg.content, msg.file);
      });
      box.scrollTop = box.scrollHeight;
    }
    function appendMessageUI(role, text, fileObj) {
      const box = document.getElementById('chat-box');
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
          badge.innerHTML = `📄 <strong>${fileObj.name}</strong> (${(fileObj.size/1024).toFixed(1)} KB)`;
          msgDiv.appendChild(badge);
        }
      }
      const contentDiv = document.createElement('div');
      contentDiv.className = 'text-content';
      if (role === 'model') {
        contentDiv.innerHTML = text ? marked.parse(text) : '<div class="typing-dots"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
      } else {
        contentDiv.innerText = text;
      }
      msgDiv.appendChild(contentDiv);
      if (text && role === 'model') {
        const wordCount = text.trim().split(/\\s+/).filter(Boolean).length;
        const readTime = Math.max(1, Math.ceil(wordCount / 200));
        const metaSpan = document.createElement('div');
        metaSpan.className = 'msg-actions';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn';
        copyBtn.innerText = '📋 Copy';
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(text);
          copyBtn.innerText = '✅ Copied';
          setTimeout(() => copyBtn.innerText = '📋 Copy', 2000);
        };
        const metaInfo = document.createElement('span');
        metaInfo.className = 'msg-meta';
        metaInfo.innerText = `${wordCount} words • ~${readTime} min read`;
        metaSpan.appendChild(copyBtn);
        metaSpan.appendChild(metaInfo);
        msgDiv.appendChild(metaSpan);
      }
      box.appendChild(msgDiv);
      box.scrollTop = box.scrollHeight;
      if (window.renderMathInElement) {
        renderMathInElement(msgDiv, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
          ]
        });
      }
      return msgDiv;
    }
    function toggleSearch() {
      webSearchEnabled = !webSearchEnabled;
      const btn = document.getElementById('search-toggle');
      btn.innerText = webSearchEnabled ? "Web Search: ON" : "Web Search: OFF";
      btn.className = webSearchEnabled ? "toggle-btn active" : "toggle-btn";
    }
    function applyChip(prefix) {
      const input = document.getElementById('user-input');
      if (input.value.trim()) {
        input.value = prefix + " " + input.value;
      } else {
        input.value = prefix + " ";
      }
      input.focus();
      autoExpand(input);
    }
    function handlePersonaChange(select) {
      if (select.value === '__NEW__') {
        const name = prompt("Enter Custom Persona Name (e.g. Code Reviewer):");
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
        alert("Speech Recognition is not supported in this browser. Try Chrome or Edge.");
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
        recognition.onstart = () => {
          isRecording = true;
          micBtn.classList.add('recording');
        };
        recognition.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          const input = document.getElementById('user-input');
          input.value = input.value ? input.value + ' ' + transcript : transcript;
          autoExpand(input);
        };
        recognition.onerror = (event) => {
          console.error("Speech error:", event.error);
          isRecording = false;
          micBtn.classList.remove('recording');
        };
        recognition.onend = () => {
          isRecording = false;
          micBtn.classList.remove('recording');
        };
        recognition.start();
      } catch (err) {
        console.error(err);
        isRecording = false;
        micBtn.classList.remove('recording');
      }
    }
    async function enhanceCurrentPrompt() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) {
        alert("Please enter a prompt draft first!");
        return;
      }
      const enhanceBtn = document.getElementById('enhance-btn');
      const originalText = enhanceBtn.innerText;
      enhanceBtn.innerText = '⏳';
      enhanceBtn.disabled = true;
      try {
        const res = await fetch('/api/enhance-prompt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text })
        });
        const data = await res.json();
        if (data.enhanced_prompt) {
          input.value = data.enhanced_prompt;
          autoExpand(input);
        } else if (data.error) {
          alert("Enhance failed: " + data.error);
        }
      } catch (err) {
        alert("Error connecting to server: " + err.message);
      } finally {
        enhanceBtn.innerText = originalText;
        enhanceBtn.disabled = false;
      }
    }
    function handleSendOrStop() {
      if (isStreaming) {
        if (activeAbortController) activeAbortController.abort();
        isStreaming = false;
        setSendButtonState(false);
      } else {
        sendMessage();
      }
    }
    function setSendButtonState(streaming) {
      const sendBtn = document.getElementById('send-btn');
      if (streaming) {
        sendBtn.innerText = 'Stop ⏹';
        sendBtn.classList.add('stop-btn');
      } else {
        sendBtn.innerText = 'Send';
        sendBtn.classList.remove('stop-btn');
      }
    }
    async function sendMessage() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text && !selectedFile) return;
      const chat = getCurrentChat();
      if (!chat) return;
      if (chat.history.length === 0) {
        chat.title = text ? (text.slice(0, 30) + (text.length > 30 ? '...' : '')) : (selectedFile ? selectedFile.name : 'New Discussion');
        renderSidebar();
      }
      const userMsg = { role: 'user', content: text, file: selectedFile };
      chat.history.push(userMsg);
      appendMessageUI('user', text, selectedFile);
      input.value = '';
      input.style.height = 'auto';
      const filePayload = selectedFile;
      clearFile();
      isStreaming = true;
      setSendButtonState(true);
      activeAbortController = new AbortController();
      const botMsgDiv = appendMessageUI('model', '', null);
      const contentDiv = botMsgDiv.querySelector('.text-content');
      const systemPrompt = document.getElementById('persona-select').value;
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: activeAbortController.signal,
          body: JSON.stringify({
            history: chat.history.slice(0, -1),
            message: text,
            file: filePayload,
            web_search: webSearchEnabled,
            system_instruction: systemPrompt
          })
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Server HTTP " + res.status);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let accText = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          accText += decoder.decode(value, { stream: true });
          contentDiv.innerHTML = marked.parse(accText);
          const box = document.getElementById('chat-box');
          box.scrollTop = box.scrollHeight;
        }
        chat.history.push({ role: 'model', content: accText });
        saveChats();
        renderChatBox();
      } catch (err) {
        if (err.name !== 'AbortError') {
          contentDiv.innerText = "Error: " + err.message;
        }
      } finally {        isStreaming = false;
        activeAbortController = null;
        setSendButtonState(false);
      }
    }
    function exportChat(format) {
      const chat = getCurrentChat();
      if (!chat || chat.history.length === 0) {
        alert("No conversation history to export!");
        return;
      }
      let dataStr = '';
      let fileName = `${chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_export`;
      if (format === 'json') {
        dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(chat, null, 2));
        fileName += '.json';
      } else {
        let md = `# ${chat.title}\n\n`;
        chat.history.forEach(m => {
          md += `### ${m.role.toUpperCase()}\n${m.content}\n\n---\n\n`;
        });
        dataStr = "data:text/markdown;charset=utf-8," + encodeURIComponent(md);
        fileName += '.md';
      }
      const dl = document.createElement('a');
      dl.setAttribute('href', dataStr);
      dl.setAttribute('download', fileName);
      document.body.appendChild(dl);
      dl.click();
      dl.remove();
    }
  </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def serve_gui():
    return HTML_CONTENT

@app.post("/api/enhance-prompt")
async def enhance_prompt(request: Request):
    try:
        body = await request.json()
        raw_prompt = body.get("prompt", "")
        if not raw_prompt.strip():
            return JSONResponse({"enhanced_prompt": raw_prompt})

        client = get_client()
        system_instruction = (
            "You are an expert prompt engineer. The user will provide a rough draft or short prompt. "
            "Rewrite it into a detailed, high-quality, clear, and effective prompt for an LLM. "
            "Output ONLY the improved prompt text itself—no greetings, no explanations, no quotes."
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=f"Improve this prompt: {raw_prompt}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        return JSONResponse({"enhanced_prompt": response.text.strip()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
        history = body.get("history", [])
        message = body.get("message", "")
        file_payload = body.get("file")
        web_search = body.get("web_search", True)
        system_instruction = body.get("system_instruction", "You are a helpful AI assistant.")

        client = get_client()

        formatted_contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            parts = []
            if msg.get("content"):
                parts.append(types.Part.from_text(text=msg["content"]))
            if parts:
                formatted_contents.append(types.Content(role=role, parts=parts))

        current_parts = []
        if message:
            current_parts.append(types.Part.from_text(text=message))

        if file_payload and "data" in file_payload:
            header, base64_data = file_payload["data"].split(",", 1)
            file_bytes = base64.b64decode(base64_data)
            mime_type = file_payload.get("type", "application/octet-stream")
            if mime_type.startswith("image/"):
                current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
            else:
                try:
                    text_content = file_bytes.decode("utf-8")
                    doc_prompt = f"\n\n[Attached File: {file_payload.get('name')}]\n{text_content}"
                    current_parts.append(types.Part.from_text(text=doc_prompt))
                except Exception:
                    current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type="application/octet-stream"))

        formatted_contents.append(types.Content(role="user", parts=current_parts))

        tools = [{"google_search": {}}] if web_search else None

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.7,
        )

        async def generate_stream():
            try:
                response = client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=formatted_contents,
                    config=config
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as err:
                yield f"\n[Error during generation: {str(err)}]"

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))