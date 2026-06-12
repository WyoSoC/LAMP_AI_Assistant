"""LAMP RAG chatbot — FastAPI backend.

POST /api/chat  {"message": str, "history": [{"role","content"}...]}
  -> {"answer": str, "sources": [{"n","title","url","heading"}...], "mode": str}

Answers are synthesized by a local Ollama model from retrieved passages, with
inline [n] citations resolved to source URLs.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .retrieval import Retriever

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))

app = FastAPI(title="LAMP RAG Chatbot")
retriever = Retriever()

_ollama = None
try:
    import ollama as _ollama_lib
    _ollama = _ollama_lib.Client(host=OLLAMA_HOST)
    # Verify connectivity — will raise if Ollama isn't reachable
    _ollama.list()
except Exception:
    _ollama = None

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the \
Learning Actively Mentoring Program (LAMP) at the University of Wyoming, using \
ONLY the provided source passages from the LAMP website.

Rules:
- Ground every claim in the passages. If the passages don't contain the answer, \
say so plainly and suggest which LAMP page might help.
- Cite sources inline with bracketed numbers like [1] or [2][3] matching the \
passage numbers. Cite at the end of each sentence or claim they support.
- Synthesize across passages when the question spans topics; don't just quote.
- Be concise and direct. Use prose; use lists only when listing items.
- Questions may be open-ended; it is fine to organize a longer answer, but stay \
within what the sources support."""


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


def _passages_block(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        head = f" — {h['heading']}" if h.get("heading") else ""
        parts.append(f"[{i}] {h['title']}{head}\nURL: {h['url']}\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def _retrieval_query(message: str, history: list[dict]) -> str:
    """Augment very short follow-ups with recent user context."""
    if len(message.split()) >= 4 or not history:
        return message
    prev_user = [h["content"] for h in history if h.get("role") == "user"]
    return (" ".join(prev_user[-1:]) + " " + message).strip()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"chunks": len(retriever.chunks), "retrieval": retriever.mode,
            "llm": MODEL if _ollama else "none (extractive fallback)"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    query = _retrieval_query(req.message.strip(), req.history)
    hits = retriever.search(query, k=TOP_K)
    if not hits:
        return {"answer": "I couldn't find anything relevant in the LAMP site corpus "
                          "for that question. Try rephrasing with LAMP-related terms.",
                "sources": [], "mode": retriever.mode}

    sources = [{"n": i + 1, "title": h["title"], "url": h["url"],
                "heading": h.get("heading", "")} for i, h in enumerate(hits)]

    if _ollama is None:
        top = hits[:3]
        ans = ["**Ollama not available — showing the most relevant passages instead "
               "of a synthesized answer.**", ""]
        for i, h in enumerate(top, 1):
            ans.append(f"[{i}] **{h['title']}**: {h['text'][:600]}…")
        return {"answer": "\n\n".join(ans), "sources": sources[:3], "mode": retriever.mode}

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": m["role"], "content": m["content"]}
             for m in req.history[-6:] if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content":
                 f"Source passages:\n\n{_passages_block(hits)}\n\n"
                 f"Question: {req.message.strip()}"})
    try:
        resp = _ollama.chat(model=MODEL, messages=msgs)
        answer = resp.message.content
    except Exception as e:
        return JSONResponse(status_code=502,
                            content={"error": f"LLM call failed: {e}"})

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    used = [s for s in sources if s["n"] in cited] or sources
    return {"answer": answer, "sources": used, "mode": retriever.mode}
