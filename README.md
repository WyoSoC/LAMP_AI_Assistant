# LAMP Website RAG Chatbot

A retrieval-augmented chatbot over the University of Wyoming **Learning Actively
Mentoring Program** website (https://www.uwyo.edu/science-initiative/lamp/). It answers
open-ended questions, synthesizes across pages, and cites every claim with a link back
to the source page. Answers are generated locally using **Ollama + Gemma 4** — no API
key or internet connection required at inference time.

## Website structure → corpus design

The LAMP site is a well-bounded subtree under `/science-initiative/lamp/` with five
content clusters, which map directly onto the corpus file naming:

| Cluster | Files | Content |
|---|---|---|
| `lamp__*` | 17 | Program overview, ELC, LA program, publications, outcomes, rubrics, people |
| `fellows__*` | 9 | Fellows Program + Summer Institutes 2016–2025, posters |
| `spectrum__*` | 38 | 31 active-learning modality pages + 7 faculty essays |
| `spotlight__*` | 17 | Monthly faculty spotlights 2017–2020 |
| `doc__*` | 5 | Strategic plans (2022, 2026), 2021 assessment, teaching-philosophy rubric, growth report (extracted from PDF) |

Each corpus file is cleaned markdown (university boilerplate removed, in-content
hyperlinks preserved) with YAML frontmatter: `url`, `title`, `section`, `fetched`.
**The `url` field is the citation anchor** — it follows every chunk through the
pipeline so answers can always link back.

## Architecture

```
scripts/crawl.py      re-crawl the live site -> data/corpus/*.md   (run on your machine)
app/ingest.py         corpus -> heading-aligned ~1400-char chunks (200 overlap)
                      -> data/chunks.jsonl  +  ChromaDB vector index (data/chroma/)
app/retrieval.py      hybrid search: BM25 (pure Python, always works) + vector
                      cosine search, merged with Reciprocal Rank Fusion;
                      max 2 chunks/document for source diversity
app/main.py           FastAPI: /api/chat retrieves top-6 passages, Gemma 4 (via Ollama)
                      synthesizes an answer with inline [n] citations
app/static/index.html chat UI; [n] markers render as links, sources listed per answer
```

Embeddings are local (all-MiniLM-L6-v2 via ChromaDB's built-in ONNX runtime — small
download on first ingest, no API cost). The LLM is also fully local via **Ollama** running
**Gemma 4** — no API keys or external calls. If the vector index is unavailable the app
automatically falls back to BM25-only retrieval. If Ollama is unreachable the chat
endpoint returns the top passages verbatim instead of a synthesized answer, so the
system degrades gracefully at every layer.

## Setup

**Prerequisites:** [Ollama](https://ollama.com) installed and running locally.

```bash
# Pull the model (one-time)
ollama pull gemma4

# Install Python dependencies and start the app
cd LAMP_AI_Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.ingest        # chunk + build vector index (corpus already included)
uvicorn app.main:app --port 8000
# open http://localhost:8000
```

## Refreshing the corpus

```bash
python scripts/crawl.py --delay 1.0   # polite re-crawl of the LAMP subtree
python -m app.ingest                  # rebuild chunks + index
```

## API

```
GET  /api/status   -> {"chunks": 459, "retrieval": "...", "llm": "..."}
POST /api/chat     {"message": "...", "history": [{"role","content"}, ...]}
                   -> {"answer": "... [1] ...", "sources": [{"n","title","url","heading"}]}
```

Config via `.env`: `OLLAMA_MODEL` (default `gemma4`), `OLLAMA_BASE_URL` (default
`http://localhost:11434`), `RAG_TOP_K` (default 6).

## Notes

- Corpus snapshot: 84 documents / 459 chunks, fetched 2026-06-10.
- `critical_thinking_pdf_resource.pdf` is image-only (no text layer); add OCR
  (e.g. `pytesseract`) to `scripts/crawl.py` if you need it.
- The grounding prompt instructs Gemma 4 to answer only from retrieved passages and
  to say so when the corpus doesn't contain an answer.
