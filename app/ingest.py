"""Ingestion pipeline: parse corpus markdown -> chunks -> indexes.

Reads data/corpus/*.md (each with YAML frontmatter carrying the source URL),
splits documents into overlapping chunks aligned to headings, writes
data/chunks.jsonl, and (optionally) builds a persistent ChromaDB vector index
using a local ONNX MiniLM embedding model. A BM25 lexical index is always
available at query time (built in-memory from chunks.jsonl by retrieval.py),
so the system works even without the embedding model.

Usage:
    python -m app.ingest            # chunk + try to build vector index
    python -m app.ingest --no-vector  # chunk only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "lamp"

CHUNK_TARGET = 1400   # chars per chunk (~350 tokens)
CHUNK_OVERLAP = 200   # chars of overlap between adjacent chunks


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a simple YAML frontmatter block (key: value lines)."""
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:]
    return meta, text.strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, text) sections on #/## headings."""
    sections: list[tuple[str, str]] = []
    current_head, buf = "", []
    for line in body.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                sections.append((current_head, "\n".join(buf).strip()))
            current_head, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((current_head, "\n".join(buf).strip()))
    return sections or [("", body)]


def pack_chunks(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pack sections into chunks of ~CHUNK_TARGET chars with overlap.

    Returns list of (heading, chunk_text). Long sections are split on
    paragraph boundaries; short adjacent sections are merged.
    """
    chunks: list[tuple[str, str]] = []
    cur_head, cur = "", ""

    def flush():
        nonlocal cur, cur_head
        if cur.strip():
            chunks.append((cur_head, cur.strip()))
        cur, cur_head = "", ""

    for head, text in sections:
        paras = re.split(r"\n\s*\n", text)
        for para in paras:
            para = para.strip()
            if not para:
                continue
            candidate = (cur + "\n\n" + para).strip() if cur else para
            if len(candidate) > CHUNK_TARGET and cur:
                tail = cur[-CHUNK_OVERLAP:]
                flush()
                cur_head = head
                cur = (tail + "\n\n" + para) if len(para) < CHUNK_TARGET else para
            else:
                if not cur:
                    cur_head = head
                cur = candidate
            # hard-split single paragraphs that exceed 2x target
            while len(cur) > 2 * CHUNK_TARGET:
                cut = cur.rfind(". ", CHUNK_TARGET, 2 * CHUNK_TARGET) + 1
                if cut <= CHUNK_OVERLAP:          # no usable sentence boundary
                    cut = 2 * CHUNK_TARGET
                piece = cur[:cut]
                cur = cur[max(CHUNK_OVERLAP + 1, cut - CHUNK_OVERLAP):]
                chunks.append((cur_head, piece.strip()))
    flush()
    return chunks


def build_chunks() -> list[dict]:
    records = []
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        sys.exit(f"No corpus files found in {CORPUS_DIR}. Run scripts/crawl.py first.")
    for fp in files:
        meta, body = parse_frontmatter(fp.read_text(encoding="utf-8"))
        url = meta.get("url", "")
        title = meta.get("title", fp.stem)
        section = meta.get("section", "")
        for i, (head, text) in enumerate(pack_chunks(split_sections(body))):
            records.append({
                "id": f"{fp.stem}::{i}",
                "doc": fp.stem,
                "url": url,
                "title": title,
                "section": section,
                "heading": head,
                "text": text,
            })
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} chunks from {len(files)} documents -> {CHUNKS_PATH}")
    return records


def build_vector_index(records: list[dict]) -> bool:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.ONNXMiniLM_L6_V2()  # downloads small model on first run
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})
        B = 64
        for i in range(0, len(records), B):
            batch = records[i:i + B]
            col.add(
                ids=[r["id"] for r in batch],
                documents=[f"{r['title']} — {r['heading']}\n{r['text']}" for r in batch],
                metadatas=[{k: r[k] for k in ("url", "title", "section", "heading")} for r in batch],
            )
            print(f"  embedded {min(i + B, len(records))}/{len(records)}", end="\r")
        print(f"\nVector index built: {col.count()} chunks -> {CHROMA_DIR}")
        return True
    except Exception as e:
        print(f"WARNING: vector index not built ({type(e).__name__}: {e}).\n"
              "The app will fall back to BM25 lexical retrieval, which works offline.")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vector", action="store_true", help="skip ChromaDB vector index")
    args = ap.parse_args()
    recs = build_chunks()
    if not args.no_vector:
        build_vector_index(recs)
