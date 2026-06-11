"""Hybrid retrieval: BM25 (always available) + ChromaDB vectors (if built).

Results from both retrievers are merged with Reciprocal Rank Fusion (RRF).
Every returned passage keeps its source URL and title so answers can cite.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "lamp"

_STOP = set("""a an and are as at be but by for from has have if in into is it its of on
or that the their there these this to was were will with what which who how why your
you we our us i me my they them he she his her do does did not no can could would should
""".split())

_token_re = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _token_re.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.doc_len[idx]
        s = 0.0
        for t in query:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = _tokens(query)
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored = [x for x in scored if x[1] > 0]
        return sorted(scored, key=lambda x: -x[1])[:k]


class Retriever:
    def __init__(self):
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"{CHUNKS_PATH} not found — run `python -m app.ingest` first.")
        self.chunks = [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8")]
        self.by_id = {c["id"]: c for c in self.chunks}
        # weight title/heading terms 3x so page topics outrank incidental mentions
        self.bm25 = BM25([_tokens(3 * (c["title"] + " " + c["heading"] + " ") + c["text"])
                          for c in self.chunks])
        self.col = None
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.col = client.get_collection(
                COLLECTION,
                embedding_function=embedding_functions.ONNXMiniLM_L6_V2())
            _ = self.col.count()
        except Exception:
            self.col = None  # vector index unavailable -> BM25 only

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + vectors)" if self.col else "BM25 only"

    def search(self, query: str, k: int = 6) -> list[dict]:
        """RRF-merge BM25 and vector results; return top-k chunk dicts with scores."""
        K = 60  # RRF constant
        fused: dict[str, float] = defaultdict(float)

        for rank, (idx, _s) in enumerate(self.bm25.top(query, k=2 * k + 4)):
            fused[self.chunks[idx]["id"]] += 1.0 / (K + rank + 1)

        if self.col:
            try:
                res = self.col.query(query_texts=[query], n_results=2 * k + 4)
                for rank, cid in enumerate(res["ids"][0]):
                    fused[cid] += 1.0 / (K + rank + 1)
            except Exception:
                pass

        ranked = sorted(fused.items(), key=lambda x: -x[1])
        out, per_doc = [], defaultdict(int)
        for cid, score in ranked:
            c = self.by_id[cid]
            if per_doc[c["doc"]] >= 2:   # diversity: max 2 chunks per document
                continue
            per_doc[c["doc"]] += 1
            c = dict(c)
            c["score"] = round(score, 5)
            out.append(c)
            if len(out) >= k:
                break
        return out
