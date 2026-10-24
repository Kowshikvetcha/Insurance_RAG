"""Per-session hybrid index (dense vectors + BM25) over search units, with parent expansion,
and the in-memory session registry with TTL eviction. Nothing is written to disk."""

from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Parent
from app.ingestion.enrich import Definition

RRF_K = 60
CATEGORY_BOOST = 1.2
EXTRA_HIT_WEIGHT = 0.15

_STOP = set("a an the of to in on for and or is are be by with as at from this that it its any if what which how does do i my me can".split())
_TOKEN_RE = re.compile(r"[a-z0-9₹%]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP:
            continue
        tokens.append(tok)
        if "-" in tok:  # "co-payment" also matches "co payment" / "copayment"
            parts = tok.split("-")
            tokens.extend(p for p in parts if p and p not in _STOP)
            tokens.append("".join(parts))
    return tokens


@dataclass
class Hit:
    parent: Parent
    score: float
    best_unit: Document
    # Retrieval diagnostics for the parent's units (observability / evaluation):
    dense_rank: int | None = None  # best rank of any of its units in the dense list (0 = top)
    dense_sim: float | None = None  # best cosine similarity of any of its units to the query
    bm25_rank: int | None = None  # best rank of any of its units in the BM25 list


class HybridIndex:
    def __init__(self, units: list[Document], parents: dict[str, Parent], embeddings: Embeddings):
        self.units = units
        self.parents = parents
        self.embeddings = embeddings
        self.matrix: np.ndarray | None = None
        self.bm25 = BM25Okapi([tokenize(u.page_content) for u in units]) if units else None

    def embed(self, batch_size: int, progress: Callable[[int, int], None] | None = None, workers: int = 4) -> None:
        texts = [u.page_content for u in self.units]
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        done = 0
        results: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]

        def run(i: int) -> int:
            results[i] = self.embeddings.embed_documents(batches[i])
            return len(batches[i])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n in pool.map(run, range(len(batches))):
                done += n
                if progress:
                    progress(done, len(texts))
        vectors = np.array([v for batch in results for v in batch], dtype=np.float32)
        if len(vectors):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        self.matrix = vectors

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k unit indices with their cosine similarity to the query."""
        if self.matrix is None or not len(self.matrix):
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        sims = self.matrix @ q
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]

    def _sparse(self, query: str, k: int) -> list[int]:
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)[:k]
        return [int(i) for i in order if scores[i] > 0]

    def search(self, query: str, k: int, boost_categories: set[str] | None = None) -> list[Hit]:
        """Reciprocal-rank fusion of dense + BM25 results, boosted by category match, then
        collapsed to parents (a parent's score is its best unit plus a little for extra hits)."""
        dense = self._dense(query, k)
        sparse = self._sparse(query, k)
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense)}
        dense_sim = dict(dense)
        bm25_rank = {idx: r for r, idx in enumerate(sparse)}
        fused: dict[int, float] = {}
        for ranking in ([idx for idx, _ in dense], sparse):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        if boost_categories:
            for idx in fused:
                if boost_categories.intersection(self.units[idx].metadata.get("categories", [])):
                    fused[idx] *= CATEGORY_BOOST
        by_parent: dict[str, list[tuple[float, int]]] = {}
        for idx, score in fused.items():
            by_parent.setdefault(self.units[idx].metadata["parent_id"], []).append((score, idx))
        hits = []
        for pid, scored in by_parent.items():
            scored.sort(reverse=True)
            # Only the next two hits count, so a big table with many rows can't win on volume alone.
            score = scored[0][0] + EXTRA_HIT_WEIGHT * sum(s for s, _ in scored[1:3])
            idxs = [i for _, i in scored]
            d_ranks = [dense_rank[i] for i in idxs if i in dense_rank]
            b_ranks = [bm25_rank[i] for i in idxs if i in bm25_rank]
            sims = [dense_sim[i] for i in idxs if i in dense_sim]
            hits.append(Hit(self.parents[pid], score, self.units[scored[0][1]],
                            dense_rank=min(d_ranks) if d_ranks else None,
                            dense_sim=max(sims) if sims else None,
                            bm25_rank=min(b_ranks) if b_ranks else None))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits


@dataclass
class Session:
    id: str
    filename: str
    index: HybridIndex
    glossary: dict[str, Definition]
    report: dict
    figures_png: dict[str, bytes] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_minutes: int):
        self.ttl = ttl_minutes * 60
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def add(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s.last_access = time.time()
            return s

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def evict_expired(self) -> int:
        cutoff = time.time() - self.ttl
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_access < cutoff]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


def new_session_id() -> str:
    return uuid.uuid4().hex
