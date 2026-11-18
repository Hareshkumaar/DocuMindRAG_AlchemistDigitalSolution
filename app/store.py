"""Storage layer: ChromaDB for vectors, SQLite for documents and chat history."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SearchHit:
    doc: Document
    score: float        # relevance to the question, 0..1
    vector: np.ndarray  # chunk embedding, used to measure how similar two hits are

    @property
    def key(self) -> str:
        return f"{self.doc.metadata['doc_id']}-{self.doc.metadata['chunk_index']}"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around Chroma. Every chunk carries session_id and doc_id
    metadata so guests never see each other's documents."""

    def __init__(
        self, embeddings: Embeddings, persist_dir: Path, batch_size: int = 16, query_prefix: str = ""
    ):
        self._batch_size = batch_size
        self._query_prefix = query_prefix  # e.g. the bge retrieval instruction; queries only
        self._db = Chroma(
            collection_name="documind",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Document]) -> None:
        # Embed in small batches to stay under API rate limits.
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            ids = [f"{c.metadata['doc_id']}-{c.metadata['chunk_index']}" for c in batch]
            self._db.add_documents(batch, ids=ids)

    def search_many(
        self, queries: list[str], session_id: str, doc_ids: list[str] | None, k: int
    ) -> list[SearchHit]:
        """Search all queries in one call. Returns every candidate with its
        relevance (0..1) and its embedding (needed for MMR diversity)."""
        conditions: list[dict] = [{"session_id": session_id}]
        if doc_ids:
            conditions.append({"doc_id": {"$in": doc_ids}})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        query_vectors = self._db.embeddings.embed_documents(
            [f"{self._query_prefix}{q}" for q in queries]
        )
        result = self._db._collection.query(
            query_embeddings=query_vectors,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        hits: list[SearchHit] = []
        for docs, metas, dists, vectors in zip(
            result["documents"], result["metadatas"], result["distances"], result["embeddings"]
        ):
            for text, meta, distance, vector in zip(docs, metas, dists, vectors):
                hits.append(SearchHit(
                    doc=Document(page_content=text, metadata=meta),
                    score=1.0 - distance,  # cosine distance -> similarity
                    vector=np.asarray(vector, dtype=float),
                ))
        return hits

    def get_chunks(self, doc_id: str) -> list[Document]:
        data = self._db.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        return sorted(docs, key=lambda d: d.metadata["chunk_index"])

    def delete_document(self, doc_id: str) -> None:
        ids = self._db.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._db.delete(ids=ids)


# ---------------------------------------------------------------------------
# SQLite database
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    file_path   TEXT,
    pages       INTEGER DEFAULT 0,
    chunks      INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'processing',
    error       TEXT,
    summary     TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     TEXT DEFAULT '[]',
    attachments TEXT DEFAULT '[]',
    found       INTEGER DEFAULT 1,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_session ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_msgs_session ON messages(session_id);
"""


class Database:
    def __init__(self, path: Path):
        self._path = path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            if "file_path" not in columns:  # databases created before file links existed
                conn.execute("ALTER TABLE documents ADD COLUMN file_path TEXT")
            msg_columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
            if "attachments" not in msg_columns:  # databases created before message attachments existed
                conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT DEFAULT '[]'")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- documents --------------------------------------------------------

    def add_document(
        self, session_id: str, name: str, file_type: str, file_hash: str, file_path: Path
    ) -> dict:
        doc = {
            "id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "name": name,
            "file_type": file_type,
            "file_hash": file_hash,
            "file_path": str(file_path),
            "created_at": _now(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, session_id, name, file_type, file_hash, file_path, created_at) "
                "VALUES (:id, :session_id, :name, :file_type, :file_hash, :file_path, :created_at)",
                doc,
            )
        return self.get_document(doc["id"], session_id)

    def update_document(self, doc_id: str, **fields) -> None:
        columns = ", ".join(f"{key} = :{key}" for key in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE documents SET {columns} WHERE id = :id", {**fields, "id": doc_id})

    def get_document(self, doc_id: str, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ? AND session_id = ?", (doc_id, session_id)
            ).fetchone()
        return dict(row) if row else None

    def find_by_hash(self, session_id: str, file_hash: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE session_id = ? AND file_hash = ? AND status = 'ready'",
                (session_id, file_hash),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ? AND session_id = ?", (doc_id, session_id))

    # --- messages ---------------------------------------------------------

    def add_message(
        self, session_id: str, role: str, content: str,
        sources: list[dict] | None = None, found: bool = True,
        attachments: list[dict] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, sources, attachments, found, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, role, content,
                    json.dumps(sources or []),
                    json.dumps(attachments or []),
                    int(found), _now(),
                ),
            )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC"
        params: tuple = (session_id,)
        if limit:
            query += " LIMIT ?"
            params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        messages = [dict(r) for r in reversed(rows)]
        for msg in messages:
            msg["sources"] = json.loads(msg["sources"])
            msg["attachments"] = json.loads(msg.get("attachments") or "[]")
            msg["found"] = bool(msg["found"])
        return messages

    def get_last_attachments(self, session_id: str) -> list[dict]:
        """Return the most recent non-empty attachment list from this session's user messages."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT attachments FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC",
                (session_id,),
            ).fetchall()
        for row in rows:
            atts = json.loads(row["attachments"] or "[]")
            if atts:
                return atts
        return []

    def clear_messages(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
