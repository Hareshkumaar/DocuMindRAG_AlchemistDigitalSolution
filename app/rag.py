"""RAG pipeline: load -> chunk -> index -> multi-query retrieve -> stream answer.

Built with LangChain runnables (prompt | llm | parser), like the reference
YouTube-transcript script, plus multi-query retrieval, streaming, a strict
relevance guardrail and sources.
"""

import asyncio
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    ANSWER_PROMPT,
    CHAT_PROMPT,
    CHAT_ROUTE,
    MULTI_QUERY_PROMPT,
    NO_DOCUMENTS_MESSAGE,
    NOT_FOUND_MESSAGE,
    REWRITE_PREFIX,
    SUMMARY_MAP_PROMPT,
    SUMMARY_REDUCE_PROMPT,
    SUMMARY_ROUTE,
    Settings,
)
from app.store import Database, SearchHit, VectorStore

TXT_SECTION_CHARS = 3000  # a TXT "section" plays the role of a PDF page
OCR_MIN_CHARS = 50        # pages with less text than this are treated as scanned images
HISTORY_CHARS = 400       # each past message is shortened to this in prompts
# When searching all documents, keep only documents whose best match is this close
# to the overall best match, so a clearly unrelated file is not mixed into the answer.
DOC_FOCUS_MARGIN = 0.08

# The hidden "sources used" line the answer model writes last. Models vary the brackets:
# [[sources: 1, 2]]   [sources: 1]   【sources: 1】
SOURCE_MARKER = re.compile(
    r"(?:\[\[|\[|【)\s*sources?\s*:\s*([\d,\s]*)(?:\]\]|\]|】)", re.IGNORECASE
)
MARKER_STARTS = ("[[", "[source", "【")
# Page citations inside summaries: (p.4)  (p.12–13)  (p.4, p.6)  (§2)
CITE_GROUP = re.compile(r"\(((?:p\.|§)\s*\d+(?:\s*(?:[-–,]|and)\s*(?:p\.|§)?\s*\d+)*)\)")

# "summarize" and its common misspellings (summrize, sumarize, summerize), "summary", "overview"
SUMMARY_INTENT = re.compile(
    r"\b(sum+[aeiy]?r[iy]?[sz]?e|sum+[ae]ry|overview)\b", re.IGNORECASE
)
# Words that do not make a summary request "about something specific".
SUMMARY_FILLER = {
    "summarize", "summarise", "summary", "overview", "give", "me", "a", "an", "the", "of",
    "this", "that", "it", "its", "these", "those", "document", "documents", "doc", "docs",
    "file", "files", "pdf", "txt", "please", "short", "brief", "quick", "quickly", "can",
    "could", "you", "write", "make", "provide", "main", "key", "points", "point", "full",
    "whole", "entire", "and", "in", "for", "about", "on", "to", "my", "what", "is", "are",
    "only", "just", "all", "both", "attached", "uploaded", "tell", "show", "describe",
    "explain", "say", "says", "content", "contents", "inside", "contains", "whats", "does", "do",
}
# Generic names for a kind of document. "summarize this quiz" names no topic, so with
# several files it is answered with "Which document should I use?".
DOC_TYPE_WORDS = {
    "quiz", "test", "exam", "assignment", "lab", "report", "paper", "notes", "note", "book",
    "handout", "worksheet", "article", "essay", "slides", "presentation", "form", "letter",
}
ALL_DOCS_PATTERN = re.compile(
    r"\b(all|both|every|each)\b[^.?!]{0,30}\b(documents?|files?|pdfs?|docs?)\b", re.IGNORECASE
)
# Words that are too common in file names to identify one document by themselves.
GENERIC_NAME_WORDS = {"file", "files", "document", "documents", "report", "final", "copy", "draft", "assignment"}
# Words that show the user is talking about a file ("the tips file", "handbook pdf").
FILE_CUE = r"(?:file|doc|docs|document|pdf|txt)"


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_llm(settings: Settings, model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Runs locally (free). The model downloads once on first use."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def parse_queries(text: str) -> list[str]:
    """Turn the LLM's one-query-per-line output into a clean list."""
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines())
    return [line for line in lines if len(line) > 3]


def is_not_found(answer: str) -> bool:
    """True only when the answer is (essentially) just the not-found sentence, so a
    partial answer that mentions a missing detail is still shown with its sources."""
    text = re.sub(r"\s+", " ", answer).strip().lower()
    target = NOT_FOUND_MESSAGE.rstrip(".").lower()
    return text.startswith(target) and len(text) <= len(target) + 40


def strip_source_marker(text: str) -> str:
    return SOURCE_MARKER.sub("", text).rstrip()


def clip(text: str, limit: int = HISTORY_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def cited_pages(summary: str) -> list[tuple[str, int]]:
    """All (unit, page) pairs cited in a summary, including ranges like (p.12–13)."""
    pages: list[tuple[str, int]] = []
    for group in CITE_GROUP.findall(summary):
        unit = "§" if "§" in group else "p."
        numbers = [int(n) for n in re.findall(r"\d+", group)]
        if re.search(r"\d\s*[-–]\s*(?:p\.|§)?\s*\d", group) and len(numbers) == 2:
            first, last = sorted(numbers)
            numbers = list(range(first, min(last, first + 9) + 1))  # cap long ranges
        pages.extend((unit, n) for n in numbers)
    return list(dict.fromkeys(pages))


def has_summary_focus(question: str, docs: list[dict]) -> bool:
    """True for requests like "summarize the outliers part": they mention a topic,
    so they are answered from the matching chunks instead of a whole-file summary."""
    name_words = {
        w for d in docs for w in re.split(r"[_\-\s.]+", d["name"].lower()) if w
    }
    words = re.findall(r"[a-z0-9]+", question.lower())
    topic = [
        w for w in words
        if len(w) > 2
        and w not in SUMMARY_FILLER
        and w not in DOC_TYPE_WORDS
        and w not in name_words
        and not SUMMARY_INTENT.fullmatch(w)  # misspelled "summarize"
    ]
    return len(topic) >= 1


def mmr_select(hits: list[SearchHit], k: int, lambda_mult: float) -> list[SearchHit]:
    """Maximal Marginal Relevance.

    Repeatedly pick the hit with the best balance of:
      relevance to the question        (hit.score)
      minus similarity to hits already picked (avoids near-duplicate chunks)
    lambda_mult = 1.0 -> pure relevance, 0.0 -> pure diversity.
    """
    remaining = sorted(hits, key=lambda h: h.score, reverse=True)
    selected: list[SearchHit] = []
    while remaining and len(selected) < k:
        def mmr_score(hit: SearchHit) -> float:
            redundancy = max((float(hit.vector @ s.vector) for s in selected), default=0.0)
            return lambda_mult * hit.score - (1 - lambda_mult) * redundancy

        pick = max(remaining, key=mmr_score)
        selected.append(pick)
        remaining.remove(pick)
    return selected


# ---------------------------------------------------------------------------
# Loading and chunking
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank lines
    return text.strip()


@lru_cache
def ocr_available() -> bool:
    """True if the Tesseract binary and tessdata are both reachable."""
    try:
        return bool(shutil.which("tesseract")) and bool(pymupdf.get_tessdata())
    except Exception:
        return False


def read_pdf_page(page: pymupdf.Page, ocr_language: str) -> str:
    """Extract text from a PDF page, falling back to OCR on image-only pages."""
    text = page.get_text()
    if len(text.strip()) < OCR_MIN_CHARS and page.get_images() and ocr_available():
        try:
            textpage = page.get_textpage_ocr(dpi=200, full=True, language=ocr_language)
            return page.get_text(textpage=textpage)
        except Exception:
            return text
    return text


def pdf_page_count(path: Path) -> int:
    with pymupdf.open(path) as pdf:
        return pdf.page_count


def load_document(path: Path, ocr_language: str = "eng") -> list[Document]:
    """Return one Document per PDF page (or per TXT section) with its number."""
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            pages = [
                Document(
                    page_content=clean_text(read_pdf_page(page, ocr_language)),
                    metadata={"page": i, "unit": "p."},
                )
                for i, page in enumerate(pdf, start=1)
            ]
    else:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        pages = [
            Document(page_content=text[i : i + TXT_SECTION_CHARS], metadata={"page": n, "unit": "§"})
            for n, i in enumerate(range(0, len(text), TXT_SECTION_CHARS), start=1)
        ]
    return [p for p in pages if p.page_content]


def short_name(filename: str) -> str:
    """'tips-087.pdf' -> 'tips-087' (shortened with '…' after 18 chars)."""
    stem = Path(filename).stem or filename
    return f"{stem[:18]}…" if len(stem) > 18 else stem


def split_documents(
    pages: list[Document], *, doc_id: str, session_id: str, doc_name: str, settings: Settings
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            doc_id=doc_id,
            session_id=session_id,
            doc_name=doc_name,
            short_name=short_name(doc_name),
            chunk_index=index,
        )
    return chunks


# ---------------------------------------------------------------------------
# Streaming helper: hides the trailing "[[sources: 1, 2]]" line from the user
# ---------------------------------------------------------------------------

class SourceMarkerFilter:
    def __init__(self) -> None:
        self._pending = ""
        self._tail = ""
        self._in_marker = False

    def feed(self, text: str) -> str:
        """Return the part of `text` that is safe to show right now."""
        if self._in_marker:
            self._tail += text
            return ""
        self._pending += text
        lowered = self._pending.lower()
        starts = [i for i in (lowered.find(m) for m in MARKER_STARTS) if i != -1]
        if starts:
            start = min(starts)
            visible, self._tail = self._pending[:start], self._pending[start:]
            self._pending, self._in_marker = "", True
            return visible
        hold = self._partial_marker_length(lowered)  # e.g. "[" or "[sour" at the very end
        cut = len(self._pending) - hold
        visible, self._pending = self._pending[:cut], self._pending[cut:]
        return visible

    def flush(self) -> str:
        """Text still held back at the end. A false alarm (e.g. a real "[" in the
        answer that was not the sources line) is released instead of being lost."""
        visible = self._pending
        if self._in_marker and not SOURCE_MARKER.search(self._tail):
            visible += self._tail
            self._tail = ""
        self._pending = ""
        return visible

    def used_ids(self) -> set[int]:
        match = SOURCE_MARKER.search(self._tail)
        if not match:
            return set()
        return {int(n) for n in re.findall(r"\d+", match.group(1))}

    @staticmethod
    def _partial_marker_length(text: str) -> int:
        longest = max(len(m) for m in MARKER_STARTS)
        for size in range(min(len(text), longest), 0, -1):
            if any(m.startswith(text[-size:]) for m in MARKER_STARTS):
                return size
        return 0


# ---------------------------------------------------------------------------
# RAG service
# ---------------------------------------------------------------------------

class RAGService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        llm: BaseChatModel | None = None,
        query_llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.settings = settings
        self.db = db
        self.llm = llm or build_llm(settings, settings.chat_model)
        self.query_llm = query_llm or build_llm(settings, settings.query_model)
        self.store = VectorStore(
            embeddings or build_embeddings(settings),
            settings.chroma_dir,
            settings.embed_batch_size,
            query_prefix=settings.query_instruction,
        )
        parser = StrOutputParser()
        # Non-streaming calls retry automatically (e.g. Groq free-tier rate limits).
        retry = {"stop_after_attempt": 3, "wait_exponential_jitter": True}
        self.answer_chain = ANSWER_PROMPT | self.llm | parser
        self.plan_chain = (
            MULTI_QUERY_PROMPT.partial(
                count=str(settings.multi_query_count),
                chat_route=CHAT_ROUTE,
                summary_route=SUMMARY_ROUTE,
                rewrite_prefix=REWRITE_PREFIX,
            )
            | self.query_llm
            | parser
        ).with_retry(**retry)
        self.chat_chain = CHAT_PROMPT | self.llm | parser
        self.map_chain = (SUMMARY_MAP_PROMPT | self.query_llm | parser).with_retry(**retry)
        self.reduce_chain = (SUMMARY_REDUCE_PROMPT | self.llm | parser).with_retry(**retry)

    # --- ingestion --------------------------------------------------------

    def ingest(self, path: Path, filename: str, session_id: str) -> dict:
        digest = file_hash(path)
        existing = self.db.find_by_hash(session_id, digest)
        if existing and not self._needs_reindex(existing, path):
            path.unlink(missing_ok=True)  # same file uploaded twice: keep the first copy
            return existing
        if existing:  # the old copy missed pages (indexed before OCR): replace it
            self.delete_document(existing["id"], session_id)

        doc = self.db.add_document(
            session_id, filename, path.suffix.lstrip(".").lower(), digest, path
        )
        try:
            pages = load_document(path, self.settings.ocr_language)
            if not pages:
                hint = (
                    ""
                    if ocr_available()
                    else " Install Tesseract OCR to read scanned or image-only PDFs."
                )
                raise ValueError(f"No readable text found.{hint}")
            chunks = split_documents(
                pages, doc_id=doc["id"], session_id=session_id,
                doc_name=filename, settings=self.settings,
            )
            self.store.add(chunks)
            self.db.update_document(doc["id"], pages=len(pages), chunks=len(chunks), status="ready")
        except Exception as exc:  # report the failure on the document card
            self.db.update_document(doc["id"], status="failed", error=str(exc))
        return self.db.get_document(doc["id"], session_id)

    @staticmethod
    def _needs_reindex(existing: dict, path: Path) -> bool:
        """True when a PDF was indexed with fewer pages than it has and OCR can now read them."""
        if existing["file_type"] != "pdf" or not ocr_available():
            return False
        try:
            return pdf_page_count(path) > (existing["pages"] or 0)
        except Exception:
            return False

    def delete_document(self, doc_id: str, session_id: str) -> bool:
        doc = self.db.get_document(doc_id, session_id)
        if not doc:
            return False
        self.store.delete_document(doc_id)
        self.db.delete_document(doc_id, session_id)
        if doc["file_path"]:
            Path(doc["file_path"]).unlink(missing_ok=True)
        return True

    def delete_session(self, session_id: str) -> None:
        """Remove everything a guest created: documents, chunks, files and chat history."""
        for doc in self.db.list_documents(session_id):
            self.delete_document(doc["id"], session_id)
        self.db.clear_messages(session_id)
        shutil.rmtree(self.settings.upload_dir / session_id, ignore_errors=True)

    # --- question answering ----------------------------------------------

    async def stream_answer(
        self,
        question: str,
        session_id: str,
        attachments: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """Yield events: token* -> sources -> done."""
        started = time.perf_counter()
        history = self._format_history(session_id)
        last_attachments = self.db.get_last_attachments(session_id)
        last_answer_doc_ids = self._last_answer_doc_ids(session_id)
        self.db.add_message(session_id, "user", question, attachments=attachments)

        ready_docs = [d for d in self.db.list_documents(session_id) if d["status"] == "ready"]
        route, payload = await self._plan(question, history, [d["name"] for d in ready_docs])

        if route == CHAT_ROUTE:  # greeting / thanks / small talk: natural reply, no search
            async for event in self._stream_chat(question, history, ready_docs, session_id, started):
                yield event
            return
        if not ready_docs:  # a real question, but nothing to search yet
            async for event in self._stream_fixed(NO_DOCUMENTS_MESSAGE, session_id, started):
                yield event
            return

        effective_question = payload[0] if route == REWRITE_PREFIX else question
        scope_doc_ids, scope_docs = self._resolve_scope(
            ready_docs, attachments, doc_ids, last_attachments, last_answer_doc_ids, effective_question
        )
        # The router's SUMMARY route, or "summarize/summary/overview" written in the request
        # (also when misspelled), even if the router sent it as a search.
        wants_summary = route == SUMMARY_ROUTE or bool(SUMMARY_INTENT.search(effective_question))
        if not scope_docs:  # no file attached, named or active
            # No file is in scope, so file-name words count as topic words here
            # ("summarize the leave policy" searches; "summarize this quiz" asks).
            if wants_summary and not has_summary_focus(effective_question, []):
                # a plain "summarize" with several files: ask which one
                reply = self._which_document_message(ready_docs)
                async for event in self._stream_fixed(reply, session_id, started, found=True):
                    yield event
                return
            scope_docs = ready_docs  # a question: search every file, best-matching files win
        if wants_summary and not has_summary_focus(effective_question, scope_docs):
            async for event in self._stream_summary(scope_docs, session_id, started):
                yield event
            return

        queries = payload if route == "RAG" else [effective_question]
        hits = await asyncio.to_thread(self._retrieve, queries, session_id, scope_doc_ids)

        if not hits:  # guardrail 1: nothing relevant, skip the LLM entirely
            answer, sources = NOT_FOUND_MESSAGE, []
            yield {"type": "token", "text": answer}
        else:
            marker = SourceMarkerFilter()
            parts: list[str] = []
            async for token in self.answer_chain.astream({
                "history": history or "(none)",
                "context": self._format_context(hits),
                "question": effective_question,
            }):
                visible = marker.feed(token)
                if visible:
                    parts.append(visible)
                    yield {"type": "token", "text": visible}
            if rest := marker.flush():
                parts.append(rest)
                yield {"type": "token", "text": rest}

            answer = strip_source_marker("".join(parts)).strip()
            sources = self._build_sources(hits, marker.used_ids())

        found = not is_not_found(answer) and bool(sources)  # guardrail 2
        if not found:
            sources = []
        self.db.add_message(session_id, "assistant", answer, sources, found)

        yield {"type": "sources", "found": found, "sources": sources, "tag": self._source_tag(sources)}
        yield {
            "type": "done",
            "queries": queries,
            "elapsed": round(time.perf_counter() - started, 2),
        }

    @staticmethod
    def _match_named_docs(text: str, ready_docs: list[dict]) -> list[dict]:
        """Documents the user clearly names:
        - the file name or its stem: "tips-087.pdf", "tips-087", "employee handbook"
        - a word only one file name contains, next to a file word: "the tips file",
          "handbook pdf", "document called handbook"
        Ordinary words alone ("how many leave days", "any tips?") never select a file,
        and numbers shared by several names ("087") never do either."""
        lowered = text.lower()
        spaced = re.sub(r"[_\-.]+", " ", lowered)
        words = {
            d["id"]: {
                w for w in re.split(r"[_\-\s.]+", Path(d["name"]).stem.lower())
                if len(w) >= 4 and not w.isdigit() and w not in GENERIC_NAME_WORDS
            }
            for d in ready_docs
        }
        counts = Counter(w for doc_words in words.values() for w in doc_words)
        matched: list[dict] = []
        for doc in ready_docs:
            name = doc["name"].lower()
            stem = Path(name).stem
            stem_spaced = re.sub(r"[_\-.]+", " ", stem).strip()
            unique = [w for w in words[doc["id"]] if counts[w] == 1]
            if (
                name in lowered
                or (len(stem) >= 3 and re.search(rf"(?<![\w-]){re.escape(stem)}(?![\w-])", lowered))
                or (" " in stem_spaced and re.search(rf"\b{re.escape(stem_spaced)}\b", spaced))
                or any(
                    re.search(rf"\b{re.escape(w)}\s+{FILE_CUE}\b", spaced)
                    or re.search(rf"\b{FILE_CUE}\s+(?:called\s+|named\s+)?{re.escape(w)}\b", spaced)
                    for w in unique
                )
            ):
                matched.append(doc)
        return matched

    @classmethod
    def _resolve_scope(
        cls,
        ready_docs: list[dict],
        attachments: list[dict] | None,
        doc_ids: list[str] | None,
        last_attachments: list[dict],
        last_answer_doc_ids: list[str],
        question: str,
    ) -> tuple[list[str] | None, list[dict]]:
        """Pick which documents to search/summarize for this turn, in priority order:
        1. files attached to this message
        2. files named in the question ("summarize haresh-087")
        3. explicit doc_ids from the client
        4. "all/both documents" -> everything
        5. the files the previous answer came from (follow-up questions)
        6. the files attached most recently
        7. the only document, if there is just one
        Otherwise return no scope: questions then search every file (best-matching files
        win) and summary requests ask the user which file they mean."""
        by_id = {d["id"]: d for d in ready_docs}

        def scoped(ids: list[str]) -> tuple[list[str], list[dict]]:
            return ids, [by_id[i] for i in ids]

        if attachments and (ids := [a["id"] for a in attachments if a["id"] in by_id]):
            return scoped(ids)
        if named := cls._match_named_docs(question, ready_docs):
            return scoped([d["id"] for d in named])
        if doc_ids and (ids := [i for i in doc_ids if i in by_id]):
            return scoped(ids)
        if ALL_DOCS_PATTERN.search(question):
            return scoped([d["id"] for d in ready_docs])
        if last_answer_doc_ids and (ids := [i for i in last_answer_doc_ids if i in by_id]):
            return scoped(ids)
        if last_attachments and (ids := [a["id"] for a in last_attachments if a["id"] in by_id]):
            return scoped(ids)
        if len(ready_docs) == 1:
            return scoped([ready_docs[0]["id"]])
        return None, []

    async def _plan(
        self, question: str, history: str, doc_names: list[str]
    ) -> tuple[str, list[str]]:
        """One fast LLM call decides the route: CHAT, SUMMARY, REWRITE, or RAG queries."""
        try:
            output = (await self.plan_chain.ainvoke({
                "documents": ", ".join(doc_names) or "(none uploaded yet)",
                "history": history or "(none)",
                "question": question,
            })).strip()
        except Exception:
            output = ""  # fall back to document search with the original question

        # A route is the WHOLE first line ("CHAT", "SUMMARY", "REWRITE: ..."), so a search
        # query that merely starts with those letters ("ChatGPT policy") is not a route.
        first_line = output.splitlines()[0].strip() if output else ""
        head = first_line.strip(" .\"'`*").upper()
        if head == CHAT_ROUTE:
            return CHAT_ROUTE, []
        if head == SUMMARY_ROUTE:
            return SUMMARY_ROUTE, []
        if head.startswith(REWRITE_PREFIX):
            rewritten = first_line.split(":", 1)[1].strip(" `\"'*")
            return REWRITE_PREFIX, [rewritten or question]

        queries = [question]
        for query in parse_queries(output)[: self.settings.multi_query_count]:
            if query.lower() not in (q.lower() for q in queries):
                queries.append(query)
        return "RAG", queries

    async def _stream_summary(
        self, docs: list[dict], session_id: str, started: float
    ) -> AsyncIterator[dict]:
        """Summarize each scoped document and stream the Markdown with cited page sources."""
        sections: list[str] = []
        sources: list[dict] = []
        seen_pages: set[tuple[str, int]] = set()

        for doc in docs:
            summary = await asyncio.to_thread(self.summarize, doc["id"], session_id)
            if not summary:
                continue
            sections.append(f"### {doc['name']}\n\n{summary}" if len(docs) > 1 else summary)

            unit = "p." if doc["file_type"] == "pdf" else "§"
            short = short_name(doc["name"])
            for u, page in cited_pages(summary) or [(unit, 1)]:
                key = (doc["id"], page)
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                sources.append({
                    "doc_id": doc["id"],
                    "file": doc["name"],
                    "short": short,
                    "unit": u,
                    "page": page,
                    "label": f"{short} {u}{page}",
                    "snippet": "",
                    "score": 1.0,
                })

        full_text = "\n\n".join(sections) or NOT_FOUND_MESSAGE
        found = bool(sections)
        if not found:
            sources = []

        for word in re.findall(r"\S+\s*", full_text):
            yield {"type": "token", "text": word}
            await asyncio.sleep(0.01)

        self.db.add_message(session_id, "assistant", full_text, sources, found)
        yield {"type": "sources", "found": found, "sources": sources, "tag": self._source_tag(sources)}
        yield {"type": "done", "queries": [], "elapsed": round(time.perf_counter() - started, 2)}

    def _last_answer_doc_ids(self, session_id: str) -> list[str]:
        """Documents cited by the most recent answer that had sources."""
        for message in reversed(self.db.get_messages(session_id, limit=self.settings.history_turns)):
            if message["role"] == "assistant" and message["sources"]:
                return list(dict.fromkeys(s["doc_id"] for s in message["sources"] if s.get("doc_id")))
        return []

    @staticmethod
    def _which_document_message(ready_docs: list[dict]) -> str:
        names = "\n".join(f"- {d['name']}" for d in ready_docs)
        return (
            f"Which document should I use? You have:\n{names}\n\n"
            "Reply with its name, attach it with the paperclip, or say \"all documents\"."
        )

    async def _stream_fixed(
        self, reply: str, session_id: str, started: float, found: bool = False
    ) -> AsyncIterator[dict]:
        """Stream a prepared message word by word. found=False shows it as an amber notice."""
        for word in re.findall(r"\S+\s*", reply):
            yield {"type": "token", "text": word}
            await asyncio.sleep(0.015)
        self.db.add_message(session_id, "assistant", reply, [], found)
        yield {"type": "sources", "found": found, "sources": [], "tag": ""}
        yield {"type": "done", "queries": [], "elapsed": round(time.perf_counter() - started, 2)}

    async def _stream_chat(
        self, question: str, history: str, ready_docs: list[dict], session_id: str, started: float
    ) -> AsyncIterator[dict]:
        if ready_docs:
            documents = ", ".join(d["name"] for d in ready_docs)
        else:
            documents = ("none uploaded yet. Briefly tell the user to attach a PDF or TXT file "
                         "with the paperclip button so you can answer questions about it")
        parts: list[str] = []
        async for token in self.chat_chain.astream(
            {"documents": documents, "history": history or "(none)", "question": question}
        ):
            parts.append(token)
            yield {"type": "token", "text": token}
        self.db.add_message(session_id, "assistant", "".join(parts).strip(), [], True)
        yield {"type": "sources", "found": True, "sources": [], "tag": ""}
        yield {"type": "done", "queries": [], "elapsed": round(time.perf_counter() - started, 2)}

    def _retrieve(
        self, queries: list[str], session_id: str, doc_ids: list[str] | None
    ) -> list[tuple[Document, float]]:
        """1. Fetch candidates for every query.
        2. Merge duplicates, keeping each chunk's best score.
        3. Drop weak matches (guardrail).
        4. MMR: pick chunks that are relevant AND different from each other."""
        best: dict[str, SearchHit] = {}
        for hit in self.store.search_many(queries, session_id, doc_ids, k=self.settings.fetch_k):
            if hit.key not in best or hit.score > best[hit.key].score:
                best[hit.key] = hit

        if self.settings.debug_retrieval:  # helps choose SCORE_THRESHOLD for your documents
            top = sorted(best.values(), key=lambda h: h.score, reverse=True)[:8]
            print(f"[retrieval] {queries[0]!r}")
            for hit in top:
                meta = hit.doc.metadata
                mark = "keep" if hit.score >= self.settings.score_threshold else "drop"
                print(f"   {hit.score:.3f} {mark}  {meta['doc_name']} {meta['unit']}{meta['page']}")

        relevant = [hit for hit in best.values() if hit.score >= self.settings.score_threshold]
        if doc_ids is None:  # searching every file: keep only the best-matching file(s)
            relevant = self._focus_on_best_documents(relevant)
        selected = mmr_select(relevant, k=self.settings.max_context_chunks,
                              lambda_mult=self.settings.mmr_lambda)
        return [(hit.doc, hit.score) for hit in selected]

    @staticmethod
    def _focus_on_best_documents(hits: list[SearchHit]) -> list[SearchHit]:
        """Drop hits from documents whose best score is clearly below the top document's,
        so an unrelated file that matched a few words is not mixed into the answer."""
        if not hits:
            return hits
        best_by_doc: dict[str, float] = {}
        for hit in hits:
            doc_id = hit.doc.metadata["doc_id"]
            best_by_doc[doc_id] = max(best_by_doc.get(doc_id, 0.0), hit.score)
        cutoff = max(best_by_doc.values()) - DOC_FOCUS_MARGIN
        return [hit for hit in hits if best_by_doc[hit.doc.metadata["doc_id"]] >= cutoff]

    def _format_history(self, session_id: str) -> str:
        """Recent conversation for the prompts. Each message is shortened so long
        summaries do not crowd out the documents or use up the token limit."""
        messages = self.db.get_messages(session_id, limit=self.settings.history_turns)
        return "\n".join(f"{m['role'].title()}: {clip(m['content'])}" for m in messages)

    @staticmethod
    def _format_context(hits: list[tuple[Document, float]]) -> str:
        return "\n\n".join(
            f"[{i}] ({doc.metadata['doc_name']}, {doc.metadata['unit']}{doc.metadata['page']})\n"
            f"{doc.page_content}"
            for i, (doc, _) in enumerate(hits, start=1)
        )

    @staticmethod
    def _build_sources(hits: list[tuple[Document, float]], used: set[int]) -> list[dict]:
        """Keep only excerpts the LLM said it used (all hits if it said nothing),
        merged so each document page appears once."""
        chosen = [hit for i, hit in enumerate(hits, start=1) if not used or i in used]
        sources: dict[tuple, dict] = {}
        for doc, score in chosen:
            meta = doc.metadata
            key = (meta["doc_id"], meta["page"])
            if key in sources:
                continue
            sources[key] = {
                "doc_id": meta["doc_id"],
                "file": meta["doc_name"],
                "short": meta["short_name"],
                "unit": meta["unit"],
                "page": meta["page"],
                "label": f"{meta['short_name']} {meta['unit']}{meta['page']}",
                "snippet": doc.page_content[:280],
                "score": round(score, 2),
            }
        return list(sources.values())

    @staticmethod
    def _source_tag(sources: list[dict]) -> str:
        """Compact bottom-right tag, e.g. 'Handbook p.12, 13 · FAQ §3'."""
        grouped: dict[str, list[str]] = {}
        for src in sources:
            name = f"{src['short']} {src['unit']}"
            grouped.setdefault(name, []).append(str(src["page"]))
        return " · ".join(f"{name}{', '.join(pages)}" for name, pages in grouped.items())

    # --- summary ----------------------------------------------------------

    def summarize(self, doc_id: str, session_id: str, refresh: bool = False) -> str | None:
        doc = self.db.get_document(doc_id, session_id)
        if not doc or doc["status"] != "ready":
            return None
        if doc["summary"] and not refresh:
            return doc["summary"]

        chunks = self.store.get_chunks(doc_id)
        limit = self.settings.summary_max_chunks
        if len(chunks) > limit:  # very long document: sample evenly to stay within rate limits
            step = len(chunks) / limit
            chunks = [chunks[int(i * step)] for i in range(limit)]
        groups = [chunks[i : i + 6] for i in range(0, len(chunks), 6)]
        texts = [
            "\n\n".join(f"[{c.metadata['unit']}{c.metadata['page']}] {c.page_content}" for c in group)
            for group in groups
        ]
        notes = self.map_chain.batch([{"text": t} for t in texts], config={"max_concurrency": 2})
        summary = self.reduce_chain.invoke({"doc_name": doc["name"], "notes": "\n".join(notes)})
        summary = strip_source_marker(summary).strip()

        self.db.update_document(doc_id, summary=summary)
        return summary

    # --- export -----------------------------------------------------------

    def export(self, session_id: str, fmt: str) -> str:
        messages = self.db.get_messages(session_id)
        if fmt == "json":
            records = [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "found": m["found"],
                    "sources": [
                        {"file": s["file"], "page": s["page"], "score": s["score"]}
                        for s in m["sources"]
                    ],
                    "timestamp": m["created_at"],
                }
                for m in messages
            ]
            return json.dumps({"session_id": session_id, "messages": records}, indent=2)

        lines = ["# DocuMind – Q&A session", ""]
        for m in messages:
            if m["role"] == "user":
                lines += [f"**Q:** {m['content']}", ""]
            else:
                lines += [f"**A:** {m['content']}", ""]
                if m["sources"]:
                    refs = "; ".join(f"{s['file']} {s['unit']}{s['page']}" for s in m["sources"])
                    lines += [f"*Sources: {refs}*", ""]
                lines += ["---", ""]
        return "\n".join(lines)
