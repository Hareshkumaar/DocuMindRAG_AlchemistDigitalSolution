"""FastAPI entry point: HTML pages + JSON/streaming API."""

import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import ALLOWED_EXTENSIONS, BASE_DIR, SAMPLE_DIR, get_settings
from app.rag import RAGService
from app.store import Database


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "rag", None) is None:  # tests may inject their own service
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your free key "
                "from https://console.groq.com/keys"
            )
        app.state.rag = RAGService(settings, Database(settings.db_path))
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def asset_version() -> int:
    """Last-modified time of the static files, added to their URLs (?v=...) so the
    browser downloads style.css/app.js again as soon as they change."""
    return max(int(path.stat().st_mtime) for path in (BASE_DIR / "static").glob("*.*"))


templates.env.globals["asset_version"] = asset_version


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


def get_session_id(
    x_session_id: str | None = Header(default=None),
    session_id: str | None = Query(default=None),
) -> str:
    """Guest session id, sent by the browser as a header (or ?session_id= for downloads)."""
    value = x_session_id or session_id
    if not value or not re.fullmatch(r"[A-Za-z0-9-]{8,64}", value):
        raise HTTPException(400, "Missing or invalid session id.")
    return value


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    attachments: list[str] | None = None
    doc_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def welcome_page(request: Request):
    return templates.TemplateResponse(request, "welcome.html")


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"active": "chat"})


@app.get("/documents", response_class=HTMLResponse)
def documents_page(request: Request):
    return templates.TemplateResponse(request, "documents.html", {"active": "documents"})


# ---------------------------------------------------------------------------
# Documents API
# ---------------------------------------------------------------------------

PRIVATE_FIELDS = {"session_id", "file_hash", "file_path"}
MEDIA_TYPES = {"pdf": "application/pdf", "txt": "text/plain; charset=utf-8"}


def _public(doc: dict) -> dict:
    """Document fields that are safe to send to the browser."""
    return {key: value for key, value in doc.items() if key not in PRIVATE_FIELDS}


def _check_capacity(rag: RAGService, session_id: str, new_files: int) -> None:
    """Only readable documents count toward the limit; failed uploads do not."""
    limit = rag.settings.max_files
    usable = [d for d in rag.db.list_documents(session_id) if d["status"] != "failed"]
    if len(usable) + new_files > limit:
        raise HTTPException(
            400,
            f"You already have {len(usable)} of {limit} documents (the limit). "
            "Delete one on the Knowledge base page, then attach it again.",
        )


async def _save_and_ingest(rag: RAGService, session_id: str, name: str, content: bytes) -> dict:
    session_dir = rag.settings.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    path.write_bytes(content)
    return _public(await run_in_threadpool(rag.ingest, path, name, session_id))


@app.post("/api/documents")
async def upload_documents(
    files: list[UploadFile],
    session_id: str = Depends(get_session_id),
    rag: RAGService = Depends(get_rag),
):
    _check_capacity(rag, session_id, len(files))
    max_bytes = rag.settings.max_file_mb * 1024 * 1024

    uploads: list[tuple[str, bytes]] = []
    for upload in files:  # validate everything before indexing anything
        name = Path(upload.filename or "document").name
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"{name}: only PDF and TXT files are supported.")
        content = await upload.read()
        if len(content) > max_bytes:
            raise HTTPException(400, f"{name}: file is larger than {rag.settings.max_file_mb} MB.")
        uploads.append((name, content))

    return [await _save_and_ingest(rag, session_id, name, content) for name, content in uploads]


@app.post("/api/samples")
async def load_samples(session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)):
    """Load the demo documents from sample_docs/ into this guest session."""
    samples = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS)
    existing = {doc["name"] for doc in rag.db.list_documents(session_id)}
    to_add = [p for p in samples if p.name not in existing]
    _check_capacity(rag, session_id, len(to_add))
    return [await _save_and_ingest(rag, session_id, p.name, p.read_bytes()) for p in to_add]


@app.get("/api/documents")
def list_documents(session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)):
    return [_public(doc) for doc in rag.db.list_documents(session_id)]


@app.get("/api/documents/{doc_id}/file")
def open_document(
    doc_id: str, session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)
):
    """Serve the original file so a source link can open it (PDFs at #page=N)."""
    doc = rag.db.get_document(doc_id, session_id)
    if not doc or not doc["file_path"] or not Path(doc["file_path"]).is_file():
        raise HTTPException(404, "Document not found.")
    return FileResponse(
        doc["file_path"],
        media_type=MEDIA_TYPES.get(doc["file_type"], "application/octet-stream"),
        filename=doc["name"],
        content_disposition_type="inline",
    )


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(
    doc_id: str, session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)
):
    if not rag.delete_document(doc_id, session_id):
        raise HTTPException(404, "Document not found.")


@app.post("/api/documents/{doc_id}/summary")
async def summarize_document(
    doc_id: str,
    refresh: bool = False,
    session_id: str = Depends(get_session_id),
    rag: RAGService = Depends(get_rag),
):
    summary = await run_in_threadpool(rag.summarize, doc_id, session_id, refresh)
    if summary is None:
        raise HTTPException(404, "Document not found or not ready.")
    return {"doc_id": doc_id, "summary": summary}


# ---------------------------------------------------------------------------
# Chat API (Server-Sent Events)
# ---------------------------------------------------------------------------

def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _resolve_attachments(
    rag: RAGService, session_id: str, attachment_ids: list[str] | None
) -> list[dict]:
    """Keep only attachment ids that belong to this session and are ready."""
    seen: set[str] = set()
    valid: list[dict] = []
    for doc_id in attachment_ids or []:
        if doc_id in seen:
            continue
        doc = rag.db.get_document(doc_id, session_id)
        if doc and doc["status"] == "ready":
            seen.add(doc_id)
            valid.append({"id": doc["id"], "name": doc["name"], "file_type": doc["file_type"]})
    return valid


@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    session_id: str = Depends(get_session_id),
    rag: RAGService = Depends(get_rag),
):
    attachments = _resolve_attachments(rag, session_id, body.attachments)

    async def event_stream():
        try:
            async for event in rag.stream_answer(
                body.question.strip(), session_id, attachments=attachments, doc_ids=body.doc_ids
            ):
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "message": f"Something went wrong: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
def history(session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)):
    return rag.db.get_messages(session_id)


@app.delete("/api/history", status_code=204)
def clear_history(session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)):
    rag.db.clear_messages(session_id)


@app.delete("/api/session", status_code=204)
async def delete_session(session_id: str = Depends(get_session_id), rag: RAGService = Depends(get_rag)):
    """End a guest session: delete its documents, chunks, files and chat."""
    await run_in_threadpool(rag.delete_session, session_id)


@app.get("/api/export")
def export(
    format: str = Query(default="md", pattern="^(md|json)$"),
    session_id: str = Depends(get_session_id),
    rag: RAGService = Depends(get_rag),
):
    content = rag.export(session_id, format)
    media = "application/json" if format == "json" else "text/markdown"
    return Response(
        content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="documind-session.{format}"'},
    )
