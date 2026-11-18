"use strict";

/* ==========================================================================
   DocuMind front-end (plain JavaScript, no framework)
   1. Session   2. API client   3. Helpers   4. Shared (documents, upload, sidebar)
   5. Welcome page   6. Chat page   7. Knowledge base page   8. Start-up
   ========================================================================== */

const ALLOWED_EXTENSIONS = [".pdf", ".txt"];
const NOT_FOUND_MESSAGE = "I could not find this information in the provided documents.";

/* 1. Session ---------------------------------------------------------------
   Two separate sessions are kept in this browser, and both survive restarts:
     guest  - the user's own files ("Start as guest"), starts empty
     sample - the demo files only ("Try sample documents")
   The backend uses the session id to keep documents and chat apart. */
const Session = (() => {
  const MODES = ["guest", "sample"];
  const MODE_KEY = "documind.mode";
  const idKey = (mode) => `documind.session.${mode}`;
  const newId = () => (crypto.randomUUID
    ? crypto.randomUUID()
    : `guest-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
  const write = (key, value) => {
    try { localStorage.setItem(key, value); } catch { /* keep for this page only */ }
  };

  // Older versions kept one shared session, which is where the samples were loaded:
  // keep it as the sample session so the guest session starts empty.
  const legacy = read("documind.session");
  if (legacy && !read(idKey("sample"))) write(idKey("sample"), legacy);
  try { localStorage.removeItem("documind.session"); } catch { /* ignore */ }

  const ids = {};
  for (const m of MODES) {
    ids[m] = read(idKey(m));
    if (!ids[m]) write(idKey(m), ids[m] = newId());
  }
  let mode = MODES.includes(read(MODE_KEY)) ? read(MODE_KEY) : "guest";

  return {
    get id() { return ids[mode]; },
    get mode() { return mode; },
    /** Switch between the guest and the sample session. */
    use(next) { mode = next; write(MODE_KEY, next); },
    /** "Clear session": give the current mode a brand-new, empty session. */
    reset() { write(idKey(mode), ids[mode] = newId()); },
  };
})();

/* 2. API client ------------------------------------------------------------ */
async function readError(response) {
  try {
    const data = await response.json();
    return typeof data.detail === "string" ? data.detail : "The request was not valid.";
  } catch {
    return `Request failed (${response.status}).`;
  }
}

const api = {
  async request(path, { method = "GET", body, json } = {}) {
    const headers = { "X-Session-Id": Session.id };
    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(json);
    }
    const response = await fetch(path, { method, headers, body });
    if (!response.ok) throw new Error(await readError(response));
    return response.status === 204 ? null : response.json();
  },

  listDocuments: () => api.request("/api/documents"),
  loadSamples: () => api.request("/api/samples", { method: "POST" }),
  deleteDocument: (id) => api.request(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  summarize: (id, refresh = false) =>
    api.request(`/api/documents/${encodeURIComponent(id)}/summary?refresh=${refresh}`, { method: "POST" }),
  deleteSession: () => api.request("/api/session", { method: "DELETE" }),
  history: () => api.request("/api/history"),
  clearHistory: () => api.request("/api/history", { method: "DELETE" }),
  exportUrl: (format) => `/api/export?format=${format}&session_id=${encodeURIComponent(Session.id)}`,
  /** Link to the original file; PDFs open at the cited page. */
  fileUrl: (docId, page, unit = "p.") =>
    `/api/documents/${encodeURIComponent(docId)}/file?session_id=${encodeURIComponent(Session.id)}`
    + (unit === "p." && page ? `#page=${page}` : ""),

  uploadFiles(files) {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return api.request("/api/documents", { method: "POST", body: form });
  },

  /** POST a question and call onEvent for every server-sent event
   *  (token | sources | done | error) as it arrives. */
  async streamChat(payload, onEvent, signal) {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "X-Session-Id": Session.id, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!response.ok || !response.body) throw new Error(await readError(response));

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("\n");
        if (data) onEvent(JSON.parse(data));
      }
    }
  },
};

/* 3. Helpers --------------------------------------------------------------- */
const $ = (selector, root = document) => root.querySelector(selector);

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Small, safe Markdown renderer: paragraphs, bullet/numbered lists, ### headings,
 *  **bold** and `code`. Input is escaped first, so no HTML from the model runs. */
function renderMarkdown(text) {
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");

  const out = [];
  let listTag = null;
  let listItems = [];
  let paraLines = [];

  const flushPara = () => {
    if (!paraLines.length) return;
    out.push(`<p>${paraLines.map(inline).join("<br>")}</p>`);
    paraLines = [];
  };
  const flushList = () => {
    if (!listTag) return;
    out.push(`<${listTag}>${listItems.join("")}</${listTag}>`);
    listTag = null;
    listItems = [];
  };

  // LLMs sometimes write headings with invisible characters or non-breaking spaces;
  // remove format characters (\p{Cf}) and turn every kind of space (\p{Zs}) into " ".
  const normalised = text
    .replace(/\p{Cf}/gu, "")
    .replace(/\p{Zs}/gu, " ")
    .replace(/\r\n?/g, "\n");

  for (const raw of normalised.trim().split("\n")) {
    const line = raw.trim().replace(/\\#/g, "#");  // "\#\# Title" -> "## Title"
    if (!line) {
      flushPara();
      flushList();
    } else if (/^#{1,6}\s*\S/.test(line)) {
      flushPara();
      flushList();
      out.push(`<h3>${inline(line.replace(/^#{1,6}\s*/, ""))}</h3>`);
    } else if (/^[-*•]\s+/.test(line)) {
      flushPara();
      if (listTag !== "ul") { flushList(); listTag = "ul"; }
      listItems.push(`<li>${inline(line.replace(/^[-*•]\s+/, ""))}</li>`);
    } else if (/^\d+[.)]\s+/.test(line)) {
      flushPara();
      if (listTag !== "ol") { flushList(); listTag = "ol"; }
      listItems.push(`<li>${inline(line.replace(/^\d+[.)]\s+/, ""))}</li>`);
    } else {
      flushList();
      paraLines.push(line);
    }
  }
  flushPara();
  flushList();
  return out.join("");
}

/** Safety net: remove a hidden "[[sources: 1, 2]]" line if one ever reaches the browser. */
const stripSourceMarker = (text) =>
  text.replace(/\s*(?:\[\[|\[|【)\s*sources?\s*:[^\]】]*(?:\]\]|\]|】)\s*$/i, "");

let toastTimer;
function toast(message, type = "info") {
  const el = $("#toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast toast--${type}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, type === "error" ? 6000 : 3500);
}

const extensionOf = (name) => name.slice(name.lastIndexOf(".")).toLowerCase();
const plural = (n, word) => {
  if (n === 1) return `${n} ${word}`;
  return `${n} ${/(?:s|x|z|ch|sh)$/i.test(word) ? `${word}es` : `${word}s`}`;
};

function formatBytes(bytes) {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* 4. Shared: documents, upload, sidebar ----------------------------------- */
const Docs = {
  items: [],
  get ready() { return this.items.filter((d) => d.status === "ready"); },

  async refresh() {
    try {
      this.items = await api.listDocuments();
    } catch (error) {
      toast(error.message, "error");
    }
    Sidebar.render(this.items);
    return this.items;
  },
};

/** Validate, upload and index files. Returns the indexed documents or null. */
async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return null;

  const unsupported = files.find((f) => !ALLOWED_EXTENSIONS.includes(extensionOf(f.name)));
  if (unsupported) {
    toast(`${unsupported.name}: only PDF and TXT files are supported.`, "error");
    return null;
  }

  toast(`Reading and indexing ${plural(files.length, "file")}…`);
  try {
    const results = await api.uploadFiles(files);
    const failed = results.find((d) => d.status === "failed");
    if (failed) toast(`${failed.name}: ${failed.error}`, "error");
    else toast(`${plural(results.length, "file")} ready to search.`, "success");
    await Docs.refresh();
    return results;
  } catch (error) {
    toast(error.message, "error");
    return null;
  }
}

const Sidebar = {
  init() {
    const md = $("#exportMd");
    const json = $("#exportJson");
    if (md) md.href = api.exportUrl("md");
    if (json) json.href = api.exportUrl("json");

    const isSample = Session.mode === "sample";
    const title = $("#sessionTitle");
    const text = $("#sessionText");
    if (title) title.textContent = isSample ? "Sample session" : "Guest session";
    if (text) {
      text.textContent = isSample
        ? "Demo files only. Choose \"Start as guest\" on the home page to use your own files."
        : "Your files and chat are kept in this browser, also after a restart.";
    }

    $("#clearSessionBtn")?.addEventListener("click", async () => {
      if (!window.confirm("Delete all your files and chat, and start a new session?")) return;
      try {
        await api.deleteSession();
      } catch (error) {
        toast(error.message, "error");
        return;
      }
      Session.reset();
      window.location.href = "/chat";
    });

    $("#newChatBtn")?.addEventListener("click", async () => {
      try {
        await api.clearHistory();
      } catch (error) {
        toast(error.message, "error");
        return;
      }
      if (document.body.dataset.page === "chat") ChatPage.reset();
      else window.location.href = "/chat";
    });
  },

  render(docs) {
    const list = $("#sidebarDocs");
    const count = $("#docCount");
    if (count) count.textContent = plural(docs.filter((d) => d.status === "ready").length, "file");
    if (!list) return;
    list.innerHTML = docs.length
      ? docs.map((d) => `
          <li title="${escapeHtml(d.name)}">
            <span class="status-dot status-dot--${escapeHtml(d.status)}"></span>
            <span>${escapeHtml(d.name)}</span>
          </li>`).join("")
      : `<li class="empty">No files yet</li>`;
  },
};

/* 5. Welcome page ---------------------------------------------------------- */
const WelcomePage = {
  init() {
    // The guest session continues (files and chat survive restarts); "Clear session"
    // in the sidebar is the explicit way to start over.
    $("#guestBtn").addEventListener("click", () => {
      Session.use("guest");  // your own files; the demo files are never added here
      window.location.href = "/chat";
    });
    $("#sampleBtn").addEventListener("click", async (e) => {
      const button = e.currentTarget;
      button.disabled = true;
      Session.use("sample");  // a separate session that holds only the demo files
      toast("Loading sample documents…");
      try {
        await api.loadSamples();  // adds only the samples that are not loaded yet
        window.location.href = "/chat";
      } catch (error) {
        toast(`${error.message} Use "Clear session" in the sidebar to start over.`, "error");
        button.disabled = false;
      }
    });
  },
};

/* 6. Chat page ------------------------------------------------------------- */

/** One question + its streamed answer. */
class TurnView {
  constructor(question, attachments = []) {
    this.el = $("#turnTemplate").content.firstElementChild.cloneNode(true);
    this.answer = $(".answer", this.el);
    this.body = $(".answer__body", this.el);
    this.meta = $(".answer__meta", this.el);
    this.refs = $(".refs", this.el);
    this.caret = Object.assign(document.createElement("span"), { className: "caret" });
    this.text = "";

    const attWrap = $(".msg__attachments", this.el);
    if (attachments.length) {
      attWrap.innerHTML = attachments.map((a) => {
        const type = a.file_type === "txt" ? "txt" : "pdf";
        return `
          <a class="att-card" href="${api.fileUrl(a.id)}" target="_blank" rel="noopener" title="${escapeHtml(a.name)}">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>
            </svg>
            <span class="file-type file-type--${type}">${type.toUpperCase()}</span>
            <span class="att-card__name">${escapeHtml(a.name)}</span>
          </a>`;
      }).join("");
      attWrap.hidden = false;
    }

    $(".turn__question", this.el).textContent = question;
    this.el.classList.add("turn--searching");
  }

  appendToken(token) {
    this.el.classList.remove("turn--searching");
    this.text += token;
    this.body.textContent = this.text;
    this.body.append(this.caret);
  }

  finish({ found, sources = [], meta = "" }) {
    this.el.classList.remove("turn--searching");
    this.caret.remove();
    this.text = stripSourceMarker(this.text);

    if (found) {
      this.body.innerHTML = renderMarkdown(this.text);
      this.body.classList.add("markdown");
    } else {
      this.answer.classList.add("answer--not-found");
      this.body.textContent = this.text.trim() || NOT_FOUND_MESSAGE;
    }
    this.meta.textContent = meta;

    if (found && sources.length) {
      this.refs.innerHTML = sources.map((s) => `
        <a class="ref" href="${api.fileUrl(s.doc_id, s.page, s.unit)}" target="_blank" rel="noopener"
           title="Open ${escapeHtml(s.file)} at ${escapeHtml(s.unit)}${s.page}">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>
          </svg>${escapeHtml(s.short)} ${escapeHtml(s.unit)}${s.page}
        </a>`).join("");
      this.refs.hidden = false;
    }
  }

  fail(message) {
    this.caret.remove();
    this.el.classList.remove("turn--searching");
    this.answer.classList.add("answer--error");
    this.body.textContent = message;
  }

}

const ChatPage = {
  attachments: [],       // [{ localId, id, name, file_type, size, status, error }]
  controller: null,      // AbortController of the answer being streamed
  nextLocalId: 1,

  async init() {
    this.main = $(".main");
    this.thread = $("#thread");
    this.empty = $("#emptyState");
    this.form = $("#askForm");
    this.attList = $("#composerAttachments");
    this.input = $("#question");
    this.sendBtn = $("#sendBtn");

    this.bindEvents();
    await Docs.refresh();

    const preselect = new URLSearchParams(window.location.search).get("doc");
    const preDoc = preselect && Docs.ready.find((d) => d.id === preselect);
    if (preDoc) {
      this.attachments.push({
        localId: this.nextLocalId++,
        id: preDoc.id,
        name: preDoc.name,
        file_type: preDoc.file_type,
        size: 0,
        status: "ready",
        error: "",
      });
      this.renderAttachments();
    }

    await this.loadHistory();
    this.input.focus();
  },

  bindEvents() {
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (this.controller) this.controller.abort();
      else this.ask(this.input.value);
    });

    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.form.requestSubmit();
      }
    });
    this.input.addEventListener("input", () => {
      this.input.style.height = "auto";
      this.input.style.height = `${this.input.scrollHeight}px`;
    });

    $("#suggestions").addEventListener("click", (e) => {
      if (e.target.matches(".chip")) this.ask(e.target.textContent);
    });

    $("#chatFiles").addEventListener("change", (e) => {
      this.attachFiles(e.target.files);
      e.target.value = "";
    });

    this.attList.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove]");
      if (!btn) return;
      const localId = Number(btn.dataset.remove);
      this.attachments = this.attachments.filter((a) => a.localId !== localId);
      this.renderAttachments();
      this.updateSendState();
    });

    this.input.addEventListener("paste", (e) => {
      const files = e.clipboardData?.files;
      if (files?.length) {
        e.preventDefault();
        this.attachFiles(files);
      }
    });

    let dragDepth = 0;
    this.main.addEventListener("dragenter", (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      dragDepth += 1;
      this.form.classList.add("composer--dragover");
    });
    this.main.addEventListener("dragover", (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
    });
    this.main.addEventListener("dragleave", (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) this.form.classList.remove("composer--dragover");
    });
    this.main.addEventListener("drop", (e) => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      dragDepth = 0;
      this.form.classList.remove("composer--dragover");
      this.attachFiles(e.dataTransfer.files);
    });
  },

  attachFiles(fileList) {
    for (const file of [...fileList]) {
      const ext = extensionOf(file.name);
      const item = {
        localId: this.nextLocalId++,
        id: null,
        name: file.name,
        file_type: ext === ".txt" ? "txt" : "pdf",
        size: file.size,
        status: "uploading",
        error: "",
      };
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        item.status = "failed";
        item.error = "only PDF and TXT files are supported.";
        this.attachments.push(item);
        this.renderAttachments();
        this.updateSendState();
        continue;
      }
      this.attachments.push(item);
      this.renderAttachments();
      this.updateSendState();
      this.uploadOne(item, file);
    }
  },

  async uploadOne(item, file) {
    try {
      const [doc] = await api.uploadFiles([file]);
      if (doc.status === "failed") {
        item.status = "failed";
        item.error = doc.error || "Indexing failed.";
      } else {
        item.id = doc.id;
        item.name = doc.name;
        item.file_type = doc.file_type;
        item.status = "ready";
        await Docs.refresh();
      }
    } catch (error) {
      item.status = "failed";
      item.error = error.message;
      toast(`${item.name}: ${error.message}`, "error");
    }
    this.renderAttachments();
    this.updateSendState();
  },

  renderAttachments() {
    if (!this.attachments.length) {
      this.attList.innerHTML = "";
      this.attList.hidden = true;
      return;
    }
    this.attList.hidden = false;
    this.attList.innerHTML = this.attachments.map((a) => {
      const type = a.file_type === "txt" ? "txt" : "pdf";
      const sizeText = formatBytes(a.size);
      const statusHtml = a.status === "uploading"
        ? `<span class="att-chip__status"><span class="spinner" aria-hidden="true"></span>Uploading…</span>`
        : a.status === "failed"
          ? `<span class="att-chip__status att-chip__status--failed" title="${escapeHtml(a.error)}">Failed: ${escapeHtml(a.error)}</span>`
          : `<span class="att-chip__status">Ready</span>`;
      const meta = [sizeText, statusHtml].filter(Boolean).join(" · ");
      return `
        <div class="att-chip ${a.status === "failed" ? "att-chip--failed" : ""}">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>
          </svg>
          <span class="file-type file-type--${type}">${type.toUpperCase()}</span>
          <div class="att-chip__info">
            <span class="att-chip__name" title="${escapeHtml(a.name)}">${escapeHtml(a.name)}</span>
            <span class="att-chip__meta">${meta}</span>
          </div>
          <button type="button" class="att-chip__remove" data-remove="${a.localId}" aria-label="Remove ${escapeHtml(a.name)}">×</button>
        </div>`;
    }).join("");
  },

  updateSendState() {
    const blocked = this.attachments.some((a) => a.status === "uploading" || a.status === "failed");
    this.sendBtn.disabled = !this.controller && blocked;
  },

  async loadHistory() {
    let messages = [];
    try {
      messages = await api.history();
    } catch (error) {
      toast(error.message, "error");
    }
    for (let i = 0; i < messages.length; i += 1) {
      if (messages[i].role !== "user") continue;
      const reply = messages[i + 1]?.role === "assistant" ? messages[i + 1] : null;
      const turn = new TurnView(messages[i].content, messages[i].attachments || []);
      turn.text = reply ? reply.content : "";
      if (reply) turn.finish({ found: reply.found, sources: reply.sources });
      else turn.fail("No answer was saved for this question.");
      this.addTurn(turn);
    }
    this.scrollToBottom(true);
  },

  addTurn(turn) {
    this.empty.hidden = true;
    this.thread.append(turn.el);
  },

  async ask(rawQuestion) {
    if (this.controller || this.attachments.some((a) => a.status === "uploading")) return;
    // Never send a message without a file the user attached: explain and keep the chip.
    const failed = this.attachments.find((a) => a.status === "failed");
    if (failed) {
      toast(`${failed.name} was not added: ${failed.error} Remove it (×) to send without it.`, "error");
      return;
    }
    const readyAttachments = this.attachments.filter((a) => a.status === "ready" && a.id);
    let question = rawQuestion.trim();
    if (!question && !readyAttachments.length) return;
    if (!question) {
      question = readyAttachments.length === 1
        ? "Summarize this document."
        : "Summarize these documents.";
    }

    this.attachments = [];
    this.renderAttachments();
    this.input.value = "";
    this.input.style.height = "auto";
    const turn = new TurnView(question, readyAttachments);
    this.addTurn(turn);
    this.scrollToBottom(true);
    this.setStreaming(true);

    const result = { found: true, sources: [], meta: "" };
    try {
      await api.streamChat(
        {
          question,
          attachments: readyAttachments.length ? readyAttachments.map((a) => a.id) : null,
        },
        (event) => {
          if (event.type === "token") {
            turn.appendToken(event.text);
            this.scrollToBottom();
          } else if (event.type === "sources") {
            result.found = event.found;
            result.sources = event.sources;
          } else if (event.type === "done") {
            result.meta = event.queries.length
              ? `${event.elapsed}s · ${plural(event.queries.length, "search")}`
              : `${event.elapsed}s`;
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        },
        this.controller.signal,
      );
      turn.finish(result);
    } catch (error) {
      if (error.name === "AbortError") turn.finish({ found: true, meta: "Stopped" });
      else turn.fail(error.message);
    } finally {
      this.setStreaming(false);
      this.scrollToBottom();
      this.input.focus();
    }
  },

  setStreaming(active) {
    this.controller = active ? new AbortController() : null;
    this.sendBtn.classList.toggle("send-btn--stop", active);
    this.sendBtn.setAttribute("aria-label", active ? "Stop" : "Send");
    this.updateSendState();
  },

  scrollToBottom(force = false) {
    const t = this.thread;
    const nearBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 160;
    if (force || nearBottom) t.scrollTop = t.scrollHeight;
  },

  reset() {
    this.controller?.abort();
    this.thread.querySelectorAll(".turn").forEach((el) => el.remove());
    this.empty.hidden = false;
    toast("Started a new chat.");
  },
};

/* 7. Knowledge base page --------------------------------------------------- */
const DocumentsPage = {
  current: null,  // document shown in the summary drawer
  summary: "",

  async init() {
    this.rows = $("#docRows");
    this.panel = $("#summaryPanel");

    $("#kbFiles").addEventListener("change", async (e) => {
      const files = [...e.target.files];
      e.target.value = "";
      this.showPending(files);
      await uploadFiles(files);
      this.render();
    });

    this.rows.addEventListener("click", (e) => {
      const button = e.target.closest("[data-action]");
      if (!button) return;
      const doc = Docs.items.find((d) => d.id === button.dataset.id);
      if (doc && button.dataset.action === "summarize") this.openSummary(doc);
      if (doc && button.dataset.action === "delete") this.remove(doc);
    });

    $("#closeSummary").addEventListener("click", () => this.closeSummary());
    $("#regenSummary").addEventListener("click", () => this.current && this.openSummary(this.current, true));
    $("#downloadSummary").addEventListener("click", () => this.downloadSummary());

    await Docs.refresh();
    this.render();
  },

  render() {
    const docs = Docs.items;
    const ready = Docs.ready;
    const pages = ready.reduce((sum, d) => sum + d.pages, 0);
    const chunks = ready.reduce((sum, d) => sum + d.chunks, 0);
    $("#kbStats").textContent = docs.length
      ? `${plural(ready.length, "file")} · ${plural(pages, "page")} · ${plural(chunks, "chunk")} indexed`
      : "No files yet. Upload PDF or TXT files to start.";

    this.rows.innerHTML = docs.length
      ? docs.map((d) => this.rowHtml(d)).join("")
      : `<tr><td colspan="5" class="table__empty">Your uploaded files will appear here.</td></tr>`;
  },

  rowHtml(doc) {
    const type = doc.file_type === "txt" ? "txt" : "pdf";
    const statusText = { ready: "Indexed", failed: doc.error || "Failed", processing: "Processing" }[doc.status];
    const selected = this.current?.id === doc.id ? " is-selected" : "";
    const actions = doc.status === "ready"
      ? `<button type="button" class="btn btn--ghost btn--sm" data-action="summarize" data-id="${doc.id}">Summarize</button>
         <a class="btn btn--ghost btn--sm" href="/chat?doc=${encodeURIComponent(doc.id)}">Ask</a>`
      : "";
    return `
      <tr class="${selected}">
        <td><span class="table__name"><span class="file-type file-type--${type}">${type.toUpperCase()}</span>${doc.status === "ready"
          ? `<a class="table__link" href="${api.fileUrl(doc.id)}" target="_blank" rel="noopener">${escapeHtml(doc.name)}</a>`
          : escapeHtml(doc.name)}</span></td>
        <td class="table__num">${doc.pages || "–"}</td>
        <td class="table__num">${doc.chunks || "–"}</td>
        <td><span class="status status--${escapeHtml(doc.status)}"><span class="status-dot status-dot--${escapeHtml(doc.status)}"></span>${escapeHtml(statusText)}</span></td>
        <td><div class="table__actions">${actions}
          <button type="button" class="btn btn--danger btn--sm" data-action="delete" data-id="${doc.id}">Delete</button>
        </div></td>
      </tr>`;
  },

  showPending(files) {
    const pending = files.map((f) => `
      <tr>
        <td><span class="table__name">${escapeHtml(f.name)}</span></td>
        <td class="table__num">–</td><td class="table__num">–</td>
        <td><span class="status status--processing"><span class="status-dot status-dot--processing"></span>Processing</span></td>
        <td></td>
      </tr>`).join("");
    if (!Docs.items.length) this.rows.innerHTML = "";
    this.rows.insertAdjacentHTML("beforeend", pending);
  },

  async openSummary(doc, refresh = false) {
    this.current = doc;
    this.render();
    this.panel.hidden = false;
    $("#summaryTitle").textContent = doc.name;
    $("#askThisFile").href = `/chat?doc=${encodeURIComponent(doc.id)}`;
    const body = $("#summaryBody");
    body.innerHTML = `<div class="skeleton"><span></span><span></span><span></span></div>`;
    try {
      const { summary } = await api.summarize(doc.id, refresh);
      if (this.current?.id !== doc.id) return;  // user switched documents meanwhile
      this.summary = summary;
      body.innerHTML = renderMarkdown(summary);
    } catch (error) {
      body.textContent = error.message;
    }
  },

  closeSummary() {
    this.panel.hidden = true;
    this.current = null;
    this.render();
  },

  downloadSummary() {
    if (!this.current || !this.summary) return;
    const blob = new Blob([`# ${this.current.name}\n\n${this.summary}\n`], { type: "text/markdown" });
    const link = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `${this.current.name.replace(/\.[^.]+$/, "")}-summary.md`,
    });
    link.click();
    URL.revokeObjectURL(link.href);
  },

  async remove(doc) {
    if (!window.confirm(`Delete "${doc.name}"? It will no longer be used in answers.`)) return;
    try {
      await api.deleteDocument(doc.id);
      if (this.current?.id === doc.id) this.closeSummary();
      toast(`${doc.name} deleted.`);
      await Docs.refresh();
      this.render();
    } catch (error) {
      toast(error.message, "error");
    }
  },
};

/* 8. Start-up -------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if (page !== "welcome") Sidebar.init();
  const pages = { welcome: WelcomePage, chat: ChatPage, documents: DocumentsPage };
  pages[page]?.init();
});
