import os
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
import uvicorn
import time
import uuid
import json

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("tool_server")

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RAG_PDF_DIR = os.getenv("RAG_PDF_DIR", "/mnt/d/AI/data/personal/pdfs")
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/mnt/d/AI/data/personal/.rag_faiss")

app = FastAPI(title="RAG Tool Server")

_embeddings: Optional[OllamaEmbeddings] = None
_vs: Optional[FAISS] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class RefreshRequest(BaseModel):
    pdf_dir: str = ""

class ShellRequest(BaseModel):
    command: str

MAX_LOG_CHARS = int(os.getenv("API_LOG_PREVIEW", "500"))

def _preview(val: str, limit: int = MAX_LOG_CHARS) -> str:
    if val is None:
        return ""
    s = val.replace("\n", " ")
    return (s[:limit] + ("..." if len(s) > limit else ""))

def _log_tool_output(tool: str, rid: str, result: str):
    log.info(f"[TOOL OUT {tool} {rid}] result_preview={_preview(result)} len={len(result)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    start = time.time()
    try:
        raw_body = await request.body()
        body_text = raw_body.decode("utf-8", errors="replace") if raw_body else ""
    except Exception as e:
        body_text = f"<body read error: {e}>"
    log.info(
        f"[API IN] id={rid} method={request.method} path={request.url.path} "
        f"query={dict(request.query_params)} body={_preview(body_text)}"
    )
    try:
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        log.info(
            f"[API OUT] id={rid} status={response.status_code} ms={elapsed:.1f}"
        )
        return response
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        log.exception(f"[API ERR] id={rid} ms={elapsed:.1f} error={e}")
        raise

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        t0 = time.time()
        _embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        log.info(f"[EMBED] model={OLLAMA_EMBED_MODEL} loaded ms={(time.time()-t0)*1000:.1f}")
    return _embeddings

def _index_path() -> Path:
    return Path(RAG_INDEX_DIR)

def _save_vs(vs: FAISS):
    p = _index_path()
    p.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(p))

def _load_vs() -> Optional[FAISS]:
    global _vs
    if _vs is not None:
        return _vs
    p = _index_path()
    if not p.exists():
        log.info("[INDEX] load skipped (path missing)")
        return None
    try:
        t0 = time.time()
        _vs = FAISS.load_local(str(p), _get_embeddings(), allow_dangerous_deserialization=True)
        log.info(f"[INDEX] loaded ms={(time.time()-t0)*1000:.1f}")
    except Exception as e:
        log.warning(f"Load index failed: {e}")
        _vs = None
    return _vs

def _build_index(pdf_dir: str) -> str:
    global _vs
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    pdf_dir = pdf_dir or RAG_PDF_DIR
    base = Path(pdf_dir).expanduser().resolve()
    if not base.exists():
        log.warning(f"[BUILD {rid}] missing_dir path={base}")
        return f"Error: PDF directory not found: {base}"
    log.info(f"[BUILD {rid}] start dir={base}")
    loader = DirectoryLoader(str(base), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()
    if not docs:
        log.info(f"[BUILD {rid}] no_pdfs ms={(time.time()-t0)*1000:.1f}")
        return f"No PDFs found in {base}"
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(chunks)
    _vs = FAISS.from_documents(chunks, _get_embeddings())
    _save_vs(_vs)
    log.info(f"[BUILD {rid}] done docs={len(docs)} chunks={len(chunks)} ms={(time.time()-t0)*1000:.1f}")
    return f"Indexed {len(chunks)} chunks from {len(docs)} PDF(s)."

@app.post("/rag/refresh")
def rag_refresh(req: RefreshRequest):
    rid = uuid.uuid4().hex[:8]
    log.info(f"[REFRESH {rid}] payload={req.dict()}")
    t0 = time.time()
    result = _build_index(req.pdf_dir)
    log.info(f"[REFRESH {rid}] result={_preview(result)} ms={(time.time()-t0)*1000:.1f}")
    _log_tool_output("rag_refresh", rid, result)
    return {"result": result}

@app.get("/rag/list")
def rag_list(folder: str = ""):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    log.info(f"[LIST {rid}] folder_param={folder!r}")
    base = Path(folder or RAG_PDF_DIR).expanduser().resolve()
    if not base.exists():
        msg = f"Error: folder not found: {base}"
        log.warning(f"[LIST {rid}] missing folder={base}")
        _log_tool_output("rag_list_pdfs", rid, msg)
        return {"result": msg}
    pdfs = sorted(str(p) for p in base.rglob("*.pdf"))
    out = "\n".join(pdfs) if pdfs else "(no PDFs found)"
    log.info(f"[LIST {rid}] count={len(pdfs)} ms={(time.time()-t0)*1000:.1f}")
    _log_tool_output("rag_list_pdfs", rid, out)
    return {"result": out}

@app.post("/rag/search")
def rag_search(req: SearchRequest):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    log.info(f"[SEARCH {rid}] query={req.query!r} k={req.k}")
    vs = _load_vs()
    if vs is None:
        log.info(f"[SEARCH {rid}] no_index -> rebuild")
        build_msg = _build_index(RAG_PDF_DIR)
        vs = _load_vs()
        if vs is None:
            msg = f"No index. {build_msg}"
            log.warning(f"[SEARCH {rid}] rebuild_failed")
            _log_tool_output("rag_search", rid, msg)
            return {"result": msg}
    retr = vs.as_retriever(search_kwargs={"k": req.k})
    docs = retr.invoke(req.query)
    if not docs:
        msg = "No relevant chunks found."
        log.info(f"[SEARCH {rid}] no_hits ms={(time.time()-t0)*1000:.1f}")
        _log_tool_output("rag_search", rid, msg)
        return {"result": msg}
    lines: List[str] = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        text = (d.page_content or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"[{i}] {src}#p{page}: {text}")
    result = "\n".join(lines)
    log.info(f"[SEARCH {rid}] hits={len(lines)} ms={(time.time()-t0)*1000:.1f}")
    _log_tool_output("rag_search", rid, result)
    return {"result": result}

@app.post("/shell")
def shell_cmd(req: ShellRequest):
    import subprocess
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    log.info(f"[SHELL {rid}] command={req.command!r}")
    allowed = {"ls", "pwd", "df", "echo"}
    parts = req.command.strip().split()
    if not parts:
        msg = "Error: empty command"
        log.warning(f"[SHELL {rid}] empty")
        _log_tool_output("shell_cmd", rid, msg)
        return {"result": msg}
    if parts[0] not in allowed:
        msg = f"Error: command '{parts[0]}' not allowed"
        log.warning(f"[SHELL {rid}] blocked cmd={parts[0]}")
        _log_tool_output("shell_cmd", rid, msg)
        return {"result": msg}
    try:
        proc = subprocess.run(req.command, shell=True, capture_output=True, text=True, check=True)
        out = proc.stdout.strip() or "Command executed successfully."
        log.info(f"[SHELL {rid}] ok rc=0 bytes={len(out)} ms={(time.time()-t0)*1000:.1f}")
        _log_tool_output("shell_cmd", rid, out)
        return {"result": out}
    except subprocess.CalledProcessError as e:
        msg = f"Error: rc={e.returncode} stderr={e.stderr.strip()}"
        log.error(f"[SHELL {rid}] fail rc={e.returncode} stderr_len={len(e.stderr or '')} ms={(time.time()-t0)*1000:.1f}")
        _log_tool_output("shell_cmd", rid, msg)
        return {"result": msg}

if __name__ == "__main__":
    uvicorn.run("tool_server:app", host="0.0.0.0", port=int(os.getenv("TOOLS_PORT", "8000")))