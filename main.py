import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGEngine

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Document RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static folder for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Single global RAG engine instance
engine = RAGEngine()


# ── Models ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    """Serve the frontend."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a document (PDF / DOCX / TXT)."""
    allowed = {".pdf", ".docx", ".txt"}
    ext     = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunk_count = engine.add_document(str(dest), file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Indexed successfully", "chunks": chunk_count, "filename": file.filename}


@app.delete("/document/{filename}")
def delete_document(filename: str):
    """Remove a document from the index."""
    path = UPLOAD_DIR / filename
    path.unlink(missing_ok=True)
    engine.remove_document(filename)
    return {"message": f"'{filename}' removed from index."}


@app.post("/query")
def query(req: QueryRequest):
    """Query across all indexed documents."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chunks = engine.retrieve(req.query, top_k=req.top_k)
    answer = engine.generate_answer(req.query, chunks)

    return {
        "answer":  answer,
        "sources": [
            {
                "source": c["source"],
                "page":   c["page"],
                "score":  round(c["score"], 4),
                "snippet": c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
            }
            for c in chunks
        ],
    }


@app.get("/stats")
def stats():
    """Return index statistics."""
    return engine.get_stats()


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    return engine.get_stats()["documents"]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)