import os
import re
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple

# Document parsers
import pypdf
import docx

# Embeddings (free, local)
from sentence_transformers import SentenceTransformer


# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks
EMBED_MODEL   = "all-MiniLM-L6-v2"   # ~80 MB, free, runs on CPU
TOP_K         = 5     # number of chunks to retrieve per query


# ── RAG Engine ─────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        print("[RAG] Loading embedding model …")
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.dim        = self.embedder.get_sentence_embedding_dimension()
        self.index      = faiss.IndexFlatL2(self.dim)
        self.chunks: List[Dict] = []   # {text, source, page}
        print(f"[RAG] Embedding model ready (dim={self.dim})")

    # ── File parsing ────────────────────────────────────────────────────────────
    def parse_pdf(self, path: str) -> List[Dict]:
        pages = []
        reader = pypdf.PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i})
        return pages

    def parse_docx(self, path: str) -> List[Dict]:
        doc   = docx.Document(path)
        pages = []
        page_num   = 1
        page_lines: List[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # treat every ~30 paragraphs as a logical "page"
            page_lines.append(text)
            if len(page_lines) >= 30:
                pages.append({"text": "\n".join(page_lines), "page": page_num})
                page_num  += 1
                page_lines = []

        if page_lines:
            pages.append({"text": "\n".join(page_lines), "page": page_num})

        return pages

    def parse_txt(self, path: str) -> List[Dict]:
        text  = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        pages = []
        page_num  = 1
        page_lines: List[str] = []

        for line in lines:
            page_lines.append(line)
            if len(page_lines) >= 50:
                pages.append({"text": "\n".join(page_lines), "page": page_num})
                page_num  += 1
                page_lines = []

        if page_lines:
            pages.append({"text": "\n".join(page_lines), "page": page_num})

        return pages

    def parse_file(self, path: str) -> List[Dict]:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(path)
        elif ext == ".docx":
            return self.parse_docx(path)
        elif ext == ".txt":
            return self.parse_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── Chunking ────────────────────────────────────────────────────────────────
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start  = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Indexing ────────────────────────────────────────────────────────────────
    def add_document(self, path: str, filename: str) -> int:
        """Parse, chunk, embed, and index a document. Returns chunk count added."""
        pages  = self.parse_file(path)
        new_chunks: List[Dict] = []

        for page_data in pages:
            for chunk_text in self.chunk_text(page_data["text"]):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 30:   # skip near-empty chunks
                    continue
                new_chunks.append({
                    "text":   chunk_text,
                    "source": filename,
                    "page":   page_data["page"],
                })

        if not new_chunks:
            return 0

        texts      = [c["text"] for c in new_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

        print(f"[RAG] Indexed '{filename}': {len(new_chunks)} chunks")
        return len(new_chunks)

    def remove_document(self, filename: str):
        """Remove all chunks belonging to a document and rebuild the index."""
        self.chunks = [c for c in self.chunks if c["source"] != filename]
        self._rebuild_index()

    def _rebuild_index(self):
        self.index = faiss.IndexFlatL2(self.dim)
        if not self.chunks:
            return
        texts      = [c["text"] for c in self.chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        self.index.add(np.array(embeddings, dtype="float32"))

    # ── Retrieval ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Return the top-k most relevant chunks for a query."""
        if self.index.ntotal == 0:
            return []

        q_emb = self.embedder.encode([query])
        q_emb = np.array(q_emb, dtype="float32")

        k         = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)

        return results

    # ── Answer generation (free, no API) ─────────────────────────────────────────
    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """
        Build a grounded answer purely from retrieved chunks.
        No external LLM API needed — uses extractive + abstractive heuristics.
        """
        if not chunks:
            return "No relevant information found in the uploaded documents."

        # Deduplicate by source+page
        seen    = set()
        unique  = []
        for c in chunks:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # Build context block
        context_parts = []
        for c in unique:
            context_parts.append(
                f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Extract sentences that are most relevant to query keywords
        query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        stop_words  = {"what","is","are","the","a","an","of","in","on","at",
                       "to","for","with","how","why","when","where","who","does","do"}
        query_words -= stop_words

        candidate_sentences: List[Tuple[float, str, str, int]] = []

        for c in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", c["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                sent_words = set(re.sub(r"[^\w\s]", "", sent.lower()).split())
                if not query_words:
                    score = 1.0
                else:
                    score = len(query_words & sent_words) / len(query_words)
                candidate_sentences.append((score, sent, c["source"], c["page"]))

        candidate_sentences.sort(key=lambda x: -x[0])
        top_sentences = candidate_sentences[:5]

        if not top_sentences or top_sentences[0][0] == 0:
            # fallback: return first chunk's opening
            first = chunks[0]
            answer_body = first["text"][:600]
            sources_used = [(first["source"], first["page"])]
        else:
            answer_body  = " ".join(s[1] for s in top_sentences)
            sources_used = list({(s[2], s[3]) for s in top_sentences})

        # Format final answer
        sources_str = "\n".join(
            f"  • {src} (page {pg})" for src, pg in sorted(sources_used)
        )

        return (
            f"{answer_body}\n\n"
            f"**Sources:**\n{sources_str}"
        )

    # ── Stats ────────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        docs = {}
        for c in self.chunks:
            docs.setdefault(c["source"], set()).add(c["page"])
        return {
            "total_chunks": len(self.chunks),
            "documents": [
                {"name": name, "pages": max(pages)}
                for name, pages in docs.items()
            ],
        }