from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DOCCHAT_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
INDEX_NAME = os.getenv("PINECONE_INDEX", "docchat-index")
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
TOP_K = 5

app = FastAPI(title="Doc-Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="docchat")
model_lock = threading.Lock()
embedding_lock = threading.Lock()
model: SentenceTransformer | None = None
embedding_device: Any = "cpu"
pinecone_indexes: dict[str, Any] = {}
pinecone_client: Pinecone | None = None
gemini_client: genai.Client | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    index_name: str = Field(min_length=1, max_length=45)


class IndexRequest(BaseModel):
    index_name: str = Field(min_length=1, max_length=45)


def get_documents() -> dict[str, dict[str, Any]]:
    if not DOCUMENTS_FILE.exists():
        return {}
    try:
        return json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_documents(documents: dict[str, dict[str, Any]]) -> None:
    temporary = DOCUMENTS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(DOCUMENTS_FILE)


def get_model() -> SentenceTransformer:
    global model, embedding_device
    if model is None:
        with model_lock:
            if model is None:
                device = "cpu"
                if EMBEDDING_DEVICE not in {"auto", "cpu", "directml"}:
                    raise ValueError("EMBEDDING_DEVICE must be 'auto', 'cpu', or 'directml'.")
                if EMBEDDING_DEVICE != "cpu":
                    try:
                        import torch_directml

                        device = torch_directml.device()
                    except Exception as error:
                        if EMBEDDING_DEVICE == "directml":
                            raise RuntimeError(
                                "DirectML was requested but is unavailable. Install the Windows "
                                "requirements and use a DirectX 12 compatible GPU with a current driver."
                            ) from error
                        print(f"DirectML unavailable ({error}); using CPU for embeddings.")
                print(f"Loading embedding model {EMBEDDING_MODEL} on {device}.")
                model = SentenceTransformer(EMBEDDING_MODEL, device=device)
                embedding_device = device
    return model


def encode_texts(inputs: str | list[str], **kwargs: Any) -> Any:
    """Encode safely; DirectML's inference-mode version-counter limitation falls back to CPU."""
    global embedding_device, model
    with embedding_lock:
        active_model = get_model()
        try:
            return active_model.encode(inputs, **kwargs)
        except RuntimeError as error:
            if "version_counter for inference tensor" not in str(error).lower():
                raise
            device_type = getattr(getattr(active_model, "device", None), "type", "")
            if device_type != "privateuseone":
                raise
            print("DirectML cannot run this Sentence Transformers inference mode; retrying embeddings on CPU.")
            try:
                active_model.to("cpu")
            except Exception:
                active_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
                model = active_model
            embedding_device = "cpu"
            return active_model.encode(inputs, **kwargs)


def get_pinecone_client() -> Pinecone:
    global pinecone_client
    if pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise HTTPException(503, "Set PINECONE_API_KEY in your .env file to enable document search.")
        pinecone_client = Pinecone(api_key=api_key)
    return pinecone_client


def get_index(index_name: str, create: bool = False) -> Any:
    validate_index_name(index_name)
    client = get_pinecone_client()
    if index_name not in pinecone_indexes:
        if not client.has_index(index_name):
            if not create:
                raise HTTPException(404, f"Pinecone index '{index_name}' was not found. Upload a PDF to create it.")
            client.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1")),
            )
        pinecone_indexes[index_name] = client.Index(index_name)
    return pinecone_indexes[index_name]


def validate_index_name(index_name: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", index_name):
        raise HTTPException(422, "Index names must be 1–45 lowercase letters, numbers, or hyphens, and start with a letter or number.")


def get_gemini() -> genai.Client:
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(503, "Set GEMINI_API_KEY in your .env file to enable AI answers.")
        gemini_client = genai.Client(api_key=api_key)
    return gemini_client


def split_text(text: str, page_number: int, max_size: int = 1100) -> list[dict[str, Any]]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict[str, Any]] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_size:
            if current:
                chunks.append({"text": current.strip(), "page": page_number})
                current = ""
            chunks.append({"text": paragraph[:max_size].strip(), "page": page_number})
            paragraph = paragraph[max_size:]
        if len(current) + len(paragraph) + 2 > max_size and current:
            chunks.append({"text": current.strip(), "page": page_number})
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append({"text": current, "page": page_number})
    return chunks


def index_pdf(path: Path, document_id: str, index_name: str) -> dict[str, int]:
    try:
        reader = PdfReader(str(path))
        chunks: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            chunks.extend(split_text(page_text, page_number))
        if not chunks:
            raise ValueError("No selectable text was found. This PDF may contain scanned images only.")
        embeddings = encode_texts([chunk["text"] for chunk in chunks], batch_size=32, show_progress_bar=False, normalize_embeddings=True).tolist()
        index = get_index(index_name, create=True)
        vectors = [
            {"id": f"{document_id}-{number}", "values": embedding, "metadata": {"text": chunk["text"], "page": chunk["page"]}}
            for number, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        for start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[start : start + 100], namespace=document_id)
        return {"pages": len(reader.pages), "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as error:
        path.unlink(missing_ok=True)
        raise ValueError(str(error)) from error


@app.get("/")
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, str]:
    return {"default_index": INDEX_NAME}


@app.post("/api/index/select")
def select_index(request: IndexRequest) -> dict[str, Any]:
    index = get_index(request.index_name)
    stats = index.describe_index_stats()
    return {"index_name": request.index_name, "exists": True, "vector_count": stats.get("total_vector_count", 0)}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return sorted(get_documents().values(), key=lambda item: item["created_at"], reverse=True)


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), index_name: str = Form(...)) -> dict[str, Any]:
    validate_index_name(index_name)
    # An existing index already holds the content and vectors; activate it without reprocessing this PDF.
    if get_pinecone_client().has_index(index_name):
        index = get_index(index_name)
        stats = index.describe_index_stats()
        return {"existing": True, "index_name": index_name, "vector_count": stats.get("total_vector_count", 0)}
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Upload a PDF file.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDFs must be 30 MB or smaller.")
    if not payload.startswith(b"%PDF-"):
        raise HTTPException(415, "This file does not appear to be a valid PDF.")
    document_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{document_id}.pdf"
    path.write_bytes(payload)
    try:
        details = await asyncio.get_running_loop().run_in_executor(executor, index_pdf, path, document_id, index_name)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    document = {
        "existing": False,
        "index_name": index_name,
        "id": document_id,
        "name": filename,
        "size": len(payload),
        "pages": details["pages"],
        "chunks": details["chunks"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    documents = get_documents()
    documents[document_id] = document
    save_documents(documents)
    return document


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, bool]:
    documents = get_documents()
    if document_id not in documents:
        raise HTTPException(404, "Document not found.")
    document = documents[document_id]
    try:
        await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: get_index(document.get("index_name", INDEX_NAME)).delete(delete_all=True, namespace=document_id),
        )
    except HTTPException:
        raise
    except Exception:
        pass
    (UPLOAD_DIR / f"{document_id}.pdf").unlink(missing_ok=True)
    del documents[document_id]
    save_documents(documents)
    return {"deleted": True}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    def answer_question() -> dict[str, Any]:
        query_vector = encode_texts(request.question, normalize_embeddings=True).tolist()
        index = get_index(request.index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces") or {"": {"vector_count": 0}}
        matches = []
        for namespace, namespace_stats in namespaces.items():
            if namespace_stats.get("vector_count", 0) == 0:
                continue
            result = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True, namespace=namespace)
            matches.extend(result.get("matches", []))
        matches = [match for match in matches if (match.get("metadata") or {}).get("text")]
        matches.sort(key=lambda match: match.get("score", 0), reverse=True)
        matches = matches[:TOP_K]
        if not matches:
            raise ValueError("No text passages were found in this index. It must contain vectors with text metadata from Doc-Chat.")
        def page_number(metadata: dict[str, Any]) -> int | None:
            page = metadata.get("page")
            if isinstance(page, bool):
                return None
            try:
                number = int(page)
            except (TypeError, ValueError):
                return None
            return number if number > 0 and str(page).strip() == str(number) else None

        passages = []
        for match in matches:
            metadata = match["metadata"]
            page = page_number(metadata)
            label = f"[Page {page}] " if page is not None else ""
            passages.append(f"{label}{metadata.get('text', '')}")
        context = "\n\n---\n\n".join(passages)
        prompt = (
            "You are Doc-Chat, an accurate document question-answering assistant.\n"
            "Use only the document excerpts below for factual claims. Do not add outside knowledge or invent details.\n"
            "Answer the user's question directly. Make the response comprehensive enough to cover the relevant evidence, "
            "while avoiding repetition and unrelated details.\n\n"
            "Output and structure:\n"
            "- Use clean Markdown with a short, descriptive heading when it improves readability.\n"
            "- For requests to summarize or explain a broad topic, organize the response into a brief overview, key points, "
            "and relevant implications, examples, or steps when the excerpts support them.\n"
            "- For a narrow question, give the direct answer first and add concise supporting details as needed.\n"
            "- Use bullets or numbered steps for separate points or procedures. Define important terms and distinguish related ideas.\n"
            "- If the excerpts do not provide enough information, state what is missing and answer only what they support.\n\n"
            "Citation rules:\n"
            "- Cite a passage only when its excerpt includes a valid numeric page label, using exactly [p. N] with that page number.\n"
            "- Never guess a page number. When a passage has no valid page number, omit its citation entirely.\n"
            "- Never output placeholders such as [Page ?], [p. ?], or any unknown-page citation.\n\n"
            f"Document excerpts:\n{context}\n\nUser question: {request.question}"
        )
        response = get_gemini().models.generate_content(model=GEMINI_MODEL, contents=prompt)
        answer = response.text or "I couldn't generate an answer for that question."
        answer = re.sub(r"\s*\[(?:page|p\.)\s*\?\]\s*", " ", answer, flags=re.IGNORECASE)
        answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
        sources = []
        seen_pages: set[int] = set()
        for match in matches:
            page = page_number(match["metadata"])
            if page is None:
                continue
            if page not in seen_pages:
                seen_pages.add(page)
                sources.append({"page": page, "preview": match["metadata"].get("text", "")[:240]})
        return {"answer": answer, "sources": sources}

    try:
        return await asyncio.get_running_loop().run_in_executor(executor, answer_question)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(502, f"Could not answer this question: {error}") from error


