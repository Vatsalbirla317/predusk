import os
import logging
import tempfile
from uuid import uuid4
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import uvicorn
from pathlib import Path

from app.config import get_settings
from app.services.rag_service import RAGService
from app.services.document_processor import DocumentProcessor
from app.models.response import ChatResponse, DocumentIngestResponse
from app.models.document import DocumentChunkWithScore
from app.services.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predusk RAG API",
    version="1.0.0",
    description="A FastAPI application with RAG capabilities using Groq and Pinecone"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Mount static files
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The message to send to the AI")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of previous messages in the conversation"
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to limit the search scope"
    )
    max_tokens: Optional[int] = Field(1000, gt=0, le=4000, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

# Initialize services
settings = get_settings()
try:
    rag_service = RAGService()
    document_processor = DocumentProcessor()
    vector_store = VectorStore()
    logger.info("Services initialized successfully")
    if settings.COHERE_API_KEY:
        logger.info("Cohere reranker is enabled")
    else:
        logger.warning("Cohere API key not found. Reranking will be disabled.")

except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    raise

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main chat interface."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Predusk RAG Chat"}
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Handle chat requests with RAG capabilities.
    """
    try:
        filters = None
        if chat_request.document_id:
            filters = {"document_id": chat_request.document_id}

        response = await rag_service.generate_response(
            query=chat_request.message,
            chat_history=chat_request.chat_history,
            top_k=5,
            filters=filters,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens
        )
        return response

    except Exception as e:
        logger.error(f"Error in RAG chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/api/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to process"),
    title: Optional[str] = Form(None, description="Document title"),
    metadata: Optional[str] = Form(
        None,
        description="Additional metadata as JSON string",
        example=json.dumps({"author": "John Doe", "category": "research"})
    )
):
    """
    Upload and process a document for the RAG system.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            doc_metadata = json.loads(metadata) if metadata else {}
            document = await document_processor.process_file(
                file_path=temp_file_path,
                source=file.filename,
                title=title or file.filename,
                metadata=doc_metadata
            )
            return DocumentIngestResponse(
                document_id=document.id,
                chunks_processed=len(document.chunks),
                total_tokens=sum(chunk.token_count for chunk in document.chunks),
                processing_time=0.0,  # Placeholder
                metadata={
                    "source": file.filename,
                    "title": title or file.filename,
                    **doc_metadata
                }
            )
        finally:
            os.unlink(temp_file_path)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata format. Must be a valid JSON object.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources/{document_id}", response_model=List[DocumentChunkWithScore])
async def get_document_sources(
    document_id: str,
    limit: int = Query(1000, description="Maximum number of sources to return"),
):
    """Retrieve all sources/chunks for a specific document."""
    try:
        chunks = await vector_store.fetch_by_metadata(
            filters={"document_id": document_id},
            top_k=limit
        )
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks from the system.
    """
    try:
        success = await vector_store.delete_chunks(document_id=document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted.")
        return {"status": "success", "message": f"Document {document_id} and its chunks deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
