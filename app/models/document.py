from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, UUID4
from uuid import uuid4

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk."""
    source: str = Field(..., description="Source of the document (e.g., filename, URL)")
    title: Optional[str] = Field(None, description="Title of the document")
    page: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Section of the document")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was processed")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DocumentChunk(DocumentMetadata):
    """A chunk of a document with vector embedding and metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    token_count: int = Field(..., description="Number of tokens in the chunk")
    chunk_index: int = Field(..., description="Index of this chunk in the document")

class DocumentChunkWithScore(DocumentChunk):
    """A document chunk with its relevance score."""
    score: float = Field(..., description="Relevance score of the chunk")

class Document(BaseModel):
    """A document with its chunks and metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the document")
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of document chunks")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was processed")
