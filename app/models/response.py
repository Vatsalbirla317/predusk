from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Citation(BaseModel):
    """A citation to a source document."""
    id: str = Field(..., description="Unique identifier for the citation")
    source: str = Field(..., description="Source of the citation")
    text: str = Field(..., description="The text being cited")
    page: Optional[int] = Field(None, description="Page number if applicable")
    score: float = Field(..., description="Relevance score of the citation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ChatMessage(BaseModel):
    """A message in the chat."""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")

class PerformanceMetrics(BaseModel):
    """Performance metrics for different stages of the RAG pipeline."""
    total_time: float = Field(..., description="Total processing time in seconds")
    retrieval_time: Optional[float] = Field(None, description="Time spent retrieving documents")
    reranking_time: Optional[float] = Field(None, description="Time spent reranking documents")
    generation_time: Optional[float] = Field(None, description="Time spent generating the response")
    post_processing_time: Optional[float] = Field(None, description="Time spent on post-processing")
    tokens_retrieved: Optional[int] = Field(None, description="Number of tokens in retrieved chunks")
    tokens_after_rerank: Optional[int] = Field(None, description="Number of tokens after reranking")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens in the generated response")
    tokens_prompt: Optional[int] = Field(None, description="Number of tokens in the prompt")
    chunks_retrieved: Optional[int] = Field(None, description="Number of chunks initially retrieved")
    chunks_after_rerank: Optional[int] = Field(None, description="Number of chunks after reranking")
    model: str = Field(..., description="Model used for generation")


class ChatResponse(BaseModel):
    """Response from the chat endpoint with performance metrics."""
    message_id: str = Field(..., description="Unique identifier for the response")
    content: str = Field(..., description="Generated response content")
    citations: List[Citation] = Field(default_factory=list, description="List of citations used in the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Performance metrics
    metrics: PerformanceMetrics = Field(..., description="Detailed performance metrics")

class DocumentIngestResponse(BaseModel):
    """Response after ingesting a document."""
    document_id: str = Field(..., description="ID of the processed document")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    total_tokens: int = Field(..., description="Total number of tokens in the document")
    processing_time: float = Field(..., description="Time taken to process the document in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
