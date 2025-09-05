import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
from datetime import datetime
from unstructured.partition.auto import partition
from ..models.document import DocumentChunk, Document
from .chunking import DocumentChunker
from .embedding import EmbeddingService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing including chunking, embedding, and storage."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[DocumentChunker] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """Initialize the document processor."""
        self.vector_store = vector_store or VectorStore()
        self.chunker = chunker or DocumentChunker()
        self.embedding_service = embedding_service or EmbeddingService()
    
    async def process_text(
        self,
        text: str,
        source: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Process a text document and store it in the vector database."""
        logger.info(f"Processing document from source: {source}")
        
        document_id = (metadata or {}).get('document_id') or self._generate_document_id(source, title)
        
        if metadata is None:
            metadata = {}
        metadata['document_id'] = document_id
        
        chunks = self.chunker.create_document_chunks(
            text=text,
            source=source,
            title=title,
            metadata=metadata.copy()
        )
        
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        await self.vector_store.upsert_chunks(chunks)
        
        document = Document(
            id=document_id,
            metadata={
                "source": source,
                "title": title,
                "processed_at": datetime.utcnow().isoformat(),
                "chunk_count": len(chunks),
                "document_id": document_id,
                **(metadata or {})
            },
            chunks=chunks
        )
        
        logger.info(f"Processed document with {len(chunks)} chunks")
        return document
    
    async def process_file(
        self,
        file_path: str,
        source: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Process a file using unstructured and store its contents."""
        path = Path(file_path)
        source = source or str(path.absolute())
        title = title or path.name
        
        logger.info(f"Processing file {file_path} with unstructured.")
        
        try:
            # Use unstructured to partition the document
            elements = partition(filename=file_path)
            text = "\n\n".join([str(el) for el in elements])
            logger.info(f"Extracted {len(text)} characters from {source}")
        except Exception as e:
            logger.error(f"Failed to process file {file_path} with unstructured: {e}")
            # As a fallback, try reading as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as fallback_e:
                logger.error(f"Fallback to read as plain text also failed: {fallback_e}")
                raise e from fallback_e

        # Generate document ID
        document_id = self._generate_document_id(source, title)
        
        # Process the extracted text
        return await self.process_text(
            text=text,
            source=source,
            title=title,
            metadata={
                **(metadata or {}),
                "document_id": document_id
            }
        )
    
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Query the vector database for relevant document chunks."""
        query_embedding = self.embedding_service.embed_texts([query_text])[0]
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        return results
    
    async def delete_document(self, source: str) -> bool:
        """Delete all chunks for a document by its source."""
        return await self.vector_store.delete_by_metadata({"source": source})
    
    @staticmethod
    def _generate_document_id(source: str, title: Optional[str] = None) -> str:
        """Generate a unique document ID based on source and title."""
        id_str = f"{source}:{title}" if title else source
        return hashlib.sha256(id_str.encode()).hexdigest()
