import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import tiktoken
from ..config import get_settings
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class TextChunk:
    """A chunk of text with its metadata."""
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = None

class DocumentChunker:
    """Handles splitting text into chunks with optional overlap."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        tokenizer: str = "cl100k_base"  # Default to tiktoken's tokenizer for GPT
    ):
        """Initialize the document chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer: Name of the tokenizer to use
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Validate chunk size and overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be smaller than chunk size ({self.chunk_size})"
            )
        
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer)
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer}: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(self.tokenizer.encode(text))
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Split text into chunks with optional overlap.
        
        Args:
            text: The text to split into chunks
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # Initialize variables
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        # Split text into sentences (naive approach, can be improved)
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Get token count for this sentence
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed the chunk size, finalize the current chunk
            if current_length + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata or {}
                ))
                
                # Start a new chunk with overlap
                overlap_text = " ".join(current_chunk[-self._get_num_sentences_for_overlap():])
                current_chunk = [overlap_text] if overlap_text else []
                current_length = self.count_tokens(overlap_text)
                start_pos = end_pos - len(overlap_text)
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _get_num_sentences_for_overlap(self) -> int:
        """Determine how many sentences to include in the overlap."""
        # This is a simple heuristic - you might want to adjust this
        return max(1, int(len(self.tokenizer.encode(" ".join([""] * 10))) * 2))
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences.
        
        This is a simple implementation and might need to be enhanced
        for production use (e.g., using NLTK, spaCy, etc.)
        """
        # Split on sentence boundaries (., !, ? followed by space and capital letter)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s for s in sentences if s.strip()]
    
    def create_document_chunks(
        self,
        text: str,
        source: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Create document chunks from text with metadata.
        
        Args:
            text: The text to chunk
            source: Source of the document
            title: Optional document title
            metadata: Additional metadata to include with each chunk
            document_id: Optional document ID to associate with all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []
            
        # Start with a clean metadata dictionary
        chunk_metadata = {}
        
        # Add standard fields
        chunk_metadata.update({
            "source": source,
            "title": title or "",
            "created_at": datetime.utcnow().isoformat(),
        })
        
        # Add document_id from either the parameter or the provided metadata
        if document_id:
            chunk_metadata['document_id'] = document_id
        elif metadata and 'document_id' in metadata:
            chunk_metadata['document_id'] = metadata['document_id']
            
        # Add any additional metadata, allowing it to override the standard fields
        if metadata:
            chunk_metadata.update(metadata)
        
        # Split text into chunks
        text_chunks = self.split_text(text, chunk_metadata)
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for i, chunk in enumerate(text_chunks):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue
                
            # Update chunk metadata with position info
            chunk_metadata.update({
                "chunk_index": i,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                **chunk.metadata
            })
            
            # Generate a unique ID for the chunk
            chunk_id = f"{document_id or source}_{i}" if document_id or source else str(i)
            
            # Create document chunk with document_id
            # Remove chunk_index from metadata to avoid duplicate parameter
            chunk_metadata.pop('chunk_index', None)
            
            # Ensure we have a valid document_id
            chunk_document_id = document_id or chunk_metadata.get('document_id')
            if not chunk_document_id:
                # If no document_id is provided, generate one from the source
                chunk_document_id = f'doc_{hash(source) & 0xffffffff}'
                
            # Create a clean metadata dictionary excluding special fields
            clean_metadata = {
                k: v for k, v in chunk_metadata.items() 
                if k not in ['document_id', 'chunk_index', 'id']
            }
            
            document_chunks.append(DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                document_id=chunk_document_id,  # Ensure document_id is always set
                token_count=self.count_tokens(chunk_text),
                chunk_index=i,  # Pass chunk_index as a direct parameter
                **clean_metadata
            ))
        
        return document_chunks
