import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from ..config import get_settings
from ..models.document import DocumentChunk, DocumentChunkWithScore

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorStore:
    """Pinecone vector store for document chunks."""
    
    def __init__(self):
        """Initialize the Pinecone client and index."""
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = settings.EMBEDDING_DIM
        self.metric = "cosine"  # Distance metric for similarity search
        self.pinecone_client = None
        
        # Initialize Pinecone client
        self._init_pinecone()
        
        # Get or create the index
        self.index = self._get_or_create_index()
    
    def _init_pinecone(self):
        """Initialize the Pinecone client."""
        if not self.api_key:
            raise ValueError("Pinecone API key must be set")
        
        try:
            self.pinecone_client = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise
    
    def _get_or_create_index(self):
        """Get an existing index or create a new one if it doesn't exist."""
        try:
            # Check if index exists
            if self.index_name in self.pinecone_client.list_indexes().names():
                logger.info(f"Using existing index: {self.index_name}")
                return self.pinecone_client.Index(self.index_name)
            
            # Create new index if it doesn't exist
            logger.info(f"Creating new index: {self.index_name}")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
            logger.info(f"Index {self.index_name} created successfully")
            return self.pinecone_client.Index(self.index_name)
        except Exception as e:
            logger.error(f"Failed to get or create index: {str(e)}")
            raise
    
    async def upsert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Upsert document chunks into the vector store."""
        if not chunks:
            return False
        try:
            vectors = []
            for chunk in chunks:
                # Build metadata dict from chunk attributes
                metadata = {
                    'text': chunk.text,
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'title': getattr(chunk, 'title', ''),
                    'source': getattr(chunk, 'source', ''),
                    'page': getattr(chunk, 'page', None),
                    'section': getattr(chunk, 'section', None),
                    'created_at': getattr(chunk, 'created_at', None),
                }
                # Add any custom_metadata fields if present
                if hasattr(chunk, 'custom_metadata') and chunk.custom_metadata:
                    metadata.update(chunk.custom_metadata)
                vector = (
                    chunk.id,  # ID
                    chunk.embedding,  # Vector
                    metadata
                )
                vectors.append(vector)
            # Upsert in batches of 100 (Pinecone's recommended batch size)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            logger.info(f"Upserted {len(chunks)} chunks into index {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error upserting chunks: {str(e)}")
            return False

    async def fetch_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 1000
    ) -> List[DocumentChunkWithScore]:
        """Fetch all chunks matching metadata filters by using a dummy query."""
        try:
            # Use a dummy vector for querying, as we only care about the metadata filter.
            # This is a common pattern for metadata-based fetching in Pinecone.
            dummy_vector = [0.0] * self.dimension
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=top_k, # Fetch a large number to get all results
                include_metadata=True,
                filter=filters
            )
            
            chunks = []
            for match in results.matches:
                metadata = match.metadata or {}
                chunk = DocumentChunkWithScore(
                    id=match.id,
                    text=metadata.get('text', ''),
                    score=match.score,
                    document_id=metadata.get('document_id', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    source=metadata.get('source', ''),
                    title=metadata.get('title', ''),
                    custom_metadata={k: v for k, v in metadata.items() if k not in ['text', 'document_id', 'chunk_index', 'title', 'source']}
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Error fetching chunks by metadata: {str(e)}")
            return []

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunkWithScore]:
        """Search for similar document chunks."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace='default',
                filter=filters
            )
            
            chunks = []
            for match in results.matches:
                metadata = match.metadata or {}
                chunk = DocumentChunkWithScore(
                    id=match.id,
                    text=metadata.get('text', ''),
                    score=match.score,
                    document_id=metadata.get('document_id', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    source=metadata.get('source', ''),
                    title=metadata.get('title', ''),
                    custom_metadata={k: v for k, v in metadata.items() if k not in ['text', 'document_id', 'chunk_index', 'title', 'source']}
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []
    
    async def delete_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            self.index.delete(
                filter={"document_id": {"$eq": document_id}}
            )
            logger.info(f"Deleted all chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
            return False
    
    async def delete_by_metadata(self, filter_dict: Dict[str, Any]) -> bool:
        """Delete chunks matching the given metadata filters."""
        if not filter_dict:
            logger.warning("No filters provided, not deleting anything")
            return False
            
        try:
            self.index.delete(
                filter=filter_dict
            )
            logger.info(f"Deleted chunks matching filters: {filter_dict}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks with filters {filter_dict}: {str(e)}")
            return False