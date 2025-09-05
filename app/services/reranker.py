import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from ..config import get_settings
from .cohere_reranker import CohereReranker as CohereRerankerImpl
from ..models.document import DocumentChunkWithScore

logger = logging.getLogger(__name__)
settings = get_settings()

class BaseReranker:
    """Base class for rerankers."""
    
    async def rerank(
        self,
        query: str,
        documents: List[Union[str, DocumentChunkWithScore]],
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Union[DocumentChunkWithScore, Dict]]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: The query text
            documents: List of document texts or DocumentChunkWithScore objects to rerank
            top_n: Number of results to return (None for all)
            **kwargs: Additional arguments for the reranker
            
        Returns:
            List of reranked results with relevance scores
        """
        raise NotImplementedError("Subclasses must implement rerank method")

class CohereReranker(BaseReranker):
    """Reranker using Cohere's rerank API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Cohere reranker.
        
        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY from settings)
            model: Reranker model to use (defaults to settings.RERANKER_MODEL)
        """
        self.impl = CohereRerankerImpl(
            api_key=api_key or settings.COHERE_API_KEY,
            model=model or settings.RERANKER_MODEL
        )

    async def rerank(
        self,
        query: str,
        documents: List[Union[str, DocumentChunkWithScore]],
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Union[DocumentChunkWithScore, Dict]]:
        """Rerank documents using Cohere's rerank API.
        
        Args:
            query: The query text
            documents: List of document texts or DocumentChunkWithScore objects to rerank
            top_n: Number of results to return (None for all)
            **kwargs: Additional arguments for the reranker
            
        Returns:
            List of reranked results with relevance scores
        """
        return await self.impl.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            **kwargs
        )

class DummyReranker(BaseReranker):
    """Dummy reranker for testing and fallback purposes."""
    
    async def rerank(
        self,
        query: str,
        documents: List[Union[str, DocumentChunkWithScore]],
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Union[DocumentChunkWithScore, Dict]]:
        """Return documents in their original order with dummy scores."""
        if not documents:
            return []
            
        top_n = min(top_n or len(documents), len(documents))
        results = documents[:top_n]
        
        # If we have DocumentChunkWithScore objects, update their scores
        if results and isinstance(results[0], DocumentChunkWithScore):
            for doc in results:
                doc.score = 1.0  # type: ignore
        
        return results

def get_reranker(use_cohere: bool = True) -> BaseReranker:
    """Get a reranker instance.
    
    Args:
        use_cohere: Whether to use Cohere reranker if available
        
    Returns:
        A reranker instance (CohereReranker or DummyReranker)
    """
    if use_cohere and settings.COHERE_API_KEY:
        return CohereReranker()
    return DummyReranker()
