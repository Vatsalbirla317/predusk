import os
import logging
import time # Import time
from typing import List, Dict, Any, Optional, Tuple # Add Tuple to imports
import cohere
from ..models.document import DocumentChunkWithScore
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class CohereReranker:
    """Reranks search results using Cohere's rerank API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v2.0",
        top_n: Optional[int] = None
    ):
        """Initialize the Cohere reranker.
        
        Args:
            api_key: Cohere API key. If not provided, will try to get from settings.
            model: The reranker model to use.
            top_n: Number of results to return after reranking.
        """
        self.api_key = api_key or settings.COHERE_API_KEY
        if not self.api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY in your environment or pass it directly.")
            
        self.model = model
        self.top_n = top_n
        self.client = cohere.Client(self.api_key)
    
    async def rerank(
        self,
        query: str,
        documents: List[DocumentChunkWithScore],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        **kwargs
    ) -> Tuple[List[DocumentChunkWithScore], float]: # Modified return type
        """Rerank a list of documents based on their relevance to the query.
        
        Args:
            query: The search query.
            documents: List of document chunks to rerank.
            top_n: Number of results to return after reranking.
            return_documents: Whether to return full document chunks or just scores.
            **kwargs: Additional arguments to pass to the Cohere rerank API.
            
        Returns:
            Tuple of (List of reranked document chunks with updated scores, reranking_time).
        """
        if not documents:
            return [], 0.0 # Return empty list and 0.0 time
            
        top_n = top_n or self.top_n or len(documents)
        
        # Convert documents to format expected by Cohere
        doc_texts = [doc.text for doc in documents]
        
        start_time = time.perf_counter() # Start timing
        
        try:
            # Call Cohere's rerank API
            results = self.client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(documents)),
                model=self.model,
                **kwargs
            )
            
            end_time = time.perf_counter() # End timing
            reranking_time = end_time - start_time
            
            # Create a mapping of text to document for easy lookup
            text_to_doc = {doc.text: doc for doc in documents}
            
            # Create new list of reranked documents with updated scores
            reranked_docs = []
            for result in results:
                original_doc = text_to_doc[result.document['text']]
                
                # Create a new document with the updated score
                reranked_doc = DocumentChunkWithScore(
                    **original_doc.dict(exclude={'score'}),  # Copy all fields except score
                    score=float(result.relevance_score)
                )
                reranked_docs.append(reranked_doc)
                
            return reranked_docs, reranking_time # Return reranked docs and time
            
        except Exception as e:
            logger.error(f"Error in Cohere reranking: {str(e)}")
            # In case of error, return original documents with a warning and 0.0 time
            logger.warning("Returning original documents due to reranking error")
            return documents[:top_n] if top_n else documents, 0.0 # Return original docs and 0.0 time
