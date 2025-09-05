import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from ..models.document import DocumentChunk, DocumentChunkWithScore
from .document_processor import DocumentProcessor
from .cohere_reranker import CohereReranker
from ..services.rag_service import calculate_token_count

logger = logging.getLogger(__name__)

class Retriever:
    """Handles document retrieval with optional reranking."""
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        reranker: Optional[Any] = None,
        top_k: int = 10,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ):
        """Initialize the retriever.
        
        Args:
            document_processor: Document processor instance
            reranker: Optional reranker instance (e.g., CohereReranker)
            top_k: Number of results to retrieve before reranking
            use_mmr: Whether to use Maximal Marginal Relevance for diversity
            mmr_lambda: Lambda parameter for MMR (0 = max diversity, 1 = max relevance)
        """
        self.document_processor = document_processor or DocumentProcessor()
        self.reranker = reranker
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: bool = True,
        rerank_top_k: Optional[int] = None
    ) -> Dict[str, Any]: # Modified return type to include metrics
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: The query text
            top_k: Number of results to retrieve from vector store (before reranking)
            filters: Optional filters to apply to the search
            use_reranker: Whether to use the reranker if available
            rerank_top_k: Number of results to return after reranking (if reranker is used)
            
        Returns:
            Dict containing list of relevant document chunks with scores and reranking metrics.
        """
        top_k = top_k or self.top_k
        
        reranking_time = 0.0
        chunks_after_rerank = 0
        tokens_after_rerank = 0

        # Get initial results from vector store
        results = await self.document_processor.query(
            query_text=query,
            top_k=top_k or self.top_k,
            filters=filters
        )

        # Fallback: If no results, try to fetch the first chunk of any document
        if not results:
            # Try to fetch the first chunk from the vector store (no filters, top_k=1)
            fallback_results = await self.document_processor.query(query_text="", top_k=1)
            if fallback_results:
                results = fallback_results

        # Apply MMR if enabled and we have multiple results
        if self.use_mmr and len(results) > 1:
            results = self._apply_mmr(query, results)

        # Apply reranker if available and enabled
        if use_reranker and self.reranker and results:
            try:
                reranked_results, reranking_time = await self.reranker.rerank( # Capture reranking_time
                    query=query,
                    documents=results,
                    top_n=rerank_top_k or len(results)
                )
                results = reranked_results # Update results with reranked ones
                chunks_after_rerank = len(results)
                tokens_after_rerank = sum(calculate_token_count(chunk.text) for chunk in results)
                logger.info(f"Reranked {len(results)} documents in {reranking_time:.4f} seconds")
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                # If reranking fails, return the original results
                results = results[:rerank_top_k] if rerank_top_k else results

        # Return top-k results and metrics
        return {
            "chunks": results[:top_k],
            "reranking_time": reranking_time,
            "chunks_after_rerank": chunks_after_rerank,
            "tokens_after_rerank": tokens_after_rerank
        }
    
    def _apply_mmr(
        self,
        query: str,
        chunks: List[DocumentChunkWithScore],
        top_k: int = None
    ) -> List[DocumentChunkWithScore]:
        """Apply Maximal Marginal Relevance to diversify results.
        
        Args:
            query: The query text
            chunks: List of document chunks with scores
            top_k: Number of results to return
            
        Returns:
            Re-ranked list of document chunks
        """
        if not chunks:
            return []
            
        # Convert chunks to a list of (chunk, score) tuples
        chunk_scores = [(chunk, chunk.score) for chunk in chunks]
        
        # Sort by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize selected chunks with the top-scoring chunk
        selected_chunks = [chunk_scores[0][0]]
        remaining_chunks = chunk_scores[1:]
        
        # While we haven't selected enough chunks and there are chunks remaining
        while len(selected_chunks) < min(top_k, len(chunks)) and remaining_chunks:
            # Calculate MMR scores for remaining chunks
            mmr_scores = []
            
            for chunk, score in remaining_chunks:
                # Calculate max similarity to already selected chunks
                max_similarity = max(
                    self._cosine_similarity(chunk.embedding, selected.embedding)
                    for selected in selected_chunks
                ) if selected_chunks else 0
                
                # Calculate MMR score
                mmr_score = (
                    self.mmr_lambda * score -
                    (1 - self.mmr_lambda) * max_similarity
                )
                mmr_scores.append((chunk, mmr_score, score))
            
            # Select chunk with highest MMR score
            selected_idx = np.argmax([score for _, score, _ in mmr_scores])
            selected_chunk, _, original_score = mmr_scores.pop(selected_idx)
            selected_chunk.score = original_score  # Keep original score for reference
            selected_chunks.append(selected_chunk)
            
            # Remove selected chunk from remaining chunks
            remaining_chunks = [
                (chunk, score) 
                for chunk, score, _ in mmr_scores
            ]
        
        return selected_chunks
    
    async def _rerank(
        self,
        query: str,
        chunks: List[DocumentChunkWithScore]
    ) -> List[DocumentChunkWithScore]:
        """Apply reranker to the retrieved chunks.
        
        Args:
            query: The query text
            chunks: List of document chunks with scores
            
        Returns:
            Re-ranked list of document chunks with updated scores
        """
        if not chunks or not self.reranker:
            return chunks
            
        try:
            # Extract text from chunks for reranking
            texts = [chunk.text for chunk in chunks]
            
            # Get reranked results
            reranked_results = await self.reranker.rerank(
                query=query,
                documents=texts,
                top_n=len(chunks)
            )
            
            # Update chunk scores based on reranker
            for i, result in enumerate(reranked_results):
                if result.index < len(chunks):
                    chunks[result.index].score = result.relevance_score
                    chunks[result.index].rerank_score = result.relevance_score
                    chunks[result.index].rank = i + 1
            
            # Sort chunks by reranker score (descending)
            chunks.sort(key=lambda x: getattr(x, 'rerank_score', x.score), reverse=True)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return chunks  # Return original chunks if reranking fails
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
            
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
