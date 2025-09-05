import logging
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx


def time_it() -> float:
    """Helper function to get current time in seconds with high precision."""
    return time.perf_counter()


def calculate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    Note: This is a rough estimation. For exact counts, use the tokenizer
    that matches the model being used.
    """
    # Rough approximation: 1 token ≈ 4 characters in English
    return max(1, len(text) // 4)
from ..config import get_settings
from ..models.document import DocumentChunkWithScore
from ..models.response import ChatResponse, Citation, PerformanceMetrics
from .retriever import Retriever
from .reranker import get_reranker
from .cohere_reranker import CohereReranker as CohereRerankerImpl

logger = logging.getLogger(__name__)
settings = get_settings()

class RAGService:
    """Service for Retrieval-Augmented Generation with citation support."""
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        cohere_model: Optional[str] = None
    ):
        """Initialize the RAG service.
        
        Args:
            retriever: Retriever instance for document retrieval
            groq_api_key: Groq API key (defaults to GROQ_API_KEY from settings)
            groq_model: Groq model to use (defaults to GROQ_MODEL from settings)
            cohere_api_key: Cohere API key for reranking (defaults to COHERE_API_KEY)
            cohere_model: Cohere reranker model (defaults to RERANKER_MODEL)
        """
        # Initialize reranker
        reranker = None
        if settings.COHERE_API_KEY or cohere_api_key:
            reranker = CohereRerankerImpl(
                api_key=cohere_api_key or settings.COHERE_API_KEY,
                model=cohere_model or settings.RERANKER_MODEL
            )
            
        # Initialize retriever with reranker
        self.retriever = retriever or Retriever(reranker=reranker)
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self.groq_model = groq_model or settings.GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = 30.0  # seconds
    
    async def generate_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **generation_kwargs
    ) -> ChatResponse:
        """Generate a response using RAG with citations and performance metrics.
        
        Args:
            query: The user's query
            chat_history: Optional list of previous messages in the conversation
            top_k: Number of document chunks to retrieve
            filters: Optional filters to apply to the retrieval
            **generation_kwargs: Additional arguments for text generation
            
        Returns:
            ChatResponse with the generated response, citations, and performance metrics
        """
        # Initialize metrics
        metrics = {
            'total_start': time_it(),
            'model': self.groq_model,
            'tokens_prompt': calculate_token_count(query)
        }
        
        try:
            # Step 1: Retrieve relevant document chunks
            metrics['retrieval_start'] = time_it()
            retrieval_results = await self.retriever.retrieve( # Modified to capture dict
                query=query,
                top_k=top_k,
                filters=filters
            )
            chunks = retrieval_results["chunks"]
            metrics['retrieval_time'] = time_it() - metrics['retrieval_start']
            metrics['chunks_retrieved'] = len(chunks)
            metrics['tokens_retrieved'] = sum(calculate_token_count(chunk.text) for chunk in chunks)
            
            # Add reranking metrics
            metrics['reranking_time'] = retrieval_results.get('reranking_time', 0.0)
            metrics['chunks_after_rerank'] = retrieval_results.get('chunks_after_rerank', 0)
            metrics['tokens_after_rerank'] = retrieval_results.get('tokens_after_rerank', 0)

            # Step 2: Format context and citations
            metrics['formatting_start'] = time_it()
            context, citations = self._format_context_and_citations(chunks)
            metrics['formatting_time'] = time_it() - metrics['formatting_start']
            
            # Step 3: Generate response with citations
            metrics['generation_start'] = time_it()
            response_text, generation_metrics = await self._generate_with_groq(
                query=query,
                context=context,
                chat_history=chat_history,
                **generation_kwargs
            )
            metrics['generation_time'] = time_it() - metrics['generation_start']
            metrics.update(generation_metrics)
            
            # Step 4: Process citations in the response
            metrics['post_processing_start'] = time_it()
            response_text, used_citations = self._process_citations(response_text, citations)
            metrics['post_processing_time'] = time_it() - metrics['post_processing_start']
            
            # Calculate total time
            metrics['total_time'] = time_it() - metrics['total_start']
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                total_time=metrics['total_time'],
                retrieval_time=metrics.get('retrieval_time'),
                reranking_time=metrics.get('reranking_time'),
                generation_time=metrics.get('generation_time'),
                post_processing_time=metrics.get('post_processing_time'),
                tokens_retrieved=metrics.get('tokens_retrieved'),
                tokens_after_rerank=metrics.get('tokens_after_rerank'),
                tokens_generated=metrics.get('tokens_generated'),
                tokens_prompt=metrics['tokens_prompt'],
                chunks_retrieved=metrics.get('chunks_retrieved'),
                chunks_after_rerank=metrics.get('chunks_after_rerank'),
                model=self.groq_model
            )
            
            # Create and return the response
            return ChatResponse(
                message_id=f"msg_{int(datetime.utcnow().timestamp())}",
                content=response_text,
                citations=[citations[ref] for ref in used_citations if ref in citations],
                metadata={
                    "retrieved_chunks": metrics.get('chunks_retrieved', 0),
                    "used_citations": len(used_citations),
                    "generation_params": generation_kwargs
                },
                metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}", exc_info=True)
            # Log metrics even in case of error
            if 'total_start' in metrics:
                metrics['total_time'] = time_it() - metrics['total_start']
                metrics['error'] = str(e)
                logger.error(f"Metrics at error: {json.dumps(metrics, indent=2, default=str)}")
            raise
            raise
    
    def _format_context_and_citations(
        self,
        chunks: List[DocumentChunkWithScore]
    ) -> Tuple[str, Dict[str, Citation]]:
        """Format document chunks into context and create citation objects.
        
        Args:
            chunks: List of document chunks with scores
            
        Returns:
            Tuple of (formatted_context, citations_dict)
        """
        if not chunks:
            return "", {}
        
        context_parts = []
        citations = {}
        
        for i, chunk in enumerate(chunks, 1):
            # Create a citation ID
            citation_id = f"[^{i}]"
            
            # Add to context with citation marker
            context_parts.append(f"Document {i} (Relevance: {chunk.score:.2f}):\n{chunk.text}\n")
            
            # Create citation object
            citations[citation_id] = Citation(
                id=f"cite_{i}",
                source=chunk.source,
                text=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                page=chunk.page,
                score=chunk.score,
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    **chunk.custom_metadata
                }
            )
        
        return "\n\n".join(context_parts), citations
    
    async def _generate_with_groq(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **generation_kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a response using the Groq API with the given context.
        
        Returns:
            A tuple of (generated_text, metrics_dict) where metrics_dict contains
            performance metrics from the generation step.
        """
        # Initialize metrics dictionary
        metrics: Dict[str, Any] = {}
        
        # Prepare messages with system prompt
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful AI assistant that provides accurate and concise answers "
                "based on the provided context. Use the citations in the format [1], [2], etc. "
                "to reference the provided context. If you don't know the answer, say so."
            )
        }
        
        # Prepare user message with context
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Build messages list
        messages = [system_message]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})
        
        # Calculate token counts
        metrics['tokens_prompt'] = calculate_token_count('\n'.join(
            f"{msg['role']}: {msg['content']}" for msg in messages
        ))
        
        # Prepare the request payload
        payload = {
            "model": self.groq_model,
            "messages": messages,
            **generation_kwargs
        }
        
        # Make the API request
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time_it()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text and usage metrics
                if "choices" in result and len(result["choices"]) > 0:
                    response_text = result["choices"][0]["message"]["content"]
                    metrics['tokens_generated'] = calculate_token_count(response_text)
                    
                    # Get usage metrics if available
                    if "usage" in result:
                        usage = result["usage"]
                        metrics.update({
                            'tokens_prompt': usage.get('prompt_tokens', metrics['tokens_prompt']),
                            'tokens_generated': usage.get('completion_tokens', metrics['tokens_generated']),
                            'tokens_total': usage.get('total_tokens')
                        })
                    
                    # Calculate generation time
                    metrics['generation_time'] = time_it() - start_time
                    
                    return response_text, metrics
                else:
                    raise ValueError("Unexpected response format from Groq API")
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Groq API: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            metrics['error'] = error_msg
            metrics['generation_time'] = time_it() - start_time if 'start_time' in locals() else None
            raise RuntimeError("Failed to generate response. Please try again later.") from e
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}", exc_info=True)
            if 'start_time' in locals():
                metrics['generation_time'] = time_it() - start_time
            metrics['error'] = str(e)
            raise RuntimeError("Failed to generate response. Please try again later.") from e
        
        finally:
            # Log the metrics
            logger.debug(f"Generation metrics: {json.dumps(metrics, indent=2, default=str)}")

    def _process_citations(
        self,
        response_text: str,
        citations: Dict[str, Citation]
    ) -> Tuple[str, List[str]]:
        """Process citations in the response text and return used citation IDs.
        
        Args:
            response_text: The generated response text
            citations: Dictionary of citation IDs to Citation objects
            
        Returns:
            Tuple of (processed_text, used_citation_ids)
        """
        if not citations:
            return response_text, []
        
        # Find all citation markers in the response
        citation_pattern = r"\\\[(\\d+)\\\]"
        matches = list(re.finditer(citation_pattern, response_text))
        used_citation_ids = set()
        
        # Process each citation marker
        for match in reversed(matches):  # Process in reverse to avoid offset issues
            citation_num = match.group(1)
            citation_id = f"[^{citation_num}]"
            
            if citation_id in citations:
                used_citation_ids.add(citation_id)
                # Replace [1] with [^1] for better formatting
                response_text = (
                    response_text[:match.start()] + 
                    f"[^{citation_num}]" + 
                    response_text[match.end():]
                )
        
        return response_text, list(used_citation_ids)
