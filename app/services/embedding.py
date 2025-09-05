import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = self._get_device(device)
        self.model = self._load_model()
        
        # Warm up the model
        self._warmup()
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device."""
        import torch
        
        if device:
            return device
            
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            # Fallback to a smaller model if the default fails
            if self.model_name != 'all-MiniLM-L6-v2':
                logger.info("Falling back to 'all-MiniLM-L6-v2' model")
                return self._load_model_with_fallback('all-MiniLM-L6-v2')
            raise
    
    def _load_model_with_fallback(self, model_name: str):
        """Attempt to load a model with a fallback to CPU if GPU fails."""
        try:
            return SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            if self.device != 'cpu':
                logger.warning(f"Failed to load model on {self.device}, falling back to CPU")
                return SentenceTransformer(model_name, device='cpu')
            raise
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        try:
            self.embed_texts(["warmup"])
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
            
        try:
            # Convert single string to list for consistent handling
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings in batches to avoid OOM errors
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=False
                )
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = batch_embeddings.tolist()
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Return the dimension from settings if available
        if hasattr(settings, 'EMBEDDING_DIM') and settings.EMBEDDING_DIM:
            return settings.EMBEDDING_DIM
        
        # Otherwise, get it from the model
        try:
            # Get embedding dimension by doing a forward pass with a dummy input
            dummy_embedding = self.embed_texts(["dummy text"])[0]
            return len(dummy_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {str(e)}")
            # Default to a common dimension if we can't determine it
            return 384  # Common dimension for models like all-MiniLM-L6-v2
