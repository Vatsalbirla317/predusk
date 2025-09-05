from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Predusk RAG"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Groq API Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "predusk-rag-index"
    PINECONE_HOST: Optional[str] = None  # Only needed for serverless indexes
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # Dimension of the embedding vectors
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000  # Number of tokens per chunk
    CHUNK_OVERLAP: int = 150  # Number of overlapping tokens between chunks
    TOP_K_RESULTS: int = 5  # Number of chunks to retrieve
    
    # Cohere Reranker (optional)
    COHERE_API_KEY: Optional[str] = None
    RERANKER_MODEL: str = "rerank-english-v2.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
