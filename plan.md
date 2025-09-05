# RAG Application with FastAPI and Groq

## Overview
This project involves building a Retrieval-Augmented Generation (RAG) application using FastAPI that integrates with Groq's LLM and Pinecone vector database. The app will allow users to upload documents, process them into vector embeddings, and query the knowledge base with citations.

## Prerequisites
- Python 3.8+
- Groq API key (from [Groq Console](https://console.groq.com/))
- Pinecone API key (from [Pinecone Console](https://app.pinecone.io/))
- Git (for version control)
- GitHub account (for deployment)

## Project Structure
```
predusk/
├── .env                    # Environment variables
├── .env.example            # Example environment variables
├── .gitignore
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration settings
│   ├── models/            # Data models
│   │   ├── __init__.py
│   │   ├── document.py    # Document and chunk models
│   │   └── response.py    # Response models
│   ├── services/          # Business logic
│   │   ├── __init__.py
│   │   ├── groq_service.py # Groq API integration
│   │   ├── vector_store.py # Pinecone integration
│   │   ├── chunking.py    # Document chunking logic
│   │   └── embedding.py   # Text embedding generation
│   ├── static/            # Static files (CSS, JS, images)
│   │   ├── styles.css
│   │   └── main.js
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── helpers.py     # Helper functions
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_api.py
│   └── test_services.py
└── templates/             # HTML templates
    ├── base.html
    ├── index.html
    └── upload.html
```

## Implementation Plan

### 1. Project Setup (Day 1)
- [x] Initialize git repository
- [x] Set up Python virtual environment
- [x] Create basic FastAPI application structure
- [x] Set up logging and configuration
- [x] Add basic error handling

### 2. Core Infrastructure (Day 1-2)
- [x] Set up Pinecone vector database
- [x] Implement document chunking (800-1200 tokens, 15% overlap)
- [x] Add embedding generation (using sentence-transformers)
- [x] Create vector store service
- [x] Implement document ingestion pipeline

### 3. RAG Implementation (Day 2-3)
- [x] Set up retriever with MMR (Maximal Marginal Relevance)
- [x] Integrate reranker (Cohere/Jina/Voyage)
- [x] Implement citation handling
- [x] Add response generation with citations
- [x] Add performance metrics collection

### 4. Web Interface (Day 3-4)
- [x] Design and implement HTML templates
- [x] Add CSS styling
- [x] Add file upload functionality
- [x] Implement citation display
- [x] Add performance metrics display
- [x] Add loading states and user feedback

### 5. Testing & Optimization (Day 4-5) - COMPLETED
- [x] Write unit tests for core components
- [x] Test retrieval accuracy
- [x] Optimize chunking parameters
- [x] Test with different document types
- [x] Implement error handling and edge cases

### 6. Deployment & Documentation (Day 5-7)
- [ ] Set up GitHub repository (Skipped for now)
- [ ] Configure deployment (Render/Railway) (Skipped for now)
- [ ] Set up environment variables (Skipped for now)
- [ ] Create comprehensive README (Skipped for now)
- [ ] Add architecture diagram (Skipped for now)
- [ ] Document API endpoints and usage (Skipped for now)

## Dependencies
```
# Core
fastapi>=0.68.0
uvicorn>=0.15.0
python-dotenv>=0.19.0
python-multipart>=0.0.5
python-decouple>=3.4

# Vector Database
pinecone-client>=3.0.0

# Text Processing
sentence-transformers>=2.2.2
unstructured>=0.10.0
nltk>=3.8.1

# Reranking
cohere>=4.0.0  # or jina-ai[reranker]

# Frontend
jinja2>=3.0.0
aiofiles>=23.1.0

# Testing
pytest>=6.2.5
pytest-asyncio>=0.21.0
httpx>=0.21.0
```

## Development Workflow
1. Create a new branch for each feature
2. Write tests for new functionality
3. Implement the feature
4. Run tests and fix any issues
5. Create a pull request
6. Deploy after code review

## Deployment Options
1. **Render** (Recommended)
   - Connect GitHub repository
   - Set environment variables
   - Deploy on push to main

2. **Railway**
   - Connect GitHub repository
   - Configure environment
   - Deploy with one click

## Next Steps
1. Set up the basic FastAPI application
2. Implement the Groq API service
3. Create the web interface
4. Test locally
5. Deploy to chosen platform

## Success Criteria
- [ ] Web app is accessible online with RAG functionality
- [ ] Users can upload documents and query the knowledge base
- [ ] Responses include proper citations to source material
- [ ] Performance metrics are displayed (latency, token counts)
- [ ] Error handling for all major failure points
- [ ] Clean, responsive UI with good UX
- [ ] Comprehensive documentation including:
  - Architecture overview
  - Setup instructions
  - API documentation
  - Deployment guide
  - Chunking and retrieval parameters
