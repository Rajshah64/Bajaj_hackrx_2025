## Advanced RAG-Powered Query-Retrieval System

A FastAPI-based document QA system that uses a hybrid Retrieval-Augmented Generation (RAG) pipeline with cross-encoder reranking to answer questions from PDF documents.

### Features

- **Hybrid Retrieval (BM25 + FAISS)**: Combines keyword and semantic search
- **Cross-Encoder Reranking**: Neural reranker for high-precision top results
- **Multi-Method PDF Extraction**: `pdfplumber`, `PyMuPDF`, and `PyPDF2`
- **Content-Aware Chunking**: Separates text and tables for cleaner context
- **Google Gemini Integration**: LLM-powered answer generation
- **Bearer Token Authentication**: Protects API endpoints
- **Async Processing**: Non-blocking retrieval and generation
- **Device-Aware Inference**: Runs on CPU or CUDA (with safe fallback)

### Technology Stack

- FastAPI, httpx
- FAISS, rank_bm25, sentence-transformers, cross-encoder
- pdfplumber, PyMuPDF (fitz), PyPDF2
- Google Generative AI (Gemini)

## Installation

1. Create and activate a virtualenv

```bash
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
# source venv/bin/activate  # Linux/Mac
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration (.env)

Create a `.env` in the project root. Minimum keys:

```env
# LLM
GEMINI_API_KEY=your_gemini_api_key_here  # If unset, a demo fallback is used

# Auth
BEARER_TOKEN=c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507

# Device selection (optional): cpu or cuda
RAG_DEVICE=cpu

# Models (optional)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large

# Logging (optional)
LOG_LEVEL=INFO
```

Notes:

- Set `RAG_DEVICE=cuda` to request GPU; it automatically falls back to CPU if CUDA is unavailable.
- The service reads `RAG_DEVICE`; you can also pass `device` when constructing `AdvancedRAGService` in code.

## Running

```bash
python main.py
```

Base URL: `http://localhost:8000/api/v1`

### Endpoints

- `GET /api/v1/` — Root
- `GET /api/v1/health` — Health check
- `GET /api/v1/status` — System status and RAG stats (device, models)
- `POST /api/v1/hackrx/run` — Main query endpoint

### Example request

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## Testing

- `python test_system.py` — Full test suite (API + direct RAG checks)
- `python test_api_quick.py` — Quick API check for critical questions
- `python test_advanced_direct.py` — Direct `AdvancedRAGService` verification

## Project Structure

```
├── main.py                     # FastAPI application
├── services/
│   ├── advanced_rag_service.py # Core hybrid retrieval + reranking
│   └── llm_service.py          # Google Gemini integration
├── test_system.py              # Comprehensive testing
├── test_api_quick.py           # Quick API test
├── test_advanced_direct.py     # Direct RAG test
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## License

MIT License
