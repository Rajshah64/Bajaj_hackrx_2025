# Advanced RAG-Powered Query-Retrieval System

A FastAPI-based intelligent document query system that uses advanced RAG (Retrieval-Augmented Generation) to provide accurate answers from PDF documents.

## Features

- **Advanced RAG Pipeline**: Hybrid retrieval combining BM25 (keyword) and FAISS (semantic) search
- **Cross-Encoder Reranking**: Uses neural models to re-rank search results for maximum relevance
- **Multi-Method PDF Processing**: Extracts text using pdfplumber, PyMuPDF, and PyPDF2
- **Content-Aware Chunking**: Intelligently separates text and tables during processing
- **Google Gemini Integration**: Generates contextual answers using LLM
- **Bearer Token Authentication**: Secure API access
- **Async Processing**: Non-blocking document processing and query handling

## Technology Stack

- **FastAPI**: Web framework
- **Google Gemini**: Large Language Model
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **BM25**: Keyword-based search
- **Cross-Encoder**: Neural reranking
- **PDF Processing**: pdfplumber, PyMuPDF, PyPDF2

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the Server

```bash
python main.py
```

### API Endpoints

- `GET /health` - Health check
- `GET /api/v1/status` - System status
- `POST /hackrx/run` - Main query endpoint

### Example Request

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## Testing

- `python test_system.py` - Full system test
- `python test_api_quick.py` - Quick API test
- `python test_advanced_direct.py` - Direct RAG service test

## Project Structure

```
├── main.py                    # FastAPI application
├── services/
│   ├── advanced_rag_service.py  # Core RAG implementation
│   └── llm_service.py          # Google Gemini integration
├── test_system.py             # Comprehensive testing
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Environment Variables

Create a `.env` file with:

```
GOOGLE_API_KEY=your_gemini_api_key
```

## License

MIT License
