from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import logging
import os
from dotenv import load_dotenv

from services.advanced_rag_service import AdvancedRAGService
from services.llm_service import LLMService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG-Powered Intelligent Query-Retrieval System",
    description="Process documents using advanced hybrid search + cross-encoder reranking",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
BEARER_TOKEN = "c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507"

# Initialize services with PURE ADVANCED RAG
advanced_rag_service = AdvancedRAGService()  # 🚀 PURE ADVANCED RAG SERVICE
llm_service = LLMService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authentication"""
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/api/v1/")
async def root():
    """Root endpoint"""
    return {"message": "Advanced RAG-Powered Intelligent Query-Retrieval System", "version": "3.0.0"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Advanced RAG system is operational", "version": "3.0.0"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Main endpoint to process document queries using PURE ADVANCED RAG
    
    This endpoint:
    1. Downloads and processes the document using multi-method PDF extraction
    2. Creates hybrid indexes (FAISS + BM25) for semantic + keyword search
    3. Uses cross-encoder reranking for maximum relevance
    4. Returns highly accurate answers (NO OLD COMPONENTS)
    """
    try:
        logger.info(f"🚀 Processing PURE ADVANCED RAG request with {len(request.questions)} questions")
        logger.info(f"   Document URL: {request.documents}")
        
        # Use PURE advanced RAG processing (no old components)
        answers = await process_queries_pure_advanced_rag(request.documents, request.questions)
        
        logger.info(f"✅ Generated {len(answers)} answers using PURE ADVANCED RAG")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Error processing queries with advanced RAG: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing queries: {str(e)}"
        )

async def process_queries_pure_advanced_rag(document_url: str, questions: List[str]) -> List[str]:
    """
    Process queries using PURE advanced RAG pipeline (no old components)
    """
    try:
        # Check if we already have the document processed
        if not advanced_rag_service.faiss_index:
            logger.info("🔄 Processing document with PURE Advanced RAG...")
            await advanced_rag_service.download_and_process_pdf(document_url)
        
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"🎯 Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Get the best context using PURE advanced RAG
                top_results = advanced_rag_service.query(question, top_k=5)
                
                if not top_results:
                    answers.append("I couldn't find relevant information in the document to answer this question.")
                    continue
                
                # Create context from top results with scoring info
                context_parts = []
                for j, result in enumerate(top_results[:5]):  # Use top 5 results
                    score = result['rerank_score']
                    chunk_type = result['metadata'].get('type', 'unknown')
                    context_parts.append(f"[Chunk {j+1} - Score: {score:.3f} - Type: {chunk_type}]\n{result['text']}")
                
                context = "\n\n---\n\n".join(context_parts)
                
                # Log the context being used
                logger.info(f"   📄 Using context from {len(context_parts)} chunks, total length: {len(context)}")
                
                # Generate answer using LLM
                answer = await llm_service.generate_answer(
                    question=question,
                    context=context,
                    document_type="policy"
                )
                
                answers.append(answer)
                logger.info(f"✅ Generated answer {i+1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {str(e)}")
                answers.append(f"I apologize, but I encountered an error while processing this question: {str(e)}")
        
        return answers
        
    except Exception as e:
        logger.error(f"Error in pure advanced RAG processing: {str(e)}")
        raise

@app.get("/api/v1/status")
async def get_status():
    """Get system status and component health"""
    try:
        # Get advanced RAG stats
        rag_stats = advanced_rag_service.get_embedding_stats()
        
        status_info = {
            "advanced_rag_service": "initialized (PURE ADVANCED)" if advanced_rag_service else "not initialized", 
            "llm_service": "initialized" if llm_service else "not initialized",
            "old_components": "REMOVED - using pure advanced RAG only"
        }
        
        return {
            "status": "operational",
            "version": "3.0.0 (PURE ADVANCED RAG)",
            "components": status_info,
            "rag_stats": rag_stats,
            "message": "Pure Advanced RAG system ready",
            "features": [
                "Hybrid Search (BM25 + FAISS)",
                "Cross-Encoder Reranking", 
                "Multi-Method PDF Processing",
                "Table-Text Separation",
                "NO OLD COMPONENTS"
            ]
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting PURE ADVANCED RAG-Powered Query-Retrieval System")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
