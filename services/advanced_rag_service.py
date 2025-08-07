import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from typing import List, Dict, Any
import logging
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import asyncio
import httpx
import io
import re

# PDF processing imports
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import os

# Use loguru for better logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.info = logging.info
    logger.error = logging.error
    logger.warning = logging.warning

class AdvancedRAGService:
    """
    An advanced RAG service implementing a pipeline of:
    1. Content-Aware PDF Chunking (separating text and tables)
    2. Hybrid Retrieval (BM25 for keywords + FAISS for semantics)
    3. Cross-Encoder Reranking for high-precision results
    """

    def __init__(self,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 reranker_model: str = "BAAI/bge-reranker-large",
                 device: str | None = None):
                #  embedding_model: str = "all-MiniLM-L6-v2",
                #  reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the service with embedding and reranking models.
        """
        try:
            # Determine and validate device
            requested_device = device or os.getenv("RAG_DEVICE")
            if requested_device:
                requested_device = requested_device.lower().strip()
            if requested_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                requested_device = "cpu"
            if requested_device not in {None, "cuda", "cpu"}:
                logger.warning(f"Unknown device '{requested_device}', falling back to auto-detect.")
                requested_device = None
            self.device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.embedding_model_name = embedding_model
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

            self.reranker_model_name = reranker_model
            self.reranker = CrossEncoder(reranker_model, device=self.device)

            # Vector index for semantic search
            self.faiss_index = None
            # Keyword index for lexical search
            self.bm25_index = None

            self.document_chunks = []
            logger.info(f"Initialized AdvancedRAGService with embedding model: '{embedding_model}' and reranker: '{reranker_model}'")

        except Exception as e:
            logger.error(f"❌ Error initializing service: {e}")
            raise

    async def download_and_process_pdf(self, pdf_url: str, chunk_strategy: str = "auto"):
        """
        Downloads a PDF from URL and processes it using unstructured library.
        
        Args:
            pdf_url (str): URL of the PDF to download and process
            chunk_strategy (str): The chunking strategy for unstructured
        """
        try:
            logger.info(f"Downloading PDF from: {pdf_url[:50]}...")
            
            # Download PDF
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()
                pdf_content = response.content
            
            logger.info(f"Downloaded {len(pdf_content)} bytes")
            
            # Save temporarily for processing
            temp_pdf_path = "temp_document.pdf"
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_content)
            
            # Process the PDF
            await self.process_and_load_pdf(temp_pdf_path, chunk_strategy)
            
            # Clean up temp file
            Path(temp_pdf_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error downloading and processing PDF: {e}")
            raise

    async def process_and_load_pdf(self, pdf_path: str, chunk_strategy: str = "auto"):
        """
        Processes a PDF using multiple extraction methods, creates intelligent chunks for text and tables,
        and builds the hybrid search indexes.

        Args:
            pdf_path (str): The file path to the PDF document.
            chunk_strategy (str): The chunking strategy (for compatibility, not used).
        """
        try:
            logger.info(f"Processing PDF '{pdf_path}' with advanced multi-method extraction...")
            
            # Use asyncio to run the CPU-intensive operation in a thread
            extracted_content = await asyncio.to_thread(self._extract_pdf_content, pdf_path)
            
            # Create intelligent chunks
            chunks = self._create_intelligent_chunks(extracted_content)
            
            self.document_chunks = [chunk for chunk in chunks if chunk['text'] and len(chunk['text'].strip()) > 20]
            logger.info(f"Successfully extracted {len(self.document_chunks)} chunks from the PDF.")

            await self._build_indexes()

        except Exception as e:
            logger.error(f"❌ Failed to process PDF and build indexes: {e}")
            raise

    def _extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract content from PDF using multiple methods"""
        try:
            # Method 1: Try pdfplumber first (best for tables)
            logger.info("Trying pdfplumber extraction...")
            content = self._extract_with_pdfplumber(pdf_path)
            if content['text'] and len(content['text']) > 100:
                logger.info(f"pdfplumber successful: {len(content['text'])} chars")
                return content
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        try:
            # Method 2: Try PyMuPDF
            logger.info("Trying PyMuPDF extraction...")
            content = self._extract_with_pymupdf(pdf_path)
            if content['text'] and len(content['text']) > 100:
                logger.info(f"PyMuPDF successful: {len(content['text'])} chars")
                return content
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        
        try:
            # Method 3: Fallback to PyPDF2
            logger.info("Trying PyPDF2 extraction...")
            content = self._extract_with_pypdf2(pdf_path)
            if content['text'] and len(content['text']) > 50:
                logger.info(f"PyPDF2 successful: {len(content['text'])} chars")
                return content
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")
        
        raise Exception("All PDF extraction methods failed")

    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber (best for tables)"""
        text_parts = []
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        # Convert table to text representation
                        table_text = self._table_to_text(table)
                        tables.append(f"[Table from Page {page_num + 1}]\n{table_text}")
        
        return {
            'text': '\n\n'.join(text_parts),
            'tables': tables,
            'method': 'pdfplumber'
        }

    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        text_parts = []
        tables = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            # Try to detect table-like structures
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    # Simple heuristic for table detection
                    if self._looks_like_table(block_text):
                        tables.append(f"[Table-like structure from Page {page_num + 1}]\n{block_text.strip()}")
        
        doc.close()
        return {
            'text': '\n\n'.join(text_parts),
            'tables': tables,
            'method': 'pymupdf'
        }

    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyPDF2 as fallback"""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return {
            'text': '\n\n'.join(text_parts),
            'tables': [],
            'method': 'pypdf2'
        }

    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to readable text"""
        if not table:
            return ""
        
        # Filter out None values and convert to strings
        clean_table = []
        for row in table:
            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(clean_row):  # Only add non-empty rows
                clean_table.append(clean_row)
        
        if not clean_table:
            return ""
        
        # Create text representation
        lines = []
        for row in clean_table:
            line = " | ".join(row)
            if line.strip():
                lines.append(line)
        
        return "\n".join(lines)

    def _looks_like_table(self, text: str) -> bool:
        """Simple heuristic to detect table-like text"""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for patterns that suggest tabular data
        pipe_count = text.count('|')
        tab_count = text.count('\t')
        number_pattern = len(re.findall(r'\d+', text))
        
        return (pipe_count > 3 or tab_count > 3 or 
                (number_pattern > 5 and len(lines) > 2))

    def _create_intelligent_chunks(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create intelligent chunks separating text and tables"""
        chunks = []
        
        # Process main text
        if content['text']:
            text_chunks = self._chunk_text(content['text'], chunk_size=800, overlap=100)
            for chunk in text_chunks:
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'type': 'text',
                        'element_type': 'TextChunk',
                        'extraction_method': content['method']
                    }
                })
        
        # Process tables separately
        for table_text in content.get('tables', []):
            chunks.append({
                'text': table_text,
                'metadata': {
                    'type': 'table',
                    'element_type': 'Table',
                    'extraction_method': content['method']
                }
            })
        
        return chunks

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

    async def _build_indexes(self):
        """
        Private method to build the FAISS and BM25 indexes from the processed document chunks.
        """
        if not self.document_chunks:
            raise ValueError("No document chunks to index. Run process_and_load_pdf first.")

        logger.info("Building FAISS and BM25 indexes...")

        # --- FAISS Index (for semantic search) ---
        texts = [chunk['text'] for chunk in self.document_chunks]
        
        # Run embedding generation in a thread to avoid blocking
        embeddings = await asyncio.to_thread(
            self.embedding_model.encode, 
            texts, 
            convert_to_tensor=False, 
            show_progress_bar=True
        )
        embeddings = np.array(embeddings).astype('float32')

        # Build FAISS index with optional GPU acceleration
        try:
            faiss.normalize_L2(embeddings)
            if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
                num_gpus = 0
                try:
                    num_gpus = faiss.get_num_gpus()
                except Exception:
                    num_gpus = 0
                if num_gpus > 0:
                    res = faiss.StandardGpuResources()
                    cpu_index = faiss.IndexFlatIP(self.embedding_dim)
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    logger.info("FAISS GPU index enabled")
                else:
                    self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                    logger.info("FAISS CPU index enabled (no GPUs detected by FAISS)")
            else:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info("FAISS CPU index enabled")
        except Exception as e:
            logger.warning(f"Could not enable FAISS GPU ({e}); falling back to CPU")
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors.")

        # --- BM25 Index (for keyword search) ---
        tokenized_corpus = [doc.lower().split(" ") for doc in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a full RAG pipeline query: Hybrid Retrieval + Reranking.

        Args:
            query (str): The user's question.
            top_k (int): The final number of top relevant documents to return.

        Returns:
            A sorted list of the most relevant chunks with their scores.
        """
        if not self.faiss_index or not self.bm25_index:
            raise RuntimeError("Indexes are not built. Call process_and_load_pdf first.")

        logger.info(f"Executing advanced query for: '{query[:50]}...'")

        # --- Stage 1: Hybrid Retrieval ---
        # Retrieve more candidates than needed for the reranker (e.g., 5x top_k)
        candidate_count = min(top_k * 10, len(self.document_chunks))

        # Keyword search with BM25
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_count]

        # Semantic search with FAISS
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        faiss_scores, faiss_top_indices = self.faiss_index.search(query_embedding, candidate_count)
        faiss_top_indices = faiss_top_indices[0]

        # Combine results using Reciprocal Rank Fusion (RRF) for robust merging
        combined_indices = self._reciprocal_rank_fusion([bm25_top_indices, faiss_top_indices])
        
        # Get unique document chunks from combined indices
        retrieved_chunks = []
        seen_indices = set()
        for i in combined_indices:
            if i not in seen_indices and i < len(self.document_chunks):
                retrieved_chunks.append(self.document_chunks[i])
                seen_indices.add(i)
                if len(retrieved_chunks) >= candidate_count:
                    break
        
        logger.info(f"Retrieved {len(retrieved_chunks)} unique candidates via hybrid search.")

        # --- Stage 2: Cross-Encoder Reranking ---
        if not retrieved_chunks:
            return []

        pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
        raw_scores = self.reranker.predict(pairs, convert_to_tensor=True, show_progress_bar=False)
        rerank_scores = torch.sigmoid(raw_scores).cpu().numpy()

        # Combine rerank scores with original chunks
        for i, chunk in enumerate(retrieved_chunks):
            chunk['rerank_score'] = float(rerank_scores[i])
            chunk['similarity_score'] = float(rerank_scores[i])  # For compatibility

        # Sort by the new rerank_score
        results = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)

        logger.info(f"Reranked results. Top result score: {results[0]['rerank_score']:.4f}")
        
        # Log top 3 results for debugging
        for i, result in enumerate(results[:3]):
            chunk_preview = result['text'][:80].replace('\n', ' ')
            chunk_type = result['metadata'].get('type', 'unknown')
            logger.info(f"   Rank {i+1}: score={result['rerank_score']:.3f} type={chunk_type} - '{chunk_preview}...'")
        
        return results[:top_k]

    def _reciprocal_rank_fusion(self, result_sets: List[np.ndarray], k: int = 60) -> List[int]:
        """
        Merges multiple ranked lists of document indices using RRF.
        """
        rrf_scores = {}
        for ranked_list in result_sets:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (k + rank + 1)

        sorted_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_scores]

    # Compatibility methods for existing interface
    def build_faiss_index(self, document_chunks: List[Dict[str, Any]]) -> None:
        """Compatibility method for existing interface"""
        self.document_chunks = document_chunks
        asyncio.run(self._build_indexes())

    def semantic_search(self, query: str, top_k: int = 10, min_score: float = 0.15) -> List[Dict[str, Any]]:
        """Compatibility method for existing interface"""
        results = self.query(query, top_k)
        # Filter by minimum score
        return [r for r in results if r.get('similarity_score', 0) >= min_score]

    def find_relevant_clauses(self, query: str, context_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Compatibility method for existing interface"""
        results = self.query(query, top_k=15)
        
        if context_keywords:
            filtered_results = []
            for result in results:
                chunk_text = result['text'].lower()
                if any(keyword.lower() in chunk_text for keyword in context_keywords):
                    result['matched_keywords'] = [
                        kw for kw in context_keywords 
                        if kw.lower() in chunk_text
                    ]
                    filtered_results.append(result)
            results = filtered_results
        
        return results

    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Compatibility method for existing interface"""
        results = self.query(query, top_k=8)
        
        if not results:
            logger.warning("No relevant context found with advanced search")
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            chunk_text = result['text']
            chunk_type = result['metadata'].get('type', 'unknown')
            
            # Add source information
            source_info = f"[ADVANCED - Type: {chunk_type}, Score: {result['rerank_score']:.3f}]"
            chunk_with_source = f"{source_info}\n{chunk_text}"
            
            if current_length + len(chunk_with_source) <= max_context_length:
                context_parts.append(chunk_with_source)
                current_length += len(chunk_with_source)
            else:
                # Add truncated version
                remaining = max_context_length - current_length
                if remaining > 100:
                    truncated = chunk_with_source[:remaining] + "..."
                    context_parts.append(truncated)
                break
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Generated ADVANCED context: {len(context)} chars from {len(context_parts)} chunks")
        
        return context

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the current embedding index"""
        try:
            if self.faiss_index is None:
                return {"status": "no_index", "message": "FAISS index not built"}
            
            return {
                "status": "ready",
                "total_vectors": self.faiss_index.ntotal,
                "embedding_dimension": self.embedding_dim,
                "embedding_model": self.embedding_model_name,
                "reranker_model": self.reranker_model_name,
                "document_chunks": len(self.document_chunks),
                "version": "advanced_rag"
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            return {"status": "error", "message": str(e)}


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    async def test_advanced_rag():
        # 1. Initialize the service
        rag_service = AdvancedRAGService()

        # 2. Process your PDF from URL
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        await rag_service.download_and_process_pdf(pdf_url)

        # 3. Ask your questions
        questions = [
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "What is the waiting period for cataract surgery?",  # The hard question
            "How does the policy define a 'Hospital'?",  # The multi-part question
            "Are there any sub-limits on room rent and ICU charges for Plan A?"  # The table question
        ]

        for q in questions:
            print(f"\n--- Query: {q} ---\n")
            # The query method performs hybrid search + reranking automatically
            top_results = rag_service.query(q, top_k=3)

            if not top_results:
                print("No relevant information found.")
                continue

            # The top result is now highly likely to be the correct answer
            best_answer_context = top_results[0]['text']
            print(f"Best Context Found (Score: {top_results[0]['rerank_score']:.4f}):\n")
            print(best_answer_context)
            print("\n" + "="*50)

    # Run the test
    asyncio.run(test_advanced_rag())