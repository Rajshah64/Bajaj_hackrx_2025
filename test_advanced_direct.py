#!/usr/bin/env python3
"""
Direct test of the AdvancedRAGService to verify it works properly
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_advanced_rag_direct():
    """Test the advanced RAG service directly"""
    try:
        print("TESTING ADVANCED RAG SERVICE DIRECTLY")
        print("=" * 80)
        
        # Import the advanced RAG service
        from services.advanced_rag_service import AdvancedRAGService
        
        # Initialize service
        logger.info("Initializing AdvancedRAGService...")
        rag_service = AdvancedRAGService()
        logger.info("Service initialized successfully")
        
        # Process document
        logger.info("Processing PDF document...")
        doc_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        await rag_service.download_and_process_pdf(doc_url)
        logger.info("Document processed successfully")
        
        # Test critical questions
        test_questions = [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for cataract surgery?",
            "How does the policy define a 'Hospital'?"
        ]
        
        print("\nTESTING QUESTIONS:")
        print("-" * 80)
        
        for i, question in enumerate(test_questions):
            print(f"\nQuestion {i+1}: {question}")
            print("-" * 60)
            
            # Get results using advanced pipeline
            results = rag_service.query(question, top_k=5)
            
            if not results:
                print("❌ No results found!")
                continue
            
            print(f"Found {len(results)} results:")
            
            for j, result in enumerate(results[:3]):  # Show top 3
                rerank_score = result['rerank_score'] * 100
                confidence = "HIGH" if rerank_score > 70 else "MEDIUM" if rerank_score > 50 else "LOW"
                chunk_type = result['metadata'].get('type', 'unknown')
                extraction_method = result['metadata'].get('extraction_method', 'unknown')
                
                print(f"\n   Rank #{j+1}:")
                print(f"      Rerank Score: {rerank_score:.1f}% ({confidence})")
                print(f"      Content Type: {chunk_type}")
                print(f"      Extraction Method: {extraction_method}")
                print(f"      Text Preview: {result['text'][:200]}...")
        
        # Test embedding stats
        print("\nEMBEDDING STATS:")
        print("-" * 80)
        stats = rag_service.get_embedding_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting direct advanced RAG test...")
    result = asyncio.run(test_advanced_rag_direct())
    print(f"\nTest result: {'SUCCESS' if result else 'FAILED'}")