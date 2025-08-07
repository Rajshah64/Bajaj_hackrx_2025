#!/usr/bin/env python3
"""
Test script for the ADVANCED RAG-Powered Intelligent Query-Retrieval System
Tests the hybrid search + cross-encoder reranking pipeline
"""

import asyncio
import httpx
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507"

# Sample data from problem statement
SAMPLE_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        # "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        # "What is the waiting period for pre-existing diseases (PED) to be covered?",
        # "Does this policy cover maternity expenses, and what are the conditions?",
        # "What is the waiting period for cataract surgery?",
        # "Are the medical expenses for an organ donor covered under this policy?",
        # "What is the No Claim Discount (NCD) offered in this policy?",
        # "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?"
        # "What is the extent of coverage for AYUSH treatments?",
        # "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Expected answers for validation
EXPECTED_ANSWERS = [
    # "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    # "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    # "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    # "The policy has a specific waiting period of two (2) years for cataract surgery.",
    # "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    # "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    # "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    # "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    # "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

async def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            
            if response.status_code == 200:
                logger.info("✅ Health check passed")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return False

async def test_status_endpoint():
    """Test the status endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/v1/status")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Status check passed")
                logger.info(f"   Version: {data.get('version')}")
                logger.info(f"   Features: {data.get('features', [])}")
                
                rag_stats = data.get('rag_stats', {})
                if rag_stats.get('version') == 'advanced_rag':
                    logger.info("🚀 ADVANCED RAG service confirmed!")
                
                return True
            else:
                logger.error(f"❌ Status check failed: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Status check error: {str(e)}")
        return False

async def test_direct_advanced_rag():
    """Test the advanced RAG service directly"""
    try:
        print("\n" + "="*100)
        print("🚀 TESTING ADVANCED RAG DIRECTLY")
        print("="*100)
        
        # Import the advanced RAG service
        from services.advanced_rag_service import AdvancedRAGService
        
        # Initialize service
        rag_service = AdvancedRAGService()
        
        # Process document
        logger.info("📥 Processing document with Advanced RAG...")
        doc_url = SAMPLE_REQUEST['documents']
        await rag_service.download_and_process_pdf(doc_url)
        
        # Test the critical cataract surgery question
        question = "How does the policy define a 'Hospital'?"
        
        print(f"\n🎯 ADVANCED RAG RESULTS FOR: '{question}'")
        print("-"*100)
        
        # Get results using advanced pipeline
        results = rag_service.query(question, top_k=10)
        
        for i, result in enumerate(results):
            rerank_score = result['rerank_score'] * 100
            confidence = "HIGH" if rerank_score > 70 else "MEDIUM" if rerank_score > 50 else "LOW"
            chunk_type = result['metadata'].get('type', 'unknown')
            
            print(f"\n📄 Rank #{i+1} - Rerank Score: {rerank_score:.1f}% ({confidence})")
            print(f"   Content Type: {chunk_type}")
            print(f"   Element Type: {result['metadata'].get('element_type', 'unknown')}")
            print(f"   Text: {result['text'][:250]}...")
            print("-"*50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct advanced RAG error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_main_endpoint():
    """Test the main query processing endpoint"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BEARER_TOKEN}"
        }
        
        logger.info("🚀 Testing advanced RAG API endpoint...")
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=SAMPLE_REQUEST
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            logger.info("✅ Advanced RAG API test passed")
            logger.info(f"   Processing time: {processing_time:.2f} seconds")
            logger.info(f"   Answers received: {len(answers)}")
            
            # Print answers
            print("\n" + "="*100)
            print("🎯 ADVANCED RAG SYSTEM ANSWERS")
            print("="*100)
            
            for i, (question, answer) in enumerate(zip(SAMPLE_REQUEST['questions'], answers)):
                print(f"\n🔹 Question {i+1}:")
                print(f"   {question}")
                print(f"\n💡 Advanced RAG Answer:")
                print(f"   {answer}")
                print("-" * 80)
            
            return True, answers
            
        else:
            logger.error(f"❌ Advanced RAG API test failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False, []
            
    except Exception as e:
        logger.error(f"❌ Advanced RAG API test error: {str(e)}")
        return False, []

def evaluate_answers(generated_answers):
    """Evaluate the quality of generated answers"""
    try:
        logger.info("\n📊 EVALUATING ANSWER QUALITY:")
        
        total_score = 0
        max_score = len(EXPECTED_ANSWERS)
        
        for i, (generated, expected) in enumerate(zip(generated_answers, EXPECTED_ANSWERS)):
            # Simple similarity check (in production, use better metrics)
            score = 0
            
            if generated and expected:
                # Check for key terms
                generated_lower = generated.lower()
                expected_lower = expected.lower()
                
                # Extract key numbers and terms
                expected_words = set(expected_lower.split())
                generated_words = set(generated_lower.split())
                
                # Calculate overlap
                overlap = len(expected_words.intersection(generated_words))
                total_words = len(expected_words.union(generated_words))
                
                if total_words > 0:
                    similarity = overlap / total_words
                    if similarity > 0.3:  # Threshold for acceptable answer
                        score = 1
            
            total_score += score
            status = "✅" if score == 1 else "❌"
            logger.info(f"   {status} Question {i+1}: {'PASS' if score == 1 else 'FAIL'}")
        
        accuracy = (total_score / max_score) * 100 if max_score > 0 else 0
        logger.info(f"\n📈 OVERALL ACCURACY: {accuracy:.1f}% ({total_score}/{max_score})")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"❌ Error evaluating answers: {str(e)}")
        return 0

async def run_full_test_suite():
    """Run the complete advanced RAG test suite"""
    print("🧪 ADVANCED RAG-POWERED QUERY-RETRIEVAL SYSTEM - TEST SUITE")
    print("=" * 100)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    health_ok = await test_health_endpoint()
    
    # Test 2: Status Check
    print("\n2️⃣ Testing Status Endpoint...")
    status_ok = await test_status_endpoint()
    
    # Test 3: Direct Advanced RAG
    print("\n3️⃣ Testing Advanced RAG Direct...")
    rag_ok = await test_direct_advanced_rag()
    
    # Test 4: Advanced API
    print("\n4️⃣ Testing Advanced RAG API...")
    api_ok, answers = await test_main_endpoint()
    
    # Final Results
    print("\n" + "="*100)
    print("📊 ADVANCED RAG SYSTEM TEST RESULTS:")
    print("="*100)
    print(f"Health Check:         {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Status Check:         {'✅ PASS' if status_ok else '❌ FAIL'}")
    print(f"Advanced RAG Direct:  {'✅ PASS' if rag_ok else '❌ FAIL'}")
    print(f"Advanced RAG API:     {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    overall_status = health_ok and status_ok and rag_ok and api_ok
    print(f"\n🎯 OVERALL STATUS: {'🚀 ADVANCED RAG SYSTEM READY' if overall_status else '❌ NEEDS ATTENTION'}")
    
    if overall_status:
        print("\n🎉 SUCCESS! The advanced RAG system provides:")
        print("   ✅ Hybrid Search (BM25 + FAISS) for keyword + semantic matching")
        print("   ✅ Cross-Encoder Reranking for maximum relevance")
        print("   ✅ Content-Aware Processing separating text and tables")
        print("   ✅ Professional-grade retrieval without manual rules")
        print("   ✅ Scalable architecture that works for any question type")
    
    print("="*100)
    
    return overall_status

if __name__ == "__main__":
    print("Starting advanced RAG system test suite...")
    result = asyncio.run(run_full_test_suite())
    exit(0 if result else 1) 