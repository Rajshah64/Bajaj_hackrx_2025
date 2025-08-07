#!/usr/bin/env python3
"""
Quick API test to verify the main.py works correctly
"""

import asyncio
import httpx
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "c1c19bb08f894ca1605c6cf9cf949ed137a2857e14dc46a322a1417058a80507"

# Test with just the problematic questions
TEST_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "How does the policy define a 'Hospital'?",
        "What is the waiting period for cataract surgery?"
    ]
}

async def test_health():
    """Test health endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/v1/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Health check passed")
                logger.info(f"   Version: {data.get('version')}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return False

async def test_status():
    """Test status endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/v1/status")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Status check passed")
                logger.info(f"   Version: {data.get('version')}")
                logger.info(f"   Features: {data.get('features', [])}")
                
                # Check if it's the pure advanced version
                if "PURE ADVANCED RAG" in data.get('version', ''):
                    logger.info("PURE ADVANCED RAG confirmed!")
                
                return True
            else:
                logger.error(f"Status check failed: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return False

async def test_api():
    """Test the main API endpoint"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BEARER_TOKEN}"
        }
        
        logger.info("Testing PURE Advanced RAG API...")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=headers,
                json=TEST_REQUEST
            )
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            logger.info("API test passed")
            logger.info(f"   Answers received: {len(answers)}")
            
            # Check answers for the problematic questions
            print("\nANSWERS FROM PURE ADVANCED RAG:")
            print("=" * 100)
            
            for i, (question, answer) in enumerate(zip(TEST_REQUEST['questions'], answers)):
                print(f"\nQuestion {i+1}:")
                print(f"   {question}")
                print(f"\nPure Advanced RAG Answer:")
                print(f"   {answer}")
                
                # Check if we still get the "OpenAI API key" fallback
                if "OpenAI API key" in answer or "valid OpenAI API key" in answer:
                    print(f"   STILL GETTING FALLBACK MESSAGE!")
                else:
                    print(f"   PROPER ANSWER GENERATED!")
                
                print("-" * 80)
            
            return True, answers
            
        else:
            logger.error(f"API test failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False, []
            
    except Exception as e:
        logger.error(f"API test error: {str(e)}")
        return False, []

async def run_quick_test():
    """Run quick test suite"""
    print("QUICK TEST OF PURE ADVANCED RAG SYSTEM")
    print("=" * 80)
    
    # Test 1: Health
    print("\nTesting Health...")
    health_ok = await test_health()
    
    # Test 2: Status  
    print("\nTesting Status...")
    status_ok = await test_status()
    
    # Test 3: API
    print("\nTesting API...")
    api_ok, answers = await test_api()
    
    # Results
    print("\n" + "="*80)
    print("QUICK TEST RESULTS:")
    print("="*80)
    print(f"Health:  {'PASS' if health_ok else 'FAIL'}")
    print(f"Status:  {'PASS' if status_ok else 'FAIL'}")
    print(f"API:     {'PASS' if api_ok else 'FAIL'}")
    
    if health_ok and status_ok and api_ok:
        print("\nPURE ADVANCED RAG SYSTEM IS WORKING!")
    else:
        print("\nSYSTEM HAS ISSUES")
    
    return health_ok and status_ok and api_ok

if __name__ == "__main__":
    print("Starting quick API test...")
    result = asyncio.run(run_quick_test())
    exit(0 if result else 1)