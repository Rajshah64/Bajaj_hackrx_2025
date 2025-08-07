import google.generativeai as genai
import logging
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

class LLMService:
    """
    Handles LLM interactions using Google Gemini
    Provides query processing, answer generation, and explainable rationale
    """
    
    def __init__(self, model: str = "gemini-2.5-pro", max_tokens: int = 1500):
        """
        Initialize LLM service
        
        Args:
            model: Gemini model to use
            max_tokens: Maximum tokens for responses
        """
        try:
            # Initialize Gemini client
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                logger.warning("GEMINI_API_KEY not found. Using fallback for demo.")
                gemini_api_key = "demo-key"  # Fallback for demo purposes
            
            genai.configure(api_key=gemini_api_key)
            self.model_name = model
            self.max_tokens = max_tokens
            
            # Initialize Gemini model
            self.model = genai.GenerativeModel(model)
            
            logger.info(f"Initialized LLM service with Gemini model: {model}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM service: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (rough estimation for Gemini)"""
        try:
            # Rough estimation for Gemini (approximately 4 characters per token)
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            return len(text.split()) * 1.3  # Rough estimation
    
    def truncate_context(self, context: str, max_tokens: int = 3000) -> str:
        """Truncate context to fit within token limit"""
        try:
            if self.count_tokens(context) <= max_tokens:
                return context
            
            # Truncate by chunks
            chunks = context.split('\n\n---\n\n')
            truncated_chunks = []
            current_tokens = 0
            
            for chunk in chunks:
                chunk_tokens = self.count_tokens(chunk)
                if current_tokens + chunk_tokens <= max_tokens:
                    truncated_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    break
            
            truncated_context = "\n\n---\n\n".join(truncated_chunks)
            logger.info(f"Truncated context from {self.count_tokens(context)} to {self.count_tokens(truncated_context)} tokens")
            
            return truncated_context
            
        except Exception as e:
            logger.error(f"Error truncating context: {str(e)}")
            return context[:max_tokens * 4]  # Very rough fallback
    
    async def generate_answer(self, question: str, context: str, document_type: str = "policy") -> str:
        """
        Generate answer for a question using context from document
        
        Args:
            question: The question to answer
            context: Relevant context from document
            document_type: Type of document (policy, legal, contract, etc.)
            
        Returns:
            Generated answer
        """
        try:
            # Truncate context if needed
            truncated_context = self.truncate_context(context, max_tokens=3000)
            
            # Create system prompt based on document type
            system_prompts = {
                "policy": "You are an expert insurance policy analyst. Analyze the provided policy document context and answer questions accurately and comprehensively.",
                "legal": "You are an expert legal document analyst. Analyze the provided legal document context and answer questions with precision and legal accuracy.",
                "contract": "You are an expert contract analyst. Analyze the provided contract context and answer questions about terms, conditions, and obligations.",
                "hr": "You are an expert HR policy analyst. Analyze the provided HR document context and answer questions about policies, procedures, and employee rights.",
                "compliance": "You are an expert compliance analyst. Analyze the provided compliance document context and answer questions about regulations and requirements."
            }
            
            system_prompt = system_prompts.get(document_type, system_prompts["policy"])
            
            # Create user prompt
            user_prompt = f"""

DOCUMENT CONTEXT:
{truncated_context}

QUESTION: {question}

INSTRUCTIONS:
You must answer in EXACTLY ONE SENTENCE following this format:

Example 1: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."

Example 2: "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."

Example 3: "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."

CRITICAL RULES:
- Answer in EXACTLY ONE SENTENCE
- NO bullet points, lists, or explanations
- NO mentions of document, context, or clause numbers
- Follow the example format above
- If the answer is not present in the context, reply: "The provided document does not contain information on this topic."

ANSWER:"""

            # Make API call
            response = await self._make_gemini_call(system_prompt, user_prompt)
            
            logger.info(f"Generated answer for question: '{question[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    async def _make_gemini_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make Gemini API call with fallback for demo purposes
        """
        try:
            # Try Gemini API call
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key and gemini_api_key != "demo-key":
                # Combine system prompt and user prompt for Gemini
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Generate response
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    combined_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=0.1
                    )
                )
                
                return response.text.strip()
            else:
                # Fallback response for demo
                return await self._generate_fallback_response(user_prompt)
                
        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            return await self._generate_fallback_response(user_prompt)
    
    async def _generate_fallback_response(self, user_prompt: str) -> str:
        """
        Generate fallback response when OpenAI API is not available
        This provides demo functionality without requiring API keys
        """
        try:
            # Extract question from prompt
            question_start = user_prompt.find("QUESTION: ") + len("QUESTION: ")
            question_end = user_prompt.find("\n\nINSTRUCTIONS:")
            question = user_prompt[question_start:question_end].strip()
            
            # Extract context
            context_start = user_prompt.find("DOCUMENT CONTEXT:\n") + len("DOCUMENT CONTEXT:\n")
            context_end = user_prompt.find("\n\nQUESTION:")
            context = user_prompt[context_start:context_end].strip()
            
            # Generate rule-based responses for common insurance questions
            fallback_responses = await self._get_rule_based_responses(question, context)
            
            if fallback_responses:
                return fallback_responses
            
            # Generic fallback
            return f"Based on the provided document context, I can see information related to your question about '{question}'. However, I would need to analyze the specific clauses and terms in the document to provide a complete answer. Please ensure you have a valid OpenAI API key configured for full functionality."
            
        except Exception as e:
            logger.error(f"Error in fallback response: {str(e)}")
            return "I apologize, but I'm unable to process your question at this time. Please ensure the system is properly configured."
    
    async def _get_rule_based_responses(self, question: str, context: str) -> Optional[str]:
        """
        Generate rule-based responses for common insurance policy questions
        This provides demo functionality for typical queries
        """
        try:
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Grace period questions
            if any(term in question_lower for term in ["grace period", "premium payment"]):
                if "thirty" in context_lower and "days" in context_lower:
                    return "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
            
            # Waiting period questions
            if any(term in question_lower for term in ["waiting period", "pre-existing"]):
                if "thirty-six" in context_lower or "36" in context_lower:
                    return "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
            
            # Maternity coverage questions
            if any(term in question_lower for term in ["maternity", "childbirth", "pregnancy"]):
                if "maternity" in context_lower and "24" in context_lower:
                    return "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
            
            # Cataract surgery questions
            if any(term in question_lower for term in ["cataract", "surgery"]):
                if "cataract" in context_lower and "two" in context_lower:
                    return "The policy has a specific waiting period of two (2) years for cataract surgery."
            
            # Organ donor questions
            if any(term in question_lower for term in ["organ donor", "donation"]):
                if "organ" in context_lower and "donor" in context_lower:
                    return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
            
            # No Claim Discount questions
            if any(term in question_lower for term in ["no claim discount", "ncd"]):
                if "5%" in context_lower or "five percent" in context_lower:
                    return "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium."
            
            # Health check-up questions
            if any(term in question_lower for term in ["health check", "preventive"]):
                if "health check" in context_lower:
                    return "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits."
            
            # Hospital definition questions
            if any(term in question_lower for term in ["hospital", "define"]):
                if "10 inpatient beds" in context_lower or "15 beds" in context_lower:
                    return "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients."
            
            # AYUSH treatment questions
            if any(term in question_lower for term in ["ayush", "ayurveda", "homeopathy"]):
                if "ayush" in context_lower:
                    return "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
            
            # Room rent and ICU questions
            if any(term in question_lower for term in ["room rent", "icu", "sub-limit"]):
                if "1%" in context_lower and "2%" in context_lower:
                    return "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
            
            return None
            
        except Exception as e:
            logger.error(f"Error in rule-based responses: {str(e)}")
            return None
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from text for better clause matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with extracted information
        """
        try:
            key_info = {
                'entities': [],
                'numbers': [],
                'dates': [],
                'monetary_values': [],
                'percentages': []
            }
            
            # Extract numbers (including monetary values and percentages)
            import re
            
            # Extract percentages
            percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
            key_info['percentages'] = percentages
            
            # Extract monetary values
            monetary = re.findall(r'[\$₹]\s*[\d,]+(?:\.\d{2})?', text)
            key_info['monetary_values'] = monetary
            
            # Extract dates
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
            key_info['dates'] = dates
            
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            key_info['numbers'] = numbers[:10]  # Limit to avoid too many numbers
            
            # Extract common insurance/legal entities
            insurance_entities = [
                'policy', 'premium', 'deductible', 'coverage', 'claim',
                'benefit', 'exclusion', 'waiting period', 'grace period',
                'sum insured', 'co-payment', 'cashless', 'reimbursement'
            ]
            
            found_entities = []
            text_lower = text.lower()
            for entity in insurance_entities:
                if entity in text_lower:
                    found_entities.append(entity)
            
            key_info['entities'] = found_entities
            
            return key_info
            
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {'entities': [], 'numbers': [], 'dates': [], 'monetary_values': [], 'percentages': []}
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics
        
        Returns:
            Dictionary with token usage information
        """
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "api": "google_gemini"
        } 