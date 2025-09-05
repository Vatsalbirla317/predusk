import os
import logging
import json
import httpx
from typing import Optional, Dict, Any
from ..config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class GroqService:
    BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        
        if not self.api_key:
            logger.error("No Groq API key provided")
            raise ValueError("Groq API key is required")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"GroqService initialized with model: {self.model}")
    
    async def list_models(self) -> list:
        """List all available models from Groq API."""
        url = f"{self.BASE_URL}/models"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []

    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using Groq's API with automatic model fallback."""
        try:
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")
                
            url = f"{self.BASE_URL}/chat/completions"
            
            # First, try with the configured model
            model = settings.GROQ_MODEL
            fallback_models = ["mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"]
            
            for current_model in [model] + fallback_models:
                try:
                    payload = {
                        "model": current_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                
                    logger.debug(f"Trying model {current_model} - Sending request to Groq API")
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            url,
                            headers=self.headers,
                            json=payload,
                            timeout=30.0
                        )
                        
                        response_data = response.json()
                        logger.debug(f"Groq API response for model {current_model}: {json.dumps(response_data, indent=2)}")
                        
                        if response.status_code == 200:
                            return response_data
                        
                        # If model not found, try next fallback
                        if response.status_code == 400 and "model not found" in str(response_data).lower():
                            logger.warning(f"Model {current_model} not found, trying next fallback")
                            continue
                            
                        # For other errors, raise an exception
                        error_msg = (
                            f"Groq API request failed with status {response.status_code}: "
                            f"{response_data.get('error', {}).get('message', 'No error message')}"
                        )
                        logger.error(error_msg)
                        raise Exception(error_msg)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response for model {current_model}: {str(e)}")
                    if current_model == fallback_models[-1]:
                        raise Exception(f"Failed to decode JSON response: {str(e)}")
                    continue
                    
                except Exception as e:
                    logger.error(f"Error with model {current_model}: {str(e)}")
                    if current_model == fallback_models[-1]:  # If this was the last fallback
                        raise Exception(f"All model attempts failed. Last error: {str(e)}")
                    continue  # Try next fallback model
                    
            # This should never be reached due to the raise in the loop
            raise Exception("Failed to generate text: No working model found")
                
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
