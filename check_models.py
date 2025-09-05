import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.groq_service import GroqService
from app.config import get_settings

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the Groq service
    groq_service = GroqService()
    
    print("Fetching available models from Groq API...")
    try:
        models = await groq_service.list_models()
        print("\nAvailable models:")
        for model in models:
            print(f"- {model}")
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return
    
    # Test the generate_text method with the first available model
    if models:
        print(f"\nTesting chat with model: {models[0]}")
        try:
            response = await groq_service.generate_text("Hello, what can you do?", max_tokens=100)
            print("\nResponse:")
            print(response.get('choices', [{}])[0].get('message', {}).get('content', 'No content'))
        except Exception as e:
            print(f"Error testing chat: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
