from abc import ABC, abstractmethod
from openai import OpenAI
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, question: str) -> str:
        pass

class OpenAILLMProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_response(self, prompt: str, question: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.0 # Lowest temperature for facts
            )
            return response.choices[0].message.content or "NOT_FOUND"
        except Exception as e:
            logger.error(f"Failed to generate LLM response via OpenAI: {e}")
            raise

class GeminiLLMProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate_response(self, prompt: str, question: str) -> str:
        try:
            full_prompt = f"{prompt}\n\nUser Question:\n{question}"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                # In google-genai, we pass config as a dict or GenerateContentConfig
                # temperature=0.0 is passed in config
                config={"temperature": 0.0}
            )
            return response.text or "NOT_FOUND"
        except Exception as e:
            logger.error(f"Failed to generate LLM response via Gemini: {e}")
            raise

def get_llm_provider() -> LLMProvider:
    if settings.LLM_PROVIDER == "openai":
        return OpenAILLMProvider()
    elif settings.LLM_PROVIDER == "gemini":
        return GeminiLLMProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
