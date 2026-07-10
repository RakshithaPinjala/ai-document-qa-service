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
            logger.error(f"Failed to generate LLM response: {e}")
            raise

def get_llm_provider() -> LLMProvider:
    if settings.LLM_PROVIDER == "openai":
        return OpenAILLMProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
