from abc import ABC, abstractmethod
from typing import List
import logging
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            # OpenAI recommends batching, we assume texts list is appropriately sized
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Failed to generate embeddings via OpenAI: {e}")
            raise

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            # The SDK returns EmbedContentResponse which has a list of embeddings
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings via Gemini: {e}")
            raise

def get_embedding_provider() -> EmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingProvider()
    elif settings.EMBEDDING_PROVIDER == "gemini":
        return GeminiEmbeddingProvider()
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
