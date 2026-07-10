from unittest.mock import patch, MagicMock
from app.services.rag_service import RAGService
from app.embeddings.provider import EmbeddingProvider
from app.llm.provider import LLMProvider
from app.vector_store.chroma import ChromaVectorStore

class MockEmbeddingProvider(EmbeddingProvider):
    def generate_embeddings(self, texts):
        return [[0.1] * 1536 for _ in texts]

class MockLLMProvider(LLMProvider):
    def __init__(self, response="Mocked Response"):
        self.response = response
    def generate_response(self, prompt, question):
        return self.response
def test_rag_service_not_found(monkeypatch):
    # Setup mocks for empty result
    mock_ep = MagicMock()
    mock_ep.generate_embeddings.return_value = [[0.1, 0.2]]
    
    mock_vs = MagicMock()
    mock_vs.search.return_value = []
    
    mock_llm = MagicMock()
    
    monkeypatch.setattr("app.services.rag_service.get_embedding_provider", lambda: mock_ep)
    monkeypatch.setattr("app.services.rag_service.get_vector_store", lambda: mock_vs)
    monkeypatch.setattr("app.services.rag_service.get_llm_provider", lambda: mock_llm)
    
    rag_service = RAGService()
    response = rag_service.query("test_id", "What is the capital of France?")
    
    assert response.status == "NOT_FOUND"
    assert response.answer is None

def test_rag_service_found(monkeypatch):
    # Setup mocks
    mock_ep = MagicMock()
    mock_ep.generate_embeddings.return_value = [[0.1, 0.2]]
    
    mock_vs = MagicMock()
    mock_vs.search.return_value = [{
        "chunk_id": "chunk1",
        "excerpt": "Paris is the capital of France.",
        "metadata": {"document_id": "test_id", "page_numbers": "1"},
        "distance": 0.1
    }]
    
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "The capital of France is Paris."
    
    monkeypatch.setattr("app.services.rag_service.get_embedding_provider", lambda: mock_ep)
    monkeypatch.setattr("app.services.rag_service.get_vector_store", lambda: mock_vs)
    monkeypatch.setattr("app.services.rag_service.get_llm_provider", lambda: mock_llm)
    
    rag_service = RAGService()
    response = rag_service.query("test_id", "What is the capital of France?")
    
    assert response.status == "FOUND"
    assert response.answer == "The capital of France is Paris."
    assert len(response.sources) == 1
    assert response.sources[0].page == 1
