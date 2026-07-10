import pytest
from unittest.mock import patch, MagicMock
from app.embeddings.provider import OpenAIEmbeddingProvider
from app.llm.provider import OpenAILLMProvider
from app.vector_store.chroma import ChromaVectorStore
from app.schemas.document import ChunkSchema
from app.pdf.extractor import PDFExtractor
import fitz

def test_embedding_provider_generate(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])])
    
    provider = OpenAIEmbeddingProvider()
    provider.client = mock_client
    
    embeddings = provider.generate_embeddings(["test1", "test2"])
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]

def test_llm_provider_generate(monkeypatch):
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.message.content = "Answer"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_message])
    
    provider = OpenAILLMProvider()
    provider.client = mock_client
    
    response = provider.generate_response("System", "User")
    assert response == "Answer"

@patch("app.vector_store.chroma.chromadb.PersistentClient")
def test_chroma_vector_store(mock_chroma_client):
    mock_collection = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    mock_chroma_client.return_value = mock_client_instance
    
    store = ChromaVectorStore()
    store.create_collection("test")
    
    chunk = ChunkSchema(chunk_id="1", page_numbers=[1], token_count=10, excerpt="test", metadata={})
    store.insert([chunk], [[0.1]])
    
    mock_collection.upsert.assert_called_once()
    
    mock_collection.query.return_value = {
        "ids": [["1"]],
        "distances": [[0.1]],
        "documents": [["test"]],
        "metadatas": [[{"page_numbers": "1"}]]
    }
    
    results = store.search([0.1], 1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"
    
    store.delete_document_vectors("test_doc")
    mock_collection.delete.assert_called_once()
