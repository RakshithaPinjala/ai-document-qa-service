import pytest
from unittest.mock import patch, MagicMock
from app.models.document import Document
from app.schemas.document import ChunkSchema

@patch("app.api.documents.PDFExtractor.extract_text_with_metadata")
@patch("app.api.documents.get_embedding_provider")
@patch("app.api.documents.get_vector_store")
def test_upload_valid_pdf(mock_get_vs, mock_get_ep, mock_extract, client, db_session):
    mock_extract.return_value = [{"text": "Sample text", "page_number": 1}]
    
    mock_ep = MagicMock()
    mock_ep.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_get_ep.return_value = mock_ep
    
    mock_vs = MagicMock()
    mock_get_vs.return_value = mock_vs
    
    # Needs a minimal valid PDF content to pass PyMuPDF checks if not mocked at router level, 
    # but we patched extract_text_with_metadata, so we just need a valid extension.
    files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert "id" in data
    
    # Check if saved to DB
    doc_id = data["id"]
    doc = db_session.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None
    assert doc.filename == "test.pdf"

@patch("app.api.documents.RAGService")
def test_query_document_not_found(mock_rag, client):
    response = client.post("/api/v1/documents/non_existent_id/query", json={"question": "What?"})
    assert response.status_code == 404

def test_delete_document_success(client, db_session):
    # Setup test doc
    doc = Document(id="test_doc_id", filename="test.pdf", status="COMPLETED", chunk_count=1)
    db_session.add(doc)
    db_session.commit()
    
    with patch("app.api.documents.get_vector_store") as mock_get_vs:
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        
        response = client.delete("/api/v1/documents/test_doc_id")
        assert response.status_code == 204
        
        # Verify deletion from DB
        assert db_session.query(Document).filter(Document.id == "test_doc_id").first() is None
        mock_vs.delete_document_vectors.assert_called_once_with("test_doc_id")

def test_get_document_success(client, db_session):
    doc = Document(id="test_doc_id_2", filename="test2.pdf", status="COMPLETED", chunk_count=2)
    db_session.add(doc)
    db_session.commit()
    
    response = client.get("/api/v1/documents/test_doc_id_2")
    assert response.status_code == 200
    assert response.json()["id"] == "test_doc_id_2"
    assert response.json()["filename"] == "test2.pdf"

def test_list_documents(client, db_session):
    doc = Document(id="test_doc_id_3", filename="test3.pdf", status="COMPLETED", chunk_count=3)
    db_session.add(doc)
    db_session.commit()
    
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert any(d["id"] == "test_doc_id_3" for d in response.json())
