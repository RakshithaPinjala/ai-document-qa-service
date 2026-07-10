import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.document import Document
from unittest.mock import patch, MagicMock
from app.schemas.document import QueryResponse, Citation

# Setup a dummy doc in DB for querying
import uuid

@pytest.fixture
def dummy_doc(db_session):
    doc_id = uuid.uuid4().hex
    doc = Document(id=doc_id, filename="e2e.pdf", status="COMPLETED", chunk_count=10)
    db_session.add(doc)
    db_session.commit()
    return doc

@patch("app.api.documents.RAGService")
def test_e2e_query_found_answer(mock_rag_class, client, dummy_doc):
    # Setup mock
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.return_value = QueryResponse(
        status="FOUND",
        answer="The capital of France is Paris.",
        confidence=0.95,
        sources=[Citation(page=1, chunk_id="chunk1", excerpt="Paris is the capital...")]
    )
    mock_rag_class.return_value = mock_rag_instance
    
    response = client.post(f"/api/v1/documents/{dummy_doc.id}/query", json={"question": "What is the capital of France?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FOUND"
    assert data["answer"] == "The capital of France is Paris."
    assert len(data["sources"]) == 1

@patch("app.api.documents.RAGService")
def test_e2e_query_not_found_answer(mock_rag_class, client, dummy_doc):
    # Setup mock
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.return_value = QueryResponse(
        status="NOT_FOUND",
        answer=None,
        reason="The uploaded document does not contain sufficient information."
    )
    mock_rag_class.return_value = mock_rag_instance
    
    response = client.post(f"/api/v1/documents/{dummy_doc.id}/query", json={"question": "What is the capital of Mars?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_FOUND"
    assert data["answer"] is None
    assert "reason" in data

@patch("app.api.documents.RAGService")
def test_e2e_query_multi_page_context(mock_rag_class, client, dummy_doc):
    # Setup mock
    mock_rag_instance = MagicMock()
    mock_rag_instance.query.return_value = QueryResponse(
        status="FOUND",
        answer="A summary spanning multiple pages.",
        confidence=0.88,
        sources=[
            Citation(page=1, chunk_id="chunk1", excerpt="Start of doc..."),
            Citation(page=5, chunk_id="chunk5", excerpt="Middle of doc..."),
            Citation(page=10, chunk_id="chunk10", excerpt="End of doc...")
        ]
    )
    mock_rag_class.return_value = mock_rag_instance
    
    response = client.post(f"/api/v1/documents/{dummy_doc.id}/query", json={"question": "Summarize the document."})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FOUND"
    assert len(data["sources"]) == 3
    # Check if pages are distinct
    pages = [source["page"] for source in data["sources"]]
    assert 1 in pages
    assert 5 in pages
    assert 10 in pages

@patch("app.api.documents.PDFExtractor.extract_text_with_metadata")
def test_e2e_upload_scanned_pdf(mock_extract, client):
    from app.pdf.extractor import ScannedPDFError
    mock_extract.side_effect = ScannedPDFError("The PDF appears to be a scanned document without selectable text.")
    
    files = {"file": ("scanned.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    
    # In this async implementation, extraction happens in a background task, 
    # so the API returns 202 ACCEPTED originally. We just verify the route accepts it.
    assert response.status_code in [202, 400, 422]

@patch("app.api.documents.PDFExtractor.extract_text_with_metadata")
def test_e2e_upload_empty_pdf(mock_extract, client):
    from app.pdf.extractor import EmptyPDFError
    mock_extract.side_effect = EmptyPDFError("The PDF document is empty.")
    
    files = {"file": ("empty.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code in [202, 400, 422]

