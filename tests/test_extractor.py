import pytest
from unittest.mock import patch, MagicMock
from app.pdf.extractor import PDFExtractor, EmptyPDFError, CorruptedPDFError, ScannedPDFError

@patch("app.pdf.extractor.fitz.open")
def test_pdf_extractor_valid(mock_fitz_open):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is a valid PDF text."
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value.__enter__.return_value = mock_doc
    
    result = PDFExtractor.extract_text_with_metadata(b"%PDF-1.4 dummy", "test.pdf")
    
    assert len(result) == 1
    assert result[0]["text"] == "This is a valid PDF text."

@patch("app.pdf.extractor.fitz.open")
def test_pdf_extractor_scanned(mock_fitz_open):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "" # No text extracted
    mock_page.get_images.return_value = [(1, 2, 3)] # Has images
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value.__enter__.return_value = mock_doc
    
    with pytest.raises(ScannedPDFError):
        PDFExtractor.extract_text_with_metadata(b"%PDF-1.4 dummy", "test.pdf")

@patch("app.pdf.extractor.fitz.open")
def test_pdf_extractor_empty(mock_fitz_open):
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 0
    mock_fitz_open.return_value.__enter__.return_value = mock_doc
    
    with pytest.raises(EmptyPDFError):
        PDFExtractor.extract_text_with_metadata(b"%PDF-1.4 dummy", "test.pdf")

def test_pdf_extractor_corrupted():
    with pytest.raises(CorruptedPDFError):
        # Passing invalid bytes will cause fitz.open to fail inside the implementation
        PDFExtractor.extract_text_with_metadata(b"not a pdf", "test.pdf")
