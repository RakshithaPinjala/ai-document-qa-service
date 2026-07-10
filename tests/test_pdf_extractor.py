import pytest
from app.pdf.extractor import PDFExtractor, CorruptedPDFError

def test_valid_pdf_header():
    assert PDFExtractor.is_valid_pdf_header(b"%PDF-1.4...") is True
    assert PDFExtractor.is_valid_pdf_header(b"Not a PDF") is False

def test_extract_corrupted_pdf():
    with pytest.raises(CorruptedPDFError):
        PDFExtractor.extract_text_with_metadata(b"Invalid PDF content", "test.pdf")
