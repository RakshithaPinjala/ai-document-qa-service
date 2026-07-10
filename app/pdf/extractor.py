import fitz # PyMuPDF
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    pass

class EmptyPDFError(PDFExtractionError):
    pass

class CorruptedPDFError(PDFExtractionError):
    pass

class ScannedPDFError(PDFExtractionError):
    pass

class PDFExtractor:
    @staticmethod
    def is_valid_pdf_header(content: bytes) -> bool:
        return content.startswith(b"%PDF-")

    @staticmethod
    def extract_text_with_metadata(content: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF document, preserving page numbers and metadata.
        Returns a list of pages with their extracted text and metadata.
        """
        if not PDFExtractor.is_valid_pdf_header(content):
            raise CorruptedPDFError("Invalid PDF header signature.")

        pages_data = []
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF {filename}: {str(e)}")
            raise CorruptedPDFError("The PDF file is corrupted and cannot be read.")

        if doc.page_count == 0:
            raise EmptyPDFError("The PDF document contains no pages.")

        total_text_length = 0
        total_images = 0

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            
            image_list = page.get_images(full=True)
            total_images += len(image_list)

            if text:
                total_text_length += len(text)
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "metadata": {
                        "filename": filename,
                    }
                })

        doc.close()

        if len(pages_data) == 0:
            if total_images > 0:
                raise ScannedPDFError("The PDF appears to be scanned. OCR is not currently supported.")
            else:
                raise EmptyPDFError("The PDF document contains no extractable text.")

        return pages_data
