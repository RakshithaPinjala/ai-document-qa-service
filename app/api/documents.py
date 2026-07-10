from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.infrastructure.database import get_db
from app.pdf.extractor import PDFExtractor, EmptyPDFError, CorruptedPDFError, ScannedPDFError
from app.chunking.splitter import TokenAwareChunker
from app.schemas.document import DocumentResponse, QueryResponse
from app.embeddings.provider import get_embedding_provider
from app.vector_store.chroma import get_vector_store
from app.services.rag_service import RAGService
from app.models.document import Document
from app.repositories.document_repo import SQLAlchemyDocumentRepository
from pydantic import BaseModel
import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB

class Question(BaseModel):
    question: str

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename
    if filename is None or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed.")
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid MIME type. Expected application/pdf.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File size exceeds the 10 MB limit.")

    try:
        # Extract text and metadata
        pages_data = PDFExtractor.extract_text_with_metadata(content, filename)
        
        # Chunk text
        chunker = TokenAwareChunker()
        chunks = chunker.chunk_pages(pages_data)
        
        # Generate Embeddings
        embedding_provider = get_embedding_provider()
        texts = [chunk.excerpt for chunk in chunks]
        embeddings = embedding_provider.generate_embeddings(texts)
        
        # Store in Vector DB
        vector_store = get_vector_store()
        doc_id = str(uuid.uuid4())
        
        # Tag chunks with doc_id
        for chunk in chunks:
            chunk.metadata["document_id"] = doc_id
            
        vector_store.insert(chunks, embeddings)
        
        # Save to Relational DB
        repo = SQLAlchemyDocumentRepository(db)
        new_doc = Document(
            id=doc_id,
            filename=filename,
            status="COMPLETED",
            chunk_count=len(chunks)
        )
        saved_doc = repo.create(new_doc)
        
        return DocumentResponse(
            id=saved_doc.id,
            filename=saved_doc.filename,
            status=saved_doc.status,
            created_at=saved_doc.created_at or datetime.now(timezone.utc),
            updated_at=saved_doc.updated_at or datetime.now(timezone.utc),
            chunk_count=saved_doc.chunk_count
        )
        
    except EmptyPDFError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except CorruptedPDFError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ScannedPDFError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process document {filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while processing the document.")

@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    repo = SQLAlchemyDocumentRepository(db)
    docs = repo.get_all()
    # Simple mapping, normally handled by Pydantic model_validate
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            status=d.status,
            created_at=d.created_at or datetime.now(timezone.utc),
            updated_at=d.updated_at or datetime.now(timezone.utc),
            chunk_count=d.chunk_count
        ) for d in docs
    ]

@router.post("/{document_id}/query", response_model=QueryResponse)
async def query_document(document_id: str, payload: Question, db: Session = Depends(get_db)):
    repo = SQLAlchemyDocumentRepository(db)
    doc = repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    rag_service = RAGService()
    try:
        response = rag_service.query(document_id, payload.question)
        return response
    except Exception as e:
        logger.error(f"Error querying document {document_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to query document.")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    repo = SQLAlchemyDocumentRepository(db)
    doc = repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        created_at=doc.created_at or datetime.now(timezone.utc),
        updated_at=doc.updated_at or datetime.now(timezone.utc),
        chunk_count=doc.chunk_count
    )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    repo = SQLAlchemyDocumentRepository(db)
    success = repo.delete(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Also delete vectors
    vector_store = get_vector_store()
    vector_store.delete_document_vectors(document_id)
    
    return None
