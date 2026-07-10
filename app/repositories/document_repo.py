from typing import List, Optional, Type
from sqlalchemy.orm import Session
from app.domain.interfaces import DocumentRepository
from app.models.document import Document

class SQLAlchemyDocumentRepository(DocumentRepository):
    def __init__(self, db: Session):
        self.db = db

    def create(self, document: Document) -> Document:
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get(self, document_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def get_all(self) -> List[Type[Document]]:
        return self.db.query(Document).all()

    def delete(self, document_id: str) -> bool:
        doc = self.get(document_id)
        if doc:
            self.db.delete(doc)
            self.db.commit()
            return True
        return False
