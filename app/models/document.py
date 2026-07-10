import uuid
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.sql import func
from app.infrastructure.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    filename = Column(String, index=True, nullable=False)
    status = Column(String, default="PROCESSING", nullable=False) # PROCESSING, COMPLETED, ERROR
    chunk_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
