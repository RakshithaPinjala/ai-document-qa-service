from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChunkSchema(BaseModel):
    chunk_id: str
    page_numbers: List[int]
    excerpt: str
    token_count: int
    metadata: dict

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = 0

class Citation(BaseModel):
    page: int
    chunk_id: str
    excerpt: str

class QueryResponse(BaseModel):
    status: str
    answer: Optional[str]
    confidence: Optional[float] = None
    sources: Optional[List[Citation]] = None
    reason: Optional[str] = None
