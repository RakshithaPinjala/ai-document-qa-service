import tiktoken
import uuid
import logging
from typing import List, Dict, Any
from app.schemas.document import ChunkSchema
from app.core.config import settings

logger = logging.getLogger(__name__)

class TokenAwareChunker:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Using cl100k_base which is standard for newer OpenAI models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[ChunkSchema]:
        """
        Chunks the text from a list of pages. Ensures chunks do not exceed chunk_size.
        Attempts to preserve paragraphs or sentence boundaries where possible (simplified here by 
        relying on the tokenizer and simple text joining).
        """
        chunks = []
        
        # We will iterate through pages and build a combined token stream 
        # while keeping track of which token belongs to which page.
        # For simplicity in this implementation, we chunk per page or across pages 
        # but track the dominant page or all pages involved.
        
        current_chunk_tokens: List[int] = []
        current_chunk_pages: set[int] = set()
        
        for page in pages:
            text = page["text"]
            page_num = page["page_number"]
            metadata = page.get("metadata", {})
            
            tokens = self.tokenizer.encode(text)
            
            idx = 0
            while idx < len(tokens):
                # How much space is left in the current chunk?
                space_left = self.chunk_size - len(current_chunk_tokens)
                
                # Take tokens to fill the chunk
                take_tokens = tokens[idx:idx + space_left]
                current_chunk_tokens.extend(take_tokens)
                current_chunk_pages.add(page_num)
                
                idx += len(take_tokens)
                
                # If chunk is full, yield it and start a new one with overlap
                if len(current_chunk_tokens) >= self.chunk_size:
                    chunk_text = self.tokenizer.decode(current_chunk_tokens)
                    chunks.append(ChunkSchema(
                        chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                        page_numbers=list(sorted(current_chunk_pages)),
                        excerpt=chunk_text,
                        token_count=len(current_chunk_tokens),
                        metadata=metadata
                    ))
                    
                    # Create next chunk with overlap
                    overlap_tokens = current_chunk_tokens[-self.overlap:] if self.overlap > 0 else []
                    current_chunk_tokens = overlap_tokens
                    # Approximate the page numbers for the overlap. 
                    # If overlap is entirely from the current page, we just use the current page.
                    current_chunk_pages = {page_num}
        
        # Handle the remaining tokens
        if current_chunk_tokens and len(current_chunk_tokens) > self.overlap:
            chunk_text = self.tokenizer.decode(current_chunk_tokens)
            chunks.append(ChunkSchema(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                page_numbers=list(sorted(current_chunk_pages)),
                excerpt=chunk_text,
                token_count=len(current_chunk_tokens),
                metadata=metadata if 'metadata' in locals() else {}
            ))
            
        return chunks
