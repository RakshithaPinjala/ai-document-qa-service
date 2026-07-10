import logging
from app.embeddings.provider import get_embedding_provider
from app.vector_store.chroma import get_vector_store
from app.llm.provider import get_llm_provider
from app.prompts.qa_prompt import QA_SYSTEM_PROMPT
from app.schemas.document import QueryResponse, Citation
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        self.vector_store = get_vector_store()
        self.llm_provider = get_llm_provider()

    def query(self, document_id: str, question: str) -> QueryResponse:
        # 1. Embed query
        query_embeddings = self.embedding_provider.generate_embeddings([question])
        query_vector = query_embeddings[0]

        # 2. Retrieve top-k chunks
        # In a real impl, we should filter by document_id. 
        # Since we use a simple chroma search, we retrieve more and filter, or use chroma's where filter.
        # Let's assume we filter after for simplicity if chroma's where isn't setup.
        # Actually, chroma supports where filter on metadata.
        retrieved_chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=settings.TOP_K,
            where={"document_id": document_id}
        )

        if not retrieved_chunks:
            return QueryResponse(
                status="NOT_FOUND",
                answer=None,
                reason="No relevant chunks found in the document."
            )

        # 3. Construct Context
        context_str = ""
        citations = []
        for chunk in retrieved_chunks:
            # Reconstruct chunk format for prompt
            pages = chunk["metadata"].get("page_numbers", "?")
            chunk_id = chunk["chunk_id"]
            excerpt = chunk["excerpt"]
            context_str += f"<chunk id=\"{chunk_id}\" pages=\"{pages}\">\n{excerpt}\n</chunk>\n\n"
            
            # Simple fallback parsing for citation pages
            try:
                page_int = int(str(pages).split(",")[0])
            except Exception:
                page_int = 1

            citations.append(Citation(
                page=page_int,
                chunk_id=chunk_id,
                excerpt=excerpt[:100] + "..." # Snippet for the citation
            ))

        prompt = QA_SYSTEM_PROMPT.format(context=context_str)

        # 4. Generate Answer
        raw_answer = self.llm_provider.generate_response(prompt, question)

        if "NOT_FOUND" in raw_answer.upper().strip():
            return QueryResponse(
                status="NOT_FOUND",
                answer=None,
                reason="The uploaded document does not contain sufficient information."
            )

        # Calculate dynamic confidence based on chunk distances
        if retrieved_chunks:
            avg_distance = sum(c.get("distance", 0.0) for c in retrieved_chunks) / len(retrieved_chunks)
            confidence = max(0.0, round(1.0 - avg_distance, 2))
        else:
            confidence = 0.0

        return QueryResponse(
            status="FOUND",
            answer=raw_answer,
            confidence=confidence,
            sources=citations
        )
