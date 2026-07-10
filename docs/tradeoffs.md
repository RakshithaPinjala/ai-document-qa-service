# Tradeoffs and Decisions

## Chunking Strategy and Size
We selected a **token-aware chunking strategy** with a target size of **500 tokens** and a **50-token overlap**.
- **Why 500 tokens?** This size strikes a balance between providing enough context for the LLM to understand the nuance of a section (unlike very small chunks of ~100 tokens) and maintaining high retrieval precision (unlike very large chunks of >1000 tokens which dilute the embedding vector's specificity).
- **Alternative Considered:** Sentence-based chunking. While semantically pure, it often results in chunks that are too small and lack surrounding context, requiring complex windowed retrieval strategies.
- **Overlap (50 tokens):** Ensures that concepts split across chunk boundaries are not lost.

## Vector Store
We selected **ChromaDB** for local development and testing, wrapped behind a `VectorStoreRepository` interface.
- **Tradeoff:** ChromaDB is lightweight and easy to embed within the application without requiring a separate service. However, for a massive-scale enterprise deployment, an external managed vector database like Pinecone, Weaviate, or Postgres with `pgvector` would offer better scalability. The interface design allows seamless swapping to `pgvector` in the future.

## LLM and Embedding Providers
We used **OpenAI** (`text-embedding-ada-002` and `gpt-4/gpt-3.5-turbo`).
- **Tradeoff:** OpenAI provides state-of-the-art out-of-the-box performance, minimizing time spent tuning models. The tradeoff is vendor lock-in and API costs. We mitigated the lock-in by using abstract `LLMProvider` and `EmbeddingProvider` interfaces.

## Cost Estimation (1,000 queries/day)
Assuming:
- Model: `gpt-3.5-turbo`
- Average query: 20 tokens
- Average retrieved context (5 chunks * 500 tokens): 2,500 tokens
- Average generated answer: 100 tokens
- Cost per 1K input tokens: ~$0.0005
- Cost per 1K output tokens: ~$0.0015

**Per Query Cost:**
- Input: 2,520 tokens * $0.0005 / 1000 = $0.00126
- Output: 100 tokens * $0.0015 / 1000 = $0.00015
- Embeddings (`text-embedding-ada-002`, $0.0001/1K tokens): 20 tokens = negligible
- Total per query: ~$0.00141

**Daily Cost (1,000 queries):** ~$1.41
**Monthly Cost:** ~$42.30

*Note: This does not include the one-time embedding cost during document upload, which depends heavily on document volume.*

## Known Limitations
1. **Multi-hop reasoning:** The system relies on semantic similarity for retrieval. Questions that require synthesizing information from highly disparate parts of a document may fail if all necessary chunks are not in the top-K retrieved results.
2. **Tables and Complex Formatting:** PyMuPDF extracts raw text. Complex tables or heavily formatted multi-column layouts might lose structural integrity, slightly degrading RAG performance on tabular data.
3. **Scanned PDFs:** The current implementation detects scanned PDFs (images without text layers) and throws a specific error, but it does not perform OCR.
