# Approach

## Understanding the Problem
The core problem is to build an enterprise-grade AI Document Q&A Service that allows users to upload complex business documents (like PDFs) and ask natural language questions to retrieve accurate, traceable answers. The key challenges are mitigating AI hallucination, handling large documents efficiently, and building a scalable, maintainable architecture.

## Task Breakdown
1. **Scaffolding and Configuration**: Set up a robust Python project using Poetry, FastAPI, and Pydantic for configuration management.
2. **Document Processing (PDF Extraction)**: Implement a mechanism to ingest PDFs and extract text while preserving critical metadata like page numbers for citations.
3. **Chunking Strategy**: Develop a token-aware text chunking strategy to break large documents into semantically meaningful pieces that fit within LLM context windows.
4. **Vector Storage and Embeddings**: Integrate an embedding provider (OpenAI) to vectorize chunks and store them in a vector database (ChromaDB) for fast similarity search.
5. **Retrieval-Augmented Generation (RAG)**: Build the core RAG logic to embed user queries, retrieve relevant chunks, inject them into a strict system prompt, and generate answers.
6. **API Layer**: Expose the functionality via RESTful FastAPI endpoints with proper error handling and validation.
7. **Testing**: Write comprehensive test suites (unit and integration) to ensure reliability.
8. **Documentation**: Document the architecture, tradeoffs, and AI usage.

## Assumptions
- The primary document format is PDF.
- We have access to the OpenAI API for embeddings (`text-embedding-ada-002`) and language models (`gpt-4` or `gpt-3.5-turbo`).
- The application will be deployed in a containerized environment (Docker).
- The documents uploaded are text-searchable (not solely scanned images, although error handling is implemented for them).

## Implementation Plan (Retrospective)
We followed a Domain-Driven Design (DDD) approach. We first defined interfaces for our external dependencies (`VectorStoreRepository`, `LLMProvider`, `EmbeddingProvider`). Then we implemented the core logic (`Chunker`, `RAGService`), followed by the infrastructure adapters (ChromaDB, OpenAI clients). Finally, we wired everything together using FastAPI dependency injection. This approach ensured that our business logic remained decoupled from specific technology choices.
