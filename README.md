# AI-Powered Document Q&A Service

A production-grade Retrieval-Augmented Generation (RAG) backend service that extracts document knowledge, stores semantic embeddings, retrieves relevant content, and generates grounded answers with strict hallucination prevention.

## Architecture

The project follows Clean Architecture and SOLID principles.
- **Presentation Layer:** FastAPI routes (`app/api/`)
- **Application Layer:** Orchestration and business logic (`app/services/`)
- **Domain Layer:** Interfaces and core schemas (`app/domain/`, `app/schemas/`)
- **Infrastructure Layer:** Database and external providers (`app/infrastructure/`, `app/embeddings/`, `app/llm/`, `app/vector_store/`)

## Features
- **PDF Upload and Validation:** Rejects corrupted, empty, or oversized PDFs. Extracts text while preserving page metadata.
- **Token-Aware Chunking:** Intelligently splits text while retaining overlap and page tracking using `tiktoken`.
- **Embeddings & Vector Search:** Uses OpenAI embeddings and ChromaDB local vector store.
- **Strict Hallucination Prevention:** The LLM is heavily instructed to answer *only* from the retrieved context or return `NOT_FOUND`.
- **Structured JSON Responses:** Including confidence scores and precise citations.

## Folder Structure
```text
app/
├── api/            # API Endpoints
├── chunking/       # Token-aware text splitting
├── core/           # Configuration and Logging
├── domain/         # Interfaces
├── embeddings/     # Embedding Providers (OpenAI)
├── infrastructure/ # DB Connection
├── llm/            # LLM Providers (OpenAI)
├── middleware/     # Custom Middleware
├── models/         # SQLAlchemy Models
├── pdf/            # PyMuPDF processing
├── prompts/        # System Prompts
├── schemas/        # Pydantic Schemas
├── services/       # RAG logic
└── vector_store/   # ChromaDB Integration
tests/              # Unit and Integration Tests
docs/               # Technical Documentation
```

## Setup & Running Locally

1. Create a `.env` file from `.env.example` and set your `OPENAI_API_KEY`.
2. Ensure you have Docker and Docker Compose installed.

### Using Docker
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

### Using Poetry (Local Python Environment)
```bash
poetry install
poetry run uvicorn app.main:app --reload
```

## Testing
Run the test suite via Poetry:
```bash
poetry run pytest
```

## Known Limitations & Future Work
- OCR is currently not supported for scanned PDFs.
- Vector store filtering by `document_id` needs strict enforcement depending on the chosen DB (currently naive in Chroma).
- Add support for pgvector alongside ChromaDB for enterprise scalability.
