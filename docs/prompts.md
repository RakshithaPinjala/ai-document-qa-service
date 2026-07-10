# AI Prompts and Evolution

## Tools Used
During the development of this project, AI assistants (specifically Antigravity / Gemini-based IDE agents) were used to bootstrap the repository, implement domain-driven design patterns, write test cases, and refactor code to eliminate placeholders.

## Core RAG Prompt
The most critical prompt in the system is the `QA_SYSTEM_PROMPT` used in the RAG service.

### Initial Iteration:
```text
You are a helpful assistant. Use the following context to answer the user's question.
Context:
{context}
```
*Issue:* The model would frequently hallucinate if the answer wasn't in the context, relying on its internal knowledge base instead.

### Final Iteration (Current):
```text
You are an expert Q&A assistant for corporate documents.
You must answer the user's question strictly using ONLY the information provided in the Context below.

Context blocks are provided in XML tags containing the chunk ID and source page numbers.
Do NOT use outside knowledge. If the answer cannot be found in the context, you must reply EXACTLY with the word "NOT_FOUND" and nothing else.

Context:
{context}
```

### How Hallucinations are Mitigated:
1. **Strict Boundary Setting:** By explicitly stating "strictly using ONLY the information provided", the LLM is constrained to the context window.
2. **Deterministic Fallback:** The instruction `reply EXACTLY with the word "NOT_FOUND"` provides a programmatic hook. The backend code intercepts this specific string and converts it into a structured HTTP response indicating the document lacks the necessary information, rather than showing the user a fabricated answer.
3. **XML Tagging:** Wrapping context chunks in `<chunk>` tags helps the LLM distinguish between the user's prompt instructions and the actual source material, mitigating prompt injection or context confusion.

## Validation and Code Corrections
During development, the AI assistant generated test mocks that occasionally assumed incorrect return types (e.g., assuming a Chroma vector store query returned a dict but mocking it to return a list).
- **Correction:** We executed `pytest`, caught the `AttributeError` or `AssertionError`, analyzed the local stack traces, and refactored the mocks using `monkeypatch` to properly mimic the database's specific return structures, ultimately achieving 100% test pass rates for the API and RAG logic.
