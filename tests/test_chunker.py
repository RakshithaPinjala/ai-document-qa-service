from app.chunking.splitter import TokenAwareChunker

def test_chunker_basic():
    chunker = TokenAwareChunker(chunk_size=10, overlap=2)
    pages = [
        {"page_number": 1, "text": "This is a simple test to see how chunking works.", "metadata": {"filename": "test.pdf"}}
    ]
    chunks = chunker.chunk_pages(pages)
    
    assert len(chunks) > 0
    # The first chunk should have exactly 10 tokens (or less if the text was short)
    # The sentence "This is a simple test to see how chunking works." has ~11-12 tokens.
    assert chunks[0].token_count == 10
    assert chunks[0].page_numbers == [1]
    assert chunks[0].metadata["filename"] == "test.pdf"
