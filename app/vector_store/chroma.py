import chromadb
from typing import List, Dict, Any, Optional
from app.domain.interfaces import VectorStoreRepository
from app.schemas.document import ChunkSchema
import logging

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStoreRepository):
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection: Optional[Any] = None

    def create_collection(self, collection_name: str) -> None:
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def insert(self, chunks: List[ChunkSchema], embeddings: List[List[float]]) -> None:
        if not self.collection:
            raise ValueError("Collection not initialized.")
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.excerpt for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta["page_numbers"] = ",".join(map(str, chunk.page_numbers))
            meta["token_count"] = chunk.token_count
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.collection:
            raise ValueError("Collection not initialized.")

        query_args = {
            "query_embeddings": [query_vector],
            "n_results": top_k
        }
        if where:
            query_args["where"] = where

        results = self.collection.query(**query_args)
        
        chunks = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "chunk_id": results["ids"][0][i],
                    "excerpt": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                })
        return chunks

    def delete_document_vectors(self, document_id: str) -> None:
        if not self.collection:
            return
        
        self.collection.delete(
            where={"filename": {"$eq": document_id}} # Assuming filename is used to track doc, should be doc_id in real impl.
        )

def get_vector_store() -> VectorStoreRepository:
    store = ChromaVectorStore()
    store.create_collection("document_chunks")
    return store
