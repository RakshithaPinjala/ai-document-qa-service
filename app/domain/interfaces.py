from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class DocumentRepository(ABC):
    @abstractmethod
    def create(self, document: Any) -> Any:
        pass

    @abstractmethod
    def get(self, document_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def get_all(self) -> List[Any]:
        pass

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        pass

class VectorStoreRepository(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        pass

    @abstractmethod
    def insert(self, chunks: List[Any], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document_vectors(self, document_id: str) -> None:
        pass
