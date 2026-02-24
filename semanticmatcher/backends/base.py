from abc import ABC, abstractmethod
from typing import Any, Union, List

class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass

class RerankerBackend(ABC):
    @abstractmethod
    def score(self, query: str, docs: list[str]) -> List[Any]:
        pass
