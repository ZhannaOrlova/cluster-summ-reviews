''' Abstract interfaces (contracts) '''
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class EmbeddingServiceInterface(ABC):
    """Abstract interface for embedding services"""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError 
    

class ClusteringServiceInterface(ABC):
    """Abstract interface for clustering services"""

    @abstractmethod
    def cluster_data(self, embeddings: np.ndarray, num_clusters: int = None):
        raise NotImplementedError
    

class DimensionalityReducerInterface(ABC):
    """Abstract interface for dimensionality reduction services"""

    @abstractmethod
    
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class SummarizationServiceInterface(ABC):
    """Abstract interface for summarization services"""

    @abstractmethod
    def summarize(self, texts: List[str]) -> str:
        raise NotImplementedError

