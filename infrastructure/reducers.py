''' Dimensionality reduction implementations '''
from domain.interfaces import DimensionalityReducerInterface
import umap
from sklearn.decomposition import PCA
import numpy as np 

class DimensionalityReducer(DimensionalityReducerInterface):
    """Reduces embeddings dimensionality using UMAP or PCA."""

    def __init__(self, method: str = "umap", **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs

        if self.method == "umap":
            self.reducer = umap.UMAP(**self.kwargs)
        elif self.method == "pca":
            self.reducer = PCA(**self.kwargs)
        else:
            raise ValueError("Unsupported reduction method. Choose 'umap' or 'pca'.")

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduces the dimensionality of embeddings."""
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        return reduced_embeddings
