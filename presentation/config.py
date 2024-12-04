''' Dependency injection setup '''
from infrastructure.embedding_service import EmbeddingService
from infrastructure.clustering_service import ClusteringService
from infrastructure.reducers import DimensionalityReducer
from infrastructure.summarization_service import SummarizationService
import os 
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# Centralized configuration
CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "clustering": {
        "min_cluster_size": 50, # Larger clusters
        "min_samples": 20, # More points required for core clusters
        "metric": "euclidean", # Distance metric aligned with embeddings
        "cluster_selection_epsilon": 0.3 # Distance metric aligned with embeddings
    },
    "reduction": {
        "method_clustering": "umap",  # 'umap' for clustering
        "n_components_clustering": 50,
        "method_visualization": "umap",  # 'pca' for visualization
        "n_components_visualization": 2,
        "n_neighbors": 30,
        "min_dist": 0.05,
        "metric": "euclidean"
    },
    "summarization_model": "sshleifer/distilbart-cnn-6-6",
    "device_embedding": "mps",  # Use 'mps' for embeddings
    "device_summarization": "mps"  # Use 'cpu' for summarization to avoid MPS issues
}

def get_embedding_service():
    """Returns an embedding service instance."""
    return EmbeddingService(model_name=CONFIG["embedding_model"])

def get_clustering_service():
    """Returns a clustering service instance with configured parameters."""
    return ClusteringService(
        min_cluster_size=CONFIG["clustering"]["min_cluster_size"],
        min_samples=CONFIG["clustering"]["min_samples"],
        metric=CONFIG["clustering"]["metric"]
    )

def get_dimensionality_reducer_for_clustering():
    """Returns a dimensionality reducer for clustering."""
    return DimensionalityReducer(
        method=CONFIG["reduction"]["method_clustering"],
        n_components=CONFIG["reduction"]["n_components_clustering"],
        n_neighbors=CONFIG["reduction"]["n_neighbors"],
        min_dist=CONFIG["reduction"]["min_dist"],
        metric=CONFIG["reduction"]["metric"],  # Pass metric
        random_state=42
    )

def get_dimensionality_reducer_for_visualization():
    """Returns a dimensionality reducer for visualization."""
    return DimensionalityReducer(
        method=CONFIG["reduction"]["method_visualization"],
        n_components=CONFIG["reduction"]["n_components_visualization"],
        metric=CONFIG["reduction"]["metric"],  # Pass metric
        random_state=42
    )

def get_summarization_service():
    """Returns a summarization service instance."""
    return SummarizationService(model_name="sshleifer/distilbart-cnn-6-6")