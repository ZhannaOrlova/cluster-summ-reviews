from domain.interfaces import ClusteringServiceInterface
from hdbscan import HDBSCAN
import numpy as np
import logging


class ClusteringService(ClusteringServiceInterface):
    """Implements clustering functionality using HDBSCAN."""

    def __init__(self, min_cluster_size: int = 20, min_samples: int = 10, metric: str = "euclidean"):
        """
        Initializes the clustering service.

        Args:
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            min_samples (int): Minimum samples per cluster for HDBSCAN.
            metric (str): Metric used for distance calculation.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        logging.info(f"Using metric: {self.metric} for clustering.")
        
    def cluster_data(self, embeddings: np.ndarray):
        """
        Clusters embeddings using HDBSCAN.

        Args:
            embeddings (np.ndarray): Embeddings to cluster.
            num_clusters (int): Not used, included for compatibility.

        Returns:
            tuple: (labels, centroids)
        """
        logging.info("Performing clustering using HDBSCAN...")

        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )
        labels = hdbscan.fit_predict(embeddings)

        # Compute centroids for each cluster (excluding noise)
        centroids = [
            embeddings[labels == cluster_label].mean(axis=0)
            for cluster_label in set(labels) if cluster_label != -1
        ]

        logging.info(f"Clustering completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
        return labels, np.array(centroids)
