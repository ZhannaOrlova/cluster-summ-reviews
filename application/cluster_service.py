'''Coordinates the process of embedding, clustering, and dimensionality reduction'''
# application/cluster_service.py
import numpy as np
import os
import logging
from typing import List, Optional
from domain.entities import Review, Cluster
from domain.interfaces import (
    EmbeddingServiceInterface,
    ClusteringServiceInterface,
    DimensionalityReducerInterface,
)

logging.basicConfig(level=logging.INFO)


class ClusterService:
    """Coordinates embedding generation, clustering, dimensionality reduction, and cluster organization."""

    def __init__(
        self,
        embeddings_service: EmbeddingServiceInterface,
        clustering_service: ClusteringServiceInterface,
        reducer_for_clustering: DimensionalityReducerInterface,
        reducer_for_visualization: DimensionalityReducerInterface,
        batch_size: int = 1000,  # Batch size for processing embeddings
    ):
        self.embeddings_service = embeddings_service
        self.clustering_service = clustering_service
        self.reducer_for_clustering = reducer_for_clustering
        self.reducer_for_visualization = reducer_for_visualization
        self.batch_size = batch_size  # For batch processing

    def get_representative_reviews(self, reviews, embeddings, labels, centroids, n_representatives=10):
        """
        Select representative reviews for each cluster.

        Args:
            reviews (list): List of review texts.
            embeddings (np.ndarray): Embeddings of the reviews.
            labels (np.ndarray): Cluster labels.
            centroids (np.ndarray): Centroids of the clusters.
            n_representatives (int): Number of representative reviews per cluster.

        Returns:
            dict: A mapping of cluster ID to representative reviews.
        """
        representatives = {}
        for cluster_id, centroid in enumerate(centroids):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]

            # Calculate distances from centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            sorted_indices = cluster_indices[np.argsort(distances)]

            # Select closest reviews
            central_reviews = sorted_indices[:n_representatives]
            representatives[cluster_id] = [reviews[idx] for idx in central_reviews]

        return representatives

    def process_texts_with_embeddings(
        self, 
        texts: List[str] = None, 
        embeddings: np.ndarray = None,  
        n_representatives: int = 5,
        save_dir: str = "data/reduced_embeddings"
    ):
        """
        Processes texts or precomputed embeddings to cluster them and reduce dimensionality.

        Args:
            texts (list[str]): List of input texts. Required if embeddings are not provided.
            embeddings (np.ndarray): Precomputed embeddings. If provided, skips embedding generation.
            num_clusters (int): Number of clusters (optional, depends on clustering algorithm).
            n_representatives (int): Number of representative reviews per cluster.
            save_dir (str): Directory to save/load reduced embeddings.

        Returns:
            tuple: Clusters, labels, centroids, reduced embeddings, and representative reviews.
        """
        # Validate input
        if embeddings is None and texts is None:
            raise ValueError("Either `texts` or `embeddings` must be provided.")

        # Generate embeddings if not provided
        if embeddings is None:
            logging.info("Generating embeddings in batches...")
            embeddings = []
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                logging.info(f"Generating embeddings for batch {start // self.batch_size + 1}...")
                batch_embeddings = self.embeddings_service.generate_embeddings(batch_texts)
                embeddings.append(batch_embeddings)
            embeddings = np.vstack(embeddings)
            logging.info(f"Total embeddings generated. Shape: {embeddings.shape}")
        else:
            logging.info(f"Using precomputed embeddings. Shape: {embeddings.shape}")

        # Check if reduced embeddings already exist
        reduced_embeddings_path = os.path.join(save_dir, "reduced_embeddings.npy")
        if os.path.exists(reduced_embeddings_path):
            reduced_embeddings_for_clustering = self.load_reduced_embeddings(reduced_embeddings_path)
            logging.info(f"Loaded reduced embeddings from {reduced_embeddings_path}.")
        else:
            # Reduce embeddings for clustering
            logging.info("Reducing embeddings for clustering...")
            reduced_embeddings_for_clustering = self.reducer_for_clustering.reduce(embeddings)
            logging.info(f"Reduction for clustering completed. Shape: {reduced_embeddings_for_clustering.shape}")

            # Save reduced embeddings
            self.save_reduced_embeddings(reduced_embeddings_for_clustering, reduced_embeddings_path)

        # Perform clustering
        logging.info("Clustering embeddings...")
        labels, centroids = self.clustering_service.cluster_data(reduced_embeddings_for_clustering)
        logging.info(f"Clustering completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

        # Organize clusters into domain entities
        logging.info("Organizing clusters into domain entities...")
        clusters = self._organize_clusters(texts or [], labels, centroids)
        logging.info(f"Organized {len(clusters)} clusters.")

        # Reduce embeddings for visualization (optional, not always needed)
        logging.info("Reducing embeddings for visualization...")
        reduced_embeddings_for_visualization = self.reducer_for_visualization.reduce(reduced_embeddings_for_clustering)
        logging.info(f"Reduction for visualization completed. Shape: {reduced_embeddings_for_visualization.shape}")

        # Select representative reviews
        logging.info("Selecting representative reviews for each cluster...")
        representative_reviews = self.get_representative_reviews(
            reviews=texts or [],
            embeddings=reduced_embeddings_for_clustering,
            labels=labels,
            centroids=centroids,
            n_representatives=n_representatives
        )
        logging.info("Representative reviews selected.")

        # Return clusters, labels, centroids, reduced embeddings for visualization, and representative reviews
        return clusters, labels, centroids, reduced_embeddings_for_visualization, representative_reviews

    def _organize_clusters(self, texts: List[str], labels: List[int], centroids: np.ndarray = None) -> List[Cluster]:
        """
        Organizes reviews into clusters.

        Args:
            texts (list[str]): List of text documents.
            labels (list[int]): Cluster labels for each document.
            centroids (np.ndarray, optional): Cluster centroids.

        Returns:
            list[Cluster]: List of organized clusters.
        """
        clusters_dict = {}
        for idx, label in enumerate(labels):
            review = Review(id=idx, text=texts[idx])
            clusters_dict.setdefault(label, []).append(review)

        clusters = [
            Cluster(
                cluster_id=label,
                reviews=reviews,
                centroid=centroids[label] if centroids is not None and label != -1 else None,
            )
            for label, reviews in clusters_dict.items()
        ]
        return clusters

    @staticmethod
    def save_reduced_embeddings(reduced_embeddings: np.ndarray, output_path: str):
        """Saves reduced embeddings to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, reduced_embeddings)
        logging.info(f"Reduced embeddings saved to {output_path}")

    @staticmethod
    def load_reduced_embeddings(input_path: str) -> np.ndarray:
        """Loads reduced embeddings from a file."""
        logging.info(f"Loading reduced embeddings from {input_path}")
        return np.load(input_path)