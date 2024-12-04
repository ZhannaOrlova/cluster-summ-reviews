# presentation/main.py

''' Entry point of the application '''
from application.cluster_service import ClusterService
from application.summarization_service import SummarizationCoordinator
from infrastructure.config import (
    get_embedding_service,
    get_clustering_service,
    get_dimensionality_reducer_for_clustering,
    get_dimensionality_reducer_for_visualization,
    get_summarization_service,
)
from infrastructure.data_loader import DataLoader
from infrastructure.visualization import VisualizationService
import logging
import os 
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    try:
        logging.info("Initializing services...")

        # Dependency injection
        embedding_service = get_embedding_service()
        clustering_service = get_clustering_service()
        reducer_for_clustering = get_dimensionality_reducer_for_clustering()
        reducer_for_visualization = get_dimensionality_reducer_for_visualization()
        summarization_service = get_summarization_service()

        # Initialize services
        cluster_service = ClusterService(
            embeddings_service=embedding_service,
            clustering_service=clustering_service,
            reducer_for_clustering=reducer_for_clustering,
            reducer_for_visualization=reducer_for_visualization,
            batch_size=1000
        )
        summarization_coordinator = SummarizationCoordinator(
            summarization_service=summarization_service
        )

        logging.info("Services initialized successfully.")

        # Load data
        data_loader = DataLoader()
        df = data_loader.load_data('data/movie_review.csv')
        texts = df['text'].tolist()
        logging.info(f"Loaded {len(texts)} reviews.")

        # Generate or load embeddings
        logging.info("Generating or loading embeddings...")
        embeddings = embedding_service.get_or_generate_embeddings(texts, save_dir="data/embeddings")

        # Process clustering
        clusters, labels, centroids, reduced_embeddings_for_visualization, representative_reviews = cluster_service.process_texts_with_embeddings(
            texts=texts,
            embeddings=embeddings,
            save_dir="data/reduced_embeddings"  # Directory for reduced embeddings
        )


        logging.info(f"Number of representative reviews: {len(representative_reviews)}")
        for cluster_id, reviews in representative_reviews.items():
            logging.info(f"Cluster {cluster_id} representatives: {reviews[:3]}")  # Log first 3 representatives

        # Summarize clusters
        logging.info("Starting summarization of clusters...")
        summarization_coordinator.summarize_clusters(clusters)
        logging.info("Summarization completed.")

        # Visualize clusters
        logging.info("Starting visualization of clusters...")
        visualization_service = VisualizationService()
        visualization_service.visualize_clusters(
            reduced_embeddings=reduced_embeddings_for_visualization,
            labels=labels,
            centroids=centroids,
            reviews=texts  # Pass actual reviews for hover text
        )
        logging.info("Visualization completed.")



    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


