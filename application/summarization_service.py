''' Handles text summarization '''
from domain.interfaces import SummarizationServiceInterface
# application/summarization_service.py

class SummarizationCoordinator:
    """ Coordinates summarization of clusters"""

    def __init__(
            self,
            summarization_service: SummarizationServiceInterface
    ):
        self.summarization_service = summarization_service

    def summarize_clusters(self, clusters: list):
        for cluster in clusters:
            texts = [review.text for review in cluster.reviews]
            print(f"Summarizing Cluster {cluster.cluster_id} with {len(texts)} reviews...")
            summary = self.summarization_service.summarize(texts)
            cluster.summary = summary 
            print(f"Cluster {cluster.cluster_id} summary completed.")
