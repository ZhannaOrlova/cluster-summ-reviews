''' Defines core domain entities '''
from typing import List, Optional

# Entity representing each text item to be clustered
class Review:
    """Represents a single review or text document."""
    def __init__(self, id: int, text: str):
        self.id = id
        self.text = text
        
# Represents a collection of review objects that belong to the same cluster + centroids for visualization and summary
class Cluster:
    """Represents a cluster containing multiple reviews"""
    def __init__(self, cluster_id: int, reviews: List['Review'], centroid=None, summary: str = None):
        self.cluster_id = cluster_id
        self.reviews = reviews # list of review objects
        self.centroid = centroid # centroid coordinates for visualization
        self.summary = summary # cluster summaries




