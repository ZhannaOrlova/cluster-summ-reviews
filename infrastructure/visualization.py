''' Visualization logic '''
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull, QhullError
import logging

class VisualizationService:
    """Creates interactive visualizations of clusters"""

    def visualize_clusters(self, reduced_embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None, reviews: list = None):
        """
        Visualizes clusters using reduced embeddings, labels, centroids, and actual reviews.

        Args:
            reduced_embeddings (np.ndarray): 2D embeddings for visualization.
            labels (np.ndarray): Cluster labels for each point (-1 for noise).
            centroids (np.ndarray): Centroids for each cluster (optional).
            reviews (list): List of reviews corresponding to each point (optional).
        """
        fig = go.Figure()

        # Define colors
        unique_labels = sorted(set(labels))
        colors = px.colors.qualitative.Plotly

        for cluster_id in unique_labels:
            if cluster_id == -1:
                # Handle noise points
                noise_points = reduced_embeddings[np.where(labels == -1)]
                if len(noise_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=noise_points[:, 0],
                        y=noise_points[:, 1],
                        mode='markers',
                        marker=dict(size=6, color='gray', opacity=0.4),
                        name='Noise',
                    ))
                continue

            # Get points and reviews in the cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = reduced_embeddings[cluster_indices]
            cluster_reviews = (
                [reviews[i] for i in cluster_indices] if reviews else [f"Point {i}" for i in cluster_indices]
            )
            color = colors[cluster_id % len(colors)]

            # Add cluster points with actual reviews as hover text
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.6),
                name=f'Cluster {cluster_id}',
                text=cluster_reviews,
                hoverinfo='text'
            ))

            # Add convex hull for the cluster
            try:
                if len(cluster_points) > 3:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        fig.add_trace(go.Scatter(
                            x=cluster_points[simplex, 0],
                            y=cluster_points[simplex, 1],
                            mode='lines',
                            line=dict(color=color, width=1),
                            showlegend=False
                        ))
            except QhullError:
                logging.warning(f"Could not compute convex hull for Cluster {cluster_id}. Points may be degenerate.")

        # Add centroids if provided
        if centroids is not None:
            for idx, centroid in enumerate(centroids):
                fig.add_trace(go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode='markers+text',
                    marker=dict(size=12, color='red', symbol='x'),
                    text=[f'Centroid {idx}'],
                    hoverinfo='text',
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            title='Cluster Visualization',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            legend_title='Clusters',
            template='plotly_white',
            hovermode='closest'
        )

        fig.show()
