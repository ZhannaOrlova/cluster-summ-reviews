''' Loads and preprocesses data '''
import pandas as pd
import re
from typing import List, Optional
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import logging
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


class DataLoader:
    """Loads, processes, deduplicates, and groups data."""

    def __init__(self, max_length=500, remove_stopwords=False, embedding_service=None, similarity_threshold=0.9):
        self.max_length = max_length
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        self.lemmatizer = WordNetLemmatizer()
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def clean_text(self, text):
        """Cleans text by removing special characters, applying lemmatization, and optionally removing stopwords."""
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)      # Remove numbers
        text = text.strip().lower()         # Strip whitespace and lowercase

        words = text.split()
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads and preprocesses data from a CSV file."""
        df = pd.read_csv(file_path, usecols=['text'], dtype=str)
        df.dropna(subset=['text'], inplace=True)

        df['text'] = df['text'].apply(self.clean_text)
        df['text'] = df['text'].str.slice(0, self.max_length)

        logging.info(df.info(memory_usage='deep'))
        return df

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizes embeddings to unit length for euclidean-based computations."""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def deduplicate_reviews(self, texts: List[str]) -> List[str]:
        """
        Removes highly similar or duplicate reviews based on "euclidean similarity."

        Args:
            texts (list[str]): List of review texts.

        Returns:
            list[str]: Deduplicated list of reviews.
        """
        logging.info("Generating embeddings for deduplication...")
        embeddings = self.embedding_service.generate_embeddings(texts)
        logging.info(f"Embeddings generated. Shape: {embeddings.shape}")

        distances = euclidean_distances(embeddings)
        similarity_matrix = 1 / (1 + distances)  # Convert distances to similarities
        logging.info("Euclidean similarity matrix computed.")

        to_remove = set()
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    to_remove.add(j)

        deduplicated_texts = [texts[i] for i in range(len(texts)) if i not in to_remove]
        logging.info(f"Deduplicated {len(texts) - len(deduplicated_texts)} reviews.")
        return deduplicated_texts

    def group_similar_reviews(self, texts: List[str], n_groups: int = 100) -> dict:
        """
        Groups similar reviews into larger groups using Agglomerative Clustering.

        Args:
            texts (list[str]): List of reviews.
            n_groups (int): Target number of groups.

        Returns:
            dict: A mapping of group ID to aggregated reviews.
        """
        logging.info("Generating embeddings for grouping...")
        embeddings = self.embedding_service.generate_embeddings(texts)
        logging.info(f"Embeddings generated. Shape: {embeddings.shape}")

        embeddings = self.normalize_embeddings(embeddings)
        distances = euclidean_distances(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            affinity='precomputed',
            linkage='average'
        )
        group_labels = clustering.fit_predict(distances)

        grouped_reviews = {}
        for idx, group_id in enumerate(group_labels):
            grouped_reviews.setdefault(group_id, []).append(texts[idx])

        logging.info(f"Grouped reviews into {n_groups} groups.")
        return grouped_reviews
