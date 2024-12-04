''' Generates text embeddings '''
from domain.interfaces import EmbeddingServiceInterface
from dotenv import load_dotenv
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import tensorflow as tf
import os
import logging
import hashlib
from typing import List


logging.basicConfig(level=logging.INFO)

load_dotenv(dotenv_path="./.env")

TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if TOKEN is None:
    raise ValueError("HUGGINGFACE_API_TOKEN is not set in the environment or .env file.")


class EmbeddingService(EmbeddingServiceInterface):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initializes the embedding service with the specified model.

        Args:
            model_name (str): Name of the Hugging Face model to use.
        """
        if tf.config.list_physical_devices('GPU'):
            self.device = '/GPU:0'  # Use GPU if available
            logging.info("Using GPU for embeddings.")
        else:
            self.device = '/CPU:0'
            logging.info("Using CPU for embeddings.")

        logging.info(f"Selected device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
            with tf.device(self.device):
                self.model = TFAutoModel.from_pretrained(model_name, token=TOKEN)
        except Exception as e:
            raise ValueError(f"Failed to initialize model or tokenizer: {e}")

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalizes embeddings to unit length.

        Args:
            embeddings (np.ndarray): Input embeddings.

        Returns:
            np.ndarray: Normalized embeddings.
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list[str]): Input texts.

        Returns:
            np.ndarray: Normalized embeddings.
        """
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input `texts` must be a non-empty list of strings.")

        inputs = self.tokenizer(
            texts,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=512
        )

        logging.info(f"Generating embeddings on device: {self.device}")

        with tf.device(self.device):
            outputs = self.model(**inputs)

        attention_mask = tf.cast(inputs['attention_mask'], dtype=tf.float32)
        pooled_output = tf.reduce_sum(
            outputs.last_hidden_state * tf.expand_dims(attention_mask, axis=-1),
            axis=1
        ) / tf.reduce_sum(attention_mask, axis=1, keepdims=True)

        embeddings = pooled_output.numpy()
        return self.normalize_embeddings(embeddings)

    @staticmethod
    def hash_texts(texts: List[str]) -> str:
        """Creates a unique hash for a list of texts."""
        return hashlib.md5("".join(sorted(texts)).encode()).hexdigest()

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, output_path: str):
        """Saves embeddings to a file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, embeddings)
            logging.info(f"Embeddings saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save embeddings: {e}")
            raise

    @staticmethod
    def load_embeddings(input_path: str) -> np.ndarray:
        """Loads embeddings from a file."""
        logging.info(f"Loading embeddings from {input_path}")
        embeddings = np.load(input_path)
        logging.info(f"Loaded embeddings of shape {embeddings.shape}")
        # Normalize loaded embeddings
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def process_batches(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Processes texts in batches to generate embeddings.

        Args:
            texts (list[str]): A flat list of text strings.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Combined embeddings for all texts.
        """
        all_embeddings = []
        num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)

        for i in range(num_batches):
            batch = texts[i * batch_size : (i + 1) * batch_size]
            logging.info(f"Processing batch {i + 1}/{num_batches} with batch size {batch_size}...")
            batch_embeddings = self.generate_embeddings(batch)
            # Normalize batch embeddings
            batch_embeddings = self.normalize_embeddings(batch_embeddings)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def get_or_generate_embeddings(self, texts: List[str], save_dir="data/embeddings") -> np.ndarray:
        """
        Generates embeddings if they don't exist, otherwise loads them from disk.

        Args:
            texts (list[str]): Input texts for which embeddings are needed.
            save_dir (str): Directory where embeddings will be saved.

        Returns:
            np.ndarray: Normalized embeddings for the input texts.
        """
        hash_key = self.hash_texts(texts)
        output_path = os.path.join(save_dir, f"{hash_key}.npy")

        if os.path.exists(output_path):
            return self.load_embeddings(output_path)

        # Generate embeddings if not saved
        embeddings = self.process_batches(texts)
        self.save_embeddings(embeddings, output_path)
        return embeddings
