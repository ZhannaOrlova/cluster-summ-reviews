''' Summarization implementation '''
# infrastructure/summarization_service.py
from domain.interfaces import SummarizationServiceInterface
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import torch
import tensorflow as tf
import os 

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf
import os

class SummarizationService:
    """Summarizes clusters using TensorFlow-based Transformers."""

    def __init__(self, model_name="sshleifer/distilbart-cnn-6-6"):
        """
        Initializes the summarization service.

        Args:
            model_name (str): Name of the model to use for summarization.
        """
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with tf.device(self.device):
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)  # Use PyTorch weights

    def summarize(self, texts, max_length=100, min_length=30):
        """
        Summarizes a list of texts.

        Args:
            texts (list[str]): List of input texts to summarize.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.

        Returns:
            str: Summarized text.
        """
        combined_text = " ".join(texts)
        inputs = self.tokenizer.encode(
            combined_text, return_tensors="tf", truncation=True, max_length=512
        )

        # Perform summarization
        with tf.device(self.device):
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
