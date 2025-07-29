"""
TULIP model operations for generating embeddings.
"""

from typing import List, Optional, Tuple
import torch
import open_clip
from PIL import Image
import numpy as np

from tulip_api.config.settings import TULIP_MODELS, DEFAULT_MODEL


class TULIPEmbedder:
    def __init__(self, model_name: Optional[str] = None):
        """Initialize TULIP model for embedding generation.

        Args:
            model_name: Name of the TULIP model to use. If None, uses default model.
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.model_config = TULIP_MODELS[self.model_name]

        # Load model and processors
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.model_config["model_path"]
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate normalized embedding for an image."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2.T))
