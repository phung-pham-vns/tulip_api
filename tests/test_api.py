"""
Tests for TULIP API endpoints.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tulip_api.api.app import app
from tulip_api.config.settings import TULIP_MODELS


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Load a sample image for testing."""
    image_path = (
        Path(__file__).parent.parent
        / "models"
        / "open_clip"
        / "images"
        / "iStock-1052880600-1024x683.jpg"
    )
    return image_path


def test_text_embedding(client):
    """Test text embedding endpoint."""
    texts = ["A photo of a red tulip", "A beautiful garden"]
    response = client.post("/embed/text", json={"texts": texts})

    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == len(texts)
    assert all(isinstance(emb, list) for emb in data["embeddings"])


def test_image_embedding(client, sample_image):
    """Test image embedding endpoint."""
    with open(sample_image, "rb") as f:
        response = client.post("/embed/image", files={"file": ("image.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)


def test_model_switching(client):
    """Test switching between different models."""
    texts = ["A test text"]

    # Test with default model
    response1 = client.post("/embed/text", json={"texts": texts})
    assert response1.status_code == 200

    # Test with specific model
    model_name = "TULIP-B-16-224"
    response2 = client.post("/embed/text", json={"texts": texts, "model_name": model_name})
    assert response2.status_code == 200


def test_invalid_model(client):
    """Test error handling for invalid model name."""
    response = client.post("/embed/text", json={"texts": ["test"], "model_name": "INVALID_MODEL"})
    assert response.status_code == 500


def test_invalid_image(client):
    """Test error handling for invalid image."""
    invalid_image = io.BytesIO(b"invalid image data")
    response = client.post(
        "/embed/image", files={"file": ("invalid.jpg", invalid_image, "image/jpeg")}
    )
    assert response.status_code == 500
