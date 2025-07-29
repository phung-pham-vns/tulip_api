"""
FastAPI application for TULIP embeddings.
"""

import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

from tulip_api.config.settings import API_TITLE, API_DESCRIPTION, API_VERSION
from tulip_api.models.embeddings import TULIPEmbedder

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Initialize model at startup
embedder = None


@app.on_event("startup")
async def startup_event():
    """Initialize TULIP model at startup."""
    global embedder
    embedder = TULIPEmbedder()


class TextRequest(BaseModel):
    """Request model for text embedding endpoint."""

    texts: List[str]
    model_name: Optional[str] = None


class TextResponse(BaseModel):
    """Response model for text embedding endpoint."""

    embeddings: List[List[float]]


class ImageResponse(BaseModel):
    """Response model for image embedding endpoint."""

    embedding: List[float]


@app.post("/embed/text", response_model=TextResponse)
async def embed_text(req: TextRequest):
    """Generate embeddings for a list of texts."""
    try:
        # Use specified model or default
        global embedder
        if req.model_name and req.model_name != embedder.model_name:
            embedder = TULIPEmbedder(model_name=req.model_name)

        # Generate embeddings
        embeddings = embedder.embed_text(req.texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/image", response_model=ImageResponse)
async def embed_image(file: UploadFile = File(...), model_name: Optional[str] = None):
    """Generate embedding for an image."""
    try:
        # Use specified model or default
        global embedder
        if model_name and model_name != embedder.model_name:
            embedder = TULIPEmbedder(model_name=model_name)

        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate embedding
        embedding = embedder.embed_image(image)
        return {"embedding": embedding[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
