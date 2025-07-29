# TULIP API

FastAPI service for [TULIP](https://tulip-berkeley.github.io) model embeddings. This service provides endpoints for generating embeddings from both text and images using the TULIP model.

## Features

- Text embedding generation
- Image embedding generation
- Support for multiple TULIP models
- Model switching at runtime
- Docker support
- Comprehensive test suite

## Project Structure

```
tulip_api/
├── src/
│   └── tulip_api/
│       ├── api/          # FastAPI application and endpoints
│       ├── config/       # Configuration and settings
│       └── models/       # Model operations and embedding logic
├── tests/               # Test suite
├── models/             # Model checkpoints
│   └── open_clip/
├── pyproject.toml      # Project configuration and dependencies
└── Dockerfile         # Container configuration
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/your-username/tulip_api.git
cd tulip_api
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

4. Download TULIP model checkpoints:
```bash
# Place the checkpoints in the models/open_clip directory
mkdir -p models/open_clip

# Download your specific TULIP checkpoints and place them in models/open_clip/
[Checkpoint Download](https://github.com/tulip-berkeley/open_clip?tab=readme-ov-file#model-checkpoints)
```

## Usage

### Running the API Server

Start the API server:

```bash
uvicorn tulip_api.api.app:app --reload
```

The API will be available at `http://localhost:8000` with the following endpoints:

- `POST /embed/text`: Generate embeddings for a list of texts
- `POST /embed/image`: Generate embeddings for an image

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Requests

1. Text Embedding:
```python
import requests

response = requests.post(
    "http://localhost:8000/embed/text",
    json={
        "texts": ["A photo of a red tulip", "A beautiful garden"],
        "model_name": "TULIP-so400m-14-384"  # Optional
    }
)
embeddings = response.json()["embeddings"]
```

2. Image Embedding:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/embed/image",
        files={"file": f}
    )
embedding = response.json()["embedding"]
```

## Development

This project uses modern Python development tools:

- [Black](https://black.readthedocs.io/) for code formatting
- [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting
- [Pytest](https://docs.pytest.org/) for testing

Format code:
```bash
black src tests
```

Lint code:
```bash
ruff check src tests
```

Run tests:
```bash
pytest
```

## Docker Deployment

Build and run the Docker container:

```bash
# Build the image
docker build -t tulip-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/models tulip-api
```
