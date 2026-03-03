# Quick Start: Vector Store Setup

## TL;DR

```bash
# 1. Install Visual C++ Redistributable (Windows only)
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation
python scripts/test_vector_store.py

# 4. Generate embeddings (first time only)
python scripts/generate_embeddings.py --reset

# 5. Done! Vector store is ready
```

## What Just Happened?

You've set up a vector database that:
- Converts movie descriptions into 384-dimensional mathematical vectors
- Stores them in ChromaDB for fast similarity search
- Enables semantic search (e.g., "action-packed sci-fi adventure")

## Usage Example

```python
from src.vector_store import MovieVectorStore

# Initialize
vs = MovieVectorStore()

# Search by description
results = vs.search_similar_movies(
    "thrilling action adventure with heroic characters",
    n_results=5
)

# Find similar movies
similar = vs.get_similar_to_movie(movie_id=1, n_results=10)
```

## Troubleshooting

### DLL Error on Windows?
Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Want to re-index?
```bash
python scripts/generate_embeddings.py --reset
```

### Check if it's working?
```bash
python scripts/test_vector_store.py
```

## Next Steps

See [VECTOR_STORE_SETUP.md](VECTOR_STORE_SETUP.md) for:
- Detailed architecture
- API reference
- Performance tuning
- Integration examples
