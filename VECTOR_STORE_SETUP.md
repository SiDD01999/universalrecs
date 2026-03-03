# Phase 1: Vector Foundation Setup Guide

This guide walks you through setting up the vector store infrastructure for UniversalRecs, moving from simple text matching to mathematical embeddings for enhanced movie recommendations.

## Overview

The vector foundation adds semantic search capabilities to UniversalRecs by:
- Converting movie descriptions and genres into 384-dimensional embeddings
- Storing these embeddings in ChromaDB for efficient similarity search
- Enabling content-based recommendations based on semantic similarity

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Vector Foundation                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  movies.csv  ──┐                                            │
│                │                                             │
│                ├─► sentence-transformers ──► 384D Vectors   │
│                │   (all-MiniLM-L6-v2)                        │
│                │                                             │
│  [Title +      │                     │                       │
│   Genres +     │                     │                       │
│   Description] │                     ▼                       │
│                │              ┌──────────────┐               │
│                └─────────────►│  ChromaDB    │               │
│                               │  Collection  │               │
│                               │              │               │
│                               │  - Vectors   │               │
│                               │  - Metadata  │               │
│                               │  - Documents │               │
│                               └──────────────┘               │
│                                     │                        │
│                                     ▼                        │
│                               Semantic Search                │
│                               - Similar movies               │
│                               - Content-based recs           │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Microsoft Visual C++ Redistributable (Windows Only)

PyTorch (required by sentence-transformers) needs the Visual C++ runtime.

**Download and Install:**
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Run the installer
- Restart your terminal after installation

### 2. Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- `chromadb` - Local vector database
- `sentence-transformers` - Embedding model library
- `torch` - PyTorch (required by sentence-transformers)

### 3. Verify Installation

Run the test script to ensure all dependencies are properly installed:

```bash
python scripts/test_vector_store.py
```

Expected output:
```
============================================================
Testing Vector Store Dependencies
============================================================

[1/6] Testing pandas import...
  [OK] pandas imported successfully

[2/6] Testing numpy import...
  [OK] numpy imported successfully

[3/6] Testing chromadb import...
  [OK] chromadb imported successfully

[4/6] Testing torch import...
  [OK] torch imported successfully (version: X.X.X)

[5/6] Testing sentence-transformers import...
  [OK] sentence-transformers imported successfully

[6/6] Testing vector_store module import...
  [OK] vector_store module imported successfully

============================================================
All dependencies installed correctly!
============================================================
```

## Usage

### Generate and Index Embeddings

To create embeddings for all movies and store them in ChromaDB:

```bash
python scripts/generate_embeddings.py --reset
```

**Options:**
- `--reset` - Delete existing collection and recreate it
- `--persist-dir PATH` - Custom directory for ChromaDB data (default: `./chroma_db`)
- `--batch-size N` - Number of movies to process at once (default: 100)

**First Run Output:**
```
============================================================
UniversalRecs - Embedding Generation & Indexing
============================================================

[1/3] Loading movie data...
  [OK] Loaded 100 movies and 500 ratings

[2/3] Initializing vector store...
Loading embedding model: all-MiniLM-L6-v2...
Created new collection 'movies'.

[3/3] Generating embeddings and indexing movies...

Indexing 100 movies into vector store...
Generating embeddings...
  Processed 100/100 movies
Storing in ChromaDB...
[OK] Successfully indexed 100 movies!
  Collection size: 100
```

### Using the Vector Store in Python

```python
from src.vector_store import MovieVectorStore
from src.data_loader import load_data

# Initialize vector store
vector_store = MovieVectorStore()

# Semantic search
query = "action-packed sci-fi adventure with heroic characters"
movie_ids, distances, metadatas = vector_store.search_similar_movies(
    query,
    n_results=5
)

# Find similar movies
similar_ids, distances, metadatas = vector_store.get_similar_to_movie(
    movie_id=1,
    n_results=10
)

# Get statistics
stats = vector_store.get_stats()
print(f"Total movies indexed: {stats['total_movies']}")
```

## File Structure

```
universalrecs/
├── src/
│   └── vector_store.py          # Core vector store module
├── scripts/
│   ├── generate_embeddings.py   # Indexing script
│   └── test_vector_store.py     # Dependency test script
├── chroma_db/                   # ChromaDB persistence (auto-created)
├── requirements.txt             # Updated with new dependencies
└── VECTOR_STORE_SETUP.md        # This file
```

## Key Features

### MovieVectorStore Class

The main class provides:

1. **Embedding Generation**
   - Model: `all-MiniLM-L6-v2` (384 dimensions)
   - Combines title, genres, and description into searchable text
   - Batch processing for efficiency

2. **ChromaDB Integration**
   - Persistent local storage
   - Cosine similarity search
   - Metadata storage alongside vectors

3. **Search Capabilities**
   - `search_similar_movies()` - Semantic search by text query
   - `get_similar_to_movie()` - Find similar movies by ID
   - `get_movie_embedding()` - Retrieve raw embedding vector

4. **Utilities**
   - `index_movies()` - Batch index from DataFrame
   - `reset_collection()` - Clear and recreate collection
   - `get_stats()` - Collection statistics

## Technical Details

### Embedding Model: all-MiniLM-L6-v2

- **Dimensions:** 384
- **Type:** Sentence Transformer
- **Use Case:** Semantic similarity
- **Speed:** Fast inference (~4000 sentences/second on CPU)
- **Size:** ~90MB download

### ChromaDB Configuration

- **Storage:** Persistent local disk
- **Distance Metric:** Cosine similarity
- **Index:** HNSW (Hierarchical Navigable Small World)
- **Metadata:** JSON format

### Text Combination Strategy

Movies are represented as:
```
"{title}. Genres: {genres}. {description}"
```

Example:
```
"Movie 15 (2015). Genres: Action, Sci-Fi, Drama. A action movie about heroic characters in a futuristic world."
```

## Performance Considerations

- **Indexing:** ~100 movies/minute (CPU, batch_size=100)
- **Search:** <100ms for top-k queries (k=10)
- **Storage:** ~1KB per movie (vector + metadata)
- **Memory:** ~500MB model + vectors

## Troubleshooting

### DLL Load Error (Windows)

**Error:**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```

**Solution:**
Install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### ChromaDB Permission Error

**Error:**
```
PermissionError: [Errno 13] Permission denied: './chroma_db'
```

**Solution:**
1. Close any processes accessing the `chroma_db` directory
2. Use `--persist-dir` to specify a different location
3. Run with appropriate permissions

### Out of Memory

**Error:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:...]
```

**Solution:**
Reduce batch size:
```bash
python scripts/generate_embeddings.py --batch-size 50
```

## Next Steps

After setting up the vector foundation:

1. **Phase 2:** Integrate vector search into the recommender engine
2. **Phase 3:** Combine collaborative filtering with semantic similarity
3. **Phase 4:** Add vector-based cold start handling
4. **Phase 5:** Implement hybrid scoring with explainability

## Testing

The vector store includes built-in testing:

```bash
# Test dependencies
python scripts/test_vector_store.py

# Test full pipeline (with test queries)
python scripts/generate_embeddings.py --reset

# Test in Python REPL
python -c "from src.vector_store import MovieVectorStore; vs = MovieVectorStore(); print(vs.get_stats())"
```

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
