"""
Vector Store Module for UniversalRecs
Handles ChromaDB integration and embedding management for movie recommendations.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json


class MovieVectorStore:
    """
    Manages movie embeddings using ChromaDB and sentence-transformers.

    Features:
    - Generates 384-dimensional embeddings using all-MiniLM-L6-v2
    - Stores embeddings with metadata in ChromaDB
    - Supports semantic search and similarity queries
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "movies",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Path to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            model_name: SentenceTransformer model to use for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name

        # Initialize sentence transformer model
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} items.")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"Created new collection '{collection_name}'.")

    def _create_movie_text(self, title: str, genres: str, description: str) -> str:
        """
        Combine movie attributes into a single text for embedding.

        Args:
            title: Movie title
            genres: Pipe-separated genres
            description: Movie description

        Returns:
            Combined text representation
        """
        # Replace pipes with commas for better readability
        genres_formatted = genres.replace("|", ", ")
        return f"{title}. Genres: {genres_formatted}. {description}"

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.

        Args:
            text: Input text

        Returns:
            384-dimensional embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def index_movies(self, movies_df: pd.DataFrame, batch_size: int = 100):
        """
        Index all movies from DataFrame into ChromaDB.

        Args:
            movies_df: DataFrame with columns [movieId, title, genres, description]
            batch_size: Number of movies to process at once
        """
        print(f"\nIndexing {len(movies_df)} movies into vector store...")

        # Prepare data for indexing
        movie_ids = []
        movie_texts = []
        metadatas = []

        for idx, row in movies_df.iterrows():
            movie_id = str(row['movieId'])
            title = row['title']
            genres = row['genres']
            description = row['description']

            # Create combined text for embedding
            movie_text = self._create_movie_text(title, genres, description)

            movie_ids.append(movie_id)
            movie_texts.append(movie_text)
            metadatas.append({
                "movieId": int(row['movieId']),
                "title": title,
                "genres": genres,
                "description": description
            })

        # Generate embeddings in batches
        print("Generating embeddings...")
        all_embeddings = []

        for i in range(0, len(movie_texts), batch_size):
            batch_texts = movie_texts[i:i + batch_size]
            batch_embeddings = self.generate_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i + batch_size, len(movie_texts))}/{len(movie_texts)} movies")

        # Add to ChromaDB in batches
        print("Storing in ChromaDB...")
        for i in range(0, len(movie_ids), batch_size):
            batch_ids = movie_ids[i:i + batch_size]
            batch_embeddings = all_embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = movie_texts[i:i + batch_size]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )

        print(f"✓ Successfully indexed {len(movie_ids)} movies!")
        print(f"  Collection size: {self.collection.count()}")

    def search_similar_movies(
        self,
        query: str,
        n_results: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for movies similar to the query text.

        Args:
            query: Search query (can be a description, genre, or natural language)
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"genres": "Action"})

        Returns:
            Tuple of (movie_ids, distances, metadatas)
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )

        # Extract results
        movie_ids = [int(meta['movieId']) for meta in results['metadatas'][0]]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        return movie_ids, distances, metadatas

    def get_movie_embedding(self, movie_id: int) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a specific movie.

        Args:
            movie_id: Movie ID

        Returns:
            Embedding vector or None if not found
        """
        try:
            result = self.collection.get(
                ids=[str(movie_id)],
                include=['embeddings']
            )
            if result['embeddings']:
                return result['embeddings'][0]
        except Exception as e:
            print(f"Error retrieving embedding for movie {movie_id}: {e}")
        return None

    def get_similar_to_movie(
        self,
        movie_id: int,
        n_results: int = 10
    ) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Find movies similar to a given movie.

        Args:
            movie_id: Reference movie ID
            n_results: Number of similar movies to return

        Returns:
            Tuple of (movie_ids, distances, metadatas)
        """
        embedding = self.get_movie_embedding(movie_id)
        if embedding is None:
            return [], [], []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results + 1  # +1 because the movie itself will be included
        )

        # Filter out the query movie itself
        movie_ids = []
        distances = []
        metadatas = []

        for i, meta in enumerate(results['metadatas'][0]):
            if int(meta['movieId']) != movie_id:
                movie_ids.append(int(meta['movieId']))
                distances.append(results['distances'][0][i])
                metadatas.append(meta)

        return movie_ids[:n_results], distances[:n_results], metadatas[:n_results]

    def reset_collection(self):
        """Delete and recreate the collection (useful for reindexing)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection '{self.collection_name}'")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_movies": self.collection.count(),
            "embedding_dimension": len(self.embedding_model.encode("test")),
            "model_name": self.model_name,
            "persist_directory": self.persist_directory
        }


def main():
    """Test the vector store functionality."""
    from data_loader import load_data

    # Load movie data
    movies, _ = load_data()

    # Initialize vector store
    vector_store = MovieVectorStore()

    # Reset and index movies
    vector_store.reset_collection()
    vector_store.index_movies(movies)

    # Print statistics
    stats = vector_store.get_stats()
    print(f"\n=== Vector Store Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test semantic search
    print(f"\n=== Testing Semantic Search ===")
    query = "action-packed sci-fi adventure with heroic characters"
    movie_ids, distances, metadatas = vector_store.search_similar_movies(query, n_results=5)

    print(f"\nQuery: '{query}'")
    print("\nTop 5 Similar Movies:")
    for i, (mid, dist, meta) in enumerate(zip(movie_ids, distances, metadatas), 1):
        print(f"{i}. {meta['title']} (ID: {mid})")
        print(f"   Genres: {meta['genres']}")
        print(f"   Distance: {dist:.4f}")

    # Test movie-to-movie similarity
    print(f"\n=== Testing Movie-to-Movie Similarity ===")
    test_movie_id = 1
    similar_ids, distances, metadatas = vector_store.get_similar_to_movie(test_movie_id, n_results=5)

    print(f"\nMovies similar to Movie ID {test_movie_id}:")
    for i, (mid, dist, meta) in enumerate(zip(similar_ids, distances, metadatas), 1):
        print(f"{i}. {meta['title']} (ID: {mid})")
        print(f"   Genres: {meta['genres']}")
        print(f"   Distance: {dist:.4f}")


if __name__ == "__main__":
    main()
