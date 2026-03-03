"""
Embedding Generation Script for UniversalRecs
Generates and indexes movie embeddings into ChromaDB vector store.
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.vector_store import MovieVectorStore


def generate_and_index_embeddings(
    reset: bool = False,
    persist_dir: str = "./chroma_db",
    batch_size: int = 100
):
    """
    Generate embeddings for all movies and index them into ChromaDB.

    Args:
        reset: If True, delete existing collection and recreate it
        persist_dir: Directory to persist ChromaDB data
        batch_size: Number of movies to process at once
    """
    print("=" * 60)
    print("UniversalRecs - Embedding Generation & Indexing")
    print("=" * 60)

    # Step 1: Load movie data
    print("\n[1/3] Loading movie data...")
    movies, ratings = load_data()
    print(f"  ✓ Loaded {len(movies)} movies and {len(ratings)} ratings")

    # Step 2: Initialize vector store
    print("\n[2/3] Initializing vector store...")
    vector_store = MovieVectorStore(persist_directory=persist_dir)

    # Reset collection if requested
    if reset:
        print("  ! Resetting existing collection...")
        vector_store.reset_collection()

    # Check if already indexed
    current_count = vector_store.collection.count()
    if current_count > 0 and not reset:
        print(f"  ! Collection already contains {current_count} movies")
        response = input("  Do you want to re-index? (y/n): ").lower()
        if response != 'y':
            print("  Skipping indexing. Use --reset to force re-indexing.")
            return

        vector_store.reset_collection()

    # Step 3: Index movies
    print("\n[3/3] Generating embeddings and indexing movies...")
    vector_store.index_movies(movies, batch_size=batch_size)

    # Display statistics
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    stats = vector_store.get_stats()
    print(f"\nVector Store Statistics:")
    print(f"  - Collection: {stats['collection_name']}")
    print(f"  - Total Movies: {stats['total_movies']}")
    print(f"  - Embedding Dimension: {stats['embedding_dimension']}")
    print(f"  - Model: {stats['model_name']}")
    print(f"  - Storage Location: {stats['persist_directory']}")

    # Run a test query
    print("\n" + "=" * 60)
    print("Running Test Query...")
    print("=" * 60)

    test_query = "thrilling action adventure with heroic characters"
    print(f"\nQuery: '{test_query}'")

    movie_ids, distances, metadatas = vector_store.search_similar_movies(
        test_query,
        n_results=5
    )

    print("\nTop 5 Similar Movies:")
    for i, (mid, dist, meta) in enumerate(zip(movie_ids, distances, metadatas), 1):
        print(f"\n{i}. {meta['title']}")
        print(f"   Movie ID: {mid}")
        print(f"   Genres: {meta['genres']}")
        print(f"   Similarity Score: {1 - dist:.4f}")  # Convert distance to similarity
        print(f"   Description: {meta['description']}")

    print("\n" + "=" * 60)
    print("All Done! Your vector store is ready to use.")
    print("=" * 60)


def main():
    """Parse arguments and run embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate and index movie embeddings into ChromaDB"
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete existing collection and recreate it'
    )

    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='Directory to persist ChromaDB data (default: ./chroma_db)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of movies to process at once (default: 100)'
    )

    args = parser.parse_args()

    try:
        generate_and_index_embeddings(
            reset=args.reset,
            persist_dir=args.persist_dir,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
