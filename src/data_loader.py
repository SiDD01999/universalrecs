import pandas as pd
import numpy as np
import os
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MOVIES_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'ratings.csv')

def create_dummy_data():
    """Generates synthetic data compatible with MovieLens schema."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Generate Movies (Items)
    # Schema: movieId, title, genres, description
    genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Sci-Fi', 'Thriller']
    
    n_movies = 100
    movies_data = []
    
    for i in range(1, n_movies + 1):
        # Pick 1-3 random genres
        movie_genres = np.random.choice(genres_list, size=np.random.randint(1, 4), replace=False)
        genres_str = "|".join(movie_genres)
        
        # Simple synthetic title
        title = f"Movie {i} ({2000 + (i % 23)})"
        
        # synthetic description for TF-IDF
        # mixing genres into a sentence
        description = f"A {movie_genres[0].lower()} movie about {('heroic ' if 'Action' in movie_genres else 'complex ')} characters in a {'futuristic' if 'Sci-Fi' in movie_genres else 'modern'} world."
        
        movies_data.append([i, title, genres_str, description])
        
    df_movies = pd.DataFrame(movies_data, columns=['movieId', 'title', 'genres', 'description'])
    df_movies.to_csv(MOVIES_FILE, index=False)
    print(f"Created {MOVIES_FILE} with {n_movies} items.")

    # 2. Generate Ratings (Interactions)
    # Schema: userId, movieId, rating, timestamp
    n_users = 20
    n_interactions = 500
    
    ratings_data = []
    
    # Ensure every user rates at least a few movies, and every movie has at least one rating (mostly)
    users = range(1, n_users + 1)
    movie_ids = df_movies['movieId'].values
    
    for _ in range(n_interactions):
        u = np.random.choice(users)
        m = np.random.choice(movie_ids)
        r = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.05, 0.1, 0.25, 0.35, 0.25]) # skewed towards positive
        timestamp = 1609459200 + np.random.randint(0, 31536000) # Random time in 2021
        
        ratings_data.append([u, m, r, timestamp])
        
    df_ratings = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating', 'timestamp'])
    # Remove duplicates (user rating same movie twice)
    df_ratings = df_ratings.drop_duplicates(subset=['userId', 'movieId'])
    
    df_ratings.to_csv(RATINGS_FILE, index=False)
    print(f"Created {RATINGS_FILE} with {len(df_ratings)} interactions.")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads movies and ratings data, generating it if necessary."""
    if not os.path.exists(MOVIES_FILE) or not os.path.exists(RATINGS_FILE):
        print("Data files not found. Generating dummy data...")
        create_dummy_data()
        
    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    
    return movies, ratings

if __name__ == "__main__":
    # Test data generation
    m, r = load_data()
    print("Movies head:")
    print(m.head())
    print("\nRatings head:")
    print(r.head())
