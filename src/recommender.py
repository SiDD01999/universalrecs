import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from .data_loader import load_data

class RecommenderEngine:
    def __init__(self):
        self.movies, self.ratings = load_data()
        self.movies.set_index('movieId', inplace=True)
        
        # Models
        self.content_sim_matrix = None
        self.collab_user_factors = None # U (User-Concept)
        self.collab_item_factors = None # Vt (Item-Concept)
        self.collab_sigma = None
        self.user_item_matrix = None
        
        # Maps
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movies.index)}
        self.idx_to_movie_id = {i: mid for i, mid in enumerate(self.movies.index)}
        
        self.train_models()

    def train_models(self):
        """Trains both Content-Based and Collaborative Filtering models."""
        print("Training Content-Based Model...")
        # 1. Content-Based: TF-IDF on Descriptions + SVD
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['description'])
        
        # SVD for Dimensionality Reduction (Latent Semantic Analysis)
        n_components_content = min(20, tfidf_matrix.shape[1] - 1)
        svd_content = TruncatedSVD(n_components=n_components_content, random_state=42)
        latent_matrix_content = svd_content.fit_transform(tfidf_matrix)
        
        # Calculate Cosine Similarity on Latent Features
        self.content_sim_matrix = cosine_similarity(latent_matrix_content)
        
        print("Training Collaborative Model...")
        # 2. Collaborative: Matrix Factorization (SVD) on User-Item Matrix
        # Drop duplicates to avoid pivot error
        ratings_unique = self.ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
        self.user_item_matrix = ratings_unique.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Only fit if we have enough data
        if not self.user_item_matrix.empty:
            X = self.user_item_matrix.values
            n_components_collab = min(10, min(X.shape) - 1)
            
            svd_collab = TruncatedSVD(n_components=n_components_collab, random_state=42)
            self.collab_user_factors = svd_collab.fit_transform(X) # User Embeddings
            self.collab_item_factors = svd_collab.components_      # Item Embeddings
        else:
            print("Warning: Not enough interaction data for Collaborative Filtering.")

    def get_popular_items(self, n=10):
        """Cold Start: Returns top rated items weighted by count."""
        # Calculate weighted rating (IMDB style or just simple mean for now)
        # Using simple mean * log(count) to boost popular items
        movie_stats = self.ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['mean', 'count']
        
        # Score = mean_rating * log(count + 1)
        movie_stats['score'] = movie_stats['mean'] * np.log(movie_stats['count'] + 1)
        top_ids = movie_stats.sort_values('score', ascending=False).head(n).index.tolist()
        
        results = []
        for mid in top_ids:
            results.append({
                'movieId': mid,
                'title': self.movies.loc[mid, 'title'],
                'genres': self.movies.loc[mid, 'genres'],
                'score': movie_stats.loc[mid, 'score'],
                'reason': 'Popular Outcome'
            })
        return results

    def recommend(self, user_id, n=10, weight_content=0.5, weight_collab=0.5):
        """Hybrid Recommendation Engine."""
        
        # 1. NEW USER CHECK
        if user_id not in self.ratings['userId'].unique():
            return self.get_popular_items(n), "Popularity (New User)"

        # 2. Collaborative Scoring
        # Predict ratings for all items for this user
        collab_scores = {}
        if self.collab_user_factors is not None:
            # Reconstruct (impute) ratings
            # Find user index in pivot table
            try:
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                user_vector = self.collab_user_factors[user_idx].reshape(1, -1)
                predicted_ratings = np.dot(user_vector, self.collab_item_factors).flatten()
                
                # Map back to movieIds
                collab_columns = self.user_item_matrix.columns
                for i, mid in enumerate(collab_columns):
                    collab_scores[mid] = predicted_ratings[i]
            except Exception as e:
                print(f"Collab error: {e}")

        # 3. Content-Based Scoring
        # Find items user liked highly (>3.5)
        user_history = self.ratings[(self.ratings['userId'] == user_id) & (self.ratings['rating'] >= 4.0)]
        liked_movies = user_history['movieId'].tolist()
        
        content_scores = {}
        explanation_sources = {} # Store which movie caused the recommendation
        
        if liked_movies:
            for liked_id in liked_movies:
                if liked_id in self.movie_id_to_idx:
                    idx = self.movie_id_to_idx[liked_id]
                    sim_scores = list(enumerate(self.content_sim_matrix[idx]))
                    
                    for i, score in sim_scores:
                        target_id = self.idx_to_movie_id[i]
                        # Accumulate similarity scores
                        if target_id not in content_scores:
                            content_scores[target_id] = 0
                            explanation_sources[target_id] = liked_id
                        content_scores[target_id] += score
                        
                        # Keep track of the strongest similarity for explanation
                        if score > 0 and (content_scores[target_id] - score < score): # Heuristic: if this new score is dominant
                             explanation_sources[target_id] = liked_id

        # 4. Hybrid Fusion
        final_scores = []
        all_movie_ids = set(self.movies.index)
        
        # Exclude items user has already seen
        seen_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        
        # Normalization helpers (simple min-max or just raw scaling)
        # SVD ratings are roughly 1-5. Cosine is 0-1 (accumulated could be higher).
        # We'll normalize roughly to 0-1 range for combination.
        
        max_collab = max(collab_scores.values()) if collab_scores else 1.0
        max_content = max(content_scores.values()) if content_scores else 1.0
        
        for mid in all_movie_ids:
            if mid in seen_movies:
                continue
                
            s_content = content_scores.get(mid, 0.0) / max_content if max_content > 0 else 0
            s_collab = collab_scores.get(mid, 0.0) / max_collab if max_collab > 0 else 0
            
            # Weighted Hybrid Score
            final_score = (s_content * weight_content) + (s_collab * weight_collab)
            
            # Determine Explanation
            reason = ""
            if s_content > s_collab:
                source_id = explanation_sources.get(mid)
                source_title = self.movies.loc[source_id, 'title'] if source_id else "movies you liked"
                reason = f"Because you liked {source_title}"
            else:
                 reason = "Users like you also enjoyed this"

            final_scores.append({
                'movieId': mid,
                'title': self.movies.loc[mid, 'title'],
                'genres': self.movies.loc[mid, 'genres'],
                'score': final_score,
                'reason': reason
            })
            
        # Sort and return
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        return final_scores[:n], "Hybrid"

    def add_feedback(self, user_id, movie_id, rating):
        """Adds a new interaction and retrains models."""
        # 1. Update in-memory
        new_row = {'userId': user_id, 'movieId': movie_id, 'rating': rating, 'timestamp': int(pd.Timestamp.now().timestamp())}
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        
        # 2. Update file (append mode would be faster but for safety we rewrite or append)
        # We'll just append to the csv
        # Re-importing inside method to be safe or assuming standard path.
        from .data_loader import RATINGS_FILE
        pd.DataFrame([new_row]).to_csv(RATINGS_FILE, mode='a', header=False, index=False)
        
        print(f"Feedback added: User {user_id} -> Item {movie_id} ({rating}*)")
        
        # 3. Retrain
        self.train_models()

    def search_items(self, query: str, n: int = 5) -> list:
        """
        Simple text search for items based on title and genres.
        Returns a list of dictionaries with 'movieId', 'title', 'genres'.
        """
        # Case-insensitive search
        mask = self.movies['title'].str.contains(query, case=False, na=False) | \
               self.movies['genres'].str.contains(query, case=False, na=False) | \
               self.movies['description'].str.contains(query, case=False, na=False)
        
        results = self.movies[mask].head(n)
        
        # Format output
        output = []
        for mid, row in results.iterrows():
            output.append({
                'movieId': mid,
                'title': row['title'],
                'genres': row['genres'],
                'score': 1.0, # Dummy score for search match
                'reason': f"Matched search query: '{query}'"
            })
        return output


if __name__ == "__main__":
    engine = RecommenderEngine()
    print("Engine trained.")
    
    # Test User 1
    recs, method = engine.recommend(1)
    print(f"\nRecommendations for User 1 ({method}):")
    for r in recs:
        print(f"- {r['title']} ({r['score']:.2f}) [{r['reason']}]")
