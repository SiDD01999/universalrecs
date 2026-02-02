import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from .recommender import RecommenderEngine

class Evaluator:
    def __init__(self, engine: RecommenderEngine):
        self.engine = engine
        self.ratings = engine.ratings
        
    def calculate_rmse(self):
        """Calculates RMSE for Collaborative Filtering part (User-Item Matrix reconstruction)."""
        # We compare the reconstructed matrix (from SVD) with the actual ratings.
        # Note: This is Training RMSE (simplification). proper way is Train/Test split.
        
        if self.engine.collab_user_factors is None:
            return float('nan')
            
        # Reconstruct full matrix
        reconstructed = np.dot(self.engine.collab_user_factors, self.engine.collab_item_factors)
        
        # We only care about errors on KNOWN ratings
        # Get indices of known ratings
        users = self.engine.user_item_matrix.index
        items = self.engine.user_item_matrix.columns
        
        y_true = []
        y_pred = []
        
        # Efficient way using the interactions dataframe
        for _, row in self.ratings.iterrows():
            u, m, r = row['userId'], row['movieId'], row['rating']
            if m in items and u in users:
                u_idx = users.get_loc(u)
                m_idx = items.get_loc(m)
                pred = reconstructed[u_idx, m_idx]
                y_true.append(r)
                y_pred.append(pred)
                
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def calculate_coverage(self, k=10):
        """Calculates Catalog Coverage: % of items that get recommended to at least one user."""
        all_items = set(self.engine.movies.index)
        recommended_items = set()
        
        users = self.ratings['userId'].unique()
        # Sample users if too many for speed
        if len(users) > 50:
            users = np.random.choice(users, 50, replace=False)
            
        for user_id in users:
            recs, _ = self.engine.recommend(user_id, n=k)
            for r in recs:
                recommended_items.add(r['movieId'])
                
        coverage = len(recommended_items) / len(all_items)
        return coverage

if __name__ == "__main__":
    print("Initializing Engine for Evaluation...")
    engine = RecommenderEngine()
    evaluator = Evaluator(engine)
    
    print("Calculating RMSE...")
    rmse = evaluator.calculate_rmse()
    print(f"RMSE (Training): {rmse:.4f}")
    
    print("Calculating Coverage (Top-10)...")
    cov = evaluator.calculate_coverage()
    print(f"Catalog Coverage: {cov:.2%}")
