# UniversalRecs
> A robust Hybrid Recommendation Engine tackling "Choice Overload" with explainable AI.

## 📖 Overview
**UniversalRecs** is a recommendation system designed to predict user preferences for unseen items. It leverages a **Hybrid Approach**, combining the strengths of Content-Based Filtering and Collaborative Filtering to provide accurate, personalized, and explainable recommendations.

The system features a **Cold Start** handler for new users and an interactive **Feedback Loop** that learns from user interactions in real-time.

## Key Features

*   **Hybrid Engine**: Combines **Content-Based** (TF-IDF + Truncated SVD) and **Collaborative Filtering** (Matrix Factorization) for robust scoring.
*   **Explainability**: Tells you *why* a recommendation was made (e.g., *"Because you liked Sci-Fi"* or *"Users like you also enjoyed this"*).
*   **Cold Start Handler**: Automatically falls back to a **Popularity-Based** model for new users with no history.
*   **Feedback Loop**: Interactive **Like/Dislike** buttons that instantly update the dataset and trigger model retraining.
*   **Evaluation Metrics**: Built-in evaluator calculating **RMSE** (Root Mean Square Error) and **Catalog Coverage** to ensure quality.
*   **Streamlit UI**: A modern, responsive dashboard with a premium dark-mode aesthetic.

## Tech Stack
*   **Language**: Python 3.x
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-learn (TF-IDF, SVD, Cosine Similarity)
*   **Frontend**: Streamlit
*   **Visualization**: Matplotlib

## Installation & Usage

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application**:
    ```bash
    python -m streamlit run app.py
    ```
4.  **Access the UI**: Open your browser at `http://localhost:8501`.

## Project Structure

```
UniversalRecs/
├── data/
│   ├── movies.csv          # Item features (ID, Title, Genres, Description)
│   └── ratings.csv         # User interactions (UserID, MovieID, Rating)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Handles data ingestion and dummy generation
│   ├── recommender.py      # Core engine (Content, Collaborative, Hybrid logic)
│   └── evaluator.py        # Metrics (RMSE, Coverage)
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Simulation
Since this is a prototype, the system includes a **Data Generator** (`data_loader.py`) that automatically instantiates synthetic "MovieLens-style" data if no files are found. This allows you to test the logic immediately without external datasets.

## Future Improvements
*   Replace dummy CSV storage with **SQLite/PostgreSQL**.
*   Implement **Deep Learning** embeddings (e.g., NeuMF) for advanced collaborative filtering.
*   Deploy as a containerized **Docker** application.
