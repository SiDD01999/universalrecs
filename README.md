# UniversalRecs
> A robust Hybrid Recommendation Engine tackling "Choice Overload" with explainable AI and Gemini-powered assistance.

## 📖 Overview
**UniversalRecs** is a recommendation system designed to predict user preferences for unseen items. It leverages a **Hybrid Approach**, combining the strengths of Content-Based Filtering and Collaborative Filtering to provide accurate, personalized, and explainable recommendations.

The system features an **AI Assistant powered by Google Gemini API**, a **Cold Start** handler for new users, and an interactive **Feedback Loop** that learns from user interactions in real-time.

## Key Features

*   **Hybrid Engine**: Combines **Content-Based** (TF-IDF + Truncated SVD) and **Collaborative Filtering** (Matrix Factorization) for robust scoring.
*   **🔍 Vector Search**: Semantic movie search using **ChromaDB** and **sentence-transformers** (384D embeddings) for content-based recommendations.
*   **🤖 Gemini AI Assistant**: An agentic chat interface built with **LangGraph** and **Google Gemini** that can search for movies and provide personalized recommendations via natural language.
*   **Explainability**: Tells you *why* a recommendation was made (e.g., *"Because you liked Movie X"* or *"Users like you also enjoyed this"*).
*   **Cold Start Handler**: Automatically falls back to a **Popularity-Based** model for new users with no history.
*   **Feedback Loop**: Interactive **Like/Dislike** buttons that instantly update the dataset and trigger model retraining.
*   **Evaluation Metrics**: Built-in evaluator calculating **RMSE** (Root Mean Square Error) and **Catalog Coverage**.
*   **Automated Testing**: Comprehensive unit tests for core logic and agent routing using `pytest`.
*   **Streamlit UI**: A modern, responsive dashboard with a premium dark-mode aesthetic and API configuration settings.

## Tech Stack
*   **LLM & Orchestration**: Google Gemini 1.5, LangChain, LangGraph
*   **Vector Store**: ChromaDB, Sentence Transformers
*   **Data Science**: Pandas, NumPy, Scikit-learn, PyTorch
*   **Frontend**: Streamlit
*   **Testing**: Pytest
*   **Environment**: Python Dotenv

## ⚙️ Installation & Usage

1.  **Clone/Download** the repository.

2.  **System Prerequisites** (Windows only):
    - Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
    - Required for PyTorch/sentence-transformers

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Vector Embeddings** (First-time setup):
    ```bash
    python scripts/generate_embeddings.py --reset
    ```
    See [VECTOR_STORE_SETUP.md](VECTOR_STORE_SETUP.md) for detailed setup guide.

5.  **Configure API Key**:
    - Create a `.env` file in the root directory.
    - Add your key: `GOOGLE_API_KEY=your_api_key_here`
    - *Alternatively, you can enter the key directly in the Streamlit Sidebar.*

6.  **Run the Application**:
    ```bash
    python -m streamlit run app.py
    ```

7.  **Access the UI**: Open your browser at `http://localhost:8501`.

## Testing
Run automated tests to verify the engine and agent logic:
```bash
python -m pytest tests/
```

## Project Structure

```
UniversalRecs/
├── data/
│   ├── movies.csv              # Item features
│   └── ratings.csv             # User interactions
├── src/
│   ├── agent.py                # LangGraph Gemini Agent
│   ├── recommender.py          # Core Engine Logic
│   ├── data_loader.py          # Data Ingestion
│   ├── evaluator.py            # Metrics
│   └── vector_store.py         # ChromaDB Vector Store
├── scripts/
│   ├── generate_embeddings.py  # Embedding Indexing Script
│   └── test_vector_store.py    # Dependency Test Script
├── tests/
│   ├── test_recommender.py     # Engine Unit Tests
│   └── test_agent.py           # Agent Routing Tests
├── chroma_db/                  # Vector Database (auto-generated)
├── app.py                      # Streamlit Entry Point
├── .env                        # API Secrets (Ignored by git)
├── README.md                   # Project Documentation
└── VECTOR_STORE_SETUP.md       # Vector Store Setup Guide
```

## Simulation
The system includes a **Data Generator** (`data_loader.py`) that automatically creates synthetic data if no files are found. This allows for immediate testing of the hybrid logic and feedback loops.

## 🔮 Roadmap & Future Improvements

### Completed
*   ✅ **Phase 1**: Vector Foundation - ChromaDB + Sentence Transformers for semantic search

### In Progress
*   🔄 **Phase 2**: Integrate vector search into recommender engine
*   🔄 **Phase 3**: Hybrid scoring with collaborative + semantic filtering

### Planned
*   📋 Replace CSV storage with **PostgreSQL**
*   📋 Implement **Deep Learning** embeddings for more complex collaborative filtering
*   📋 Multi-modal embeddings (posters, trailers, reviews)
*   📋 Deploy as a containerized **Docker** application
*   📋 A/B testing framework for recommendation strategies
