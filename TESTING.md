# Testing Walkthrough: UniversalRecs

This guide explains how to run the application and test its core features.

## 🚀 1. Running the Application
The application is already running in the background. If you need to restart it, run:
```bash
python -m streamlit run app.py
```
**Access the UI**: Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🧪 2. Testing Scenarios

### Scenario A: New User (Cold Start)
1.  In the **Sidebar**, select **"New User"** (or ensure it's selected by default).
2.  **Observation**:
    -   You should see a message: *"Welcome! As a new user, we'll show you what's popular."*
    -   The recommendations list should display **Top Picks** based on popularity (highest weighted ratings).
    -   **Engine Mode** should display: `Popularity (New User)`.
    -   **Reason** on cards should be: `Popular Outcome`.

### Scenario B: Existing User (Personalized)
1.  In the **Sidebar**, select any existing user (e.g., **User 1**, **User 5**).
2.  **Observation**:
    -   The page updates to show personalized recommendations for that user.
    -   **Engine Mode** should display: `Hybrid`.
    -   **Reason** on cards should be either:
        -   `Because you liked [Movie]` (Content-Based influence)
        -   `Users like you also enjoyed this` (Collaborative influence)

### Scenario C: Feedback Loop (Real-time Learning)
1.  Select an **Existing User** (e.g., User 1).
2.  Find a recommended movie and click **"👍 Like"**.
3.  **Observation**:
    -   A toast notification appears: *"Liked [Movie]! Retraining..."*.
    -   The app reloads. The recommended list might change slightly as the model retrains with this new positive interaction.
4.  Find another movie and click **"👎 Dislike"**.
    -   Confirm the app updates and acknowledges the negative feedback.

### Scenario D: Evaluation Models
1.  In the right column, under **Engine Stats**, click **"Calculate Metrics"**.
2.  **Observation**:
    -   Wait for the spinner *"Evaluating..."*.
    -   **RMSE Error**: Should be a value roughly between `0.5` and `1.5` (lower is better, dependent on data).
    -   **Catalog Coverage**: Should show the percentage of items the system is capable of recommending.

## 🧪 3. Automated Testing
We use `pytest` for automated unit testing of the recommender engine and agent logic.

### Running Tests
To run all tests (ensure you are in the project root):
```bash
python -m pytest tests/
```

**Note on AI Agent Tests**:
The `test_agent.py` requires a working `torch` installation and an `OPENAI_API_KEY` (or it will use the keyword fallback). If you encounter DLL initialization errors with `torch`, ensure you have the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed.

### Test Coverage
- `test_recommender.py`: Verifies hybrid logic, feedback loops, and data loading.
- `test_agent.py`: Verifies tool-calling logic and routing.
