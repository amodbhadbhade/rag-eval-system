# AI-Gurukul RAG Evaluation System

A comprehensive system for diagnosing, evaluating, and improving the performance of Retrieval-Augmented Generation (RAG) pipelines. This tool provides an API backend and an interactive Streamlit dashboard to test RAG outputs against multiple metrics using **RAGAS** and **DeepEval** frameworks.

## Features

* **Automated Dataset Generation:** Upload PDFs and automatically generate test Q&A pairs (ground truth datasets) using LLMs.
* **Multi-Framework Evaluation:** Choose between RAGAS and DeepEval for scoring.
* **Comprehensive Metrics:** Evaluate Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Similarity, and Answer Correctness.
* **Reference-Free Evaluation:** Evaluate real-world queries where ground truth is not available (measures Faithfulness and Relevancy).
* **Diagnostic Analytics:** View historical performance, metric distributions, and identify degrading configurations.
* **AI-Powered Recommendations:** Receive actionable, LLM-generated recommendations to fix failing RAG metrics (e.g., chunk size tuning, prompt fixes, top-k adjustments).
* **Flexible LLM Support:** Works with OpenAI models (GPT-4o) or local/cloud Ollama models.

## Project Structure

* `/backend`: FastAPI application handling dataset generation, evaluation logic, database interactions (MongoDB), and AI recommendations.
* `/dashboard`: Streamlit application providing the interactive user interface.

## Prerequisites

* Python 3.9+
* MongoDB (Local or Atlas instance)
* OpenAI API Key (if using OpenAI models)
* Ollama (if using local models, e.g., `llama3`, `nomic-embed-text`)

## Setup and Installation

### 1. Backend

Navigate to the backend directory, set up your virtual environment, and install dependencies:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory with your configuration. Example:
```env
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=mongodb://localhost:27017
EVALUATOR_LLM=gpt-4o
OLLAMA_MODE=False
EMBEDDING_MODEL=text-embedding-3-small
GENERATOR_LLM=gpt-4o-mini
CRITIC_LLM=gpt-4o-mini
GET_RECOMMENDATIONS_FROM_LLM=True
```

Run the backend server:
```bash
python -m uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Dashboard

Open a new terminal, navigate to the dashboard directory, set up the virtual environment, and install dependencies:

```bash
cd dashboard
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Create a `.env` file in the `dashboard/` directory if you need to point to a remote backend:
```env
BACKEND_URL=http://localhost:8000
```

Run the Streamlit dashboard:
```bash
streamlit run app.py
```
The dashboard will open in your browser at `http://localhost:8501`.
