# Doctor-Patient Matching System

A web application that matches patients with doctors based on symptoms using Mistral AI embeddings and ChromaDB vector search.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys and configuration:
     - `MISTRAL_API_KEY`: Your Mistral AI API key
     - `CHROMA_API_KEY`: Your ChromaDB API key
     - `CHROMA_TENANT`: Your ChromaDB tenant ID
     - Other configuration values (optional, defaults provided)

3. Run the application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Features

- **Symptom-based matching**: Enter patient symptoms and find the best matching doctor
- **Vector similarity search**: Uses Mistral embeddings to find semantically similar doctors
- **AI recommendations**: Get AI-powered explanations for doctor recommendations
- **Multiple results**: View top-k matching doctors with similarity scores
- **Visual match scores**: Color-coded match indicators

## How it works

1. Patient enters symptoms
2. Symptoms are converted to embeddings using Mistral Embed API
3. Vector similarity search in ChromaDB finds matching doctors
4. Results are ranked by similarity score
5. AI generates recommendation explanation (optional)

## Configuration

API keys and database credentials are stored in the `.env` file (not committed to git for security). 
- Copy `.env.example` to `.env` and fill in your actual credentials
- The `.env` file is automatically ignored by git (see `.gitignore`)
- Required variables: `MISTRAL_API_KEY`, `CHROMA_API_KEY`, `CHROMA_TENANT`
- Optional variables have defaults: `CHROMA_DATABASE`, `COLLECTION_NAME`, `EMBEDDING_MODEL`, `CHAT_MODEL`


