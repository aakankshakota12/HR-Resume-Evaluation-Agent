# src/retrieval/search.py
from .resume_parser import load_resumes
from .embeddings import EmbeddingModel
from .faiss_index import ResumeIndex
import os

# Initialize global instances (lazy loading recommended in production)
embedder = EmbeddingModel()
vector_db = ResumeIndex()

def ingest_resumes(resume_folder_path):
    """
    Full pipeline: Parse PDF -> Clean -> Embed -> Index
    """
    # 1. Parse
    raw_data = load_resumes(resume_folder_path)
    if not raw_data:
        print("No resumes found to ingest.")
        return

    # 2. Extract text for embedding
    texts = [item['text'] for item in raw_data]

    # 3. Generate Embeddings
    vectors = embedder.get_embeddings(texts)

    # 4. Add to Index
    vector_db.add_resumes(vectors, raw_data)
    
    # Optional: Save to disk so you don't re-parse every time
    # vector_db.save_index()

def query_resumes(query_text, top_k=3):
    """
    Takes a query (e.g., "Python developer with AI skills") 
    and returns top matching resumes.
    """
    # 1. Embed query
    query_vec = embedder.get_query_embedding(query_text)
    
    # 2. Search FAISS
    results = vector_db.search(query_vec, k=top_k)
    
    return results

# Quick test block to verify it works when running this file directly
if __name__ == "__main__":
    # Point this to your actual folder
    TEST_RESUME_FOLDER = "Resume" 
    
    # Run ingestion
    ingest_resumes(TEST_RESUME_FOLDER)
    
    # Run a test search
    user_query = "machine learning engineer with python"
    matches = query_resumes(user_query)
    
    print(f"\nQuery: {user_query}")
    print("-" * 30)
    for match in matches:
        print(f"File: {match['resume'].get('id')}")
        print(f"Score: {match['score']:.4f}")
        print(f"Snippet: {match['resume'].get('text')[:100]}...")
        print("-" * 10)