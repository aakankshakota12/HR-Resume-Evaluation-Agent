# src/retrieval/search.py
from .resume_parser import load_resumes
from .embeddings import EmbeddingModel
from .faiss_index import ResumeIndex
import os

# Initialize components
embedder = EmbeddingModel()
vector_db = ResumeIndex()

# FIX: Try to load existing index immediately
vector_db.load_index()

def ingest_resumes(resume_folder_path):
    """
    Full pipeline: Parse -> Embed -> Index -> Save
    """
    raw_data = load_resumes(resume_folder_path)
    if not raw_data:
        print("No resumes found to ingest.")
        return

    # Check for duplicates (simple check based on ID)
    # (Optional: In production you would filter out IDs that already exist in metadata_map)
    
    texts = [item['text'] for item in raw_data]
    
    print("Generating embeddings...")
    vectors = embedder.get_embeddings(texts)

    vector_db.add_resumes(vectors, raw_data)
    
    # FIX: Save index immediately after adding
    vector_db.save_index()

def query_resumes(query_text, top_k=3):
    query_vec = embedder.get_query_embedding(query_text)
    results = vector_db.search(query_vec, k=top_k)
    return results

if __name__ == "__main__":
    TEST_RESUME_FOLDER = "Resume" 
    
    # Run ingestion (this will now save to disk)
    ingest_resumes(TEST_RESUME_FOLDER)
    
    user_query = "machine learning engineer with python"
    matches = query_resumes(user_query)
    
    print(f"\nQuery: {user_query}")
    print("-" * 30)
    for match in matches:
        print(f"File ID: {match['resume'].get('id')}") # Now using Hash ID
        print(f"Filename: {match['resume'].get('filename')}")
        print(f"Score: {match['score']:.4f}") # Now Cosine Similarity (Higher is better)
        print("-" * 10)